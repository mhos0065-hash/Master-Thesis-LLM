# =============================================================================
#  spatial_outlier_iterative_r_all_years.R
#  NEW TASK 4 (harder version): Iterative-r spatial outlier detection
#  (Lu, C.T., Chen, D., & Kou, Y. (2003). "Algorithms for spatial outlier
#  detection." Proceedings of the 3rd IEEE International Conference on
#  Data Mining (ICDM), 597-600.)
#
#  This SUPERSEDES spatial_outlier_z_all_years.R (the simple neighbor-
#  difference z-test) -- that method is too easy for LLMs to get right
#  (single subtraction + threshold), giving no discriminating power
#  between models. This version fixes the "masking" problem the simple
#  method suffers from: a true outlier can distort the very neighborhood
#  average used to detect it, hiding itself (or hiding/falsely-flagging
#  OTHER nearby locations -- "swamping"). Iterative-r fixes this by
#  fitting a regression, excluding likely outliers, then refitting.
#
#  Method (fixed at exactly ONE refit round, deliberately, for fair
#  comparison across models -- the original paper allows further rounds
#  until convergence, but a fixed single round keeps every LLM solving
#  the identical, unambiguous problem):
#    1. For each borough, compute its own value AND its neighbors'
#       average value (the spatial lag).
#    2. Fit OLS regression: own_value ~ neighbor_avg, using ALL boroughs.
#    3. Compute residuals, standardize into z-scores (pass 1).
#    4. Flag any borough with |z| > 1.96 in pass 1 as a CANDIDATE outlier.
#    5. Refit the SAME regression, this time EXCLUDING the candidates
#       flagged in step 4 -- this is the key step that avoids masking.
#    6. Apply this refined line to ALL boroughs (including the excluded
#       candidates) to get FINAL predicted values and residuals.
#    7. Standardize the final residuals into z-scores and classify:
#       |z| > 2.576 -> outlier (99%), |z| > 1.96 -> outlier (95%),
#       otherwise not significant.
#
#  Both pass-1 (naive) and final (post-refit) z-scores are saved, so you
#  can directly show real cases in your data where the refit CHANGED the
#  classification -- concrete evidence the masking correction matters.
#
#  SAFE FOR YOUR EXISTING PIPELINE: reads the same borough boundaries as
#  your other scripts (read-only), writes to NEW, distinctly-named output
#  files. Does not touch global_moran_*, local_moran_*, getisord_*,
#  bivariate_*, spatial_outlier_z_* (the simpler version), or
#  borough_adjacency.csv.
#
#  Input:  PREP/2021.csv, PREP/2022.csv, PREP/2023.csv (only 2023.csv
#          provided so far; add earlier years the same way, or trim
#          YEARS below to just "2023")
#
#  Output: figures/spatial_outlier_iterative_r_<year>.csv    (one per year)
#          figures/spatial_outlier_iterative_r_all_years.csv (stacked, all years)
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
PREP         <- "sustainability/dataset"
YEARS        <- c("2021", "2022", "2023")   # trim to c("2023") if that's all you have right now
OUT_DIR      <- "figures"
Z_CANDIDATE  <- 1.96   # threshold for flagging a PASS-1 candidate to exclude before refitting
Z_THRESH_95  <- 1.96   # final significance threshold, 95%
Z_THRESH_99  <- 2.576  # final significance threshold, 99%

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

IND_TO_COL <- c(
  NO2       = "no2_annual_mean",
  Gas       = "gas_kwh_per_consumer",
  Energy    = "energy_kwh_per_capita",
  GHG       = "ghg_tco2e_per_capita",
  Transport = "transport_kwh_per_capita",
  Crime     = "crime_per_1k",
  Waste     = "waste_tonnes_per_capita"
)
indicators <- names(IND_TO_COL)

clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Load borough boundaries + standard Queen contiguity weights ───────────
boroughs_sf_raw <- st_read(
  "https://github.com/LingruFeng/GIS_assessment/blob/main/London_Boroughs.gpkg?raw=true",
  quiet = TRUE
)

boroughs_sf <- boroughs_sf_raw %>%
  st_transform(4326) %>%
  rename(geometry = geom) %>%
  st_set_geometry("geometry") %>%
  mutate(borough = clean_borough(name)) %>%
  select(borough, geometry)

nb_queen <- poly2nb(boroughs_sf, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

cat("Neighbour summary:\n")
print(summary(nb_queen))

# ── Iterative-r spatial outlier helper ─────────────────────────────────
compute_iterative_r_outlier <- function(sf_obj, col, lw) {
  vals    <- sf_obj[[col]]
  missing <- is.na(vals)
  vals_imp <- vals
  vals_imp[missing] <- mean(vals, na.rm = TRUE)

  neighbor_avg <- lag.listw(lw, vals_imp, zero.policy = TRUE)

  # --- pass 1: fit on everyone ---
  fit1 <- lm(vals_imp ~ neighbor_avg)
  resid1 <- residuals(fit1)
  z1 <- as.numeric(scale(resid1))

  # --- identify candidates to exclude before refitting ---
  candidate <- abs(z1) > Z_CANDIDATE & !missing

  # --- refit EXCLUDING candidates (this is the step that fixes masking) ---
  if (sum(!candidate) >= 3) {   # need enough points left to fit a line
    fit2 <- lm(vals_imp[!candidate] ~ neighbor_avg[!candidate])
    intercept2 <- coef(fit2)[1]
    slope2     <- coef(fit2)[2]
  } else {
    # fallback: if almost everyone got flagged, keep the pass-1 line
    intercept2 <- coef(fit1)[1]
    slope2     <- coef(fit1)[2]
  }

  # --- apply refined line to EVERYONE (including excluded candidates) ---
  predicted_final <- intercept2 + slope2 * neighbor_avg
  residual_final  <- vals_imp - predicted_final
  z_final <- as.numeric(scale(residual_final))

  outlier_type <- case_when(
    missing               ~ "No data",
    z_final >  Z_THRESH_99 ~ "Outlier - high (99%)",
    z_final >  Z_THRESH_95 ~ "Outlier - high (95%)",
    z_final < -Z_THRESH_99 ~ "Outlier - low (99%)",
    z_final < -Z_THRESH_95 ~ "Outlier - low (95%)",
    TRUE                    ~ "Not significant"
  )

  # flag cases where the refit actually changed the classification --
  # concrete evidence of masking/swamping correction in your real data
  was_flagged_pass1  <- abs(z1) > Z_THRESH_95 & !missing
  is_flagged_final    <- outlier_type %in% c("Outlier - high (99%)", "Outlier - high (95%)",
                                              "Outlier - low (99%)", "Outlier - low (95%)")
  masking_correction <- case_when(
    missing ~ "No data",
    was_flagged_pass1 & !is_flagged_final ~ "Unmasked by refit (was flagged, now not)",
    !was_flagged_pass1 & is_flagged_final ~ "Revealed by refit (was NOT flagged, now is)",
    TRUE ~ "No change"
  )

  tibble(
    borough              = sf_obj$borough,
    own_value            = round(vals_imp, 4),
    neighbor_avg         = round(neighbor_avg, 4),
    z_score_pass1        = round(z1, 4),
    z_score_final        = round(z_final, 4),
    outlier_type         = outlier_type,
    masking_correction   = masking_correction
  )
}

# ── Main loop: one pass per year, all 7 indicators ─────────────────────
outlier_all_years <- list()

for (yr in YEARS) {

  panel_path <- file.path(PREP, paste0(yr, ".csv"))
  if (!file.exists(panel_path)) {
    cat("\nSkipping", yr, "-- file not found:", panel_path, "\n")
    next
  }

  cat("\n============================================================\n")
  cat(" YEAR:", yr, "\n")
  cat("============================================================\n")

  panel_yr <- read_csv(panel_path, show_col_types = FALSE) %>%
    mutate(borough = clean_borough(borough))

  spatial_df <- boroughs_sf %>%
    left_join(panel_yr %>% select(borough, all_of(unname(IND_TO_COL))), by = "borough")

  outlier_yr <- lapply(indicators, function(ind) {
    col <- IND_TO_COL[[ind]]
    res <- compute_iterative_r_outlier(spatial_df, col, lw)
    res %>% mutate(year = yr, Indicator = ind, .before = 1)
  }) %>% bind_rows()

  cat("\n--- Iterative-r outlier counts,", yr, "---\n")
  print(outlier_yr %>% count(Indicator, outlier_type) %>%
          pivot_wider(names_from = outlier_type, values_from = n, values_fill = 0))

  cat("\n--- Masking correction cases,", yr, "(where the refit changed the answer) ---\n")
  print(outlier_yr %>% filter(masking_correction != "No change", masking_correction != "No data") %>%
          select(Indicator, borough, z_score_pass1, z_score_final, masking_correction))

  write.csv(outlier_yr, file.path(OUT_DIR, paste0("spatial_outlier_iterative_r_", yr, ".csv")), row.names = FALSE)
  outlier_all_years[[yr]] <- outlier_yr
}

outlier_all <- bind_rows(outlier_all_years)
write.csv(outlier_all, file.path(OUT_DIR, "spatial_outlier_iterative_r_all_years.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" - figures/spatial_outlier_iterative_r_<year>.csv        (7 indicators x 33 boroughs each)\n")
cat(" - figures/spatial_outlier_iterative_r_all_years.csv      (7 indicators x 33 boroughs x N years)\n")
cat("============================================================\n")
