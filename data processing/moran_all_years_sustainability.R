# =============================================================================
#  moran_all_years.R
#  Computes Global Moran's I AND Local Moran's I (LISA) for all seven
#  sustainability indicators, once per year (2021, 2022, 2023).
#
#  Combines the logic of your two existing scripts:
#   - Global Moran's I:  moran.mc() + moran.test(), 999 permutations   (script 1)
#   - Local Moran's I:   localmoran_perm(), 999 permutations, HH/LL/HL/LH/ns  (script 2)
#
#  Input:  PREP/2021.csv, PREP/2022.csv, PREP/2023.csv
#          (already-wrangled panel files — one row per borough per year,
#           columns: borough, no2_annual_mean, gas_kwh_per_consumer,
#           energy_kwh_per_capita, ghg_tco2e_per_capita,
#           transport_kwh_per_capita, crime_per_1k, waste_tonnes_per_capita)
#
#  Output: figures/global_moran_results_<year>.csv   (one per year)
#          figures/global_moran_results_all_years.csv (stacked, all years)
#          figures/local_moran_results_<year>.csv    (one per year, all 7 indicators)
#          figures/local_moran_results_all_years.csv (stacked, all years)
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
PREP        <- "sustainability/dataset"   # folder holding 2021.csv, 2022.csv, 2023.csv
YEARS       <- c("2021", "2022", "2023")
OUT_DIR     <- "figures"
NSIM        <- 999                            # permutations for both global and local tests
P_THRESH    <- 0.10                           # LISA significance threshold (exploratory)

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# Indicator display name -> column name in the panel CSVs
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

# ── Borough name cleaner (same convention as your app / wrangle script) ───
clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Load borough boundaries once (shared across all years) ────────────────
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

# ── Build spatial weights matrix once (Queen contiguity, row-standardised) ─
nb_queen <- poly2nb(boroughs_sf, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

cat("Neighbour summary:\n")
print(summary(nb_queen))

# ── Local Moran's I (LISA) helper — identical logic to your script 2,
#    generalised to accept any indicator column ─────────────────────────
compute_lisa <- function(sf_obj, col, lw, nsim = NSIM, p_thresh = P_THRESH) {
  vals     <- sf_obj[[col]]
  missing  <- is.na(vals)
  vals_imp <- vals
  vals_imp[missing] <- mean(vals, na.rm = TRUE)

  lisa  <- localmoran_perm(vals_imp, lw, nsim = nsim,
                            zero.policy = TRUE, alternative = "two.sided")
  z     <- scale(vals_imp)[, 1]
  lag_z <- lag.listw(lw, z, zero.policy = TRUE)
  p_col <- grep("Pr\\(z != E\\(Ii\\)\\)", colnames(lisa), value = TRUE)
  p_val <- lisa[, p_col[length(p_col)]]

  cluster <- case_when(
    missing            ~ "No data",
    p_val >= p_thresh  ~ "ns",
    z > 0 & lag_z > 0  ~ "HH",
    z < 0 & lag_z < 0  ~ "LL",
    z > 0 & lag_z < 0  ~ "HL",
    z < 0 & lag_z > 0  ~ "LH",
    TRUE               ~ "ns"
  )

  tibble(
    borough = sf_obj$borough,
    z_score = round(z, 4),
    lag_z   = round(lag_z, 4),
    Ii      = round(lisa[, "Ii"], 4),
    p_value = round(p_val, 4),
    cluster = cluster
  )
}

# ── Global Moran's I helper — identical logic to your script 1,
#    generalised to accept any indicator column ────────────────────────
compute_global_moran <- function(vals, lw, nsim = NSIM) {
  n_missing <- sum(is.na(vals))
  if (n_missing > 0) {
    vals[is.na(vals)] <- mean(vals, na.rm = TRUE)
  }

  moran_mc  <- moran.mc(x = vals, listw = lw, nsim = nsim, zero.policy = TRUE)
  moran_std <- moran.test(x = vals, listw = lw, zero.policy = TRUE)

  tibble(
    Moran_I        = round(moran_mc$statistic, 4),
    Expected_I     = round(moran_std$estimate["Expectation"], 4),
    Variance       = round(moran_std$estimate["Variance"], 6),
    p_value_MC     = round(moran_mc$p.value, 4),
    p_value_std    = round(moran_std$p.value, 4),
    n_missing      = n_missing,
    Significant    = if_else(moran_mc$p.value < 0.05, "Yes", "No"),
    Interpretation = case_when(
      moran_mc$statistic >  0.3 & moran_mc$p.value < 0.05 ~ "Strong clustering",
      moran_mc$statistic >  0.1 & moran_mc$p.value < 0.05 ~ "Moderate clustering",
      moran_mc$statistic >  0   & moran_mc$p.value < 0.05 ~ "Weak clustering",
      moran_mc$statistic <= 0                              ~ "Dispersed/Random",
      TRUE                                                 ~ "Not significant"
    )
  )
}

# ── Main loop: one pass per year, all 7 indicators computed together ──────
global_all_years <- list()
local_all_years  <- list()

for (yr in YEARS) {

  cat("\n============================================================\n")
  cat(" YEAR:", yr, "\n")
  cat("============================================================\n")

  panel_path <- file.path(PREP, paste0(yr, ".csv"))
  panel_yr   <- read_csv(panel_path, show_col_types = FALSE) %>%
    mutate(borough = clean_borough(borough))

  # Join this year's indicator values onto the borough boundaries
  spatial_df <- boroughs_sf %>%
    left_join(panel_yr %>% select(borough, all_of(unname(IND_TO_COL))),
              by = "borough")

  cat("Spatial df rows:", nrow(spatial_df), "\n")

  # --- Global Moran's I for all 7 indicators, this year ------------------
  global_yr <- lapply(indicators, function(ind) {
    col  <- IND_TO_COL[[ind]]
    vals <- as.numeric(spatial_df[[col]])
    res  <- compute_global_moran(vals, lw, nsim = NSIM)
    res %>% mutate(year = yr, Indicator = ind, .before = 1)
  }) %>% bind_rows() %>% arrange(desc(Moran_I))

  cat("\n--- Global Moran's I,", yr, "---\n")
  print(global_yr, n = 7)

  write.csv(global_yr,
            file.path(OUT_DIR, paste0("global_moran_results_", yr, ".csv")),
            row.names = FALSE)

  global_all_years[[yr]] <- global_yr

  # --- Local Moran's I (LISA) for all 7 indicators, this year ------------
  local_yr <- lapply(indicators, function(ind) {
    col <- IND_TO_COL[[ind]]
    res <- compute_lisa(spatial_df, col, lw, nsim = NSIM, p_thresh = P_THRESH)
    res %>% mutate(year = yr, Indicator = ind, .before = 1)
  }) %>% bind_rows()

  cat("\n--- Local Moran's I (LISA) cluster counts,", yr, "---\n")
  print(local_yr %>% count(Indicator, cluster) %>% pivot_wider(names_from = cluster, values_from = n, values_fill = 0))

  write.csv(local_yr,
            file.path(OUT_DIR, paste0("local_moran_results_", yr, ".csv")),
            row.names = FALSE)

  local_all_years[[yr]] <- local_yr
}

# ── Stack all years into two combined CSVs ─────────────────────────────────
global_moran_all <- bind_rows(global_all_years)
local_moran_all  <- bind_rows(local_all_years)

write.csv(global_moran_all,
          file.path(OUT_DIR, "global_moran_results_all_years.csv"),
          row.names = FALSE)
write.csv(local_moran_all,
          file.path(OUT_DIR, "local_moran_results_all_years.csv"),
          row.names = FALSE)

adj_pairs <- do.call(rbind, lapply(seq_along(nb_queen), function(i) {
  if (nb_queen[[i]][1] != 0) {
    data.frame(borough = boroughs_sf$borough[i],
               neighbor = boroughs_sf$borough[nb_queen[[i]]])
  }
}))
write.csv(adj_pairs, "borough_adjacency.csv", row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved per-year and combined CSVs to:", OUT_DIR, "\n")
cat("============================================================\n")
cat(" - global_moran_results_<year>.csv        (7 indicators x 1 row each)\n")
cat(" - global_moran_results_all_years.csv      (7 indicators x 3 years = 21 rows)\n")
cat(" - local_moran_results_<year>.csv          (7 indicators x 33 boroughs each)\n")
cat(" - local_moran_results_all_years.csv       (7 indicators x 33 boroughs x 3 years)\n")
