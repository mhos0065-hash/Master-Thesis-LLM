# =============================================================================
#  moran_no2_crime.R
#  ADDITIONAL Spatial Correlation Detection ground truth, built from the
#  quarterly NO2 and half-yearly crime long-format files generated
#  earlier this session (no2_quarterly.csv, crime_halfyearly.csv).
#
#  Identical method to moran_all_years.R: real spdep::moran.mc() +
#  moran.test() for Global Moran's I, real spdep::localmoran_perm() for
#  Local Moran's I (LISA), same Queen contiguity weights, same 999
#  permutations, same mean-imputation-with-reported-count convention for
#  missing boroughs, same HH/LL/HL/LH/ns classification logic. The ONLY
#  change is the input: instead of one row per (year, indicator) from
#  wide annual panels, this loops over 34 individual PERIODS (12 NO2
#  quarters + 22 crime half-years) from the two long-format files, each
#  period treated the way one (year, indicator) combination was treated
#  in the original script.
#
#  BOTH Global and Local Moran's I are computed and saved, since which
#  one "Spatial Correlation Detection" actually uses is not yet
#  confirmed -- this way no re-run is needed once that's clarified;
#  whichever file matches the existing manifest's ground_truth
#  structure is the one to use for manifest2 generation.
#
#  Input:  no2_quarterly.csv     (time_period, borough, avg_no2)
#          crime_halfyearly.csv  (time, Borough, population, crime_rate)
#
#  Output: figures/global_moran_no2_crime_all_periods.csv
#          (34 rows: one per period, matching global_moran_results_all_years.csv's shape)
#          figures/local_moran_no2_crime_all_periods.csv
#          (34 periods x 33 boroughs, matching local_moran_results_all_years.csv's shape)
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

## ============================ CONFIG ====================================
NO2_PATH    <- "sustainability/dataset/additional test cases/no2_quarterly.csv"
CRIME_PATH  <- "sustainability/dataset/additional test cases/crime_halfyearly.csv"
OUT_DIR     <- "figures"
NSIM        <- 999
P_THRESH    <- 0.10   # LISA significance threshold, same as moran_all_years.R
## ==========================================================================

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

## ── Load borough boundaries + standard Queen contiguity weights
##    (identical source/setup to moran_all_years.R) ──────────────────────
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

## ── Global Moran's I helper -- IDENTICAL logic to moran_all_years.R's
##    compute_global_moran(), unchanged ──────────────────────────────────
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

## ── Local Moran's I (LISA) helper -- IDENTICAL logic to
##    moran_all_years.R's compute_lisa(), unchanged ─────────────────────
compute_lisa <- function(sf_obj, vals, lw, nsim = NSIM, p_thresh = P_THRESH) {
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

## ── Build a (borough -> value) vector, aligned to boroughs_sf order,
##    for one period from one of the two long-format source files ──────
values_for_period <- function(df, period_col, period_val, value_col, borough_col) {
  sub <- df %>% filter(.data[[period_col]] == period_val) %>%
    select(borough = all_of(borough_col), value = all_of(value_col))
  tibble(borough = boroughs_sf$borough) %>%
    left_join(sub, by = "borough") %>%
    pull(value)
}

global_results <- list()
local_results  <- list()

## ── NO2: all 12 quarters, in chronological order ────────────────────────
cat("\n============================================================\n")
cat(" NO2 (quarterly, 12 periods)\n")
cat("============================================================\n")

no2_raw <- read_csv(NO2_PATH, show_col_types = FALSE) %>%
  mutate(borough = clean_borough(borough))

no2_periods <- no2_raw %>%
  distinct(time_period) %>%
  mutate(
    yr = as.integer(str_extract(time_period, "\\d{4}")),
    q  = as.integer(str_extract(time_period, "(?<=Q)\\d"))
  ) %>%
  arrange(yr, q) %>%
  pull(time_period)

for (p in no2_periods) {
  vals <- values_for_period(no2_raw, "time_period", p, "avg_no2", "borough")

  g <- compute_global_moran(vals, lw, nsim = NSIM) %>%
    mutate(period = p, Indicator = "NO2", .before = 1)
  cat(sprintf("  [Global] %s : Moran's I = %.4f, p = %.4f, %s\n",
              p, g$Moran_I, g$p_value_MC, g$Interpretation))
  global_results[[length(global_results) + 1]] <- g

  l <- compute_lisa(boroughs_sf, vals, lw, nsim = NSIM, p_thresh = P_THRESH) %>%
    mutate(period = p, Indicator = "NO2", .before = 1)
  local_results[[length(local_results) + 1]] <- l
}

## ── Crime: all 22 half-years, in chronological order ───────────────────
cat("\n============================================================\n")
cat(" Crime (half-yearly, 22 periods)\n")
cat("============================================================\n")

crime_raw <- read_csv(CRIME_PATH, show_col_types = FALSE) %>%
  mutate(Borough = clean_borough(Borough))

crime_periods <- crime_raw %>%
  distinct(time) %>%
  mutate(
    yr   = as.integer(str_extract(time, "\\d{4}")),
    half = if_else(str_starts(time, "first"), 1L, 2L)
  ) %>%
  arrange(yr, half) %>%
  pull(time)

for (p in crime_periods) {
  vals <- values_for_period(crime_raw, "time", p, "crime_rate", "Borough")

  g <- compute_global_moran(vals, lw, nsim = NSIM) %>%
    mutate(period = p, Indicator = "crime", .before = 1)
  cat(sprintf("  [Global] %s : Moran's I = %.4f, p = %.4f, %s\n",
              p, g$Moran_I, g$p_value_MC, g$Interpretation))
  global_results[[length(global_results) + 1]] <- g

  l <- compute_lisa(boroughs_sf, vals, lw, nsim = NSIM, p_thresh = P_THRESH) %>%
    mutate(period = p, Indicator = "crime", .before = 1)
  local_results[[length(local_results) + 1]] <- l
}

## ── Save both outputs ────────────────────────────────────────────────
global_moran_no2_crime <- bind_rows(global_results)
local_moran_no2_crime  <- bind_rows(local_results)

write.csv(global_moran_no2_crime,
          file.path(OUT_DIR, "global_moran_no2_crime_all_periods.csv"),
          row.names = FALSE)
write.csv(local_moran_no2_crime,
          file.path(OUT_DIR, "local_moran_no2_crime_all_periods.csv"),
          row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" -", file.path(OUT_DIR, "global_moran_no2_crime_all_periods.csv"),
    sprintf("  (%d periods: %d NO2 + %d crime)\n",
            length(no2_periods) + length(crime_periods), length(no2_periods), length(crime_periods)))
cat(" -", file.path(OUT_DIR, "local_moran_no2_crime_all_periods.csv"),
    sprintf("  (%d periods x 33 boroughs = %d rows)\n",
            length(no2_periods) + length(crime_periods),
            (length(no2_periods) + length(crime_periods)) * 33))
cat(" BOTH global and local are saved -- use whichever matches the existing\n")
cat(" Spatial Correlation Detection manifest's ground-truth structure.\n")
cat("============================================================\n")