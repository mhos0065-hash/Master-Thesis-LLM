# =============================================================================
#  spatial_change_spillover_no2_crime.R
#  ADDITIONAL Change Clustering Detection ground truth, built from the
#  quarterly NO2 and half-yearly crime long-format files generated
#  earlier this session (no2_quarterly.csv, crime_halfyearly.csv).
#
#  Identical method to spatial_change_spillover_all_years.R: real
#  spdep::moran.test() on delta_value = value[period_b] - value[period_a],
#  same Queen contiguity weights, same mean-imputation-with-reported-count
#  convention for missing boroughs. The ONLY change is the input: instead
#  of annual year-pairs from wide per-year panels, this uses CONSECUTIVE
#  time periods within each long-format file, at each dataset's own
#  native granularity (quarter-over-quarter for NO2, half-year-over-
#  half-year for crime) -- directly parallel to your existing
#  consecutive-year-pair design, just at finer resolution.
#
#  Input:  no2_quarterly.csv     (time_period, borough, avg_no2)
#          crime_halfyearly.csv  (time, Borough, population, crime_rate)
#
#  Output: figures/spatial_change_spillover_no2_crime_<pair>.csv
#          (per-borough delta values, for transparency)
#          figures/spatial_change_spillover_no2_crime_all_pairs.csv
#          (stacked summary: one row per consecutive period-pair, with
#          the real computed Moran's I, p-value, and significance)
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
OUT_ALL     <- "spatial_change_spillover_no2_crime_all_pairs.csv"
## ==========================================================================

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

## ── Load borough boundaries + standard Queen contiguity weights
##    (identical source/setup to spatial_change_spillover_all_years.R,
##    for full consistency with every other task's adjacency structure) ──
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

## ── Reusable delta + Moran's I computation, identical logic to
##    spatial_change_spillover_all_years.R's inner lapply body -----------
compute_change_spillover <- function(period_label_a, period_label_b, value_a, value_b) {
  delta_df <- tibble(borough = boroughs_sf$borough) %>%
    left_join(tibble(borough = names(value_a), value_a = unname(value_a)), by = "borough") %>%
    left_join(tibble(borough = names(value_b), value_b = unname(value_b)), by = "borough") %>%
    mutate(delta_value = value_b - value_a)

  spatial_df <- boroughs_sf %>% left_join(delta_df, by = "borough")

  missing_count <- sum(is.na(spatial_df$delta_value))
  delta_imputed <- spatial_df$delta_value
  delta_imputed[is.na(delta_imputed)] <- mean(delta_imputed, na.rm = TRUE)

  mt <- moran.test(delta_imputed, lw, zero.policy = TRUE)
  moran_i    <- unname(mt$estimate["Moran I statistic"])
  moran_pval <- mt$p.value
  clustered  <- moran_pval < 0.05

  tibble(
    period_pair    = paste0(period_label_a, "_to_", period_label_b),
    n_missing      = missing_count,
    moran_i        = round(moran_i, 4),
    p_value        = round(moran_pval, 4),
    clustered      = clustered,
    borough_deltas = paste(spatial_df$borough, round(delta_imputed, 4), sep = ":", collapse = "; ")
  )
}

results_all <- list()

## ── NO2: consecutive quarterly pairs, in chronological order ───────────
cat("\n============================================================\n")
cat(" NO2 (quarterly, consecutive pairs)\n")
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

for (i in seq_len(length(no2_periods) - 1)) {
  p_a <- no2_periods[i]
  p_b <- no2_periods[i + 1]

  va <- no2_raw %>% filter(time_period == p_a) %>% { setNames(.$avg_no2, .$borough) }
  vb <- no2_raw %>% filter(time_period == p_b) %>% { setNames(.$avg_no2, .$borough) }

  res <- compute_change_spillover(p_a, p_b, va, vb) %>% mutate(indicator = "NO2", .before = 1)
  cat(sprintf("  %s -> %s : Moran's I = %.4f, p = %.4f, clustered = %s, n_missing = %d\n",
              p_a, p_b, res$moran_i, res$p_value, res$clustered, res$n_missing))

  write.csv(res %>% select(-borough_deltas),
            file.path(OUT_DIR, paste0("spatial_change_spillover_no2_crime_", res$period_pair, ".csv")),
            row.names = FALSE)
  results_all[[length(results_all) + 1]] <- res
}

## ── Crime: consecutive half-year pairs, in chronological order ─────────
cat("\n============================================================\n")
cat(" Crime (half-yearly, consecutive pairs)\n")
cat("============================================================\n")

crime_raw <- read_csv(CRIME_PATH, show_col_types = FALSE) %>%
  rename(borough = Borough, time_period = time) %>%
  mutate(borough = clean_borough(borough))

crime_periods <- crime_raw %>%
  distinct(time_period) %>%
  mutate(
    yr   = as.integer(str_extract(time_period, "\\d{4}")),
    half = if_else(str_starts(time_period, "first"), 1L, 2L)
  ) %>%
  arrange(yr, half) %>%
  pull(time_period)

for (i in seq_len(length(crime_periods) - 1)) {
  p_a <- crime_periods[i]
  p_b <- crime_periods[i + 1]

  va <- crime_raw %>% filter(time_period == p_a) %>% { setNames(.$crime_rate, .$borough) }
  vb <- crime_raw %>% filter(time_period == p_b) %>% { setNames(.$crime_rate, .$borough) }

  res <- compute_change_spillover(p_a, p_b, va, vb) %>% mutate(indicator = "crime", .before = 1)
  cat(sprintf("  %s -> %s : Moran's I = %.4f, p = %.4f, clustered = %s, n_missing = %d\n",
              p_a, p_b, res$moran_i, res$p_value, res$clustered, res$n_missing))

  write.csv(res %>% select(-borough_deltas),
            file.path(OUT_DIR, paste0("spatial_change_spillover_no2_crime_", res$period_pair, ".csv")),
            row.names = FALSE)
  results_all[[length(results_all) + 1]] <- res
}

## ── Combine, save the summary needed for manifest generation ───────────
change_summary <- bind_rows(results_all)
write.csv(change_summary, file.path(OUT_DIR, OUT_ALL), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" -", file.path(OUT_DIR, "spatial_change_spillover_no2_crime_<pair>.csv"),
    "  (one per period-pair, no borough_deltas column)\n")
cat(" -", file.path(OUT_DIR, OUT_ALL),
    sprintf("  (%d NO2 pairs + %d crime pairs = %d rows total)\n",
            length(no2_periods) - 1, length(crime_periods) - 1, nrow(change_summary)))
cat(" borough_deltas column (in the all_pairs file only) lists every borough's real computed\n")
cat(" delta value, e.g. 'Camden:1.23; Barnet:-0.45; ...' -- needed later for chunk generation\n")
cat(" and per-borough ground truth verification, exactly as in your existing all_years file.\n")
cat("============================================================\n")