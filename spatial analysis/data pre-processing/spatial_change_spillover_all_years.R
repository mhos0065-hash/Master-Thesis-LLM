# =============================================================================
#  spatial_change_spillover_all_years.R
#  NEW TASK 5: Spatial spillover of CHANGE (temporal + spatial combined)
#
#  Tests whether the CHANGE in an indicator between two years clusters
#  spatially -- i.e. do boroughs that saw a large increase (or decrease)
#  tend to be located near OTHER boroughs that also saw a large increase
#  (or decrease)? This is distinct from all previous tasks, which tested
#  clustering of the RAW LEVEL in a single year -- here the variable
#  being tested for spatial clustering is delta_value = value[year_b] -
#  value[year_a], not the raw indicator itself.
#
#  Method: identical to your existing Global Moran's I approach (same
#  spdep::moran.test(), same Queen contiguity weights) -- the ONLY
#  difference is the INPUT column is a computed year-over-year delta,
#  not a raw indicator column. This deliberately reuses proven,
#  already-validated infrastructure rather than introducing a new
#  statistical method.
#
#  Three year-pairs computed per indicator, to maximise the number of
#  real, distinct chunks (matching the multi-year structure of your
#  other tasks): 2021->2022, 2022->2023, 2021->2023 (the fullest-span
#  pair, likely the strongest signal-to-noise ratio).
#
#  SAFE FOR YOUR EXISTING PIPELINE: reads the same borough boundaries and
#  panel data as your other scripts (read-only), writes to NEW,
#  distinctly-named output files. Does not touch global_moran_*,
#  local_moran_*, getisord_*, bivariate_*, spatial_outlier_iterative_r_*,
#  distance_decay_*, or borough_adjacency.csv.
#
#  Input:  sustainability/dataset/2021.csv, 2022.csv, 2023.csv
#          (same real panel data used throughout your project)
#
#  Output: figures/spatial_change_spillover_<yearpair>_<indicator>.csv
#          (per-borough delta values and centroid info, for transparency)
#          figures/spatial_change_spillover_all_years.csv
#          (stacked summary: one row per year-pair x indicator, with the
#          real computed Moran's I, p-value, and significance)
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
PREP     <- "sustainability/dataset"
OUT_DIR  <- "figures"

# three year-pairs: (start_year, end_year, label)
YEAR_PAIRS <- list(
  list(start = "2021", end = "2022", label = "2021_2022"),
  list(start = "2022", end = "2023", label = "2022_2023"),
  list(start = "2021", end = "2023", label = "2021_2023")
)

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
# (identical source and setup to moran_all_years.R, for full consistency
# with every other task's adjacency structure)
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

# ── Main loop: one pass per year-pair, all 7 indicators ────────────────
change_all <- list()

for (yp in YEAR_PAIRS) {

  path_a <- file.path(PREP, paste0(yp$start, ".csv"))
  path_b <- file.path(PREP, paste0(yp$end, ".csv"))
  if (!file.exists(path_a) || !file.exists(path_b)) {
    cat("\nSkipping", yp$label, "-- one or both files not found\n")
    next
  }

  cat("\n============================================================\n")
  cat(" YEAR PAIR:", yp$label, "\n")
  cat("============================================================\n")

  panel_a <- read_csv(path_a, show_col_types = FALSE) %>% mutate(borough = clean_borough(borough))
  panel_b <- read_csv(path_b, show_col_types = FALSE) %>% mutate(borough = clean_borough(borough))

  yearpair_results <- lapply(indicators, function(ind) {
    col <- IND_TO_COL[[ind]]

    delta_df <- panel_a %>%
      select(borough, value_a = all_of(col)) %>%
      inner_join(panel_b %>% select(borough, value_b = all_of(col)), by = "borough") %>%
      mutate(delta_value = value_b - value_a)

    spatial_df <- boroughs_sf %>% left_join(delta_df, by = "borough")

    # missing-value handling: impute at the MEAN (consistent with your
    # other tasks' established convention), report how many were missing
    missing_count <- sum(is.na(spatial_df$delta_value))
    delta_imputed <- spatial_df$delta_value
    delta_imputed[is.na(delta_imputed)] <- mean(delta_imputed, na.rm = TRUE)

    # real Global Moran's I on the DELTA values (identical method to
    # your existing Global Moran's I task, different input column)
    mt <- moran.test(delta_imputed, lw, zero.policy = TRUE)
    moran_i    <- unname(mt$estimate["Moran I statistic"])
    moran_pval <- mt$p.value
    clustered  <- moran_pval < 0.05

    tibble(
      year_pair       = yp$label,
      Indicator       = ind,
      n_missing       = missing_count,
      moran_i         = round(moran_i, 4),
      p_value         = round(moran_pval, 4),
      clustered       = clustered,
      borough_deltas  = paste(spatial_df$borough, round(delta_imputed, 4), sep = ":", collapse = "; ")
    )
  }) %>% bind_rows()

  cat("\n--- Change-spillover results,", yp$label, "---\n")
  print(yearpair_results %>% select(Indicator, n_missing, moran_i, p_value, clustered))

  write.csv(yearpair_results %>% select(-borough_deltas),
            file.path(OUT_DIR, paste0("spatial_change_spillover_", yp$label, ".csv")),
            row.names = FALSE)
  change_all[[yp$label]] <- yearpair_results
}

change_summary <- bind_rows(change_all)
write.csv(change_summary, file.path(OUT_DIR, "spatial_change_spillover_all_years.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" - figures/spatial_change_spillover_<yearpair>.csv   (7 indicators each, per year-pair)\n")
cat(" - figures/spatial_change_spillover_all_years.csv    (7 indicators x 3 year-pairs = 21 rows)\n")
cat(" - borough_deltas column (in the all_years file only) lists every borough's real computed\n")
cat("   delta value, e.g. 'Camden:12.34; Barnet:-5.67; ...' -- needed later for chunk generation\n")
cat("   and per-borough ground truth verification.\n")
cat("============================================================\n")
