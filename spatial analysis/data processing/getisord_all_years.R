# =============================================================================
#  getisord_all_years.R
#  NEW TASK 3: Getis-Ord Gi* hot-spot analysis (spdep::localG) -- the
#  standard method behind ArcGIS's "Hot Spot Analysis" tool. Two
#  foundational papers, cite both:
#    Getis, A. and Ord, J.K. (1992). "The analysis of spatial association
#      by use of distance statistics." Geographical Analysis, 24(3), 189-206.
#      (original G/G* statistic)
#    Ord, J.K. and Getis, A. (1995). "Local spatial autocorrelation
#      statistics: distributional issues and an application." Geographical
#      Analysis, 27(4), 286-306. (the LOCAL Gi/Gi* variant -- this is what
#      is actually computed below; note author order is Ord THEN Getis for
#      this specific paper, though the statistic is colloquially called
#      "Getis-Ord Gi*" in common usage).
#
#  Tests something genuinely DIFFERENT from
#  Local Moran's I: instead of asking "is this borough SIMILAR to its
#  neighbors" (Moran's I), Gi* asks "is there an unusually intense
#  CONCENTRATION of high (or low) values in this borough AND its
#  neighbors combined, regardless of similarity." A borough can show a
#  real Gi* hot spot even if it doesn't individually "match" any one
#  neighbor closely -- it's a magnitude/concentration test, not a
#  similarity test.
#
#  SAFE FOR YOUR EXISTING PIPELINE: reads the same borough boundaries as
#  moran_all_years.R (read-only), writes to NEW, distinctly-named output
#  files. Does not touch global_moran_*, local_moran_*, or
#  borough_adjacency.csv.
#
#  Input:  PREP/2021.csv, PREP/2022.csv, PREP/2023.csv (only 2023.csv
#          provided so far; add earlier years the same way, or trim
#          YEARS below to just "2023")
#
#  Output: figures/getisord_results_<year>.csv    (one per year, 7 indicators x 33 boroughs)
#          figures/getisord_results_all_years.csv (stacked, all years)
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
PREP     <- "sustainability/dataset"
YEARS    <- c("2021", "2022", "2023")   # trim to c("2023") if that's all you have right now
OUT_DIR  <- "figures"

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

# ── Load borough boundaries ─────────────────────────────────────────────
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

# ── Two weight structures: standard (for reference) and SELF-INCLUDED
#    (required for Gi* specifically -- Gi* includes the focal borough in
#    its own neighborhood sum, unlike Local Moran's I) ─────────────────
nb_queen      <- poly2nb(boroughs_sf, queen = TRUE)
nb_queen_self <- include.self(nb_queen)          # Gi* requires this
lw_star       <- nb2listw(nb_queen_self, style = "W", zero.policy = TRUE)

cat("Neighbour summary (standard, for reference):\n")
print(summary(nb_queen))

# ── Getis-Ord Gi* helper ────────────────────────────────────────────────
compute_getis_ord <- function(sf_obj, col, lw_star) {
  vals    <- sf_obj[[col]]
  missing <- is.na(vals)
  vals_imp <- vals
  vals_imp[missing] <- mean(vals, na.rm = TRUE)

  gi_star <- localG(vals_imp, lw_star, zero.policy = TRUE)
  z <- as.numeric(gi_star)

  # Standard critical values (Ord & Getis 1995): |z|>1.96 -> 95%, |z|>2.576 -> 99%
  cluster <- case_when(
    missing         ~ "No data",
    z >  2.576      ~ "Hot spot (99%)",
    z >  1.96       ~ "Hot spot (95%)",
    z < -2.576      ~ "Cold spot (99%)",
    z < -1.96       ~ "Cold spot (95%)",
    TRUE            ~ "Not significant"
  )

  tibble(
    borough  = sf_obj$borough,
    gi_star_z = round(z, 4),
    cluster  = cluster
  )
}

# ── Main loop: one pass per year, all 7 indicators ─────────────────────
getisord_all_years <- list()

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

  getisord_yr <- lapply(indicators, function(ind) {
    col <- IND_TO_COL[[ind]]
    res <- compute_getis_ord(spatial_df, col, lw_star)
    res %>% mutate(year = yr, Indicator = ind, .before = 1)
  }) %>% bind_rows()

  cat("\n--- Getis-Ord Gi* cluster counts,", yr, "---\n")
  print(getisord_yr %>% count(Indicator, cluster) %>%
          pivot_wider(names_from = cluster, values_from = n, values_fill = 0))

  write.csv(getisord_yr, file.path(OUT_DIR, paste0("getisord_results_", yr, ".csv")), row.names = FALSE)
  getisord_all_years[[yr]] <- getisord_yr
}

getisord_all <- bind_rows(getisord_all_years)
write.csv(getisord_all, file.path(OUT_DIR, "getisord_results_all_years.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" - figures/getisord_results_<year>.csv        (7 indicators x 33 boroughs each)\n")
cat(" - figures/getisord_results_all_years.csv      (7 indicators x 33 boroughs x N years)\n")
cat("============================================================\n")
