# =============================================================================
#  bivariate_moran_all_years.R
#  NEW TASK 1: Bivariate spatial correlation between pairs of sustainability
#  indicators, using Lee's L statistic (spdep::lee.mc) -- a real, published
#  method (Lee 2001, J Geograph Syst) built specifically to extend Moran's I
#  to two variables. Answers: "does high X in one borough associate with
#  high/low Y in NEIGHBORING boroughs?" -- genuinely different from Global/
#  Local Moran's I, which only ever look at ONE variable against itself.
#
#  For each year, computes Lee's L for all 21 unordered pairs of your 7
#  indicators (Crime, Waste, NO2, GHG, Gas, Energy, Transport).
#
#  SAFE FOR YOUR EXISTING PIPELINE: reads the same borough boundaries/
#  adjacency as moran_all_years.R (read-only, nothing modified), writes to
#  NEW, distinctly-named output files -- does not touch global_moran_*,
#  local_moran_*, or borough_adjacency.csv in any way.
#
#  Input:  PREP/2021.csv, PREP/2022.csv, PREP/2023.csv (same panel files --
#          only 2023.csv provided so far; add 2021/2022 the same way if
#          you want all 3 years, or trim YEARS below to just "2023")
#
#  Output: figures/bivariate_moran_<year>.csv     (one per year, 21 rows)
#          figures/bivariate_moran_all_years.csv  (stacked, all years)
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
NSIM     <- 999
P_THRESH <- 0.05

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

# all 21 unordered pairs (7 choose 2)
indicator_pairs <- combn(indicators, 2, simplify = FALSE)

# ── Borough name cleaner (identical convention to your existing scripts) ──
clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Load borough boundaries + spatial weights (same as moran_all_years.R) ─
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

print(summary(lw))

# ── Bivariate spatial correlation helper (Lee's L) ─────────────────────────
compute_lee <- function(x, y, lw, nsim = NSIM) {
  n_missing_x <- sum(is.na(x))
  n_missing_y <- sum(is.na(y))
  if (n_missing_x > 0) x[is.na(x)] <- mean(x, na.rm = TRUE)
  if (n_missing_y > 0) y[is.na(y)] <- mean(y, na.rm = TRUE)

  lee_mc <- lee.mc(x = x, y = y, listw = lw, nsim = nsim, zero.policy = TRUE,
                    alternative = "two.sided")

  tibble(
    Lee_L          = round(as.numeric(lee_mc$statistic), 4),
    p_value_MC     = round(lee_mc$p.value, 4),
    n_missing_x    = n_missing_x,
    n_missing_y    = n_missing_y,
    Significant    = if_else(lee_mc$p.value < P_THRESH, "Yes", "No"),
    Interpretation = case_when(
      lee_mc$p.value < P_THRESH & as.numeric(lee_mc$statistic) > 0 ~
        "Positive bivariate spatial association (high-X areas near high-Y areas)",
      lee_mc$p.value < P_THRESH & as.numeric(lee_mc$statistic) < 0 ~
        "Negative bivariate spatial association (high-X areas near low-Y areas)",
      TRUE ~ "No significant bivariate spatial association"
    )
  )
}

# ── Main loop: one pass per year, all 21 indicator pairs ───────────────────
bivariate_all_years <- list()

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

  bivariate_yr <- lapply(indicator_pairs, function(pair) {
    ind_x <- pair[1]; ind_y <- pair[2]
    col_x <- IND_TO_COL[[ind_x]]; col_y <- IND_TO_COL[[ind_y]]
    x <- as.numeric(spatial_df[[col_x]])
    y <- as.numeric(spatial_df[[col_y]])
    res <- compute_lee(x, y, lw, nsim = NSIM)
    res %>% mutate(year = yr, indicator_x = ind_x, indicator_y = ind_y, .before = 1)
  }) %>% bind_rows() %>% arrange(desc(abs(Lee_L)))

  cat("\n--- Bivariate spatial correlation (Lee's L),", yr, "-- top 5 by |L| ---\n")
  print(head(bivariate_yr, 5))

  write.csv(bivariate_yr, file.path(OUT_DIR, paste0("bivariate_moran_", yr, ".csv")), row.names = FALSE)
  bivariate_all_years[[yr]] <- bivariate_yr
}

bivariate_moran_all <- bind_rows(bivariate_all_years)
write.csv(bivariate_moran_all, file.path(OUT_DIR, "bivariate_moran_all_years.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" - figures/bivariate_moran_<year>.csv        (21 pairs x 1 row each)\n")
cat(" - figures/bivariate_moran_all_years.csv      (21 pairs x N years)\n")
cat("============================================================\n")
