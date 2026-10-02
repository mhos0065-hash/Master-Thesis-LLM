# =============================================================================
#  getisord_no2_crime_quarterly.R
#  Getis-Ord Gi* (spdep::localG) ground truth for the two long-format
#  files built separately this session: no2_quarterly.csv (time_period,
#  borough, avg_no2) and crime_halfyearly.csv (time, Borough, population,
#  crime_rate). Same statistic, same weights structure, same thresholds
#  as getisord_all_years.R -- just applied to per-period long tables
#  instead of per-year wide panels, and combined into ONE output file.
#
#  Getis, A. and Ord, J.K. (1992). "The analysis of spatial association
#    by use of distance statistics." Geographical Analysis, 24(3), 189-206.
#  Ord, J.K. and Getis, A. (1995). "Local spatial autocorrelation
#    statistics: distributional issues and an application." Geographical
#    Analysis, 27(4), 286-306. (the LOCAL Gi/Gi* variant computed below)
#
#  Input:  no2_quarterly.csv      (time_period, borough, avg_no2)
#          crime_halfyearly.csv   (time, Borough, population, crime_rate)
#
#  Output: getisord_no2_crime_ground_truth.csv
#          (columns: time_period, Indicator, borough, gi_star_z, cluster)
#          -- exactly one file, as requested.
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
NO2_PATH   <- "sustainability/dataset/additional test cases/no2_quarterly.csv"
CRIME_PATH <- "sustainability/dataset/additional test cases/crime_halfyearly.csv"
OUT_PATH   <- "figures/getisord_no2_crime_ground_truth.csv"

clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Load borough boundaries (identical to getisord_all_years.R) ─────────
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

# ── Self-included Queen weights -- required for Gi*, identical to
#    getisord_all_years.R (Gi* includes the focal borough in its own
#    neighborhood sum, unlike Local Moran's I) ──────────────────────────
nb_queen      <- poly2nb(boroughs_sf, queen = TRUE)
nb_queen_self <- include.self(nb_queen)
lw_star       <- nb2listw(nb_queen_self, style = "W", zero.policy = TRUE)

cat("Neighbour summary (standard, for reference):\n")
print(summary(nb_queen))

# ── Getis-Ord Gi* helper -- identical logic to getisord_all_years.R,
#    just takes a plain numeric vector directly rather than a column
#    name, since the two input files aren't wide panels ───────────────
compute_getis_ord <- function(sf_obj, vals, lw_star) {
  missing  <- is.na(vals)
  vals_imp <- vals
  vals_imp[missing] <- mean(vals, na.rm = TRUE)
  
  gi_star <- localG(vals_imp, lw_star, zero.policy = TRUE)
  z <- as.numeric(gi_star)
  
  # Standard critical values (Ord & Getis 1995): |z|>1.96 -> 95%, |z|>2.576 -> 99%
  cluster <- case_when(
    missing    ~ "No data",
    z >  2.576 ~ "Hot spot (99%)",
    z >  1.96  ~ "Hot spot (95%)",
    z < -2.576 ~ "Cold spot (99%)",
    z < -1.96  ~ "Cold spot (95%)",
    TRUE       ~ "Not significant"
  )
  
  tibble(
    borough   = sf_obj$borough,
    gi_star_z = round(z, 4),
    cluster   = cluster
  )
}

# ── NO2: one Gi* pass per quarterly time_period ─────────────────────────
no2_raw <- read_csv(NO2_PATH, show_col_types = FALSE) %>%
  mutate(borough = clean_borough(borough))

no2_periods <- unique(no2_raw$time_period)

no2_results <- lapply(no2_periods, function(tp) {
  panel_tp <- no2_raw %>% filter(time_period == tp) %>% select(borough, avg_no2)
  
  spatial_df <- boroughs_sf %>% left_join(panel_tp, by = "borough")
  
  res <- compute_getis_ord(spatial_df, spatial_df$avg_no2, lw_star)
  res %>% mutate(time_period = tp, Indicator = "No2", .before = 1)
}) %>% bind_rows()

cat("\n--- NO2 Gi* cluster counts (all quarters) ---\n")
print(no2_results %>% count(time_period, cluster) %>%
        pivot_wider(names_from = cluster, values_from = n, values_fill = 0))

# ── Crime: one Gi* pass per half-year time period ───────────────────────
crime_raw <- read_csv(CRIME_PATH, show_col_types = FALSE) %>%
  rename(borough = Borough, time_period = time) %>%
  mutate(borough = clean_borough(borough))

crime_periods <- unique(crime_raw$time_period)

crime_results <- lapply(crime_periods, function(tp) {
  panel_tp <- crime_raw %>% filter(time_period == tp) %>% select(borough, crime_rate)
  
  spatial_df <- boroughs_sf %>% left_join(panel_tp, by = "borough")
  
  res <- compute_getis_ord(spatial_df, spatial_df$crime_rate, lw_star)
  res %>% mutate(time_period = tp, Indicator = "crime", .before = 1)
}) %>% bind_rows()

cat("\n--- Crime Gi* cluster counts (all half-years) ---\n")
print(crime_results %>% count(time_period, cluster) %>%
        pivot_wider(names_from = cluster, values_from = n, values_fill = 0))

# ── Combine into the single requested output file ───────────────────────
getisord_no2_crime <- bind_rows(no2_results, crime_results) %>%
  select(time_period, Indicator, borough, gi_star_z, cluster)

write.csv(getisord_no2_crime, OUT_PATH, row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:", OUT_PATH, "\n")
cat(" Rows:", nrow(getisord_no2_crime),
    "(expect", length(no2_periods), "NO2 periods x 33 +",
    length(crime_periods), "crime periods x 33 =",
    (length(no2_periods) + length(crime_periods)) * 33, ")\n")
cat("============================================================\n")