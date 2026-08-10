# =============================================================================
#  distance_decay_cyclehire.R
#  NEW TASK 2: Distance-decay (gravity model) analysis on real Cycle Hire
#  trip data. Tests whether trip volume between borough pairs falls off as
#  the distance between them increases -- the foundational relationship in
#  transportation geography (the "gravity model"). Foundational reference:
#    Wilson, A.G. (1971). "A family of spatial interaction models, and
#    associated developments." Environment and Planning A, 3(1), 1-32.
#  (the standard modern formalization of the gravity/spatial-interaction
#  model; the framework itself traces further back to Reilly 1929 and
#  Zipf 1946, but Wilson 1971 is the most commonly cited source for the
#  form used here). Uses REAL station
#  coordinates already present in your uploaded journeys_with_boroughs.csv
#  (StartLat/StartLon/EndLat/EndLon), which nothing in your pipeline has
#  used yet -- everything so far has only used the borough LABELS.
#
#  SAFE FOR YOUR EXISTING PIPELINE: reads journeys_with_boroughs.csv
#  read-only, writes to a NEW, distinctly-named output file. Does not
#  touch cyclehire_windows/, borough_adjacency.csv, or any Moran's I output.
#
#  Method: borough centroids are estimated as the mean lat/lon of all
#  stations within that borough (computed directly from real station data,
#  not looked up externally). For every ordered borough pair with at least
#  one real trip, compute the great-circle (haversine) distance between
#  centroids and the real trip count. Fit trips ~ distance in log-log form
#  (the standard gravity-model specification) and test the correlation.
#
#  Input:  journeys_with_boroughs.csv (same real file used throughout)
#  Output: figures/distance_decay_cyclehire.csv        (borough-pair level detail)
#          figures/distance_decay_summary.csv           (one row: overall test result)
# =============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(stringr)

# ── Configuration ─────────────────────────────────────────────────────────
JOURNEYS_PATH <- "cycles/dataset/cyclehire_windows/journeys_with_boroughs.csv"
OUT_DIR       <- "figures"
NSIM          <- 999

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Haversine great-circle distance (km), no external package needed ──────
haversine_km <- function(lat1, lon1, lat2, lon2) {
  R <- 6371  # Earth radius, km
  to_rad <- function(deg) deg * pi / 180
  dlat <- to_rad(lat2 - lat1)
  dlon <- to_rad(lon2 - lon1)
  a <- sin(dlat / 2)^2 + cos(to_rad(lat1)) * cos(to_rad(lat2)) * sin(dlon / 2)^2
  2 * R * asin(sqrt(pmin(1, a)))
}

# ── Load real journey data ──────────────────────────────────────────────
cat("Loading journeys...\n")
journeys <- read_csv(JOURNEYS_PATH, show_col_types = FALSE) %>%
  rename(
    StartStationId = `StartStation Id`, EndStationId = `EndStation Id`,
    StartLat = StartLat, StartLon = StartLon, EndLat = EndLat, EndLon = EndLon
  ) %>%
  mutate(StartBorough = clean_borough(StartBorough), EndBorough = clean_borough(EndBorough)) %>%
  filter(!is.na(StartBorough), !is.na(EndBorough))

cat("Real trips loaded:", nrow(journeys), "\n")

# ── Borough centroids: mean lat/lon of all stations actually in that
#    borough, computed directly from your real station data ──────────────
start_pts <- journeys %>% select(borough = StartBorough, lat = StartLat, lon = StartLon)
end_pts   <- journeys %>% select(borough = EndBorough,   lat = EndLat,   lon = EndLon)
all_pts   <- bind_rows(start_pts, end_pts) %>% filter(!is.na(lat), !is.na(lon))

centroids <- all_pts %>%
  group_by(borough) %>%
  summarise(centroid_lat = mean(lat), centroid_lon = mean(lon), n_points = n(), .groups = "drop")

cat("\nBorough centroids (from real station coordinates):\n")
print(centroids %>% select(borough, centroid_lat, centroid_lon))

# ── Build the borough-pair OD table: real trip counts + real distances ────
od_counts <- journeys %>%
  filter(StartBorough != EndBorough) %>%   # cross-borough trips only -- distance-decay is
                                            # about trips BETWEEN places, not within one
  count(StartBorough, EndBorough, name = "n_trips")

od_table <- od_counts %>%
  left_join(centroids %>% rename(start_lat = centroid_lat, start_lon = centroid_lon),
            by = c("StartBorough" = "borough")) %>%
  left_join(centroids %>% rename(end_lat = centroid_lat, end_lon = centroid_lon),
            by = c("EndBorough" = "borough")) %>%
  mutate(distance_km = haversine_km(start_lat, start_lon, end_lat, end_lon)) %>%
  filter(distance_km > 0)  # drop any zero-distance artifacts

cat("\nBorough pairs with real trips:", nrow(od_table), "\n")
cat("Distance range (km):", round(min(od_table$distance_km), 2), "to", round(max(od_table$distance_km), 2), "\n")

write.csv(od_table, file.path(OUT_DIR, "distance_decay_cyclehire.csv"), row.names = FALSE)

# ── Gravity-model test: does log(trips) decline with log(distance)? ───────
# Standard log-log specification -- the real, established form used in
# transportation gravity models (trips ~ distance^-beta)
od_table_log <- od_table %>% mutate(log_trips = log(n_trips), log_distance = log(distance_km))

fit <- lm(log_trips ~ log_distance, data = od_table_log)
slope       <- round(coef(fit)[["log_distance"]], 4)
slope_pval  <- round(summary(fit)$coefficients["log_distance", "Pr(>|t|)"], 4)
r_squared   <- round(summary(fit)$r.squared, 4)
correlation <- round(cor(od_table_log$log_trips, od_table_log$log_distance), 4)

cat("\n--- Gravity model fit: log(trips) ~ log(distance) ---\n")
cat("Slope (beta):", slope, "| p-value:", slope_pval, "| R-squared:", r_squared,
    "| Pearson correlation:", correlation, "\n")

interpretation <- case_when(
  slope_pval < 0.05 & slope < 0 ~ "DISTANCE_DECAY_PRESENT -- trip volume significantly decreases with distance (real gravity-model relationship)",
  slope_pval < 0.05 & slope > 0 ~ "POSITIVE_DISTANCE_RELATIONSHIP -- trip volume significantly INCREASES with distance (unusual, worth investigating)",
  TRUE ~ "NO_DISTANCE_DECAY -- no significant relationship between trip volume and distance"
)
cat(interpretation, "\n")

summary_row <- tibble(
  n_borough_pairs = nrow(od_table),
  slope_beta      = slope,
  slope_p_value   = slope_pval,
  r_squared       = r_squared,
  pearson_correlation = correlation,
  significant     = if_else(slope_pval < 0.05, "Yes", "No"),
  interpretation  = interpretation
)
write.csv(summary_row, file.path(OUT_DIR, "distance_decay_summary.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat(" DONE. Saved:\n")
cat(" - figures/distance_decay_cyclehire.csv   (borough-pair level: trips, distance)\n")
cat(" - figures/distance_decay_summary.csv     (one row: overall test result)\n")
cat("============================================================\n")
