# =============================================================================
#  cyclehire_containment.R
#  Computes Global Moran's I AND Local Moran's I (LISA) for SELF-CONTAINMENT
#  RATIO (trips that start AND end in the same borough, as a share of all
#  trips that start there), once per 7-day sliding window.
#
#  SELF-SUFFICIENT: reads ONLY the window chunk files in cyclehire_windows/
#  (from generate_cyclehire_windows.py) -- no reference to the original full
#  dataset. Self-containment ratio needs no station-count normalization: it
#  is already a proportion (both numerator and denominator scale with that
#  borough's own activity level), so it does not need the per-station
#  correction that cyclehire_start_station.R / cyclehire_end_station.R use.
#
#  Input:  cyclehire_windows/windows_index.csv
#          cyclehire_windows/window_w###.csv
#
#  Output: output_containment/Global_moran_all_windows.csv
#          output_containment/Local_moran_all_windows.csv
# =============================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(tidyverse)

# ── Configuration ─────────────────────────────────────────────────────────
WINDOWS_DIR   <- "cycles/dataset/cyclehire_windows"
WINDOWS_INDEX <- file.path(WINDOWS_DIR, "windows_index.csv")
OUT_DIR       <- "output_containment"
NSIM          <- 999
P_THRESH      <- 0.10
METRICS <- list(
  list(col = "self_containment_ratio", name = "SelfContainmentRatio"),
  list(col = "self_containment_index", name = "SelfContainmentIndex")
)

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

CYCLEHIRE_BOROUGHS <- c(
  "Camden", "City of London", "Hackney", "Hammersmith and Fulham",
  "Islington", "Kensington and Chelsea", "Lambeth", "Newham",
  "Southwark", "Tower Hamlets", "Wandsworth", "Westminster"
)

# ── Borough name cleaner ────────────────────────────────────────────────
clean_borough <- function(x) {
  x %>% str_to_title() %>%
    str_replace("\\bAnd\\b",  "and")  %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b",   "of")
}

# ── Load borough boundaries once, restrict to the Cycle Hire subset ───────
boroughs_sf_raw <- st_read(
  "https://github.com/LingruFeng/GIS_assessment/blob/main/London_Boroughs.gpkg?raw=true",
  quiet = TRUE
)

boroughs_sf <- boroughs_sf_raw %>%
  st_transform(4326) %>%
  rename(geometry = geom) %>%
  st_set_geometry("geometry") %>%
  mutate(borough = clean_borough(name)) %>%
  select(borough, geometry) %>%
  filter(borough %in% CYCLEHIRE_BOROUGHS)

cat("Boroughs in spatial subset:", nrow(boroughs_sf), "(expected", length(CYCLEHIRE_BOROUGHS), ")\n")
missing_boroughs <- setdiff(CYCLEHIRE_BOROUGHS, boroughs_sf$borough)
if (length(missing_boroughs) > 0) {
  stop("Borough name(s) not found in boundary file: ", paste(missing_boroughs, collapse = ", "))
}

# ── Build spatial weights matrix ONCE on the 12-borough subgraph ──────────
nb_queen <- poly2nb(boroughs_sf, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)
cat("Neighbour summary:\n")
print(summary(nb_queen))

# ── Local Moran's I (LISA) helper ──────────────────────────────────────
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

  tibble(borough = sf_obj$borough, z_score = round(z, 4), lag_z = round(lag_z, 4),
         Ii = round(lisa[, "Ii"], 4), p_value = round(p_val, 4), cluster = cluster)
}

# ── Global Moran's I helper ─────────────────────────────────────────────
compute_global_moran <- function(vals, lw, nsim = NSIM) {
  n_missing <- sum(is.na(vals))
  if (n_missing > 0) vals[is.na(vals)] <- mean(vals, na.rm = TRUE)

  moran_mc  <- moran.mc(x = vals, listw = lw, nsim = nsim, zero.policy = TRUE)
  moran_std <- moran.test(x = vals, listw = lw, zero.policy = TRUE)

  tibble(
    Moran_I = round(moran_mc$statistic, 4), Expected_I = round(moran_std$estimate["Expectation"], 4),
    Variance = round(moran_std$estimate["Variance"], 6), p_value_MC = round(moran_mc$p.value, 4),
    p_value_std = round(moran_std$p.value, 4), n_missing = n_missing,
    Significant = if_else(moran_mc$p.value < 0.05, "Yes", "No"),
    Interpretation = case_when(
      moran_mc$statistic >  0.3 & moran_mc$p.value < 0.05 ~ "Strong clustering",
      moran_mc$statistic >  0.1 & moran_mc$p.value < 0.05 ~ "Moderate clustering",
      moran_mc$statistic >  0   & moran_mc$p.value < 0.05 ~ "Weak clustering",
      moran_mc$statistic <= 0                              ~ "Dispersed/Random",
      TRUE                                                 ~ "Not significant"
    )
  )
}

# ── Self-containment ratio for one window ──────────────────────────────
compute_self_containment <- function(trips_df, boroughs) {
  per_borough <- trips_df %>%
    filter(StartBorough %in% boroughs) %>%
    group_by(StartBorough) %>%
    summarise(n_departures = n(), n_self_contained = sum(EndBorough == StartBorough, na.rm = TRUE),
              .groups = "drop")

  # Expected self-contained trips under RANDOM redistribution: if a trip's
  # destination were chosen at random (proportional to each borough's share
  # of ALL arrivals this window), what fraction of borough i's departures
  # would be expected to land back in i purely by chance? This corrects for
  # borough size/geometry -- a small borough naturally has a small "arrival
  # share" and thus a small expected self-containment, so a small OBSERVED
  # ratio for it isn't automatically "real" local-travel behavior.
  arrivals_by_borough <- trips_df %>%
    filter(EndBorough %in% boroughs) %>%
    count(EndBorough, name = "n_arrivals")
  total_arrivals <- sum(arrivals_by_borough$n_arrivals)

  tibble(borough = boroughs) %>%
    left_join(per_borough, by = c("borough" = "StartBorough")) %>%
    left_join(arrivals_by_borough, by = c("borough" = "EndBorough")) %>%
    mutate(
      n_departures     = replace_na(n_departures, 0),
      n_self_contained = replace_na(n_self_contained, 0),
      n_arrivals       = replace_na(n_arrivals, 0),
      self_containment_ratio = n_self_contained / n_departures,
      arrival_share           = n_arrivals / total_arrivals,               # expected-by-chance rate
      expected_self_contained = n_departures * arrival_share,
      # Self-Containment Index: observed / expected. >1 = genuine excess
      # local-loop behavior beyond what borough size/share alone predicts.
      self_containment_index  = n_self_contained / expected_self_contained
    )
}

# ── Main loop: one pass per window ──────────────────────────────────────
windows_index <- read_csv(WINDOWS_INDEX, show_col_types = FALSE)
global_all <- list()
local_all  <- list()

for (i in seq_len(nrow(windows_index))) {
  win <- windows_index[i, ]
  cat("\n=== WINDOW:", win$window_id, "(", win$window_start, "to", win$window_end, ") ===\n")

  trips_df <- read_csv(file.path(WINDOWS_DIR, win$filepath), show_col_types = FALSE) %>%
    mutate(StartBorough = clean_borough(StartBorough), EndBorough = clean_borough(EndBorough))

  sc <- compute_self_containment(trips_df, boroughs_sf$borough)
  cat("Trips:", nrow(trips_df), "\n")
  cat("  SelfContainmentRatio range: ",
      round(min(sc$self_containment_ratio, na.rm = TRUE), 3), "-",
      round(max(sc$self_containment_ratio, na.rm = TRUE), 3), "\n")
  cat("  SelfContainmentIndex range: ",
      round(min(sc$self_containment_index, na.rm = TRUE), 3), "-",
      round(max(sc$self_containment_index, na.rm = TRUE), 3), "\n")

  spatial_df <- boroughs_sf %>% left_join(sc, by = "borough")

  for (m in METRICS) {
    global_win <- compute_global_moran(spatial_df[[m$col]], lw, nsim = NSIM) %>%
      mutate(year = year(as.Date(win$window_start)), window_id = win$window_id,
             window_start = win$window_start, window_end = win$window_end,
             Indicator = m$name, .before = 1)
    print(global_win)
    global_all[[paste(win$window_id, m$name)]] <- global_win

    local_win <- compute_lisa(spatial_df, m$col, lw, nsim = NSIM, p_thresh = P_THRESH) %>%
      mutate(year = year(as.Date(win$window_start)), window_id = win$window_id,
             window_start = win$window_start, window_end = win$window_end,
             Indicator = m$name, .before = 1)
    print(local_win %>% count(cluster))
    local_all[[paste(win$window_id, m$name)]] <- local_win
  }
}

write.csv(bind_rows(global_all), file.path(OUT_DIR, "Global_moran_all_windows.csv"), row.names = FALSE)
write.csv(bind_rows(local_all),  file.path(OUT_DIR, "Local_moran_all_windows.csv"),  row.names = FALSE)
cat("\nDONE. Output in", OUT_DIR, "/\n")
