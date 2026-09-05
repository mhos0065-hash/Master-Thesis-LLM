## ground_truth_all_methods_changespillover.R
## ==============================================
## For each change-spillover chunk (one variable, one year-pair per
## chunk), computes ground-truth CLUSTERED/NOT-CLUSTERED verdicts for
## the OTHER candidate methods models used, testing spatial clustering
## of the YEAR-OVER-YEAR CHANGE (not the raw level) in each variable.
##
## Global Moran's I is NOT recomputed -- it is copied verbatim from the
## manifest's ground_truth_metric_json (moran_i, p_value), same
## principle as Lee's L in bivariate / Gi* in getisord / Iterative-R in
## outlierr, for zero deviation risk.
##
## Methods computed fresh, all via spdep directly (not hand-derived
## formulas):
##   - Geary's C (spdep::geary.mc): an alternative global spatial-
##     autocorrelation statistic. Different scale from Moran's I
##     (centered at 1, typically ranges roughly 0-2, vs Moran's I's
##     roughly -1 to 1), but tests the same underlying null hypothesis
##     (no spatial autocorrelation in the change values).
##   - Join Count Statistic (spdep::joincount.mc): tests clustering of
##     the DIRECTION of change (increase vs decrease) rather than its
##     magnitude -- converts each borough's change into a binary
##     increase/decrease factor first. A genuinely different question
##     (direction-only, ignores magnitude) but a legitimate, named,
##     commonly-used alternative for exactly this kind of "do similar
##     categories cluster" question.
##   - Neighbor-average correlation (non-standard): Pearson correlation
##     between each borough's own change and its direct neighbors'
##     mean change, with significance via a 999-permutation test (NOT
##     spdep, since this isn't a spatial-weights-based test at all --
##     just a plain correlation with a resampling p-value). Flagged as
##     "non-standard" because it ignores the full spatial weight
##     structure entirely.
##
## NOT computed (falls back to the manifest's Global Moran's I answer
## if a model used one of these): Local Moran's I / LISA -- this task
## specifically asks about GLOBAL clustering (spatial_granularity ==
## "global" for every chunk), and a local statistic has no single,
## non-arbitrary way to collapse into one global yes/no answer without
## inventing an aggregation rule this task doesn't specify. Rather than
## guess, it's left unverified, consistent with how structurally-
## mismatched methods were handled in the bivariate/getisord tasks.
##
## REQUIRES: spdep, jsonlite (jsonlite is used to parse the manifest's
## ground_truth_metric_json column). Install once with:
##   install.packages(c("spdep", "jsonlite"))
##
## EDIT the CONFIG block, then source/run this file.

library(spdep)
library(jsonlite)

## ============================ CONFIG ============================
ADJACENCY_FILE  <- "chunks_csv/borough_adjacency.csv"
MANIFEST_FILE   <- "chunks_csv/changespillover/changespillover_manifest.csv"
OUTPUT_FILE     <- "ground truth/ground_truth_all_methods_changespillover.csv"
N_PERMUTATIONS  <- 999
ALPHA           <- 0.05
# ==================================================================

manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)

adj <- read.csv(ADJACENCY_FILE, stringsAsFactors = FALSE)
adj_cols <- names(adj)[1:2]
all_boroughs <- sort(unique(c(adj[[adj_cols[1]]], adj[[adj_cols[2]]])))

build_listw_for <- function(boroughs_present) {
  idx <- setNames(seq_along(boroughs_present), boroughs_present)
  nb_list <- setNames(vector("list", length(boroughs_present)), boroughs_present)
  for (b in boroughs_present) nb_list[[b]] <- integer(0)

  sub_adj <- adj[adj[[adj_cols[1]]] %in% boroughs_present & adj[[adj_cols[2]]] %in% boroughs_present, ]
  for (i in seq_len(nrow(sub_adj))) {
    a <- sub_adj[[adj_cols[1]]][i]; b <- sub_adj[[adj_cols[2]]][i]
    nb_list[[a]] <- unique(c(nb_list[[a]], idx[[b]]))
    nb_list[[b]] <- unique(c(nb_list[[b]], idx[[a]]))
  }
  nb <- structure(lapply(nb_list, function(x) if (length(x) == 0) 0L else x), class = "nb")
  attr(nb, "region.id") <- boroughs_present
  nb2listw(nb, style = "W", zero.policy = TRUE)
}

## Reads a changespillover chunk csv and computes the CHANGE (delta)
## between window_start and window_end for each borough. Assumes the
## csv has a borough id column plus year columns matching the window
## years -- EDIT this if your actual chunk files use different column
## naming (only the manifest's file *paths* were available when writing
## this, not the actual data, so verify column names against one real
## chunk file before trusting this blindly).
compute_change <- function(csv_path, year_start, year_end) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
  borough_col <- names(d)[1]
  col_start <- as.character(year_start)
  col_end <- as.character(year_end)
  if (!(col_start %in% names(d)) || !(col_end %in% names(d))) {
    # fall back to guessing by substring match if exact year isn't a column name
    col_start <- names(d)[grepl(year_start, names(d))][1]
    col_end <- names(d)[grepl(year_end, names(d))][1]
  }
  change <- d[[col_end]] - d[[col_start]]
  setNames(change, d[[borough_col]])
}

results <- list()

for (i in seq_len(nrow(manifest))) {
  chunk_id <- manifest$chunk_id[i]
  csv_path <- manifest$csv_path[i]
  if (!file.exists(csv_path)) {
    message("Skipping ", chunk_id, " -- file not found: ", csv_path)
    next
  }

  change <- tryCatch(
    compute_change(csv_path, manifest$window_start[i], manifest$window_end[i]),
    error = function(e) NULL
  )
  if (is.null(change) || any(is.na(change))) {
    change <- change[!is.na(change)]
  }
  if (is.null(change) || length(change) < 8) {
    message("Skipping ", chunk_id, " -- couldn't compute change or too few boroughs")
    next
  }

  boroughs_present <- intersect(all_boroughs, names(change))
  change <- change[boroughs_present]
  lw <- tryCatch(build_listw_for(boroughs_present), error = function(e) NULL)
  if (is.null(lw)) {
    message("Skipping ", chunk_id, " -- couldn't build weights")
    next
  }

  ## --- Geary's C (999-permutation Monte Carlo, matching the Global
  ## Moran's I convention implied by the manifest / task prompt) ---
  geary_stat <- NA; geary_p <- NA
  gc <- tryCatch(geary.mc(change, lw, nsim = N_PERMUTATIONS, zero.policy = TRUE), error = function(e) NULL)
  if (!is.null(gc)) {
    geary_stat <- as.numeric(gc$statistic)
    geary_p <- gc$p.value
  }
  geary_clustered <- !is.na(geary_p) && geary_p < ALPHA

  ## --- Join Count Statistic (binary increase/decrease direction) ---
  joincount_stat <- NA; joincount_p <- NA
  direction_factor <- factor(ifelse(change > 0, "increase", "decrease"))
  if (length(unique(direction_factor)) == 2) {
    jc <- tryCatch(joincount.mc(direction_factor, lw, nsim = N_PERMUTATIONS, zero.policy = TRUE),
                    error = function(e) NULL)
    if (!is.null(jc) && length(jc) >= 1) {
      # joincount.mc returns one mc.sim object per same-category pair;
      # use the "increase-increase" same-category join count as the
      # summary statistic (a same-direction clustering test)
      same_cat <- jc[[1]]
      joincount_stat <- as.numeric(same_cat$statistic)
      joincount_p <- same_cat$p.value
    }
  }
  joincount_clustered <- !is.na(joincount_p) && joincount_p < ALPHA

  ## --- Neighbor-average correlation (non-standard, non-spatial-weight) ---
  nb_names <- lapply(boroughs_present, function(b) {
    row <- adj[adj[[adj_cols[1]]] == b | adj[[adj_cols[2]]] == b, ]
    unique(c(row[[adj_cols[1]]], row[[adj_cols[2]]]))[
      unique(c(row[[adj_cols[1]]], row[[adj_cols[2]]])) != b]
  })
  names(nb_names) <- boroughs_present
  neighbor_avg_change <- sapply(boroughs_present, function(b) {
    n <- nb_names[[b]]
    n <- n[n %in% names(change)]
    if (length(n) == 0) NA else mean(change[n])
  })
  valid <- !is.na(neighbor_avg_change)
  corr_stat <- NA; corr_p <- NA
  if (sum(valid) > 5) {
    obs_r <- suppressWarnings(cor(change[valid], neighbor_avg_change[valid]))
    perm_rs <- replicate(N_PERMUTATIONS, {
      suppressWarnings(cor(sample(change[valid]), neighbor_avg_change[valid]))
    })
    perm_rs <- perm_rs[!is.na(perm_rs)]
    if (!is.na(obs_r) && length(perm_rs) > 0) {
      corr_stat <- obs_r
      corr_p <- (sum(abs(perm_rs) >= abs(obs_r)) + 1) / (length(perm_rs) + 1)
    }
  }
  corr_clustered <- !is.na(corr_p) && corr_p < ALPHA

  results[[length(results) + 1]] <- data.frame(
    chunk_id = chunk_id,
    method = c("Geary's C", "Join Count Statistic", "Neighbor-average correlation (non-standard)"),
    gt_stat = c(geary_stat, joincount_stat, corr_stat),
    gt_p = c(geary_p, joincount_p, corr_p),
    gt_clustered = c(geary_clustered, joincount_clustered, corr_clustered),
    stringsAsFactors = FALSE
  )
}

computed <- do.call(rbind, results)

## --- Global Moran's I: copied verbatim from the manifest, NOT recomputed ---
moran_stats <- sapply(manifest$ground_truth_metric_json, function(j) {
  tryCatch(fromJSON(j)$moran_i, error = function(e) NA)
})
moran_ps <- sapply(manifest$ground_truth_metric_json, function(j) {
  tryCatch(fromJSON(j)$p_value, error = function(e) NA)
})

moran_from_manifest <- data.frame(
  chunk_id = manifest$chunk_id,
  method = "Global Moran's I",
  gt_stat = moran_stats,
  gt_p = moran_ps,
  gt_clustered = !is.na(moran_ps) & moran_ps < ALPHA,
  stringsAsFactors = FALSE
)

out <- rbind(computed, moran_from_manifest)
out <- out[order(out$chunk_id, out$method), ]

write.csv(out, OUTPUT_FILE, row.names = FALSE)
cat("Wrote", nrow(out), "rows to", OUTPUT_FILE, "\n")
cat("(", nrow(moran_from_manifest), "of those rows are Global Moran's I, copied verbatim from the manifest )\n")
