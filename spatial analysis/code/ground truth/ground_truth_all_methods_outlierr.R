## ground_truth_all_methods_outlierr.R
## =====================================
## For each outlier-detection chunk (one target borough per chunk),
## computes ground-truth outlier verdicts for the OTHER candidate
## methods models used, in a simple long format (chunk_id, method,
## gt_direction, gt_z, gt_significant) that plugs directly into
## score_outlierr.py.
##
## Iterative-R is NOT recomputed -- it is copied verbatim from the
## manifest's ground_truth_direction / ground_truth_z_score columns,
## same principle as Lee's L in the bivariate task and Gi* in getisord,
## for zero deviation risk from your existing ground truth.
##
## Methods computed fresh (all via listw/regression built directly from
## the adjacency + value data, not hand-derived from memory of formulas
## where a real package function exists):
##   - Single-pass OLS regression (no masking correction): own ~
##     neighbor_avg fit ONCE on all boroughs, no exclusion/refit step.
##     This is the naive method the task prompt explicitly warns
##     against ("masking bias") -- computing it lets you quantify how
##     much masking actually shifts the answer vs Iterative-R.
##   - Robust regression: MASS::rlm() (Huber M-estimator, iteratively
##     reweighted least squares) -- a legitimate alternative way to
##     correct for masking bias, without the explicit two-stage
##     exclude-and-refit procedure.
##   - Neighbor mean/std z-score: z = (own_value - mean(neighbor
##     values)) / sd(neighbor values) for the TARGET borough's direct
##     neighbors only -- no regression at all, a simpler and more naive
##     comparison.
##   - Local Moran's I / LISA: spdep::localmoran(), with the target
##     borough's spatial-outlier QUADRANT (not cluster) mapped to a
##     direction -- High-Low quadrant (own value high, neighbors low)
##     -> outlier_high; Low-High -> outlier_low; High-High/Low-Low
##     (clusters, not outliers by this definition) -> not_outlier
##     regardless of significance.
##
## NOT computed (left to fall back to the manifest's Iterative-R answer
## if a model used one of these): Getis-Ord Gi/Gi* (measures local
## concentration with neighbors, not deviation FROM neighbors -- a
## different question), Mahalanobis distance, IQR, MAD (these need an
## assumed multivariate/reference-set definition this task's data
## doesn't naturally specify -- rather than guess, they're left
## unverified, consistent with how Pearson/univariate Moran's I were
## handled in the bivariate task).
##
## REQUIRES: spdep, MASS (MASS ships with base R by default).
##
## EDIT the CONFIG block, then source/run this file.

library(spdep)
library(MASS)

## ============================ CONFIG ============================
ADJACENCY_FILE  <- "chunks_csv/borough_adjacency.csv"
MANIFEST_FILE   <- "chunks_csv/outlierr/outlier_iterative_r_manifest.csv"
OUTPUT_FILE     <- "ground truth/ground_truth_all_methods_outlierr.csv"
Z_THRESHOLD     <- 1.96
# ==================================================================

manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)

adj <- read.csv(ADJACENCY_FILE, stringsAsFactors = FALSE)
adj_cols <- names(adj)[1:2]
all_boroughs <- sort(unique(c(adj[[adj_cols[1]]], adj[[adj_cols[2]]])))

build_neighbor_list <- function(boroughs_present) {
  nb_list <- setNames(vector("list", length(boroughs_present)), boroughs_present)
  for (b in boroughs_present) nb_list[[b]] <- character(0)
  sub_adj <- adj[adj[[adj_cols[1]]] %in% boroughs_present & adj[[adj_cols[2]]] %in% boroughs_present, ]
  for (i in seq_len(nrow(sub_adj))) {
    a <- sub_adj[[adj_cols[1]]][i]; b <- sub_adj[[adj_cols[2]]][i]
    nb_list[[a]] <- unique(c(nb_list[[a]], b))
    nb_list[[b]] <- unique(c(nb_list[[b]], a))
  }
  nb_list
}

build_listw_for <- function(boroughs_present, nb_names_list) {
  idx <- setNames(seq_along(boroughs_present), boroughs_present)
  nb <- lapply(boroughs_present, function(b) {
    n <- nb_names_list[[b]]
    if (length(n) == 0) 0L else unname(idx[n])
  })
  nb <- structure(nb, class = "nb")
  attr(nb, "region.id") <- boroughs_present
  nb2listw(nb, style = "W", zero.policy = TRUE)
}

direction_from_z <- function(z, significant) {
  if (is.na(z) || !significant) return("not_outlier")
  if (z > 0) "outlier_high" else "outlier_low"
}

results <- list()

for (i in seq_len(nrow(manifest))) {
  chunk_id <- manifest$chunk_id[i]
  csv_path <- manifest$csv_path[i]
  target <- manifest$target_borough[i]
  if (!file.exists(csv_path)) {
    message("Skipping ", chunk_id, " -- file not found: ", csv_path)
    next
  }
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
  borough_col <- names(d)[1]
  value_col   <- names(d)[sapply(d, is.numeric)][1]
  d <- d[!is.na(d[[value_col]]), ]
  boroughs_present <- intersect(all_boroughs, d[[borough_col]])
  if (!(target %in% boroughs_present) || length(boroughs_present) < 8) {
    message("Skipping ", chunk_id, " -- target missing or too few boroughs")
    next
  }
  d <- d[match(boroughs_present, d[[borough_col]]), ]
  x <- setNames(d[[value_col]], boroughs_present)
  target_idx <- which(boroughs_present == target)

  nb_names <- build_neighbor_list(boroughs_present)
  neighbor_avg <- sapply(boroughs_present, function(b) {
    n <- nb_names[[b]]
    n <- n[n %in% names(x)]
    if (length(n) == 0) NA else mean(x[n])
  })

  valid <- !is.na(neighbor_avg)
  x_valid <- x[valid]
  na_valid <- neighbor_avg[valid]
  boroughs_valid <- boroughs_present[valid]
  target_valid_idx <- which(boroughs_valid == target)

  ## --- Single-pass OLS regression (no masking correction) ---
  single_pass_z <- NA
  if (length(target_valid_idx) == 1 && length(x_valid) > 5) {
    fit <- tryCatch(lm(x_valid ~ na_valid), error = function(e) NULL)
    if (!is.null(fit)) {
      resid_std <- rstandard(fit)  # leverage-adjusted standardized residual
      single_pass_z <- resid_std[target_valid_idx]
    }
  }
  single_pass_sig <- !is.na(single_pass_z) && abs(single_pass_z) > Z_THRESHOLD
  single_pass_dir <- direction_from_z(single_pass_z, single_pass_sig)

  ## --- Robust regression: MASS::rlm (Huber M-estimator) ---
  robust_z <- NA
  if (length(target_valid_idx) == 1 && length(x_valid) > 5) {
    rfit <- tryCatch(rlm(x_valid ~ na_valid, maxit = 100), error = function(e) NULL)
    if (!is.null(rfit)) {
      resid_r <- residuals(rfit)
      scale_r <- rfit$s
      if (!is.null(scale_r) && scale_r > 0) {
        robust_z <- resid_r[target_valid_idx] / scale_r
      }
    }
  }
  robust_sig <- !is.na(robust_z) && abs(robust_z) > Z_THRESHOLD
  robust_dir <- direction_from_z(robust_z, robust_sig)

  ## --- Neighbor mean/std z-score (non-regression), target only ---
  nbr_z <- NA
  target_neighbors <- nb_names[[target]]
  target_neighbors <- target_neighbors[target_neighbors %in% names(x)]
  if (length(target_neighbors) >= 2) {
    nbr_vals <- x[target_neighbors]
    nbr_sd <- sd(nbr_vals)
    if (!is.na(nbr_sd) && nbr_sd > 0) {
      nbr_z <- (x[[target]] - mean(nbr_vals)) / nbr_sd
    }
  }
  nbr_sig <- !is.na(nbr_z) && abs(nbr_z) > Z_THRESHOLD
  nbr_dir <- direction_from_z(nbr_z, nbr_sig)

  ## --- Local Moran's I / LISA, target's spatial-outlier quadrant ---
  lm_z <- NA
  lm_dir <- "not_outlier"
  lw <- tryCatch(build_listw_for(boroughs_valid, nb_names), error = function(e) NULL)
  if (!is.null(lw) && length(x_valid) > 5) {
    lmres <- tryCatch(localmoran(x_valid, lw, zero.policy = TRUE), error = function(e) NULL)
    if (!is.null(lmres)) {
      zx <- scale(x_valid)[, 1]
      lag <- lag.listw(lw, zx, zero.policy = TRUE)
      Ii <- lmres[target_valid_idx, "Ii"]
      Z.Ii <- lmres[target_valid_idx, "Z.Ii"]
      lm_z <- Z.Ii
      lm_sig <- !is.na(Z.Ii) && abs(Z.Ii) > Z_THRESHOLD
      is_outlier_quadrant <- Ii < 0  # HL or LH quadrant, not a cluster
      if (lm_sig && is_outlier_quadrant) {
        lm_dir <- if (zx[target_valid_idx] > 0) "outlier_high" else "outlier_low"
      }
    }
  }

  results[[length(results) + 1]] <- data.frame(
    chunk_id = chunk_id,
    method = c("Single-pass OLS regression (no masking correction)",
               "Robust regression (Huber/RANSAC/Theil-Sen)",
               "Neighbor mean/std z-score (non-regression)",
               "Local Moran's I / LISA"),
    gt_direction = c(single_pass_dir, robust_dir, nbr_dir, lm_dir),
    gt_z = c(single_pass_z, robust_z, nbr_z, lm_z),
    gt_significant = c(single_pass_sig, robust_sig, nbr_sig, !is.na(lm_z) && abs(lm_z) > Z_THRESHOLD),
    stringsAsFactors = FALSE
  )
}

computed <- do.call(rbind, results)

## --- Iterative-R: copied verbatim from the manifest, NOT recomputed ---
iterative_r_from_manifest <- data.frame(
  chunk_id = manifest$chunk_id,
  method = "Iterative-R (Lu, Chen & Kou 2003)",
  gt_direction = manifest$ground_truth_direction,
  gt_z = manifest$ground_truth_z_score,
  gt_significant = manifest$ground_truth_direction != "not_outlier",
  stringsAsFactors = FALSE
)

out <- rbind(computed, iterative_r_from_manifest)
out <- out[order(out$chunk_id, out$method), ]

write.csv(out, OUTPUT_FILE, row.names = FALSE)
cat("Wrote", nrow(out), "rows to", OUTPUT_FILE, "\n")
cat("(", nrow(iterative_r_from_manifest), "of those rows are Iterative-R, copied verbatim from the manifest )\n")
