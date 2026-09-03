## ground_truth_all_methods.R
## ===========================
## For each bivariate chunk (Energy vs Crime, etc.), computes ground-truth
## values for the OTHER candidate statistics your models used (Bivariate/
## Cross-Moran's I, Pearson correlation, univariate Moran's I).
##
## Lee's L is NOT recomputed here -- it is copied verbatim from the
## manifest's existing ground_truth_lee_l / ground_truth_p_value /
## ground_truth_is_significant columns, so there is zero risk of
## deviation from the ground truth you already have.
##
## Only Lee's L and Bivariate/Cross-Moran's I actually test the manifest's
## question ("are high-X boroughs located near high/low-Y boroughs?").
## Plain Pearson correlation and plain univariate Moran's I are included for
## completeness/diagnostics only -- they answer a different question and
## should be flagged as methodologically invalid regardless of how close
## their number lands, per the review above.
##
## REQUIRES: spdep (for nb2listw / spatial weights). Install once with:
##   install.packages("spdep")
##
## EDIT the CONFIG block, then source/run this file.

library(spdep)

## ============================ CONFIG ============================
CHUNKS_DIR      <- "chunks_csv/bivariate"        # folder with per-chunk X/Y csvs
ADJACENCY_FILE  <- "chunks_csv/borough_adjacency.csv"
MANIFEST_FILE   <- "chunks_csv/bivariate/bivariate_manifest.csv"
OUTPUT_FILE     <- "ground truth/ground_truth_all_methods_bivariate.csv"
N_PERMUTATIONS  <- 999
# ==================================================================

manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)

## --- Build the spatial weights list (W) from the adjacency table ---
## Assumes adjacency_file has two columns of borough names that are
## neighbors of each other (edit column names if yours differ).
adj <- read.csv(ADJACENCY_FILE, stringsAsFactors = FALSE)
adj_cols <- names(adj)[1:2]

boroughs <- sort(unique(c(adj[[adj_cols[1]]], adj[[adj_cols[2]]])))
nb_list <- setNames(vector("list", length(boroughs)), boroughs)
for (b in boroughs) nb_list[[b]] <- integer(0)

idx <- setNames(seq_along(boroughs), boroughs)
for (i in seq_len(nrow(adj))) {
  a <- adj[[adj_cols[1]]][i]; b <- adj[[adj_cols[2]]][i]
  nb_list[[a]] <- unique(c(nb_list[[a]], idx[[b]]))
  nb_list[[b]] <- unique(c(nb_list[[b]], idx[[a]]))
}
nb <- structure(lapply(nb_list, function(x) if (length(x) == 0) 0L else x),
                 class = "nb")
attr(nb, "region.id") <- boroughs
lw <- nb2listw(nb, style = "W", zero.policy = TRUE)
W  <- listw2mat(lw)

## --- Generic permutation test wrapper: works for ANY statistic fn ---
perm_test <- function(x, y, stat_fn, n_perm = N_PERMUTATIONS) {
  observed <- stat_fn(x, y)
  perm_stats <- replicate(n_perm, stat_fn(x, sample(y)))
  # two-sided empirical p-value
  p <- (sum(abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1)
  list(stat = observed, p = p)
}

## --- Statistic definitions ---
## NOTE: Lee's L is intentionally NOT computed here. Per instruction,
## its ground truth is copied directly from the manifest's existing
## ground_truth_lee_l / ground_truth_p_value / ground_truth_is_significant
## columns below, so there is zero risk of it deviating from what you
## already have. Only the other three methods are computed fresh.

# Bivariate / Cross-Moran's I -- raw X_i vs lagged Y_j (asymmetric)
cross_moran_stat <- function(x, y) {
  zx <- scale(x)[, 1]; zy <- scale(y)[, 1]
  n <- length(x)
  S0 <- sum(W)
  num <- n * sum(zx * (W %*% zy))
  den <- S0 * sqrt(sum(zx^2) * sum(zy^2))
  as.numeric(num / den)
}

# Plain Pearson correlation (NO spatial weights -- included for reference
# only; does not test the manifest's spatial-neighbor question)
pearson_stat <- function(x, y) cor(x, y, method = "pearson")

# Plain univariate Moran's I of X alone (reference only -- ignores Y
# entirely, cannot answer a bivariate question by construction)
moran_x_stat <- function(x, y) {
  zx <- scale(x)[, 1]
  n <- length(x); S0 <- sum(W)
  as.numeric((n / S0) * sum((W %*% zx) * zx) / sum(zx^2))
}

METHODS <- list(
  "Bivariate / Cross Moran's I" = cross_moran_stat,
  "Pearson correlation"        = pearson_stat,
  "Moran's I (X only, ref.)"   = moran_x_stat
)

## --- Run for every chunk in the manifest ---
results <- list()
for (i in seq_len(nrow(manifest))) {
  chunk_id <- manifest$chunk_id[i]
  csv_path <- manifest$csv_path[i]
  if (!file.exists(csv_path)) {
    message("Skipping ", chunk_id, " -- file not found: ", csv_path)
    next
  }
  d <- read.csv(csv_path, stringsAsFactors = FALSE)

  # Assumes the chunk csv has a Borough id column plus two numeric value
  # columns (X and Y). Edit these column names to match your actual files.
  borough_col <- names(d)[1]
  value_cols  <- names(d)[sapply(d, is.numeric)]
  if (length(value_cols) < 2) {
    message("Skipping ", chunk_id, " -- couldn't find 2 numeric columns")
    next
  }
  x <- d[[value_cols[1]]]
  y <- d[[value_cols[2]]]

  # Reorder to match the adjacency's borough ordering
  ord <- match(boroughs, d[[borough_col]])
  x <- x[ord]; y <- y[ord]

  for (method_name in names(METHODS)) {
    fn <- METHODS[[method_name]]
    r <- tryCatch(perm_test(x, y, fn), error = function(e) list(stat = NA, p = NA))
    results[[length(results) + 1]] <- data.frame(
      chunk_id = chunk_id,
      method   = method_name,
      gt_stat  = r$stat,
      gt_p     = r$p,
      gt_significant = !is.na(r$p) && r$p < 0.05
    )
  }
}

out <- do.call(rbind, results)

## --- Lee's L: copied directly from the manifest, NOT recomputed ---
## This guarantees zero deviation from what you already have.
lee_from_manifest <- data.frame(
  chunk_id       = manifest$chunk_id,
  method         = "Lee's L",
  gt_stat        = manifest$ground_truth_lee_l,
  gt_p           = manifest$ground_truth_p_value,
  gt_significant = toupper(as.character(manifest$ground_truth_is_significant)) == "TRUE"
)

out <- rbind(out, lee_from_manifest)
out <- out[order(out$chunk_id, out$method), ]

write.csv(out, OUTPUT_FILE, row.names = FALSE)
cat("Wrote", nrow(out), "rows to", OUTPUT_FILE, "\n")
cat("(", nrow(lee_from_manifest), "of those rows are Lee's L, copied verbatim from the manifest -- not recomputed )\n")
