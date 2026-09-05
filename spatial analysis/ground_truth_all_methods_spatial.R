## ground_truth_all_methods_spatial.R
## =====================================
## For each spatial-correlation chunk (Global OR Local Moran's I, one
## variable/year per chunk), computes ground-truth CLUSTERED/NOT
## verdicts for the OTHER candidate methods models used.
##
## Global Moran's I and Local Moran's I / LISA are NOT recomputed --
## both are copied verbatim from the manifest (ground_truth_metric_json
## for global; the significant_cluster_boroughs existence test for
## local), same principle as every other task's designated method.
##
## IMPORTANT sign-direction note: Moran's I and Geary's C use OPPOSITE
## conventions for "clustered". Moran's I: clustering means I > 0 (this
## manifest's own ground_truth_rule is explicit that p<0.05 alone is
## NOT enough -- a significant NEGATIVE I means dispersion, not
## clustering, and must not count as CLUSTERED). Geary's C: clustering
## means C < 1 (C > 1 means dispersion). Both directions are applied
## correctly below -- getting this backwards would silently invert
## every Geary's C verdict.
##
## GLOBAL methods computed fresh (via spdep directly):
##   - Geary's C (spdep::geary.mc)
##   - Join Count Statistic (spdep::joincount.mc, on a binary high/low
##     split of the raw values around the median)
##   - Neighbor-average correlation (non-standard): plain Pearson
##     correlation between each borough's own value and its neighbors'
##     mean value, significance via a 999-permutation test.
##
## LOCAL methods computed fresh:
##   - Getis-Ord Gi/Gi* existence test: computes Gi* for every borough
##     (spdep::localG(), self-inclusive weights) and asks "does at
##     least one |z| > 1.96 exist" -- a genuinely different definition
##     from Local Moran's I (Gi* detects local CONCENTRATION with
##     neighbors, not value-similarity CLUSTERS), but a plausible
##     alternative a model might reach for when asked to find specific
##     standout boroughs.
##
## REQUIRES: spdep, jsonlite. Install once with:
##   install.packages(c("spdep", "jsonlite"))
##
## EDIT the CONFIG block, then source/run this file.

library(spdep)
library(jsonlite)

## ============================ CONFIG ============================
MANIFEST_FILE   <- "chunks_csv/manifest.csv"
OUTPUT_FILE     <- "ground truth/ground_truth_all_methods_spatial.csv"
N_PERMUTATIONS  <- 999
ALPHA_GLOBAL    <- 0.05
ALPHA_LOCAL     <- 0.10  # matches this task's own Local Moran's I convention
Z_THRESHOLD     <- 1.96

## Adjacency file resolution, in priority order:
##   1. ADJACENCY_OVERRIDES -- a named list matching on a substring found
##      in chunk_id, checked FIRST. Use this for datasets that need a
##      DIFFERENT adjacency file than whatever the manifest says -- e.g.
##      cyclehire only covers 12 of the 33 boroughs, on a queen-contiguity
##      subgraph recomputed on just those 12 (see cyclehire_containment.R),
##      NOT the standard 33-borough file. Add an entry here once you have
##      that file (see the chat for how to export it from your own script).
##   2. Otherwise, falls back to manifest$adjacency_path per chunk (this
##      is correct for the moran_* chunks, which DO use the standard file).
## Set the RHS to "" or comment out an entry to disable it and fall back
## to manifest$adjacency_path for that dataset.
ADJACENCY_OVERRIDES <- list(
  "cyclehire" = "chunks_csv/borough_adjacency.csv"   # <- put the exported cyclehire adjacency CSV path here, e.g.
                      #    "chunks_csv/cyclehire_adjacency.csv"
)
# ==================================================================

manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)

## Adjacency is read PER CHUNK from manifest$adjacency_path, not a
## single global file -- different chunks (e.g. cyclehire, which uses
## a different spatial unit than the 33 London boroughs) may specify a
## different adjacency file. Cached so the same file isn't re-read for
## every chunk that shares it.
.adjacency_cache <- new.env()
get_adjacency_for <- function(path) {
  if (!exists(path, envir = .adjacency_cache)) {
    if (!file.exists(path)) return(NULL)
    a <- read.csv(path, stringsAsFactors = FALSE)
    assign(path, a, envir = .adjacency_cache)
  }
  get(path, envir = .adjacency_cache)
}

## Resolves the adjacency path for one chunk: checks ADJACENCY_OVERRIDES
## for a substring match against chunk_id first (skipping any entry set
## to ""), otherwise uses the manifest's own per-row adjacency_path.
resolve_adjacency_path <- function(chunk_id, manifest_path) {
  for (key in names(ADJACENCY_OVERRIDES)) {
    override_path <- ADJACENCY_OVERRIDES[[key]]
    if (nzchar(override_path) && grepl(key, chunk_id, fixed = TRUE)) {
      return(override_path)
    }
  }
  manifest_path
}

build_listw_for <- function(boroughs_present, adj, adj_cols, include_self = FALSE) {
  idx <- setNames(seq_along(boroughs_present), boroughs_present)
  nb_list <- setNames(vector("list", length(boroughs_present)), boroughs_present)
  for (b in boroughs_present) nb_list[[b]] <- integer(0)

  sub_adj <- adj[adj[[adj_cols[1]]] %in% boroughs_present & adj[[adj_cols[2]]] %in% boroughs_present, ]
  for (i in seq_len(nrow(sub_adj))) {
    a <- sub_adj[[adj_cols[1]]][i]; b <- sub_adj[[adj_cols[2]]][i]
    nb_list[[a]] <- unique(c(nb_list[[a]], idx[[b]]))
    nb_list[[b]] <- unique(c(nb_list[[b]], idx[[a]]))
  }
  if (include_self) {
    for (b in boroughs_present) nb_list[[b]] <- unique(c(nb_list[[b]], idx[[b]]))
  }
  nb <- structure(lapply(nb_list, function(x) if (length(x) == 0) 0L else x), class = "nb")
  attr(nb, "region.id") <- boroughs_present
  nb2listw(nb, style = "W", zero.policy = TRUE)
}

read_values <- function(csv_path) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
  borough_col <- names(d)[1]
  value_col <- names(d)[sapply(d, is.numeric)][1]
  d <- d[!is.na(d[[value_col]]), ]
  setNames(d[[value_col]], d[[borough_col]])
}

results <- list()

for (i in seq_len(nrow(manifest))) {
  chunk_id <- manifest$chunk_id[i]
  csv_path <- manifest$csv_path[i]
  granularity <- manifest$spatial_granularity[i]
  if (!file.exists(csv_path)) {
    message("Skipping ", chunk_id, " -- file not found: ", csv_path)
    next
  }

  adjacency_path <- resolve_adjacency_path(chunk_id, manifest$adjacency_path[i])
  adj <- get_adjacency_for(adjacency_path)
  if (is.null(adj)) {
    message("Skipping ", chunk_id, " -- adjacency file not found: ", adjacency_path)
    next
  }
  adj_cols <- names(adj)[1:2]
  all_boroughs <- sort(unique(c(adj[[adj_cols[1]]], adj[[adj_cols[2]]])))

  x <- tryCatch(read_values(csv_path), error = function(e) NULL)
  if (is.null(x) || length(x) < 8) {
    message("Skipping ", chunk_id, " -- couldn't read values or too few boroughs")
    next
  }
  boroughs_present <- intersect(all_boroughs, names(x))
  if (length(boroughs_present) < 8) {
    message("Skipping ", chunk_id, " -- fewer than 8 names in the data match the adjacency file",
            " (", length(boroughs_present), " matched) -- check adjacency_path is the right file",
            " for this chunk's spatial unit")
    next
  }
  x <- x[boroughs_present]

  if (granularity == "global") {
    lw <- tryCatch(build_listw_for(boroughs_present, adj, adj_cols), error = function(e) NULL)
    if (is.null(lw)) { message("Skipping ", chunk_id, " -- weights failed"); next }

    ## --- Geary's C (999-perm Monte Carlo) -- clustering means C < 1 ---
    geary_stat <- NA; geary_p <- NA
    gc <- tryCatch(geary.mc(x, lw, nsim = N_PERMUTATIONS, zero.policy = TRUE), error = function(e) NULL)
    if (!is.null(gc)) { geary_stat <- as.numeric(gc$statistic); geary_p <- gc$p.value }
    geary_clustered <- !is.na(geary_p) && !is.na(geary_stat) && geary_p < ALPHA_GLOBAL && geary_stat < 1

    ## --- Join Count Statistic (binary high/low split around median) ---
    joincount_stat <- NA; joincount_p <- NA
    direction_factor <- factor(ifelse(x > median(x), "high", "low"))
    joincount_clustered <- FALSE
    if (length(unique(direction_factor)) == 2) {
      jc <- tryCatch(joincount.mc(direction_factor, lw, nsim = N_PERMUTATIONS, zero.policy = TRUE),
                      error = function(e) NULL)
      if (!is.null(jc) && length(jc) >= 1) {
        same_cat <- jc[[1]]
        joincount_stat <- as.numeric(same_cat$statistic)
        joincount_p <- same_cat$p.value
        joincount_clustered <- !is.na(joincount_p) && joincount_p < ALPHA_GLOBAL
      }
    }

    ## --- Neighbor-average correlation (non-standard) ---
    nb_names <- lapply(boroughs_present, function(b) {
      row <- adj[adj[[adj_cols[1]]] == b | adj[[adj_cols[2]]] == b, ]
      cand <- unique(c(row[[adj_cols[1]]], row[[adj_cols[2]]]))
      cand[cand != b]
    })
    names(nb_names) <- boroughs_present
    neighbor_avg <- sapply(boroughs_present, function(b) {
      n <- nb_names[[b]]; n <- n[n %in% names(x)]
      if (length(n) == 0) NA else mean(x[n])
    })
    valid <- !is.na(neighbor_avg)
    corr_stat <- NA; corr_p <- NA
    if (sum(valid) > 5) {
      obs_r <- suppressWarnings(cor(x[valid], neighbor_avg[valid]))
      perm_rs <- replicate(N_PERMUTATIONS, suppressWarnings(cor(sample(x[valid]), neighbor_avg[valid])))
      perm_rs <- perm_rs[!is.na(perm_rs)]
      if (!is.na(obs_r) && length(perm_rs) > 0) {
        corr_stat <- obs_r
        corr_p <- (sum(abs(perm_rs) >= abs(obs_r)) + 1) / (length(perm_rs) + 1)
      }
    }
    corr_clustered <- !is.na(corr_p) && !is.na(corr_stat) && corr_p < ALPHA_GLOBAL && corr_stat > 0

    results[[length(results) + 1]] <- data.frame(
      chunk_id = chunk_id,
      method = c("Geary's C", "Join Count Statistic", "Neighbor-average correlation (non-standard)"),
      gt_stat = c(geary_stat, joincount_stat, corr_stat),
      gt_clustered = c(geary_clustered, joincount_clustered, corr_clustered),
      stringsAsFactors = FALSE
    )

  } else if (granularity == "local") {
    ## --- Getis-Ord Gi* existence test (self-inclusive weights) ---
    lw_star <- tryCatch(build_listw_for(boroughs_present, adj, adj_cols, include_self = TRUE),
                         error = function(e) NULL)
    if (is.null(lw_star)) {
      message("Skipping Getis-Ord for ", chunk_id, " -- weights failed (Local Moran's I still",
              " copied from manifest below, unaffected)")
      next
    }
    gi <- tryCatch(as.numeric(localG(x, lw_star, zero.policy = TRUE)), error = function(e) NULL)
    if (is.null(gi)) {
      message("Skipping Getis-Ord for ", chunk_id, " -- localG() failed")
      next
    }
    gi_exists <- any(!is.na(gi) & abs(gi) > Z_THRESHOLD)

    results[[length(results) + 1]] <- data.frame(
      chunk_id = chunk_id,
      method = "Getis-Ord Gi / Gi*",
      gt_stat = NA,
      gt_clustered = gi_exists,  # a real TRUE/FALSE at this point -- never a silently-defaulted NA-as-FALSE
      stringsAsFactors = FALSE
    )
  }
}

computed <- do.call(rbind, results)

## --- Global Moran's I: copied verbatim from the manifest ---
global_idx <- manifest$spatial_granularity == "global"
moran_stats <- rep(NA, nrow(manifest))
moran_ps <- rep(NA, nrow(manifest))
moran_stats[global_idx] <- sapply(manifest$ground_truth_metric_json[global_idx], function(j) {
  tryCatch(fromJSON(j)$moran_i, error = function(e) NA)
})
moran_ps[global_idx] <- sapply(manifest$ground_truth_metric_json[global_idx], function(j) {
  tryCatch(fromJSON(j)$p_value, error = function(e) NA)
})
moran_clustered <- global_idx & !is.na(moran_ps) & moran_ps < ALPHA_GLOBAL & !is.na(moran_stats) & moran_stats > 0

global_from_manifest <- data.frame(
  chunk_id = manifest$chunk_id[global_idx],
  method = "Global Moran's I",
  gt_stat = moran_stats[global_idx],
  gt_clustered = moran_clustered[global_idx],
  stringsAsFactors = FALSE
)

## --- Local Moran's I / LISA: copied verbatim from the manifest ---
## Existence test: ground_truth_present already IS this test, computed
## by the manifest generator via the significant_cluster_boroughs dict.
local_idx <- manifest$spatial_granularity == "local"
local_from_manifest <- data.frame(
  chunk_id = manifest$chunk_id[local_idx],
  method = "Local Moran's I / LISA",
  gt_stat = NA,
  gt_clustered = as.logical(manifest$ground_truth_present[local_idx]),
  stringsAsFactors = FALSE
)

out <- rbind(computed, global_from_manifest, local_from_manifest)
out <- out[order(out$chunk_id, out$method), ]

write.csv(out, OUTPUT_FILE, row.names = FALSE)
cat("Wrote", nrow(out), "rows to", OUTPUT_FILE, "\n")
cat("(", nrow(global_from_manifest), "Global Moran's I +", nrow(local_from_manifest),
    "Local Moran's I rows copied verbatim from the manifest )\n")
