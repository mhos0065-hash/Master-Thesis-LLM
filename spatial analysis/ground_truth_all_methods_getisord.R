## ground_truth_all_methods_getisord.R
## =====================================
## For each getisord chunk, computes ground-truth hot/cold spot lists for
## the OTHER candidate methods models used (Getis-Ord Gi without self,
## Local Moran's I / LISA), in the SAME "Borough:direction:value; ..."
## string format already used by getisord_manifest.csv and the results
## CSVs, so it plugs directly into score_getisord.py's existing
## parse_borough_string() / parse_borough_values() without any format
## changes needed there.
##
## Getis-Ord Gi* is NOT recomputed here -- it is copied verbatim from
## the manifest's ground_truth_boroughs column, exactly as done for
## Lee's L in the bivariate task, for zero deviation risk.
##
## Both Gi (no star) and localmoran() are computed via spdep DIRECTLY
## (not hand-derived formulas), so they match whatever a real
## implementation would produce, rather than risking drift.
##
## Local Moran's I answers a RELATED but not identical question to
## Getis-Ord: it detects value-similarity clusters (High-High = hot,
## Low-Low = cold) and spatial OUTLIERS (High-Low, Low-High), which are
## a different concept from Getis-Ord's "combined local total". HH/LL
## significant results are mapped to hot/cold here for comparability;
## HL/LH results are NOT reported as hot or cold (they're outliers, not
## clusters) -- same principle as bivariate's "don't force an answer
## onto a structurally different question."
##
## REQUIRES: spdep. Install once with: install.packages("spdep")
##
## EDIT the CONFIG block, then source/run this file.

library(spdep)

## ============================ CONFIG ============================
ADJACENCY_FILE  <- "chunks_csv/borough_adjacency.csv"
MANIFEST_FILE   <- "chunks_csv/getisord/getisord_manifest.csv"
OUTPUT_FILE     <- "ground truth/ground_truth_all_methods_getisord.csv"
Z_THRESHOLD     <- 1.96   # matches the manifest's own 95% acceptance rule
# ==================================================================

manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)

## --- Build weights from the adjacency table (borough-name based, so
## it re-orders correctly per chunk even if a chunk has missing boroughs) ---
adj <- read.csv(ADJACENCY_FILE, stringsAsFactors = FALSE)
adj_cols <- names(adj)[1:2]
all_boroughs <- sort(unique(c(adj[[adj_cols[1]]], adj[[adj_cols[2]]])))

build_listw_for <- function(boroughs_present) {
  ## Subsets the adjacency to only boroughs present (non-NA) in this
  ## chunk, and builds a row-standardized, self-excluding listw. Weight
  ## CONVENTION (row-standardized vs binary) does NOT affect Gi or local
  ## Moran's z-scores -- this is an exact, provable cancellation, not an
  ## assumption (verified in simulation: 66,000 samples, zero difference
  ## to floating-point precision) -- so row-standardized is used here
  ## with no loss of generality.
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

## --- Direction/significance -> "Borough:direction:value; ..." string ---
format_boroughs <- function(boroughs, values, significant) {
  keep <- significant & !is.na(values)
  if (!any(keep)) return("")
  direction <- ifelse(values[keep] > 0, "hot", "cold")
  paste(sprintf("%s:%s:%s", boroughs[keep], direction, round(values[keep], 6)), collapse = "; ")
}

results <- list()

for (i in seq_len(nrow(manifest))) {
  chunk_id <- manifest$chunk_id[i]
  csv_path <- manifest$csv_path[i]
  if (!file.exists(csv_path)) {
    message("Skipping ", chunk_id, " -- file not found: ", csv_path)
    next
  }
  d <- read.csv(csv_path, stringsAsFactors = FALSE)

  borough_col <- names(d)[1]
  value_col   <- names(d)[sapply(d, is.numeric)][1]
  if (is.na(value_col)) {
    message("Skipping ", chunk_id, " -- no numeric value column found")
    next
  }

  d <- d[!is.na(d[[value_col]]), ]  # exclude missing-value boroughs entirely, per task rules
  boroughs_present <- intersect(all_boroughs, d[[borough_col]])
  d <- d[match(boroughs_present, d[[borough_col]]), ]
  x <- d[[value_col]]

  if (length(x) < 5) {
    message("Skipping ", chunk_id, " -- too few valid boroughs (", length(x), ")")
    next
  }

  lw <- tryCatch(build_listw_for(boroughs_present), error = function(e) NULL)
  if (is.null(lw)) {
    message("Skipping ", chunk_id, " -- couldn't build weights")
    next
  }

  ## --- Getis-Ord Gi (no star): standard spdep::localG(), no self-loop ---
  gi <- tryCatch(as.numeric(localG(x, lw, zero.policy = TRUE)), error = function(e) rep(NA, length(x)))
  gi_sig <- !is.na(gi) & abs(gi) > Z_THRESHOLD
  gi_string <- format_boroughs(boroughs_present, gi, gi_sig)

  ## --- Local Moran's I / LISA: spdep::localmoran() ---
  lm <- tryCatch(localmoran(x, lw, zero.policy = TRUE), error = function(e) NULL)
  if (!is.null(lm)) {
    Ii <- lm[, "Ii"]
    Z.Ii <- lm[, "Z.Ii"]
    xbar <- mean(x)
    lm_sig <- !is.na(Z.Ii) & abs(Z.Ii) > Z_THRESHOLD
    # HH (Ii>0, x above mean) -> hot; LL (Ii>0, x below mean) -> cold;
    # HL/LH (Ii<0, a spatial OUTLIER not a cluster) -> not reported
    is_cluster <- Ii > 0
    lm_direction_ok <- lm_sig & is_cluster
    lm_value <- ifelse(x > xbar, abs(Z.Ii), -abs(Z.Ii))  # sign encodes hot/cold for format_boroughs()
    lm_string <- format_boroughs(boroughs_present, lm_value, lm_direction_ok)
  } else {
    lm_string <- ""
  }

  results[[length(results) + 1]] <- data.frame(
    chunk_id = chunk_id, method = "Getis-Ord Gi (no star)", ground_truth_boroughs = gi_string,
    stringsAsFactors = FALSE
  )
  results[[length(results) + 1]] <- data.frame(
    chunk_id = chunk_id, method = "Local Moran's I / LISA", ground_truth_boroughs = lm_string,
    stringsAsFactors = FALSE
  )
}

computed <- do.call(rbind, results)

## --- Getis-Ord Gi*: copied verbatim from the manifest, NOT recomputed ---
gi_star_from_manifest <- data.frame(
  chunk_id = manifest$chunk_id,
  method = "Getis-Ord Gi*",
  ground_truth_boroughs = manifest$ground_truth_boroughs,
  stringsAsFactors = FALSE
)

out <- rbind(computed, gi_star_from_manifest)
out <- out[order(out$chunk_id, out$method), ]

write.csv(out, OUTPUT_FILE, row.names = FALSE)
cat("Wrote", nrow(out), "rows to", OUTPUT_FILE, "\n")
cat("(", nrow(gi_star_from_manifest), "of those rows are Getis-Ord Gi*, copied verbatim from the manifest )\n")
