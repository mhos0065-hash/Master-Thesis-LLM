## ground_truth_all_methods_getisord.R
## =====================================
## Computes ground-truth hot/cold spot lists for the OTHER candidate
## methods models used (Getis-Ord Gi without self, Local Moran's I /
## LISA), for BOTH the original manifest and manifest2, in one run.
## Getis-Ord Gi* is copied verbatim from each manifest's own
## ground_truth_boroughs column -- never recomputed, zero deviation risk.
##
## FIX applied: manifest2's csv_path column points at a folder
## (chunks_csv/getisord_no2_crime/) that doesn't match where the actual
## chunk files live (chunks_csv/getisord/) -- a mismatch from how the
## manifest was originally generated, not a real missing-file problem.
## Rather than editing the manifest or moving files, this version
## resolves every chunk file by FILENAME inside a single real chunks
## directory, ignoring whatever folder the manifest's csv_path claims.
## This means BOTH manifests -- and any future one with the same kind
## of path mismatch -- work without ever touching the manifest or the
## file layout again.
##
## REQUIRES: spdep. Install once with: install.packages("spdep")

library(spdep)

## ============================ CONFIG ============================
ADJACENCY_FILE  <- "chunks_csv/borough_adjacency.csv"

# The ONE real folder where every getisord chunk CSV actually lives,
# regardless of what any manifest's csv_path column says.
CHUNKS_DIR <- "chunks_csv/getisord"

JOBS <- list(
  list(manifest = "chunks_csv/getisord/getisord_manifest.csv",
       output   = "ground truth/ground_truth_all_methods_getisord.csv"),
  list(manifest = "chunks_csv/getisord/getisord_manifest2.csv",
       output   = "ground truth/ground_truth_all_methods_getisord_additional.csv")
)

Z_THRESHOLD <- 1.96   # matches each manifest's own 95% acceptance rule
# ==================================================================

## Resolves a chunk file by filename inside CHUNKS_DIR, regardless of
## what directory the manifest's csv_path column points to. Falls back
## to the manifest's own path only if a same-named file genuinely isn't
## found in CHUNKS_DIR either -- so a real "file doesn't exist anywhere"
## case still gets reported honestly, not silently masked.
resolve_chunk_path <- function(csv_path) {
  candidate <- file.path(CHUNKS_DIR, basename(csv_path))
  if (file.exists(candidate)) return(candidate)
  if (file.exists(csv_path)) return(csv_path)
  candidate  # neither exists; return the CHUNKS_DIR guess so the
             # missing-file report below shows a sensible path
}

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

format_boroughs <- function(boroughs, values, significant) {
  keep <- significant & !is.na(values)
  if (!any(keep)) return("")
  direction <- ifelse(values[keep] > 0, "hot", "cold")
  paste(sprintf("%s:%s:%s", boroughs[keep], direction, round(values[keep], 6)), collapse = "; ")
}

cat("Working directory:", getwd(), "\n")
cat("Chunks directory (single source of truth):", CHUNKS_DIR, "\n")
if (!dir.exists(CHUNKS_DIR)) {
  cat("*** WARNING: CHUNKS_DIR does not exist -- check this path before continuing ***\n")
} else {
  cat("CHUNKS_DIR contains", length(list.files(CHUNKS_DIR, pattern = "\\.csv$")), "CSV files\n")
}
cat("\n")

for (job in JOBS) {
  MANIFEST_FILE <- job$manifest
  OUTPUT_FILE   <- job$output

  cat("============================================================\n")
  cat("Processing manifest:", MANIFEST_FILE, "\n")
  cat("============================================================\n")

  manifest <- read.csv(MANIFEST_FILE, stringsAsFactors = FALSE)
  manifest$csv_path_resolved <- vapply(manifest$csv_path, resolve_chunk_path, character(1))

  found   <- sum(file.exists(manifest$csv_path_resolved))
  missing <- nrow(manifest) - found
  cat(found, "of", nrow(manifest), "chunk CSV files found in", CHUNKS_DIR, ";", missing, "genuinely missing.\n")
  if (missing > 0) {
    cat("Files not found anywhere (in CHUNKS_DIR or at the manifest's own path):\n")
    print(basename(manifest$csv_path[!file.exists(manifest$csv_path_resolved)]))
  }
  cat("\n")

  results <- list()
  skip_reasons <- character(0)

  for (i in seq_len(nrow(manifest))) {
    chunk_id <- manifest$chunk_id[i]
    csv_path <- manifest$csv_path_resolved[i]

    if (!file.exists(csv_path)) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- file not found:", csv_path))
      next
    }
    d <- read.csv(csv_path, stringsAsFactors = FALSE)

    borough_col <- names(d)[1]
    numeric_cols <- names(d)[sapply(d, is.numeric)]
    if (length(numeric_cols) == 0) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- no numeric value column found"))
      next
    }
    value_col <- numeric_cols[1]

    d <- d[!is.na(d[[value_col]]), ]
    boroughs_present <- intersect(all_boroughs, d[[borough_col]])
    d <- d[match(boroughs_present, d[[borough_col]]), ]
    x <- d[[value_col]]

    if (length(x) < 5) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- too few valid boroughs:", length(x)))
      next
    }

    lw <- tryCatch(build_listw_for(boroughs_present), error = function(e) NULL)
    if (is.null(lw)) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- couldn't build weights"))
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
      is_cluster <- Ii > 0
      lm_direction_ok <- lm_sig & is_cluster
      lm_value <- ifelse(x > xbar, abs(Z.Ii), -abs(Z.Ii))
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

  cat(length(results) / 2, "of", nrow(manifest), "chunks successfully processed for Gi/LISA.\n")
  if (length(skip_reasons) > 0) {
    cat(length(skip_reasons), "chunks skipped:\n")
    cat(paste(" -", skip_reasons, collapse = "\n"), "\n")
  }
  cat("\n")

  computed <- if (length(results) > 0) do.call(rbind, results) else
    data.frame(chunk_id = character(0), method = character(0), ground_truth_boroughs = character(0))

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
  cat("(", nrow(gi_star_from_manifest), "of those rows are Getis-Ord Gi*, copied verbatim from the manifest )\n\n")
}