## ground_truth_all_methods_changespillover.R
## ==============================================
## For each change-spillover chunk, computes ground-truth
## CLUSTERED/NOT-CLUSTERED verdicts for the OTHER candidate methods,
## for BOTH the original manifest and manifest2, in one run. Global
## Moran's I is copied verbatim from each manifest's own
## ground_truth_metric_json -- never recomputed, zero deviation risk.
##
## FIX applied: the previous version matched change-period columns by
## searching for the YEAR as a substring in column names. This breaks
## silently (computes delta = 0 for every borough, no error, no skip)
## whenever window_start == window_end -- which is the normal case for
## manifest2's within-year quarter/half pairs (e.g. Q1_2021 -> Q2_2021).
## This version instead reads the EXACT period labels straight out of
## each manifest's own ground_truth_metric_json (period_start/period_end
## for manifest2, year_start/year_end for the original manifest) and
## matches columns against those exact labels -- never a substring
## guess. A same-year pair now resolves correctly instead of silently
## computing a false "no change" everywhere.
##
## Also applies the same chunk-file-resolution fix used for the
## Getis-Ord ground truth script: chunk files are located by FILENAME
## inside CHUNKS_DIR, not by trusting whatever folder each manifest's
## csv_path column happens to say -- so a future folder-name mismatch
## (like the one that hit getisord_manifest2) can't silently break this
## script the same way.
##
## Methods computed fresh, all via spdep directly (unchanged from the
## original): Geary's C, Join Count Statistic (direction of change),
## Neighbor-average correlation (non-standard, permutation-tested).
## Local Moran's I / LISA is still NOT computed, for the same reason as
## before: this task is GLOBAL clustering only, and a local statistic
## has no single non-arbitrary global collapse rule.
##
## REQUIRES: spdep, jsonlite. Install once with:
##   install.packages(c("spdep", "jsonlite"))

library(spdep)
library(jsonlite)

## ============================ CONFIG ============================
ADJACENCY_FILE <- "chunks_csv/borough_adjacency.csv"

# The ONE real folder where every changespillover chunk CSV actually
# lives, regardless of what any manifest's csv_path column says.
CHUNKS_DIR <- "chunks_csv/changespillover"

JOBS <- list(
  list(manifest = "chunks_csv/changespillover/changespillover_manifest.csv",
       output   = "ground truth/ground_truth_all_methods_changespillover.csv"),
  list(manifest = "chunks_csv/changespillover/changespillover_manifest2.csv",
       output   = "ground truth/ground_truth_all_methods_changespillover_additional.csv")
)

N_PERMUTATIONS <- 999
ALPHA          <- 0.05
# ==================================================================

resolve_chunk_path <- function(csv_path) {
  candidate <- file.path(CHUNKS_DIR, basename(csv_path))
  if (file.exists(candidate)) return(candidate)
  if (file.exists(csv_path)) return(csv_path)
  candidate
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

## Resolves the exact start/end period LABELS for a chunk from its
## manifest row's ground_truth_metric_json -- tries manifest2's key
## names (period_start/period_end) first, then falls back to the
## original manifest's key names (year_start/year_end). Returns NULL
## if neither is present, so the caller can report a real failure
## rather than silently guessing.
resolve_period_labels <- function(metric_json) {
  parsed <- tryCatch(fromJSON(metric_json), error = function(e) NULL)
  if (is.null(parsed)) return(NULL)
  if (!is.null(parsed$period_start) && !is.null(parsed$period_end)) {
    return(list(start = as.character(parsed$period_start), end = as.character(parsed$period_end)))
  }
  if (!is.null(parsed$year_start) && !is.null(parsed$year_end)) {
    return(list(start = as.character(parsed$year_start), end = as.character(parsed$year_end)))
  }
  NULL
}

## Finds the value column for a given period label inside a chunk's
## column names. Tries, in order: an EXACT match on the label itself
## (original manifest: columns literally named "2021", "2022"), then
## an exact match on "<label>_value" (manifest2: columns named
## "Q3_2021_value"). Never falls back to a bare substring search --
## if neither exact form is found, returns NA so the caller can skip
## and report the chunk rather than silently mismatching columns.
find_period_column <- function(col_names, label) {
  if (label %in% col_names) return(label)
  labeled <- paste0(label, "_value")
  if (labeled %in% col_names) return(labeled)
  NA_character_
}

compute_change <- function(csv_path, period_start, period_end) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
  borough_col <- names(d)[1]
  
  col_start <- find_period_column(names(d), period_start)
  col_end   <- find_period_column(names(d), period_end)
  
  if (is.na(col_start) || is.na(col_end)) {
    return(list(change = NULL,
                error = sprintf("could not find exact columns for '%s' / '%s' among: %s",
                                period_start, period_end, paste(names(d), collapse = ", "))))
  }
  
  change <- d[[col_end]] - d[[col_start]]
  list(change = setNames(change, d[[borough_col]]), error = NULL)
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
    cat("Files not found anywhere:\n")
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
    
    labels <- resolve_period_labels(manifest$ground_truth_metric_json[i])
    if (is.null(labels)) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- could not read period labels from ground_truth_metric_json"))
      next
    }
    
    change_result <- tryCatch(compute_change(csv_path, labels$start, labels$end),
                              error = function(e) list(change = NULL, error = conditionMessage(e)))
    if (is.null(change_result$change)) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "-", change_result$error))
      next
    }
    change <- change_result$change
    change <- change[!is.na(change)]
    
    if (length(change) < 8) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- too few valid boroughs after removing NA:", length(change)))
      next
    }
    
    boroughs_present <- intersect(all_boroughs, names(change))
    change <- change[boroughs_present]
    lw <- tryCatch(build_listw_for(boroughs_present), error = function(e) NULL)
    if (is.null(lw)) {
      skip_reasons <- c(skip_reasons, paste(chunk_id, "- couldn't build weights"))
      next
    }
    
    ## --- Geary's C (999-permutation Monte Carlo) ---
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
        same_cat <- jc[[1]]
        joincount_stat <- as.numeric(same_cat$statistic)
        joincount_p <- same_cat$p.value
      }
    }
    joincount_clustered <- !is.na(joincount_p) && joincount_p < ALPHA
    
    ## --- Neighbor-average correlation (non-standard) ---
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
  
  cat(length(results), "of", nrow(manifest), "chunks successfully processed for Geary's C / Join Count / Neighbor-avg.\n")
  if (length(skip_reasons) > 0) {
    cat(length(skip_reasons), "chunks skipped:\n")
    cat(paste(" -", skip_reasons, collapse = "\n"), "\n")
  }
  cat("\n")
  
  computed <- if (length(results) > 0) do.call(rbind, results) else
    data.frame(chunk_id = character(0), method = character(0), gt_stat = numeric(0),
               gt_p = numeric(0), gt_clustered = logical(0))
  
  ## --- Global Moran's I: copied verbatim from the manifest ---
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
  cat("(", nrow(moran_from_manifest), "of those rows are Global Moran's I, copied verbatim from the manifest )\n\n")
}