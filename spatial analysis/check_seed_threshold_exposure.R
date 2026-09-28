## =============================================================================
## check_seed_threshold_exposure.R
##
## Checks how many chunks, per method, have a ground-truth p-value close
## to the 0.05 significance threshold -- these are the chunks where
## Monte Carlo permutation seed variation could plausibly flip a verdict
## between CLUSTERED/NOT CLUSTERED (or the equivalent for each task).
##
## Run this BEFORE deciding whether to build any seed-variation handling
## into the scoring scripts -- if the affected share is small, it's
## better documented as a stated limitation than engineered around.
##
## HOW TO USE:
##   1. Edit ground_truth_dir below if your files live somewhere else.
##   2. Run the whole script: source("check_seed_threshold_exposure.R")
##      or Rscript check_seed_threshold_exposure.R
##   3. For each file, it FIRST prints the column names and a few sample
##      rows, so you can see the exact p-value column name if the
##      auto-detection below doesn't find it.
##   4. It THEN attempts to auto-detect the p-value column (tries
##      "gt_p", "p_value", "gt_significant", "p" in that order) and
##      prints the near-threshold summary per method automatically.
##   5. If auto-detection fails for a file, it tells you so -- rerun
##      just that file's check manually with the real column name (see
##      the check_near_threshold() function at the bottom).
## =============================================================================

if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
library(dplyr)

## ---- CONFIG ------------------------------------------------------------
## Edit this path if your ground truth CSVs live somewhere else.
ground_truth_dir <- "ground truth"

files_to_check <- c(
  "ground_truth_all_methods_bivariate.csv",
  "ground_truth_all_methods_spatial.csv",
  "ground_truth_all_methods_getisord.csv",
  "ground_truth_all_methods_changespillover.csv",
  "ground_truth_all_methods_outlierr.csv"
)

candidate_p_cols <- c("gt_p", "p_value", "gt_significant", "p")

## ---- FUNCTIONS -----------------------------------------------------------

inspect_file <- function(path) {
  cat("\n============================================================\n")
  cat("FILE:", path, "\n")
  cat("============================================================\n")
  if (!file.exists(path)) {
    cat("  >>> FILE NOT FOUND -- check the path/filename.\n")
    return(NULL)
  }
  d <- read.csv(path, stringsAsFactors = FALSE)
  cat("Columns:", paste(names(d), collapse = ", "), "\n\n")
  cat("First 3 rows:\n")
  print(utils::head(d, 3))
  d
}

check_near_threshold <- function(d, p_col, method_col = "method", band = 0.02) {
  if (!(p_col %in% names(d))) {
    cat("  >>> Column '", p_col, "' not found -- skipping.\n", sep = "")
    return(invisible(NULL))
  }
  if (!(method_col %in% names(d))) {
    method_col <- NULL  # fall back to a single overall summary if no method column
  }
  d2 <- d %>%
    filter(!is.na(.data[[p_col]])) %>%
    mutate(near_threshold = abs(.data[[p_col]] - 0.05) < band)

  if (is.null(method_col)) {
    result <- d2 %>%
      summarise(
        n_total            = n(),
        n_near_threshold   = sum(near_threshold),
        pct_near_threshold = round(100 * n_near_threshold / n_total, 1)
      )
  } else {
    result <- d2 %>%
      group_by(.data[[method_col]]) %>%
      summarise(
        n_total            = n(),
        n_near_threshold   = sum(near_threshold),
        pct_near_threshold = round(100 * n_near_threshold / n_total, 1),
        .groups = "drop"
      )
  }
  print(result)
  invisible(result)
}

auto_detect_and_check <- function(d) {
  found_col <- candidate_p_cols[candidate_p_cols %in% names(d)][1]
  if (is.na(found_col)) {
    cat("  >>> Could not auto-detect a p-value column from:",
        paste(candidate_p_cols, collapse = ", "), "\n")
    cat("  >>> Look at the column list above and re-run manually, e.g.:\n")
    cat("      check_near_threshold(d, p_col = \"<real_column_name>\")\n")
    return(invisible(NULL))
  }
  cat("\nAuto-detected p-value column: '", found_col, "'\n", sep = "")
  cat("Near-threshold summary (|p - 0.05| < 0.02), by method:\n")
  check_near_threshold(d, p_col = found_col)
}

## ---- RUN FOR EVERY FILE ----------------------------------------------------

all_results <- list()
for (fname in files_to_check) {
  path <- file.path(ground_truth_dir, fname)
  d <- inspect_file(path)
  if (!is.null(d)) {
    auto_detect_and_check(d)
    all_results[[fname]] <- d
  }
}

cat("\n============================================================\n")
cat("Done. If any file's p-value column wasn't auto-detected, use:\n")
cat("  check_near_threshold(all_results[[\"<filename>\"]], p_col = \"<real_column_name>\")\n")
cat("to run it manually with the correct column name.\n")
