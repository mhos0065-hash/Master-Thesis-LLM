# =============================================================================
# plot_single_vs_multi.R
# =============================================================================
# Reads the two "sectioned report" style CSV files (one per condition),
# extracts the EVIDENCE-FILTERED results for each of the 6 tasks, lets you
# select which tasks to include in the comparison, averages accuracy and F1
# across the selected tasks per model, and plots two grouped bar charts
# (accuracy, F1) comparing single-turn vs multi-turn.
#
# Assumes both CSV files are in the same directory as this script.
# Uses only base R (no external packages required).
# =============================================================================

single_file <- "Follow up_summary_single agent_single_shot.csv"
multi_file  <- "Follow up_summary_single agent_multi_shot.csv"

# ---------------------------------------------------------------------------
# >>> TASK SELECTION PROVISION <<<
# Only the tasks listed here will be included in the average. Comment out
# (or remove) any task you don't want considered. Names must match exactly
# how they appear as section headers in the CSV files.
# ---------------------------------------------------------------------------
selected_tasks <- c(
  "2. Bivariate spatial correlation",
  "3. GetISord (Hotspot detection)",
  "6. Changes pill over analysis",
   "1. Global and Local spatial correlation (Global and Local moran I)",
   "4. Spatial outlier analysis"
  # "5. Spatial Distance decay analysis",
)

# ---------------------------------------------------------------------------
# Parser: reads a "sectioned report" CSV (task headers, "without/after
# filtering" markers, a "model" header row, then data rows) and returns a
# clean, tidy data frame with one row per (task, filter, model).
# ---------------------------------------------------------------------------
parse_report <- function(path, source_mode) {
  lines <- readLines(path, encoding = "latin1", warn = FALSE)

  current_task <- NA
  current_filter <- NA
  in_data_block <- FALSE
  out_rows <- list()

  for (line in lines) {
    fields <- strsplit(line, ",")[[1]]
    fields <- trimws(gsub('"', '', fields))
    non_empty <- fields[fields != ""]
    first_text <- if (length(non_empty) > 0) non_empty[1] else ""

    if (grepl("^[0-9]+\\.\\s", first_text)) {
      current_task <- trimws(first_text)
      in_data_block <- FALSE
      next
    }
    if (grepl("without filtering", first_text, ignore.case = TRUE)) {
      current_filter <- "raw"
      in_data_block <- FALSE
      next
    }
    if (grepl("after filtering|with filtering", first_text, ignore.case = TRUE)) {
      current_filter <- "evidence"
      in_data_block <- FALSE
      next
    }
    if (trimws(first_text) == "model") {
      in_data_block <- TRUE
      next
    }
    if (in_data_block && nzchar(trimws(first_text)) && grepl("/", first_text)) {
      vals <- non_empty
      if (length(vals) >= 12) {
        out_rows[[length(out_rows) + 1]] <- data.frame(
          source_mode = source_mode,
          task        = current_task,
          filter      = current_filter,
          model       = vals[1],
          n           = suppressWarnings(as.numeric(vals[2])),
          n_ambiguous = suppressWarnings(as.numeric(vals[3])),
          correct     = suppressWarnings(as.numeric(vals[5])),
          accuracy    = suppressWarnings(as.numeric(vals[9])),
          precision   = suppressWarnings(as.numeric(vals[10])),
          recall      = suppressWarnings(as.numeric(vals[11])),
          f1          = suppressWarnings(as.numeric(vals[12])),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  do.call(rbind, out_rows)
}

single_df <- parse_report(single_file, "single")
multi_df  <- parse_report(multi_file, "multi")
combined  <- rbind(single_df, multi_df)

# de-duplicate in case a section appears twice in a source file (keep last)
combined <- combined[!duplicated(combined[, c("source_mode","task","filter","model")],
                                  fromLast = TRUE), ]

# keep only evidence-filtered rows, and only the selected tasks
ef <- combined[combined$filter == "evidence" & combined$task %in% selected_tasks, ]

cat("Tasks actually found and included:\n")
print(unique(ef$task))
cat("\n")

# ---------------------------------------------------------------------------
# Average accuracy and F1 across the selected tasks, per model, per condition
# ---------------------------------------------------------------------------
agg <- aggregate(cbind(accuracy, f1) ~ model + source_mode, data = ef, FUN = mean, na.rm = TRUE)

models <- sort(unique(agg$model))
acc_mat <- matrix(NA, nrow = 2, ncol = length(models),
                   dimnames = list(c("single","multi"), models))
f1_mat  <- acc_mat

for (m in models) {
  s_row <- agg[agg$model == m & agg$source_mode == "single", ]
  m_row <- agg[agg$model == m & agg$source_mode == "multi", ]
  if (nrow(s_row) == 1) { acc_mat["single", m] <- s_row$accuracy; f1_mat["single", m] <- s_row$f1 }
  if (nrow(m_row) == 1) { acc_mat["multi", m]  <- m_row$accuracy; f1_mat["multi", m]  <- m_row$f1 }
}

cat("Average accuracy (single vs multi), per model:\n")
print(round(acc_mat, 3))
cat("\nAverage F1 (single vs multi), per model:\n")
print(round(f1_mat, 3))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
short_names <- sub("^.*/", "", models)  # drop the "provider/" prefix for readable labels

png("accuracy_single_vs_multi.png", width = 1000, height = 650, res = 120)
par(mar = c(9, 4.5, 4, 1))
bp <- barplot(acc_mat, beside = TRUE, names.arg = short_names, las = 2,
              col = c("#2a78d6", "#eb6834"), ylim = c(0, 1),
              ylab = "Accuracy (evidence-filtered)",
              main = "Accuracy: single-turn vs multi-turn\n(averaged across selected tasks)",
              cex.names = 0.85)
legend("topright", legend = c("Single-turn", "Multi-turn"), fill = c("#2a78d6", "#eb6834"), bty = "n")
dev.off()

png("f1_single_vs_multi.png", width = 1000, height = 650, res = 120)
par(mar = c(9, 4.5, 4, 1))
bp <- barplot(f1_mat, beside = TRUE, names.arg = short_names, las = 2,
              col = c("#2a78d6", "#eb6834"), ylim = c(0, 1),
              ylab = "F1 score (evidence-filtered)",
              main = "F1 score: single-turn vs multi-turn\n(averaged across selected tasks)",
              cex.names = 0.85)
legend("topright", legend = c("Single-turn", "Multi-turn"), fill = c("#2a78d6", "#eb6834"), bty = "n")
dev.off()

cat("\nSaved: accuracy_single_vs_multi.png, f1_single_vs_multi.png\n")
