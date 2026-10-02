# =============================================================================
# table_mean_f1_sd_4configs.R
# Builds the "Mean F1 score (+/- standard deviation) across the five insight
# categories, by model and configuration" table (Table 8 layout).
#
#   Rows    : 8 models, grouped by family (Generalist / Reasoning augmented)
#   Columns : Family | Model | single-turn | multi-turn | ReAct | SpatialInsightAgent
#   Cell    : mean and standard deviation of the model's evidence-filtered F1
#             over the 5 tasks (spatial correlation, cross-variable clustering,
#             spatial hotspot, spatial outlier, change clustering).
#
# Reads ONLY the block "After filtering based on correct evidence" of each
# task section, and ONLY its F1 column (never the "Without filtering" block).
# No figures are produced.
#
# EDIT ONLY THE FOUR PATHS BELOW (same paths as radar_appendix_4configs.R).
# Outputs (folder output_tables/):
#   table8_mean_f1_sd.csv   the table (opens correctly in Excel)
#   table8_mean_f1_sd.tex   a LaTeX tabular in the style of your current Table 8
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(purrr)

# ---- EDIT THESE FOUR PATHS -------------------------------------------------
CONFIG_FILES <- c(
  "single-turn"         = "Follow up_summary_single agent_single_turn.csv",
  "multi-turn"          = "Follow up_summary_single agent_multi_turn.csv",
  "ReAct"               = "Follow up_summary-ReAct.csv",
  "SpatialInsightAgent" = "Follow up_summary_multi_agent.csv"
)
config_levels <- names(CONFIG_FILES)   # = column order of the table

missing_files <- CONFIG_FILES[!file.exists(CONFIG_FILES)]
if (length(missing_files) > 0) {
  stop("File(s) not found in working directory (", getwd(), "):\n  ",
       paste(missing_files, collapse = "\n  "))
}

# ---- Tasks (matched by DESCRIPTION, so the leading "N." never matters) -----
target_tasks <- c(
  "spatial correlation detection"              = "Spatial Correlation",
  "cross-variable clustering detection"        = "Cross-variable Clustering",
  "spatial hotspot detection"                  = "Spatial Hotspot",
  "spatial outlier detection"                  = "Spatial Outlier",
  "spatiotemporal change clustering detection" = "Change Clustering"
)
task_levels <- unname(target_tasks)

# ---- Models: CSV key -> name shown in the table, in the table's row order ---
model_table <- tibble::tribble(
  ~key,                                ~model,          ~family,
  "deepseek/deepseek-chat-v3-0324",    "DeepSeek V3",   "Generalist",
  "openai/gpt-4o-mini",                "GPT 4o mini",   "Generalist",
  "moonshotai/kimi-k2.6",              "Kimi K2.6",     "Generalist",
  "meta-llama/llama-3.1-8b-instruct",  "Llama 3.1 8B",  "Generalist",
  "deepseek/deepseek-r1-0528",         "DeepSeek R1",   "Reasoning augmented",
  "openai/o3",                         "o3",            "Reasoning augmented",
  "openai/o4-mini",                    "o4 mini",       "Reasoning augmented",
  "qwen/qwen3-32b",                    "Qwen3 32B",     "Reasoning augmented"
)
target_models <- setNames(model_table$model, model_table$key)
model_levels  <- model_table$model

# ---- Parser: F1 from the "After filtering based on correct evidence" block --
# The F1 column is located per task section as the first header cell that
# starts with "f1" (case-insensitive). This is needed because the column name
# differs between and inside files (e.g. "F1_Evidence" for the first tasks and
# "f1_evidence_filtered" for the later ones in the ReAct file).
is_task_header <- function(cell) grepl("^\\d+\\.\\s*[A-Za-z]", cell)
strip_leading_number <- function(s) tolower(trimws(sub("^\\d+\\.\\s*", "", s)))

parse_config_csv <- function(path, config_name, n_cols_guess = 25) {
  raw <- read_delim(
    path, delim = ",", col_names = paste0("V", seq_len(n_cols_guess)),
    col_types = cols(.default = col_character()), quote = "\"",
    escape_double = TRUE, locale = locale(encoding = "latin1"),
    na = character(), trim_ws = TRUE, progress = FALSE
  )
  raw <- as.data.frame(raw, stringsAsFactors = FALSE)
  raw[is.na(raw)] <- ""
  
  records <- list()
  current_task <- NA_character_
  in_after <- FALSE
  f1_col <- NA_integer_
  f1_name <- NA_character_
  have_header <- FALSE
  
  for (i in seq_len(nrow(raw))) {
    row <- as.character(unlist(raw[i, ]))
    header_idx <- which(is_task_header(row))[1]
    if (!is.na(header_idx)) {
      key <- strip_leading_number(row[header_idx])
      current_task <- if (key %in% names(target_tasks)) target_tasks[[key]] else NA_character_
      in_after <- FALSE; f1_col <- NA_integer_; f1_name <- NA_character_; have_header <- FALSE
      next
    }
    if (is.na(current_task)) next
    joined <- tolower(paste(row, collapse = " "))
    if (grepl("after filtering based on correct evidence", joined, fixed = TRUE)) {
      in_after <- TRUE; f1_col <- NA_integer_; f1_name <- NA_character_; have_header <- FALSE
      next
    }
    if (!in_after) next
    if (!have_header) {
      if (any(tolower(row) == "model")) {
        f1_col  <- which(startsWith(tolower(row), "f1"))[1]
        f1_name <- if (!is.na(f1_col)) row[f1_col] else NA_character_
        have_header <- TRUE
      }
      next
    }
    model_key <- tolower(trimws(row[1]))
    if (model_key %in% names(target_models) && !is.na(f1_col) && f1_col <= length(row)) {
      f1_val <- suppressWarnings(as.numeric(row[f1_col]))
      if (!is.na(f1_val)) {
        records[[length(records) + 1]] <- data.frame(
          config = config_name, task = current_task,
          model = target_models[[model_key]], f1 = f1_val, f1_header = f1_name,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  bind_rows(records)
}

df_all <- imap_dfr(CONFIG_FILES, ~ parse_config_csv(.x, .y))

# ---- Completeness check: exactly one value per config x task x model -------
full_grid <- expand.grid(config = config_levels, task = task_levels, model = model_levels,
                         stringsAsFactors = FALSE)
counts <- df_all %>% count(config, task, model)
problems <- full_grid %>%
  left_join(counts, by = c("config", "task", "model")) %>%
  mutate(n = ifelse(is.na(n), 0L, n)) %>%
  filter(n != 1L)
if (nrow(problems) > 0) {
  print(as.data.frame(problems %>% arrange(config, task, model)), row.names = FALSE)
  stop("Expected exactly 1 F1 value per config/task/model (n = 0 means missing, n > 1 means duplicated). ",
       "See the rows printed above; usually a task-header or model-name spelling differs in that file.")
}

cat("\nF1 column read (After-filtering block) per configuration and task:\n")
print(as.data.frame(df_all %>% distinct(config, task, f1_header) %>% arrange(config, task)),
      row.names = FALSE)

# ---- Mean +/- SD over the 5 tasks ------------------------------------------
# sd() is the SAMPLE standard deviation (n - 1), which is what reproduces the
# values in the paper's Table 8.
fmt_txt <- function(m, s) sprintf("%.1f \u00b1 %.1f", m, s)
fmt_txt <- function(m, s) sprintf("%.2f \u00b1 %.2f", m, s)

summary_df <- df_all %>%
  group_by(config, model) %>%
  summarise(mean_f1 = mean(f1), sd_f1 = sd(f1), n_tasks = n(), .groups = "drop") %>%
  mutate(txt = fmt_txt(mean_f1, sd_f1), tex = fmt_tex(mean_f1, sd_f1))
stopifnot(all(summary_df$n_tasks == length(task_levels)))

widen <- function(col) {
  summary_df %>%
    select(model, config, all_of(col)) %>%
    pivot_wider(names_from = config, values_from = all_of(col))
}

table8 <- model_table %>%
  left_join(widen("txt"), by = "model") %>%
  select(Family = family, Model = model, all_of(config_levels))

# ---- Console view (family label shown once, as in the paper) ----------------
print_tab <- table8 %>% mutate(Family = ifelse(duplicated(Family), "", Family))
cat("\nMean F1 score (\u00b1 standard deviation) across the five insight categories:\n\n")
print(as.data.frame(print_tab), row.names = FALSE, right = FALSE)

# ---- Save CSV ---------------------------------------------------------------
dir.create("output_tables", showWarnings = FALSE)
write_excel_csv(table8, "output_tables/table8_mean_f1_sd.csv")

# ---- Save LaTeX tabular (same structure as the current Table 8) -------------
tex_wide <- model_table %>% left_join(widen("tex"), by = "model")
tex_lines <- c(
  "\\begin{tabular}{llcccc}",
  "\\toprule",
  paste0("Family & Model & ", paste(config_levels, collapse = " & "), " \\\\"),
  "\\midrule"
)
families <- unique(model_table$family)
for (fam in families) {
  rows <- tex_wide %>% filter(family == fam)
  tex_lines <- c(tex_lines, sprintf("\\multirow{%d}{*}{%s}", nrow(rows), fam))
  for (i in seq_len(nrow(rows))) {
    cells <- vapply(config_levels, function(cfg) rows[[cfg]][i], character(1))
    tex_lines <- c(tex_lines,
                   paste0(" & ", rows$model[i], " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  if (fam != tail(families, 1)) tex_lines <- c(tex_lines, "\\midrule")
}
tex_lines <- c(tex_lines, "\\bottomrule", "\\end{tabular}")
writeLines(tex_lines, "output_tables/table8_mean_f1_sd.tex")

cat("\nSaved: output_tables/table8_mean_f1_sd.csv\nSaved: output_tables/table8_mean_f1_sd.tex\n")