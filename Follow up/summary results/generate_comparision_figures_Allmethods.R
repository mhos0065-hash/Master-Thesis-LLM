## =============================================================================
## generate_comparison_figures_AllMethods.R
##
## Reads the four "Follow_up_summary_*.csv" result files (single agent
## single turn, single agent multi turn, ReAct baseline, SpatialInsightAgent)
## and produces TWO publication ready bar chart figures per insight category,
## one for accuracy and one for F1 score, both using the evidence filtered
## ("After filtering based on correct evidence") block, for every model.
## Also writes a category level overview (test cases, ambiguous reply rate,
## evidence rate) to an Excel file.
##
## Usage:
##   Rscript generate_comparison_figures_AllMethods.R
##
## Requirements: ggplot2, dplyr, tidyr, writexl, RColorBrewer
##   install.packages(c("ggplot2","dplyr","tidyr","writexl","RColorBrewer"))
## =============================================================================

if (!requireNamespace("writexl", quietly = TRUE)) {
  install.packages("writexl")
}
library(writexl)

if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
  install.packages("RColorBrewer")
}
library(RColorBrewer)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

## ---- 0. CONFIG --------------------------------------------------------------

# Folder containing the 4 input CSVs (edit if needed)
setwd("D:/Australia/Monash/Semester 4/Research stream/Master Thesis/Master-Thesis-LLM/Follow up/summary results")

getwd()
list.files(pattern = "\\.csv$", ignore.case = TRUE)

input_dir <- "."

# Folder where the figures and the overview spreadsheet will be saved
output_dir <- "output_figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Map: file name -> friendly configuration label (no hyphens)
files <- c(
  "Follow up_summary_single agent_single_turn.csv" = "Single agent single turn",
  "Follow up_summary_single agent_multi_turn.csv"   = "Single agent multi turn",
  "Follow up_summary-ReAct.csv"                     = "ReAct baseline",
  "Follow up_summary_multi_agent.csv"               = "SpatialInsightAgent"
)

# Canonical task names, in the order the 6 blocks appear in every file
# (hyphens removed; "Outlier detection" expanded to match the taxonomy
# name used throughout the thesis)
task_names <- c(
  "1" = "Spatial correlation detection",
  "2" = "Cross variable clustering detection",
  "3" = "Spatial hotspot detection",
  "4" = "Spatial outlier detection",
  "5" = "Spatiotemporal change clustering detection"
)

# Fixed model order (consistent across all 4 files), and clean, hyphen free
# display labels used only for plotting
model_order <- c(
  "deepseek/deepseek-chat-v3-0324",
  "openai/gpt-4o-mini",
  "moonshotai/kimi-k2.6",
  "meta-llama/llama-3.1-8b-instruct",
  "deepseek/deepseek-r1-0528",
  "openai/o3",
  "openai/o4-mini",
  "qwen/qwen3-32b"
)

model_labels <- c(
  "deepseek/deepseek-chat-v3-0324"   = "DeepSeek V3",
  "openai/gpt-4o-mini"               = "GPT 4o mini",
  "moonshotai/kimi-k2.6"             = "Kimi K2.6",
  "meta-llama/llama-3.1-8b-instruct" = "Llama 3.1 8B",
  "deepseek/deepseek-r1-0528"        = "DeepSeek R1",
  "openai/o3"                        = "o3",
  "openai/o4-mini"                   = "o4 mini",
  "qwen/qwen3-32b"                   = "Qwen3 32B"
)

# Configuration order: single agent single turn, single agent multi turn,
# ReAct baseline third, SpatialInsightAgent fourth
config_order <- c(
  "Single agent single turn",
  "Single agent multi turn",
  "ReAct baseline",
  "SpatialInsightAgent"
)

# Colour palette from ColorBrewer, one colour per configuration.
# "Paired" is the qualitative palette ColorBrewer's own tool flags as
# officially colourblind safe at exactly 4 classes (Dark2 and Set2 are
# only certified safe at 3 classes or fewer). brewer.pal() must be asked
# for the FULL 12 colour palette here, then subset down to 4; asking for
# only 4 upfront returns just the first two light/dark pairs, and
# indexing beyond that silently yields NA (blank swatches, missing bars).
config_colors <- setNames(brewer.pal(12, "Paired")[c(2, 8, 10, 4)], config_order)

## ---- 1. PARSER ---------------------------------------------------------------
## The CSVs are "report style": task titles + two labelled sub tables
## ("Without filtering..." and "After filtering based on correct evidence...")
## live inside one sheet, with blank separator rows in between. Column
## headers differ slightly in wording across tasks/files, but position is
## always fixed once you reach a data table:
##   V1 = model, V2 = N_Query, V3 = Reply_ambiguous, V6 = pct_with_evidence,
##   V9 = accuracy_evidence_filtered, V12 = f1_evidence_filtered

parse_after_filtering_tables <- function(path, config_label) {
  
  raw <- read.csv(
    path, header = FALSE, colClasses = "character",
    fill = TRUE, quote = "\"", na.strings = NULL,
    stringsAsFactors = FALSE, blank.lines.skip = FALSE,
    encoding = "UTF-8"
  )
  
  trim <- function(x) trimws(gsub("\t", " ", x, fixed = TRUE))
  
  v4 <- trim(raw[[4]])
  v1 <- trim(raw[[1]])
  
  section_start <- grepl("^[1-6]\\.", v4)
  section_rows  <- which(section_start)
  section_num   <- substring(v4[section_rows], 1, 1)
  
  keep <- !duplicated(section_num)
  section_rows <- section_rows[keep]
  section_num  <- section_num[keep]
  
  section_end <- c(section_rows[-1] - 1, nrow(raw))
  
  out <- list()
  
  for (i in seq_along(section_rows)) {
    
    r0 <- section_rows[i]
    r1 <- section_end[i]
    block <- raw[r0:r1, , drop = FALSE]
    b4 <- trim(block[[4]])
    b1 <- trim(block[[1]])
    
    af_idx <- which(b4 == "After filtering based on correct evidence")
    
    if (length(af_idx) > 0) {
      hdr_candidates <- which(b1 == "model" & seq_len(nrow(block)) > af_idx[1])
      if (length(hdr_candidates) == 0) next
      hdr_idx <- hdr_candidates[1]
    } else {
      hdr_candidates <- which(b1 == "model")
      if (length(hdr_candidates) < 2) next
      hdr_idx <- hdr_candidates[2]
    }
    
    j <- hdr_idx + 1
    data_rows <- integer(0)
    while (j <= nrow(block) && b1[j] != "") {
      data_rows <- c(data_rows, j)
      j <- j + 1
    }
    if (length(data_rows) == 0) next
    
    sub <- block[data_rows, , drop = FALSE]
    task_id <- section_num[i]
    
    df <- data.frame(
      task     = task_names[[task_id]],
      task_num = as.integer(task_id),
      config   = config_label,
      model    = trim(sub[[1]]),
      n_query           = suppressWarnings(as.integer(trim(sub[[2]]))),
      reply_ambiguous   = suppressWarnings(as.integer(trim(sub[[3]]))),
      pct_with_evidence = suppressWarnings(as.numeric(trim(sub[[6]]))),
      accuracy = suppressWarnings(as.numeric(trim(sub[[9]]))),
      f1       = suppressWarnings(as.numeric(trim(sub[[12]]))),
      stringsAsFactors = FALSE
    )
    
    df <- df[df$model %in% model_order, ]
    out[[task_id]] <- df
  }
  
  bind_rows(out)
}

## ---- 2. READ + COMBINE ALL 4 FILES -------------------------------------------

all_data <- bind_rows(lapply(names(files), function(fname) {
  path <- file.path(input_dir, fname)
  parse_after_filtering_tables(path, files[[fname]])
}))

## ---- 2c. NORMALISE ACCURACY / F1 SCALE ---------------------------------------
## Some source files report accuracy/F1 already multiplied by 100 (a
## percentage) rather than as a 0-1 proportion, inconsistent with the
## other files (confirmed: Follow_up_summary_multi_agent.csv reports
## accuracy_Evidence around 50-80, while the other three files report
## 0.2-0.8 for the same cells). This is detected per configuration, not
## per row, since it is a systemic difference between source files, not
## a random glitch: if a configuration's median accuracy exceeds 1, the
## whole configuration is assumed to be on a 0-100 scale and is divided
## back down to 0-1, so every configuration is on the same scale before
## any plotting or export happens. This is a workaround; the real fix
## belongs in whatever script generates the affected CSV.
scale_check <- all_data %>%
  group_by(config) %>%
  summarise(median_accuracy = median(accuracy, na.rm = TRUE), .groups = "drop")
print(scale_check)

needs_rescale <- scale_check$config[scale_check$median_accuracy > 1.5]
if (length(needs_rescale) > 0) {
  message("Rescaling accuracy/F1 from 0-100 to 0-1 for: ",
          paste(needs_rescale, collapse = ", "))
  all_data <- all_data %>%
    mutate(
      accuracy = ifelse(config %in% needs_rescale, accuracy / 100, accuracy),
      f1       = ifelse(config %in% needs_rescale, f1 / 100, f1)
    )
}

## ---- 2b. CATEGORY LEVEL OVERVIEW TABLE ---------------------------------------
## Ambiguous rate and evidence rate are pooled across all model x config
## cells. Because N_Query is identical for every cell within a category,
## this pooled rate is equal to the unweighted average across those cells,
## so no distortion is introduced by reporting it this way. Computed here,
## before the config/model columns are turned into factors, since neither
## factor is needed for this summary.

overview <- all_data %>%
  group_by(task, task_num) %>%
  summarise(
    N                 = first(n_query),                 # constant within a category
    models_x_configs  = n(),
    ambiguous_pct     = 100 * sum(reply_ambiguous, na.rm = TRUE) /
      (first(n_query) * n()),
    evidence_rate_pct = 100 * sum(pct_with_evidence * n_query, na.rm = TRUE) /
      sum(n_query, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(task_num) %>%
  select(Category = task, N, `Models x Configs` = models_x_configs,
         `Ambiguous (%)` = ambiguous_pct, `Evidence rate (%)` = evidence_rate_pct)

write_xlsx(overview, file.path(output_dir, "category_overview.xlsx"))
message("Saved: ", file.path(output_dir, "category_overview.xlsx"))

## ---- 3. FACTOR LEVELS FOR PLOTTING --------------------------------------------

all_data$config <- factor(all_data$config, levels = config_order)
all_data$model  <- factor(all_data$model, levels = model_order,
                          labels = unname(model_labels[model_order]))

## ---- 4. PUBLICATION THEME -----------------------------------------------------

theme_pub <- theme_classic(base_size = 13, base_family = "sans") +
  theme(
    plot.title      = element_text(size = 16, face = "bold", hjust = 0,
                                   margin = margin(b = 4)),
    plot.subtitle   = element_text(size = 14, colour = "steelblue4",
                                   margin = margin(b = 10)),
    axis.title.y    = element_text(size = 12.5, margin = margin(r = 8)),
    axis.title.x    = element_blank(),
    axis.text.x     = element_text(size = 10.5, angle = 40, hjust = 1,
                                   margin = margin(t = 4)),
    axis.text.y     = element_text(size = 10.5),
    legend.title    = element_blank(),
    legend.text     = element_text(size = 11),
    legend.position = "bottom",
    plot.margin     = margin(t = 8, r = 14, b = 8, l = 8)
  )

## ---- 5. PLOT: ONE ACCURACY FIGURE AND ONE F1 FIGURE PER TASK ------------------
## Values are converted to percentages (0-100) for display, since the
## y axis and subtitle/label now read "Accuracy (%)" / "F1 score (%)"
## rather than the raw 0-1 proportion.

make_metric_figure <- function(task_label, task_num, metric_col, metric_name) {
  
  d <- all_data %>%
    filter(task == task_label) %>%
    select(model, config, value = all_of(metric_col)) %>%
    mutate(value = value * 100)
  
  if (nrow(d) == 0 || all(is.na(d$value))) {
    message("Skipped (no data): ", task_label, " - ", metric_name)
    return(invisible(NULL))
  }
  
  p <- ggplot(d, aes(x = model, y = value, fill = config)) +
    geom_col(position = position_dodge2(width = 0.75, padding = 0.1),
             width = 0.75, na.rm = TRUE) +
    scale_fill_manual(values = config_colors, drop = FALSE) +
    scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.04))) +
    labs(
      title    = paste0("Task: ", task_label),
      subtitle = paste0(metric_name, " after filtering on correct evidence"),
      x = NULL, y = metric_name
    ) +
    theme_pub +
    guides(fill = guide_legend(nrow = 2))
  
  fname <- sprintf("%02d_%s_%s.png", task_num,
                   gsub("[^A-Za-z0-9]+", "_", task_label),
                   tolower(gsub("[^A-Za-z0-9]+", "", metric_name)))
  
  ggsave(file.path(output_dir, fname), plot = p,
         width = 8.5, height = 5.8, dpi = 300)
  
  message("Saved: ", file.path(output_dir, fname))
}

invisible(Map(function(task_label, task_num) {
  make_metric_figure(task_label, task_num, "accuracy", "Accuracy (%)")
  make_metric_figure(task_label, task_num, "f1",       "F1 score (%)")
}, task_names, as.integer(names(task_names))))

message("\nAll figures written to: ", normalizePath(output_dir))