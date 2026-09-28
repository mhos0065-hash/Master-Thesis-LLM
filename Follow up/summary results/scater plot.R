## =============================================================================
## generate_grounding_scatter_rawF1.R
##
## Reads the same four "Follow_up_summary_*.csv" files as
## generate_comparison_figures_AllMethods.R, using the SAME file/task
## parsing structure, but extracts accuracy/F1 from the "Without
## filtering based on correct evidence" block instead of the "After
## filtering..." block that script used.
##
## WHY THIS MATTERS: the original script's grounding-vs-correctness
## scatter plotted evidence rate against EVIDENCE-FILTERED F1 -- a
## verdict there can only count as correct if it also cited real
## evidence, which makes evidence rate and F1 mechanically related
## (a model with a 10% evidence rate is structurally incapable of much
## more than ~10% evidence-filtered F1, regardless of its actual
## reasoning quality). Using RAW F1 here instead -- correctness judged
## purely against ground truth, with no evidence requirement -- keeps
## evidence rate and F1 genuinely independent, so a positive
## relationship between them is an actual finding, not a built-in
## consequence of the metric's definition. See the "Grounding versus
## Correctness" thesis subsection for the argument this figure supports.
##
## This script produces ONLY the scatter plot. It does not generate the
## per-category bar charts, the family/F1 heatmap, or the category
## overview spreadsheet that the original script also produces --
## if you need those, keep using the original script alongside this one.
##
## Usage:
##   Rscript generate_grounding_scatter_rawF1.R
##
## Requirements: ggplot2, dplyr, tidyr, RColorBrewer
##   install.packages(c("ggplot2","dplyr","tidyr","RColorBrewer"))
## =============================================================================

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
## Identical to the original script -- same folder, same file mapping,
## same task names, same model order/labels, same config order/colours.

setwd("D:/Australia/Monash/Semester 4/Research stream/Master Thesis/Master-Thesis-LLM/Follow up/summary results")

getwd()
list.files(pattern = "\\.csv$", ignore.case = TRUE)

input_dir <- "."

output_dir <- "output_figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

files <- c(
  "Follow up_summary_single agent_single_turn.csv" = "Single agent single turn",
  "Follow up_summary_single agent_multi_turn.csv"   = "Single agent multi turn",
  "Follow up_summary-ReAct.csv"                     = "ReAct baseline",
  "Follow up_summary_multi_agent.csv"               = "SpatialInsightAgent"
)

task_names <- c(
  "1" = "Spatial correlation detection",
  "2" = "Cross variable clustering detection",
  "3" = "Spatial hotspot detection",
  "4" = "Spatial outlier detection",
  "5" = "Spatiotemporal change clustering detection"
)

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

config_order <- c(
  "Single agent single turn",
  "Single agent multi turn",
  "ReAct baseline",
  "SpatialInsightAgent"
)

config_colors <- setNames(brewer.pal(12, "Paired")[c(2, 8, 10, 4)], config_order)

## ---- 1. PARSER (MODIFIED) -----------------------------------------------------
## Same "report style" structure as the original script: task titles +
## two labelled sub-tables per task, separated by blank rows. Column
## positions are IDENTICAL in both sub-tables (confirmed directly
## against the raw CSV):
##   V1 = model, V2 = N_Query, V3 = Reply_ambiguous, V6 = pct_with_evidence,
##   V9 = accuracy, V12 = F1
##
## ONLY CHANGE from the original parser: this looks for the marker
## "Without filtering based on correct evidence" (the RAW table) instead
## of "After filtering based on correct evidence" (the evidence-filtered
## table), and reads accuracy/F1 from that block instead.

parse_without_filtering_tables <- function(path, config_label) {
  
  raw <- read.csv(
    path, header = FALSE, colClasses = "character",
    fill = TRUE, quote = "\"", na.strings = NULL,
    stringsAsFactors = FALSE, blank.lines.skip = FALSE,
    encoding = "UTF-8"
  )
  
  trim <- function(x) trimws(gsub("\t", " ", x, fixed = TRUE))
  
  v4 <- trim(raw[[4]])
  v1 <- trim(raw[[1]])
  
  section_start <- grepl("^[1-5]\\.", v4)
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
    
    # CHANGED: locate the RAW (unfiltered) sub-table's marker instead of
    # the evidence-filtered one.
    wf_idx <- which(b4 == "Without filtering based on correct evidence")
    
    if (length(wf_idx) > 0) {
      hdr_candidates <- which(b1 == "model" & seq_len(nrow(block)) > wf_idx[1])
      if (length(hdr_candidates) == 0) next
      hdr_idx <- hdr_candidates[1]
    } else {
      # Fallback: if the marker text doesn't match exactly (e.g. wording
      # differs slightly in some file), take the FIRST "model" header in
      # the block -- confirmed from the file structure that the raw
      # table's header always comes before the evidence-filtered one.
      hdr_candidates <- which(b1 == "model")
      if (length(hdr_candidates) == 0) next
      hdr_idx <- hdr_candidates[1]
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
  parse_without_filtering_tables(path, files[[fname]])
}))

## ---- 2c. NORMALISE ACCURACY / F1 SCALE ---------------------------------------
## Same rationale as the original script: some source files report
## accuracy/F1 already multiplied by 100, inconsistent with others that
## report a 0-1 proportion. Detected per configuration (a systemic
## difference between source files, not a random glitch) and rescaled
## so every configuration is on the same 0-1 scale before plotting.

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

## ---- 3. FACTOR LEVELS FOR PLOTTING --------------------------------------------

all_data$config <- factor(all_data$config, levels = config_order)
all_data$model  <- factor(all_data$model, levels = model_order,
                          labels = unname(model_labels[model_order]))

## ---- 4. GROUNDING VS CORRECTNESS SCATTER (RAW F1) -----------------------------
## Each point is one real model, configuration, category combination.
## No value is averaged or pooled here. F1 here is RAW (not evidence
## filtered) -- see the header comment for why this is the correct
## choice for this specific figure.
##
## All four quadrant annotations use annotate("label", ...) rather than
## annotate("text", ...): this draws a white background box behind the
## text, keeping it legible regardless of how many points fall
## underneath it. The "reasoning failure" label is also repositioned to
## the BOTTOM-RIGHT (high evidence, low F1) to match what it describes --
## the original script placed this text in the top-left, which is low
## evidence/high F1, the opposite of its own label.

category_labels <- c(
  "Spatial correlation detection"              = "Sp. Corr",
  "Cross variable clustering detection"        = "Cross Var",
  "Spatial hotspot detection"                  = "Sp. Hotspot",
  "Spatial outlier detection"                  = "Sp. Outlier",
  "Spatiotemporal change clustering detection" = "Change Clu"
)

scatter_data <- all_data %>%
  filter(task %in% names(category_labels)) %>%
  transmute(
    model, config,
    category      = category_labels[task],
    evidence_rate = pct_with_evidence * 100,
    f1_pct        = f1 * 100
  )

p_scatter <- ggplot(scatter_data, aes(x = evidence_rate, y = f1_pct,
                                      colour = config)) +
  geom_point(size = 2.2, alpha = 0.75) +
  geom_vline(xintercept = 50, linetype = "dashed", colour = "black", linewidth = 0.4) +
  geom_hline(yintercept = 50, linetype = "dashed", colour = "black", linewidth = 0.4) +
  annotate("label", x = 73, y = 2, label = "High evidence, low F1\n(reasoning failure)",
           hjust = 0, vjust = 0, size = 3.5, colour = "black", fontface = "italic",
           fill = "white", label.size = 0, alpha = 0.85) +
  annotate("label", x = 2, y = 14, label = "Low evidence\n(grounding failure)",
           hjust = 0, vjust = 0, size = 3.5, colour = "black", fontface = "italic",
           fill = "white", label.size = 0, alpha = 0.85) +
  annotate("label", x = 2, y = 97, label = "Low evidence, high F1\n(ungrounded correctness)",
           hjust = 0, vjust = 1, size = 3.5, colour = "black", fontface = "italic",
           fill = "white", label.size = 0, alpha = 0.85) +
  annotate("label", x = 52, y = 97, label = "High evidence, high F1\n(grounded correctness)",
           hjust = 0, vjust = 1, size = 3.5, colour = "black", fontface = "italic",
           fill = "white", label.size = 0, alpha = 0.85) +
  scale_colour_manual(values = config_colors, name = NULL) +
  scale_x_continuous(limits = c(0, 100), expand = expansion(mult = c(0.01, 0.03))) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0.01, 0.03))) +
  labs(x = "Evidence rate (%)", y = "F1 score (%)") +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "bottom",
    axis.title      = element_text(size = 12.5),
    axis.text       = element_text(size = 10.5),
    legend.text     = element_text(size = 10.5)
  ) +
  guides(colour = guide_legend(nrow = 2, override.aes = list(size = 3, alpha = 1)))

ggsave(file.path(output_dir, "grounding_vs_correctness_rawF1.png"), plot = p_scatter,
       width = 6.5, height = 6, dpi = 300)
message("Saved: ", file.path(output_dir, "grounding_vs_correctness_rawF1.png"))