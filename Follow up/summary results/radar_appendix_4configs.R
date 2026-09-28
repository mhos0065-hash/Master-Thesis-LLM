# =============================================================================
# radar_appendix_4configs.R
# Generates TWO publication-ready appendix figures, each a 2x2 grid of radar
# plots across all four configurations:
#   Row 1: Single-Agent Single-Turn | Single-Agent Multi-Turn
#   Row 2: ReAct Baseline            | SpatialInsightAgent
#   Figure 1 (accuracy): reads the 'accuracy_*' evidence-filtered column
#   Figure 2 (F1):        reads the 'f1_*' evidence-filtered column
#
# EDIT ONLY THE FOUR PATHS BELOW to point at your real CSV files, in this
# exact order (Single-Turn, Multi-Turn, ReAct, SpatialInsightAgent). Every
# other setting (colors, task list, model list) is shared across all four.
#
# NOTE: element sizing (font sizes, line width, point/label size, legend key
# dimensions) below is matched to radar_single_agent_single_turn_8models.R;
# overall canvas size (ggsave width/height/dpi), margins, and the 2-row
# legend layout are unchanged from before, since those are page-layout
# choices for fitting 4 panels on one appendix page rather than per-element
# sizing.
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(patchwork)   # install.packages("patchwork") if not already installed

# ---- EDIT THESE FOUR PATHS -------------------------------------------------
CONFIG_FILES <- c(
  "Single-Agent\nSingle-Turn" = "Follow up_summary_single agent_single_turn.csv",
  "Single-Agent\nMulti-Turn"  = "Follow up_summary_single agent_multi_turn.csv",
  "ReAct Baseline"             = "Follow up_summary-ReAct.csv",
  "SpatialInsightAgent"        = "Follow up_summary_multi_agent.csv"
)
# Panel order (top-left, top-right, bottom-left, bottom-right) = order above.
config_levels <- names(CONFIG_FILES)

# ---- Shared task / model vocabulary (same across all four configs) --------
target_tasks <- c(
  "spatial correlation detection"              = "Spatial\nCorrelation",
  "cross-variable clustering detection"        = "Cross-variable\nClustering",
  "spatial hotspot detection"                  = "Spatial\nHotspot",
  "spatial outlier detection"                  = "Spatial\nOutlier",
  "spatiotemporal change clustering detection" = "Change\nClustering"
)
task_levels <- unname(target_tasks)

target_models <- c(
  "deepseek/deepseek-r1-0528"        = "DeepSeek-R1",
  "deepseek/deepseek-chat-v3-0324"   = "DeepSeek-V3",
  "openai/gpt-4o-mini"               = "GPT-4o-mini",
  "moonshotai/kimi-k2.6"             = "Kimi-K2.6",
  "meta-llama/llama-3.1-8b-instruct" = "Llama-3.1-8B",
  "openai/o3"                        = "o3",
  "openai/o4-mini"                   = "o4-mini",
  "qwen/qwen3-32b"                   = "Qwen3-32B"
)
model_levels <- unname(target_models)

cols <- c(
  "DeepSeek-V3"  = "#0072B2", "DeepSeek-R1"  = "#D55E00",
  "Llama-3.1-8B" = "#009E73", "Kimi-K2.6"    = "#CC79A7",
  "GPT-4o-mini"  = "#E69F00", "o3"           = "#999999",
  "o4-mini"      = "#000000", "Qwen3-32B"    = "#56B4E9"
)

# ---- Generic parser: pulls BOTH accuracy and F1 in one pass ---------------
# Matches task sections by DESCRIPTION TEXT (ignoring the leading "N. "),
# so numbering drift between files (e.g. a file labelling change-clustering
# as "5." vs "6.") can never silently drop a whole task again. Locates the
# accuracy/F1 columns generically within the "After filtering based on
# correct evidence" block via a case-insensitive startsWith() match, so it
# is robust to the column-name variants already seen within one file alone
# (e.g. "accuracy_Evidence" vs "accuracy_evidence_filtered").
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
  acc_col <- NA_integer_
  f1_col  <- NA_integer_
  have_header <- FALSE
  
  for (i in seq_len(nrow(raw))) {
    row <- as.character(unlist(raw[i, ]))
    header_idx <- which(is_task_header(row))[1]
    if (!is.na(header_idx)) {
      key <- strip_leading_number(row[header_idx])
      current_task <- if (key %in% names(target_tasks)) target_tasks[[key]] else NA_character_
      in_after <- FALSE; acc_col <- NA_integer_; f1_col <- NA_integer_; have_header <- FALSE
      next
    }
    if (is.na(current_task)) next
    joined <- tolower(paste(row, collapse = " "))
    if (grepl("after filtering based on correct evidence", joined, fixed = TRUE)) {
      in_after <- TRUE; acc_col <- NA_integer_; f1_col <- NA_integer_; have_header <- FALSE
      next
    }
    if (!in_after) next
    if (!have_header) {
      if (any(tolower(row) == "model")) {
        hdr <- tolower(row)
        acc_col <- which(startsWith(hdr, "accuracy"))[1]
        f1_col  <- which(startsWith(hdr, "f1"))[1]
        have_header <- TRUE
      }
      next
    }
    model_key <- tolower(trimws(row[1]))
    if (model_key %in% names(target_models)) {
      acc_val <- if (!is.na(acc_col) && acc_col <= length(row)) suppressWarnings(as.numeric(row[acc_col])) else NA_real_
      f1_val  <- if (!is.na(f1_col)  && f1_col  <= length(row)) suppressWarnings(as.numeric(row[f1_col]))  else NA_real_
      if (!is.na(acc_val) && !is.na(f1_val)) {
        records[[length(records) + 1]] <- data.frame(
          config = config_name, task = current_task, model = target_models[[model_key]],
          accuracy = acc_val, f1 = f1_val
        )
      }
    }
  }
  bind_rows(records)
}

# ---- Parse all four files, with a clear diagnostic if anything is missing --
df_all <- imap_dfr(CONFIG_FILES, parse_config_csv) %>%
  mutate(
    config = factor(config, levels = config_levels),
    task   = factor(task,   levels = task_levels),
    model  = factor(model,  levels = model_levels)
  )

expected_n <- length(config_levels) * length(task_levels) * length(model_levels)
if (nrow(df_all) != expected_n) {
  full_grid <- expand.grid(config = config_levels, task = task_levels, model = model_levels,
                           stringsAsFactors = FALSE)
  have <- df_all %>% transmute(config = as.character(config), task = as.character(task),
                               model = as.character(model), found = TRUE)
  missing <- full_grid %>% left_join(have, by = c("config", "task", "model")) %>% filter(is.na(found))
  print(missing %>% arrange(config, task, model))
  stop(sprintf(
    "Expected %d rows (4 configs x %d tasks x %d models), got %d. See the missing config/task/model combinations printed above -- check that column header, on that output above, e.g. a task name or model key spelling differs in the offending file.",
    expected_n, length(task_levels), length(model_levels), nrow(df_all)
  ))
}
print(df_all %>% select(config, task, model, accuracy, f1) %>% arrange(config, task, model))

# ---- Shared radar geometry (independent of config, computed once) ---------
n_tasks <- length(task_levels)
angle_lookup <- setNames(seq(0, 2 * pi, length.out = n_tasks + 1)[1:n_tasks], task_levels)

circle_radii <- c(25, 50, 75, 100)
theta_seq <- seq(0, 2 * pi, length.out = 200)
grid_circles <- do.call(rbind, lapply(circle_radii, function(r) {
  data.frame(angle = theta_seq, value = r, r_id = r)
}))
grid_spokes <- data.frame(
  angle = rep(unname(angle_lookup), each = 2),
  value = rep(c(0, 100), times = n_tasks),
  spoke_id = rep(names(angle_lookup), each = 2)
)
radial_labels <- data.frame(angle = 0, y = circle_radii,
                            label = c("25", "50", "75", "100"))

coord_radar <- function(theta = "x", start = 0, direction = 1) {
  theta <- match.arg(theta, c("x", "y"))
  r <- if (theta == "x") "y" else "x"
  ggproto("CoordRadar", CoordPolar, theta = theta, r = r,
          start = start, direction = sign(direction),
          is_linear = function(coord) TRUE, clip = "off")
}

# ---- One radar panel for a given config + metric ---------------------------
build_panel <- function(df_all, config_name, metric_col, panel_title) {
  df_sub <- df_all %>%
    filter(config == config_name) %>%
    transmute(task, model, value = .data[[metric_col]]) %>%
    mutate(angle = angle_lookup[as.character(task)])
  
  df_closed <- df_sub %>%
    group_by(model) %>%
    group_modify(~ bind_rows(.x, .x[1, ])) %>%
    mutate(angle = ifelse(row_number() == n(), 2 * pi, angle)) %>%
    ungroup()
  
  ggplot() +
    geom_path(data = grid_circles, aes(x = angle, y = value, group = r_id),
              color = "grey78", linewidth = 0.4) +
    geom_path(data = grid_spokes, aes(x = angle, y = value, group = spoke_id),
              color = "grey78", linewidth = 0.4) +
    geom_polygon(data = df_closed, aes(x = angle, y = value, group = model),
                 fill = NA, linewidth = 0) +
    geom_line(data = df_closed, aes(x = angle, y = value, group = model, color = model),
              linewidth = 1.0, alpha = 0.95) +
    geom_point(data = df_closed, aes(x = angle, y = value, color = model),
               size = 1.8) +
    geom_label(data = radial_labels, aes(x = angle, y = y, label = label),
               inherit.aes = FALSE, size = 4.5, color = "grey30",
               label.size = 0, label.padding = unit(0.1, "lines"),
               fill = alpha("white", 0.85), hjust = 1.15, vjust = 0.00) +
    scale_color_manual(values = cols, name = NULL, drop = FALSE) +
    scale_x_continuous(breaks = unname(angle_lookup), labels = names(angle_lookup),
                       limits = c(0, 2 * pi)) +
    scale_y_continuous(limits = c(0, 112), breaks = seq(0, 100, 25), expand = c(0, 0)) +
    coord_radar() +
    labs(title = panel_title) +
    theme_void(base_size = 13) +
    theme(
      plot.title  = element_text(size = 13, face = "bold", hjust = 0.5,
                                 margin = margin(b = 6)),
      axis.text.x = element_text(size = 12, face = "bold", color = "black",
                                 margin = margin(t = 5, b = 8), lineheight = 0.95),
      legend.position   = "bottom",
      legend.title      = element_blank(),
      legend.text       = element_text(size = 10.5),
      legend.key.width  = unit(0.9, "cm"),
      legend.key.height = unit(0.45, "cm"),
      legend.spacing.x  = unit(0.2, "cm"),
      plot.margin       = margin(t = 10, r = 24, b = 6, l = 24)
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE,
                                override.aes = list(linewidth = 1.4)))
}

# ---- Assemble one 2x2 figure for a given metric ----------------------------
build_figure <- function(df_all, metric_col) {
  panels <- lapply(config_levels, function(cfg) {
    build_panel(df_all, cfg, metric_col, gsub("\n", " ", cfg))
  })
  ((panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]])) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = "a") &
    theme(legend.position = "bottom",
          plot.tag = element_text(face = "bold", size = 15))
}

fig_accuracy <- build_figure(df_all, "accuracy")
fig_f1       <- build_figure(df_all, "f1")

ggsave("output_figures/appendix_radar_accuracy_4configs.png", fig_accuracy,
       width = 11, height = 12.5, dpi = 450, bg = "white")
ggsave("output_figures/appendix_radar_f1_4configs.png", fig_f1,
       width = 11, height = 12.5, dpi = 450, bg = "white")

cat("\nSaved: radar_accuracy_4configs.png\nSaved: radar_f1_4configs.png\n")