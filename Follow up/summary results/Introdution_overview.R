# =============================================================================
# radar_single_agent_single_turn_8models.R
# Canvas and font sizes recalibrated: large enough that text won't blur when
# placed at column width, small enough that nothing overflows or collides.
# Legend switched to 4 rows x 2 columns so long model names never truncate.
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

CSV_PATH <- "Follow up_summary_single agent_single_shot.csv"

target_tasks <- c(
  "1. spatial correlation detection"              = "Spatial\nCorrelation",
  "2. cross-variable clustering detection"        = "Cross-variable\nClustering",
  "3. spatial hotspot detection"                  = "Spatial\nHotspot",
  "4. spatial outlier detection"                  = "Spatial\nOutlier",
  "6. spatiotemporal change clustering detection" = "Change\nClustering"
)
task_levels <- unname(target_tasks)

target_models <- c(
  "deepseek/deepseek-chat-v3-0324"   = "DeepSeek-V3",
  "deepseek/deepseek-r1-0528"        = "DeepSeek-R1",
  "meta-llama/llama-3.1-8b-instruct" = "Llama-3.1-8B",
  "moonshotai/kimi-k2.6"             = "Kimi-K2.6",
  "openai/gpt-4o-mini"               = "GPT-4o-mini",
  "openai/o3"                        = "o3",
  "openai/o4-mini"                   = "o4-mini",
  "qwen/qwen3-32b"                   = "Qwen3-32B"
)
model_levels <- unname(target_models)

n_cols_guess <- 20
raw <- read_delim(
  CSV_PATH, delim = ",", col_names = paste0("V", seq_len(n_cols_guess)),
  col_types = cols(.default = col_character()), quote = "\"",
  escape_double = TRUE, locale = locale(encoding = "latin1"),
  na = character(), trim_ws = TRUE, progress = FALSE
)
raw <- as.data.frame(raw, stringsAsFactors = FALSE)
raw[is.na(raw)] <- ""

is_task_header <- function(cell) grepl("^\\d+\\.\\s+[A-Za-z]", cell)

records <- list()
current_task <- NA_character_
in_after <- FALSE
f1_col <- NA_integer_
have_header <- FALSE

for (i in seq_len(nrow(raw))) {
  row <- as.character(unlist(raw[i, ]))
  header_idx <- which(is_task_header(row))[1]
  if (!is.na(header_idx)) {
    key <- tolower(trimws(row[header_idx]))
    current_task <- if (key %in% names(target_tasks)) target_tasks[[key]] else NA_character_
    in_after <- FALSE; f1_col <- NA_integer_; have_header <- FALSE
    next
  }
  if (is.na(current_task)) next
  joined <- tolower(paste(row, collapse = " "))
  if (grepl("after filtering based on correct evidence", joined, fixed = TRUE)) {
    in_after <- TRUE; f1_col <- NA_integer_; have_header <- FALSE
    next
  }
  if (!in_after) next
  if (!have_header) {
    if (any(tolower(row) == "model")) {
      hdr <- tolower(row)
      f1_col <- which(startsWith(hdr, "f1"))[1]
      have_header <- TRUE
    }
    next
  }
  model_key <- tolower(trimws(row[1]))
  if (model_key %in% names(target_models) && !is.na(f1_col) && f1_col <= length(row)) {
    val <- suppressWarnings(as.numeric(row[f1_col]))
    if (!is.na(val)) {
      records[[length(records) + 1]] <- data.frame(
        task = current_task, model = target_models[[model_key]], value = val
      )
    }
  }
}

df_long <- bind_rows(records) %>%
  mutate(task = factor(task, levels = task_levels), model = factor(model, levels = model_levels))
stopifnot(nrow(df_long) == length(task_levels) * length(model_levels))
print(df_long %>% pivot_wider(names_from = task, values_from = value))

n_tasks <- length(task_levels)
angle_lookup <- setNames(seq(0, 2 * pi, length.out = n_tasks + 1)[1:n_tasks], task_levels)

df_plot <- df_long %>% mutate(angle = angle_lookup[as.character(task)])
df_closed <- df_plot %>%
  group_by(model) %>%
  group_modify(~ bind_rows(.x, .x[1, ])) %>%
  mutate(angle = ifelse(row_number() == n(), 2 * pi, angle)) %>%
  ungroup()

coord_radar <- function(theta = "x", start = 0, direction = 1) {
  theta <- match.arg(theta, c("x", "y"))
  r <- if (theta == "x") "y" else "x"
  ggproto("CoordRadar", CoordPolar, theta = theta, r = r,
          start = start, direction = sign(direction),
          is_linear = function(coord) TRUE, clip = "off")
}

circle_radii <- c(0.25, 0.50, 0.75, 1.00)
theta_seq <- seq(0, 2 * pi, length.out = 200)
grid_circles <- do.call(rbind, lapply(circle_radii, function(r) {
  data.frame(angle = theta_seq, value = r, r_id = r)
}))
grid_spokes <- data.frame(
  angle = rep(unname(angle_lookup), each = 2),
  value = rep(c(0, 1), times = n_tasks),
  spoke_id = rep(names(angle_lookup), each = 2)
)
radial_labels <- data.frame(angle = 0, y = circle_radii,
                            label = c("0.25", "0.50", "0.75", "1.00"))

cols <- c(
  "DeepSeek-V3"  = "#0072B2", "DeepSeek-R1"  = "#D55E00",
  "Llama-3.1-8B" = "#009E73", "Kimi-K2.6"    = "#CC79A7",
  "GPT-4o-mini"  = "#E69F00", "o3"           = "#B8860B",
  "o4-mini"      = "#000000", "Qwen3-32B"    = "#56B4E9"
)

p <- ggplot() +
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
             inherit.aes = FALSE, size = 3.0, color = "grey30",
             label.size = 0, label.padding = unit(0.1, "lines"),
             fill = alpha("white", 0.85), hjust = 1.15, vjust = -0.25) +
  scale_color_manual(values = cols) +
  scale_x_continuous(breaks = unname(angle_lookup), labels = names(angle_lookup),
                     limits = c(0, 2 * pi)) +
  scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
  coord_radar() +
  theme_void(base_size = 13) +
  theme(
    axis.text.x       = element_text(size = 12, face = "bold", color = "black",
                                     margin = margin(t = 5, b = 8), lineheight = 0.95),
    legend.position    = "bottom",
    legend.title       = element_blank(),
    legend.text        = element_text(size = 10.5),
    legend.key.width   = unit(0.9, "cm"),
    legend.key.height  = unit(0.45, "cm"),
    legend.spacing.x   = unit(0.2, "cm"),
    plot.margin        = margin(t = 30, r = 70, b = 15, l = 70)
  ) +
  guides(color = guide_legend(nrow = 4, byrow = TRUE, override.aes = list(linewidth = 1.4)))

ggsave("radar_single_agent_single_turn_8models.png", p,
       width = 6.2, height = 6.8, dpi = 380, bg = "white")