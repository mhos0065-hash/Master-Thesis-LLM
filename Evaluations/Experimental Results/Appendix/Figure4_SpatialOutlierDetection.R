# =====================================================================
# Figure -- Task 4: Spatial Outlier Detection (Masking-Corrected)
# =====================================================================
# Tests whether a location deviates from what its spatial neighbourhood
# PREDICTS, via x_hat_i = alpha + beta * x_bar_N(i) (Equation 2, main
# text). Because a true outlier included in the shared regression fit
# can pull the line toward itself (masking bias), this implements the
# exact two-pass correction described in Section 3.1.3: fit once, flag
# |first-pass residual| > 1.96 as candidates, REFIT excluding them, then
# REAPPLY the refined line to every location (including candidates) to
# obtain the final residual z_i used for classification.
#
# NOTE: R is not available in the authoring sandbox this script was
# written in, so this has NOT been execution-tested end to end --
# please run it locally and check the console output / saved figure
# before using it.
# =====================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(scales)
library(ggrepel)

source("01_wrangle.R")   # only needed for clean_borough(); see note in
# Figure2_CrossVariableClustering.R

# ── Load borough boundaries ───────────────────────────────────
boroughs_sf_raw <- st_read(
  "https://github.com/LingruFeng/GIS_assessment/blob/main/London_Boroughs.gpkg?raw=true",
  quiet = TRUE
)

boroughs_sf <- boroughs_sf_raw %>%
  st_transform(4326) %>%
  rename(geometry = geom) %>%
  st_set_geometry("geometry") %>%
  mutate(borough = clean_borough(name)) %>%
  select(borough, geometry)

# ── Load variable ────────────────────────────────────────────
# Recorded Crime Rate, 2023. City of London has no recorded-crime data
# and is dropped by the inner join below, consistent with Table 2 of
# the main text.
crime_raw <- read.csv("2023.csv", stringsAsFactors = FALSE)

crime_df <- crime_raw %>%
  select(borough, Crime = crime_per_1k) %>%
  mutate(Crime = as.numeric(as.character(Crime))) %>%
  filter(!is.na(Crime))

spatial_df <- boroughs_sf %>%
  inner_join(crime_df, by = "borough")

# ── Spatial weights and spatial lag ─────────────────────────
nb_queen <- poly2nb(spatial_df, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)
lag_x    <- lag.listw(lw, spatial_df$Crime, zero.policy = TRUE)

x <- spatial_df$Crime

# ── PASS 1: fit on ALL locations, flag candidates ───────────
fit1  <- lm(x ~ lag_x)
res1  <- residuals(fit1)
z1    <- scale(res1)[, 1]
z_thresh <- 1.96

is_candidate <- abs(z1) > z_thresh
cat(sprintf("\nPass 1: %d candidate(s) flagged for exclusion: %s\n",
            sum(is_candidate), paste(spatial_df$borough[is_candidate], collapse = ", ")))

# ── PASS 2: REFIT excluding candidates ──────────────────────
fit2 <- lm(x[!is_candidate] ~ lag_x[!is_candidate])
cat(sprintf("Pass 1 slope = %.3f  ->  Pass 2 (refined) slope = %.3f\n",
            coef(fit1)[2], coef(fit2)[2]))

# ── REAPPLY the refined line to EVERY location, including the
#    excluded candidates, to obtain the final residual z_i ─────
x_hat_final <- coef(fit2)[1] + coef(fit2)[2] * lag_x
res_final   <- x - x_hat_final
z_final     <- scale(res_final)[, 1]

spatial_df <- spatial_df %>%
  mutate(
    lag_x = lag_x,
    z1 = z1,
    z_final = z_final,
    was_candidate = is_candidate,
    cluster = case_when(
      z_final >  z_thresh ~ "outlier_high",
      z_final < -z_thresh ~ "outlier_low",
      TRUE                 ~ "not_outlier"
    )
  )

cat("\nFinal outlier classification:\n"); print(table(spatial_df$cluster))

# ── Shared palette ───────────────────────────────────────────
cluster_colors <- c("outlier_high" = "#D7191C", "outlier_low" = "#2C7BB6",
                    "not_outlier" = "#BDBDBD", "no_data" = "white")
cluster_levels <- c("outlier_high", "outlier_low", "not_outlier")
cluster_labels <- c("outlier_high" = "Outlier-High", "outlier_low" = "Outlier-Low",
                    "not_outlier" = "Not an outlier", "no_data" = "No data (City of London)")

BASE <- 26

# ── Panel (a): outlier map ──────────────────────────────────
sf_data <- spatial_df %>% mutate(cluster = factor(cluster, levels = cluster_levels))
present <- levels(droplevels(sf_data$cluster))
cols    <- cluster_colors[present]

nodata_sf <- boroughs_sf %>%
  filter(!borough %in% spatial_df$borough) %>%
  mutate(cluster = "no_data")

map_sf <- sf_data %>%
  mutate(cluster = as.character(cluster)) %>%
  select(borough, cluster) %>%
  bind_rows(select(nodata_sf, borough, cluster)) %>%
  mutate(cluster = factor(cluster, levels = c(cluster_levels, "no_data")))

present_map <- c(present, "no_data")
cols_map    <- cluster_colors[present_map]

centroids <- sf_data %>%
  filter(cluster %in% c("outlier_high", "outlier_low")) %>%
  mutate(
    cx = st_coordinates(st_centroid(geometry))[, 1],
    cy = st_coordinates(st_centroid(geometry))[, 2]
  ) %>%
  st_drop_geometry()

map_a <- ggplot(map_sf) +
  geom_sf(aes(fill = cluster), color = "white", linewidth = 0.6) +
  geom_sf(data = filter(map_sf, cluster == "no_data"), fill = NA,
          color = "grey50", linewidth = 0.5, inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  scale_fill_manual(
    values = cols_map, drop = FALSE, name = "Classification",
    breaks = present_map, labels =  cluster_labels[present_map],
    guide = guide_legend(
      override.aes = list(color = "grey40", linewidth = 0.5),
      keywidth = unit(1.4, "cm"), keyheight = unit(1.4, "cm")
    )
  ) +
  geom_label_repel(
    data = centroids, aes(x = cx, y = cy, label = borough, fill = cluster),
    color = "white", fontface = "bold", size = BASE * 0.36,
    label.size = 0.3, label.r = unit(0.15, "lines"),
    box.padding = 0.8, point.padding = 0.5, force = 6, force_pull = 0.2,
    max.overlaps = Inf, show.legend = FALSE, min.segment.length = 0, seed = 200
  ) +
  labs(tag = "a", subtitle = "Recorded Crime Rate (per 1,000 residents), 2023") +
  theme_void(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 3, color = "grey20", face = "italic",
                                 margin = margin(t = 4, b = 4)),
    legend.position = "right",
    legend.text = element_text(size = BASE + 2),
    legend.title = element_text(size = BASE + 3, face = "bold"),
    plot.margin = margin(0, 0, 0, 0)
  )

# ── Panel (b): masking-correction scatter ───────────────────
# Shows BOTH the naive first-pass line (dashed, includes candidates)
# and the refined, masking-corrected line (solid, candidates excluded
# then reapplied to all) -- this is the single most important visual
# for explaining WHY this task needs a two-pass correction at all: a
# true outlier pulls the naive line toward itself, understating its
# own deviation.
xlo <- min(lag_x, na.rm = TRUE) - 5
xhi <- max(lag_x, na.rm = TRUE) + 5
ylo <- min(x, na.rm = TRUE) - 5
yhi <- max(x, na.rm = TRUE) + 8

scatter_b <- ggplot(sf_data, aes(x = lag_x, y = Crime)) +
  coord_cartesian(xlim = c(xlo, xhi), ylim = c(ylo, yhi), expand = FALSE) +
  geom_point(aes(color = cluster), size = 4.5, alpha = 0.9) +
  geom_abline(aes(slope = coef(fit1)[2], intercept = coef(fit1)[1],
                  linetype = "Pass 1 (naive, includes outliers)"),
              color = "grey45", linewidth = 1.3,  key_glyph = "path") +
  geom_abline(aes(slope = coef(fit2)[2], intercept = coef(fit2)[1],
                  linetype = "Pass 2 (refined, masking-corrected)"),
              color = "#CC3333", linewidth = 1.6,  key_glyph = "path") +
  scale_linetype_manual(
    name = NULL,
    values = c("Pass 1 (naive, includes outliers)" = "dashed",
               "Pass 2 (refined, masking-corrected)" = "solid"),
    guide = guide_legend(
      override.aes = list(color = c("grey45", "#CC3333"), linewidth = c(1.1, 1.4)),
      order = 2, nrow = 2
    )
  ) +
  scale_color_manual(values = cols, guide = "none") +
  geom_text_repel(
    data = filter(sf_data, cluster != "not_outlier"),
    aes(label = borough, color = cluster), size = BASE * 0.32, fontface = "bold",
    box.padding = 0.6, point.padding = 0.4, force = 2, force_pull = 0.5,
    min.segment.length = 0.2, max.overlaps = Inf,
    xlim = c(xlo, xhi), ylim = c(ylo, yhi), show.legend = FALSE, seed = 77
  ) +
  labs(tag = "b",
       subtitle = "Crime rate vs. spatial lag, with two-pass regression fit",
       x = "Spatial lag of crime rate (mean of neighbours)",
       y = "Crime rate (per 1,000 residents)") +
  theme_classic(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 3, color = "grey20", face = "italic",
                                 margin = margin(t = 4, b = 4)),
    plot.title.position = "plot",
    axis.title = element_text(size = BASE + 3, face = "bold"),
    axis.text = element_text(size = BASE + 1),
    legend.position = "bottom",
    legend.text = element_text(size = BASE + 1),
    legend.key.width = unit(1.8, "cm"),
    plot.margin = margin(0, 40, 4, 70)
  )

# ── Combine ───────────────────────────────────────────────────
combined <- (map_a | scatter_b) + plot_layout(widths = c(1.3, 1))

caption_text <- paste0(
  "(a) Map: red = borough whose crime rate is an outlier relative to its neighbours; grey = not an outlier; white = no data (City of London).\n",
  "(b) Scatter: each point is a borough; x = mean crime rate of its neighbouring boroughs (spatial lag), y = its own crime rate.\n",
  "Dashed line = Pass 1 fit on all boroughs; solid line = Pass 2 refit after excluding the Pass 1 candidates (|z| > 1.96).\n",
  "Outlier = final standardised residual z beyond \u00b11.96 (Pass 2 line applied to every borough, including the excluded candidates); n = ",
  nrow(sf_data), " boroughs."
) %>% str_wrap(width = 200)

final_plot <- combined +
  plot_annotation(
    title = "Spatial Outlier Detection: Recorded Crime Rate Across London Boroughs (2023)",
    subtitle = "Masking-Corrected Spatial Regression  \u00b7  Queen Contiguity Spatial Lag",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = BASE + 11, face = "bold", hjust = 0.5,
                                margin = margin(t = 4, b = 8)),
      plot.subtitle = element_text(size = BASE + 4, hjust = 0.5, color = "grey30",
                                   margin = margin(t = 0, b = 10)),
      plot.caption = element_text(size = BASE - 1, hjust = 0, color = "grey15",
                                  lineheight = 1.4, margin = margin(t = 8)),
      plot.margin = margin(6, 12, 6, 12)
    )
  )

# ── Save ──────────────────────────────────────────────────────
ggsave(
  "figures/Figure_SpatialOutlierDetection_Crime.png",
  plot = final_plot, width = 32, height = 16, dpi = 300, bg = "white"
)

cat("\nSaved: figures/Figure_SpatialOutlierDetection_Crime.png\n")