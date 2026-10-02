# =====================================================================
# Figure -- Task 3: Spatial Hotspot Detection (Getis-Ord Gi*)
# =====================================================================
# Tests whether a location, TOGETHER WITH its neighbours (the location
# includes itself in its own neighbourhood -- the "star" in Gi*), forms
# an unusually intense LOCAL CONCENTRATION relative to the CITYWIDE
# AVERAGE. Distinct from Task 1 (LISA): Gi* measures the magnitude of
# a local sum against a global baseline, not similarity between
# neighbouring values.
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
# Greenhouse gas emissions per capita, 2023.
ghg_raw <- read.csv("2023.csv", stringsAsFactors = FALSE)

ghg_df <- ghg_raw %>%
  select(borough, GHG = ghg_tco2e_per_capita) %>%
  mutate(GHG = as.numeric(as.character(GHG))) %>%
  filter(!is.na(GHG))

spatial_df <- boroughs_sf %>%
  inner_join(ghg_df, by = "borough")

# ── Spatial weights INCLUDING SELF (required for Gi*, not plain Gi) ─
nb_queen      <- poly2nb(spatial_df, queen = TRUE)
nb_queen_star <- include.self(nb_queen)          # adds i to its own neighbour set
lw_star       <- nb2listw(nb_queen_star, style = "W", zero.policy = TRUE)

# ── Getis-Ord Gi* ────────────────────────────────────────────
# spdep::localG() on a self-inclusive listw returns Gi* directly as a
# standardised z-score, matching Equation (1) in the main text.
gi_star <- localG(spatial_df$GHG, lw_star, zero.policy = TRUE)
gi_vals <- as.numeric(gi_star)

z_95 <- 1.96    # 95% two-tailed critical value
z_99 <- 2.576   # 99% two-tailed critical value -- matches manifest exactly

missing_mask <- is.na(spatial_df$GHG)

spatial_df <- spatial_df %>%
  mutate(
    gi = gi_vals,
    cluster = case_when(
      missing_mask         ~ "No data",
      gi >  z_99            ~ "HOT (99%)",
      gi >  z_95            ~ "HOT (95%)",
      gi < -z_99            ~ "COLD (99%)",
      gi < -z_95            ~ "COLD (95%)",
      TRUE                   ~ "ns"
    )
  )

cat("\nGi* hotspot classification:\n"); print(table(spatial_df$cluster))

# ── Shared palette ───────────────────────────────────────────
cluster_colors <- c(
  "HOT (99%)" = "#A50021", "HOT (95%)" = "#F4A582",
  "COLD (99%)" = "#08306B", "COLD (95%)" = "#92C5DE",
  "ns" = "#BDBDBD", "No data" = "#F5F0E8"
)
cluster_levels <- c("HOT (99%)", "HOT (95%)", "COLD (99%)", "COLD (95%)", "ns", "No data")

BASE <- 26

# ── Panel (a): hotspot map ──────────────────────────────────
sf_data <- spatial_df %>% mutate(cluster = factor(cluster, levels = cluster_levels))
present <- cluster_levels   # show every category in the legend, even if empty for this variable
cols    <- cluster_colors[present]

centroids <- sf_data %>%
  filter(cluster %in% c("HOT (99%)", "HOT (95%)", "COLD (99%)", "COLD (95%)")) %>%
  mutate(
    cx = st_coordinates(st_centroid(geometry))[, 1],
    cy = st_coordinates(st_centroid(geometry))[, 2]
  ) %>%
  st_drop_geometry()

anchor_xy <- st_coordinates(st_centroid(st_union(sf_data$geometry)))[1, ]
bb <- st_bbox(sf_data)

# one opaque dummy feature per level, parked well outside the map window
dummy_sf <- st_as_sf(
  tibble(
    cluster = factor(cluster_levels, levels = cluster_levels),
    x = as.numeric(bb["xmin"]) - 10,
    y = as.numeric(bb["ymin"]) - 10
  ),
  coords = c("x", "y"), crs = st_crs(sf_data)
)

# geom_sf() does not reliably show unused factor levels in the legend,
# even with scale drop = FALSE (a known ggplot2 limitation). Force all
# 6 categories to register by adding one fully invisible dummy point
# per level, anchored at a single valid map coordinate.

map_a <- ggplot(sf_data) +
  geom_sf(aes(fill = cluster), color = "white", linewidth = 0.6,
          key_glyph = draw_key_rect) +
  geom_sf(data = dummy_sf, aes(fill = cluster), shape = 22,
          size = 2, colour = NA, key_glyph = draw_key_rect) +
  coord_sf(xlim = c(bb["xmin"], bb["xmax"]),
           ylim = c(bb["ymin"], bb["ymax"]), expand = FALSE) +
  scale_fill_manual(
    values = cluster_colors,
    limits = cluster_levels,
    drop = FALSE,
    name = "Significance",
    guide = guide_legend(
      keywidth = unit(0.95, "cm"),
      keyheight = unit(0.95, "cm")
    )
  ) +
  geom_label_repel(
    data = centroids, aes(x = cx, y = cy, label = borough, fill = cluster),
    color = "white", fontface = "bold", size = BASE * 0.34,
    label.size = 0.3, label.r = unit(0.15, "lines"),
    box.padding = 0.8, point.padding = 0.5, force = 6, force_pull = 0.2,
    max.overlaps = Inf, show.legend = FALSE, min.segment.length = 0, seed = 200
  ) +
  labs(tag = "a", subtitle = "GHG Emissions (per capita tCO2e), 2023") +
  theme_void(base_size = BASE) +
  theme(
    plot.tag = element_text(
      size = BASE + 6,
      face = "bold",
      margin = margin(r = 4)
    ),
    plot.subtitle = element_text(
      size = BASE + 3,
      color = "grey20",
      face = "italic",
      margin = margin(t = 4, b = 4)
    ),
    
    legend.position = "right",
    legend.title = element_text(
      size = BASE,
      face = "bold"
    ),
    legend.text = element_text(
      size = BASE - 2
    ),
    legend.key = element_blank(),
    legend.key.size = unit(0.95, "cm"),
    legend.spacing.y = unit(0.15, "cm"),
    
    plot.margin = margin(0, 0, 0, 0)
  )

# ── Panel (b): ranked Gi* lollipop chart ────────────────────
# Shows every borough's Gi* value against the +-1.96 threshold lines,
# making explicit WHY each borough was or was not classified as a
# hotspot -- the distance from the threshold, not just the label.
rank_df <- sf_data %>%
  st_drop_geometry() %>%
  arrange(gi) %>%
  mutate(borough = factor(borough, levels = borough))

scatter_b <- ggplot(rank_df, aes(x = gi, y = borough, color = cluster)) +
  geom_vline(xintercept = c(-z_95, z_95), color = "grey40",
             linewidth = 0.8, linetype = "dashed") +
  geom_vline(xintercept = c(-z_99, z_99), color = "grey40",
             linewidth = 0.8, linetype = "dotted") +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = gi, y = borough, yend = borough), linewidth = 1) +
  geom_point(size = 4.2) +
  scale_color_manual(values = cols, guide = "none") +
  annotate("text", x = z_95, y = Inf, label = "95%", hjust = 0.5, vjust = -0.4,
           size = BASE * 0.28, color = "grey30") +
  annotate("text", x = -z_95, y = Inf, label = "95%", hjust = 0.5, vjust = -0.4,
           size = BASE * 0.28, color = "grey30") +
  annotate("text", x = z_99, y = Inf, label = "99%", hjust = 0.5, vjust = -0.4,
           size = BASE * 0.28, color = "grey30") +
  annotate("text", x = -z_99, y = Inf, label = "99%", hjust = 0.5, vjust = -0.4,
           size = BASE * 0.28, color = "grey30") +
  labs(tag = "b", x = expression(paste("Getis-Ord ", G[i]^"*", " (z-score)")), y = NULL) +
  theme_classic(base_size = BASE) +  coord_cartesian(clip = "off") +
  theme(
    axis.title = element_text(size = BASE + 3, face = "bold"),
    axis.text.x = element_text(size = BASE + 1),
    axis.text.y = element_text(size = BASE - 3),
    plot.margin = margin(6, 20, 4, 6)
  )

# ── Combine ───────────────────────────────────────────────────
combined <- (map_a | scatter_b) + plot_layout(widths = c(1.3, 1))

caption_text <- paste0(
  "(a) Map of each borough's hotspot/coldspot result. (b) Each borough's own Gi* z-score, ranked, ",
  "showing how far it sits from the significance cutoffs.\n",
  "\u2022HOT/COLD = a borough and its neighbours together are unusually high or low  ",
  "\u202299% / 95% = confidence level  \u2022ns = not significant  ",
  "\u2022No data = missing source data, excluded from scoring. \n",
  "Gi* compares each borough's own neighbourhood, including itself, against the citywide average ",
  "(n = ", nrow(sf_data), " boroughs)."
) %>% str_wrap(width = 200)

final_plot <- combined +
  plot_annotation(
    title = "Spatial Hotspot Detection: GHG Emissions Across London Boroughs (2023)",
    subtitle = "Getis-Ord Gi* Local Statistic  \u00b7  Self-Inclusive Queen Contiguity Weights",
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
  "figures/Figure_SpatialHotspotDetection_GHG.png",
  plot = final_plot, width = 32, height = 16, dpi = 300, bg = "white"
)

cat("\nSaved: figures/Figure_SpatialHotspotDetection_GHG.png\n")
