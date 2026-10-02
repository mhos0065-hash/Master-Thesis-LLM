# =====================================================================
# Figure -- Task 5: Spatiotemporal Change Clustering Detection
# =====================================================================
# Tests whether the CHANGE in a variable between two time points, in
# both size and direction, clusters spatially, using a signed delta,
# delta_x_i = x_i^(t2) - x_i^(t1), rather than a raw snapshot.
#
#   Panel (a): raw delta map (what changed)
#   Panel (b): Local Moran's I / LISA on the delta (where it clusters)
#   Panel (c): Moran scatterplot; slope = Global Moran's I (the task
#              verdict, same moran.test call as the ground-truth script)
#
# City of London has no data: it is kept in the spatial weights and
# mean-imputed (as in the ground-truth script) but shown as "No data".
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

# ── Load BOTH time points and compute the signed delta ─────────
# Waste Generation, 2021 -> 2023.
waste_2021 <- read.csv("2021.csv", stringsAsFactors = FALSE) %>%
  select(borough, Waste_t1 = waste_tonnes_per_capita) %>%
  mutate(Waste_t1 = as.numeric(as.character(Waste_t1)))

waste_2023 <- read.csv("2023.csv", stringsAsFactors = FALSE) %>%
  select(borough, Waste_t2 = waste_tonnes_per_capita) %>%
  mutate(Waste_t2 = as.numeric(as.character(Waste_t2)))

change_df <- waste_2021 %>%
  inner_join(waste_2023, by = "borough") %>%
  filter(!is.na(Waste_t1), !is.na(Waste_t2)) %>%
  mutate(Delta = Waste_t2 - Waste_t1)

# left_join keeps City of London (Delta = NA) so it stays in the weights
spatial_df <- boroughs_sf %>%
  left_join(change_df, by = "borough")

# ── Spatial weights matrix ────────────────────────────────────
nb_queen <- poly2nb(spatial_df, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

# ── LISA on the DELTA ─────────────────────────────────────────
compute_lisa <- function(sf_obj, col, lw, nsim = 999, p_thresh = 0.05) {
  vals     <- sf_obj[[col]]
  missing  <- is.na(vals)
  vals_imp <- vals
  vals_imp[missing] <- mean(vals, na.rm = TRUE)
  
  lisa  <- localmoran_perm(vals_imp, lw, nsim = nsim,
                           zero.policy = TRUE, alternative = "two.sided")
  z     <- scale(vals_imp)[, 1]
  lag_z <- lag.listw(lw, z, zero.policy = TRUE)
  p_col <- grep("Pr\\(z != E\\(Ii\\)\\)", colnames(lisa), value = TRUE)
  p_val <- lisa[, p_col[length(p_col)]]
  
  cluster <- case_when(
    missing            ~ "No data",
    p_val >= p_thresh  ~ "ns",
    z > 0 & lag_z > 0 ~ "HH",
    z < 0 & lag_z < 0 ~ "LL",
    z > 0 & lag_z < 0 ~ "HL",
    z < 0 & lag_z > 0 ~ "LH",
    TRUE               ~ "ns"
  )
  
  list(
    sf    = sf_obj %>% mutate(cluster = cluster, z = z, lag_z = lag_z,
                              p_val = p_val, val_raw = vals),
    z     = z, lag_z = lag_z
  )
}

delta_lisa <- compute_lisa(spatial_df, "Delta", lw, p_thresh = 0.05)
cat("\nChange-cluster counts:\n"); print(table(delta_lisa$sf$cluster))

# Polygon(s) with no data (City of London), used to draw an outline
nodata_sf <- filter(delta_lisa$sf, cluster == "No data")

# ── Global Moran's I on the signed delta (task verdict) ────────
delta_imp <- spatial_df$Delta
delta_imp[is.na(delta_imp)] <- mean(delta_imp, na.rm = TRUE)
mt      <- moran.test(delta_imp, lw, zero.policy = TRUE)   # same call as ground truth
moran_I <- unname(mt$estimate["Moran I statistic"])
moran_p <- mt$p.value
cat(sprintf("\nGlobal Moran's I = %.4f, p = %.4f -> %s\n", moran_I, moran_p,
            ifelse(moran_p < 0.05, "CLUSTERED", "NOT CLUSTERED")))
moran_txt <- sprintf("Moran scatterplot: Global Moran's I = %.3f, %s", moran_I,
                     ifelse(moran_p < 0.001, "p < 0.001", sprintf("p = %.3f", moran_p)))

# ── Shared palette & levels ───────────────────────────────────
cluster_colors <- c(
  "HH" = "#D7191C", "LH" = "#2C7BB6", "LL" = "#08306B",
  "HL" = "#F46D43", "ns" = "#BDBDBD", "No data" = "white"
)
cluster_levels <- c("HH", "HL", "LH", "LL", "ns", "No data")

cluster_labels <- c("HH" = "High\u2013High", "HL" = "High\u2013Low", "LH" = "Low\u2013High",
                    "LL" = "Low\u2013Low", "ns" = "Not significant", "No data" = "No data")

BASE <- 26

# ── Panel (a): raw delta map (diverging scale -- increase vs decrease) ─
raw_map <- ggplot(delta_lisa$sf) +
  geom_sf(aes(fill = val_raw), color = "white", linewidth = 0.6) +
  geom_sf(data = nodata_sf, fill = NA, color = "grey50", linewidth = 0.5, inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  scale_fill_gradient2(
    low = "#2C7BB6", mid = "#F5F0E8", high = "#D7191C", midpoint = 0, na.value = "white",
    name = "\u0394 Waste\n(tonnes/capita)",
    guide = guide_colorbar(barheight = unit(4, "cm"))
  ) +
  labs(tag = "a", subtitle = "Raw Change, 2021 \u2192 2023 (blue = decrease, red = increase)") +
  theme_void(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 1, color = "grey20", face = "italic",
                                 margin = margin(t = 4, b = 4)),
    legend.position = "right",
    legend.text = element_text(size = BASE + 1),
    legend.title = element_text(size = BASE + 2, face = "bold"),
    plot.margin = margin(0, 30, 0, 0)
  )

# ── Panel (b): LISA cluster map of the delta ────────────────
sf_data <- delta_lisa$sf %>% mutate(cluster = factor(cluster, levels = cluster_levels))
present <- levels(droplevels(sf_data$cluster))
cols    <- cluster_colors[present]

centroids <- sf_data %>%
  filter(cluster %in% c("HH", "LL", "HL", "LH")) %>%
  mutate(
    cx = st_coordinates(st_centroid(geometry))[, 1],
    cy = st_coordinates(st_centroid(geometry))[, 2]
  ) %>%
  st_drop_geometry()

lisa_map <- ggplot(sf_data) +
  geom_sf(aes(fill = cluster), color = "white", linewidth = 0.6) +
  geom_sf(data = nodata_sf, fill = NA, color = "grey50", linewidth = 0.5, inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  scale_fill_manual(
    values = cols, drop = FALSE, name = "Change\ncluster\n(p < 0.05)", breaks = present,
    labels = cluster_labels[present],
    guide = guide_legend(
      override.aes = list(color = "grey40", linewidth = 0.5),
      keywidth = unit(1.4, "cm"), keyheight = unit(1.4, "cm")
    )
  ) +
  geom_label_repel(
    data = centroids, aes(x = cx, y = cy, label = borough, fill = cluster),
    color = "white", fontface = "bold", size = BASE * 0.32,
    label.size = 0.3, label.r = unit(0.15, "lines"),
    box.padding = 0.8, point.padding = 0.5, force = 6, force_pull = 0.2,
    max.overlaps = Inf, show.legend = FALSE, min.segment.length = 0, seed = 200
  ) +
  labs(tag = "b", subtitle = "LISA on \u0394 Waste (Local Moran's I of the delta)") +
  theme_void(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 1, color = "grey20", face = "italic",
                                 margin = margin(t = 4, b = 4)),
    legend.position = "right",
    legend.text = element_text(size = BASE + 1),
    legend.title = element_text(size = BASE + 2, face = "bold"),
    plot.margin = margin(0, 0, 0, 0)
  )

# ── Panel (c): Moran scatterplot of the delta ───────────────
z <- delta_lisa$z; lag_z <- delta_lisa$lag_z
df_scatter <- tibble(
  z = z, lag_z = lag_z, cluster = sf_data$cluster, borough = sf_data$borough
) %>% filter(cluster != "No data")

xlo <- min(df_scatter$z, na.rm = TRUE) - 0.6
xhi <- max(df_scatter$z, na.rm = TRUE) + 0.6
ylo <- min(df_scatter$lag_z, na.rm = TRUE) - 0.35
yhi <- max(df_scatter$lag_z, na.rm = TRUE) + 0.35

slope_lab <- "Moran regression slope (= Global Moran's I)"

scatter_c <- ggplot(df_scatter, aes(x = z, y = lag_z)) +
  coord_cartesian(xlim = c(xlo, xhi), ylim = c(ylo, yhi), expand = FALSE) +
  geom_hline(yintercept = 0, color = "grey55", linewidth = 0.7, linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.7, linetype = "dashed") +
  annotate("text", x = xhi - 0.1, y = yhi - 0.05, label = "High\u2013High", hjust = 1, vjust = 1,
           size = BASE * 0.28, color = "grey40", fontface = "italic") +
  annotate("text", x = xlo + 0.1, y = yhi - 0.05, label = "Low\u2013High",  hjust = 0, vjust = 1,
           size = BASE * 0.28, color = "grey40", fontface = "italic") +
  annotate("text", x = xlo + 0.1, y = ylo + 0.05, label = "Low\u2013Low",   hjust = 0, vjust = 0,
           size = BASE * 0.28, color = "grey40", fontface = "italic") +
  annotate("text", x = xhi - 0.1, y = ylo + 0.05, label = "High\u2013Low",  hjust = 1, vjust = 0,
           size = BASE * 0.28, color = "grey40", fontface = "italic") +
  geom_point(aes(color = cluster), size = 4.5, alpha = 0.9) +
  geom_abline(aes(slope = moran_I, intercept = 0, linetype = slope_lab),
              color = "#CC3333", linewidth = 1.6, key_glyph = "path") +
  scale_linetype_manual(
    name = NULL, values = setNames("solid", slope_lab),
    guide = guide_legend(override.aes = list(color = "#CC3333", linewidth = 1.4), order = 2)
  ) +
  scale_color_manual(values = cols, guide = "none") +
  geom_text_repel(
    data = filter(df_scatter, cluster %in% c("HH", "LL", "HL", "LH")),
    aes(label = borough, color = cluster), size = BASE * 0.3, fontface = "bold",
    box.padding = 0.6, point.padding = 0.8, force = 2, force_pull = 0.5,
    min.segment.length = 0.2, max.overlaps = Inf,
    xlim = c(xlo, xhi), ylim = c(ylo, yhi), show.legend = FALSE, seed = 77
  ) +
  labs(tag = "c", subtitle = moran_txt,
       x = "\u0394 Waste (standardised z-score)",
       y = "Spatial lag of neighbor's standardised \u0394 Waste") +
  theme_classic(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 1, color = "grey20", face = "italic",
                                 margin = margin(t = 4, b = 4)),
    plot.title.position = "plot",
    axis.title = element_text(size = BASE - 2, face = "bold"),
    axis.text = element_text(size = BASE - 1),
    legend.position = "bottom",
    legend.text = element_text(size = BASE + 1),
    legend.key.width = unit(1.6, "cm"),
    plot.margin = margin(0, 20, 4, 70)
  )

# ── Combine (three panels: raw change | LISA cluster | scatter) ──
combined <- (raw_map | lisa_map | scatter_c) + plot_layout(widths = c(1, 1.25, 1))

caption_text <- paste0(
  "(a) Raw change in waste per capita, 2021 to 2023 (\u0394 Waste = 2023 \u2212 2021; red = increase, blue = decrease; white = no data, City of London).\n",
  "(b) Local Moran's I (LISA) on \u0394 Waste, p < 0.05: High\u2013High = above-average change among above-average neighbours; Low\u2013Low = below-average among below-average; High\u2013Low / Low\u2013High = local outliers.\n",
  "(c) Moran scatterplot: each borough's standardised \u0394 Waste (x) vs the mean of its neighbours (y); the slope of the line is Global Moran's I, which decides whether change is clustered.\n",
  "Above/below average = relative to the London mean change. Queen contiguity weights; LISA p-values from 999 permutations; n = ",
  sum(sf_data$cluster != "No data"), " boroughs."
) %>% str_wrap(width = 200)

final_plot <- combined +
  plot_annotation(
    title = "Spatiotemporal Change Clustering: Waste Generation Across London Boroughs (2021\u21922023)",
    subtitle = "Global Moran's I on Signed Delta (task verdict)  \u00b7  Local Moran's I / LISA (cluster locations)  \u00b7  Queen Contiguity Weights",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = BASE + 10, face = "bold", hjust = 0.5,
                                margin = margin(t = 4, b = 8)),
      plot.subtitle = element_text(size = BASE + 3, hjust = 0.5, color = "grey30",
                                   margin = margin(t = 0, b = 10)),
      plot.caption = element_text(size = BASE, hjust = 0, color = "grey15",
                                  lineheight = 1.4, margin = margin(t = 8)),
      plot.margin = margin(6, 12, 6, 12)
    )
  )

# ── Save ──────────────────────────────────────────────────────
dir.create("figures", showWarnings = FALSE)
ggsave(
  "figures/Figure_ChangeClusteringDetection_Waste.png",
  plot = final_plot, width = 36, height = 14, dpi = 300, bg = "white"
)

cat("\nSaved: figures/Figure_ChangeClusteringDetection_Waste.png\n")