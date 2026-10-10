# =====================================================================
# Figure -- Task 2: Cross-Variable Clustering (Lee's L)
# =====================================================================
# Tests whether two DIFFERENT variables co-vary systematically across
# space (Section 3.1.3, main text): a borough's value on variable X is
# compared not to its own neighbourhood's X, but to its neighbourhood's
# Y -- the bivariate analogue of Local Moran's I. This is the ONLY
# bivariate task in the taxonomy.
#
# NOTE: R is not available in the authoring sandbox this script was
# written in, so this has NOT been execution-tested end to end --
# please run it locally and check the console output / saved figure
# before using it. Column names below are taken directly from the
# uploaded 2023.csv.
# =====================================================================

library(sf)
library(spdep)
library(dplyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(scales)
library(ggrepel)

source("01_wrangle.R")   # only needed if clean_borough() lives there;
                          # otherwise remove this line and define
                          # clean_borough() yourself, or skip if the
                          # 2023.csv borough names already match
                          # boroughs_sf$name exactly.

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

# ── Load the two variables for Cross-Variable Clustering ───────
# Gas Consumption x Waste Generation, 2023 -- both are per-consumer /
# per-capita normalised measures from the same borough-level table.
bivar_raw <- read.csv("2023.csv", stringsAsFactors = FALSE)

bivar_df <- bivar_raw %>%
  select(borough, X = gas_kwh_per_consumer, Y = waste_tonnes_per_capita) %>%
  mutate(
    X = as.numeric(as.character(X)),
    Y = as.numeric(as.character(Y))
  ) %>%
  filter(!is.na(X), !is.na(Y))

# ── Join to spatial, keep only boroughs with BOTH variables ────
spatial_df <- boroughs_sf %>%
  left_join(bivar_df, by = "borough")

# Record which boroughs are missing BEFORE imputing -- these must
# still be classified "No data" later, even though a value has to be
# filled in now so lee()/lag.listw() don't error on non-finite input
# (same approach as Task 1's compute_lisa(): impute with the mean,
# but the imputed value never drives that borough's OWN displayed
# classification -- only its neighbours' spatial lag).
missing_mask <- is.na(spatial_df$X) | is.na(spatial_df$Y)

spatial_df <- spatial_df %>%
  mutate(
    X = ifelse(is.na(X), mean(X, na.rm = TRUE), X),
    Y = ifelse(is.na(Y), mean(Y, na.rm = TRUE), Y)
  )

# ── Spatial weights matrix ──────────────────────────────────────
nb_queen <- poly2nb(spatial_df, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

# ── Lee's L: global statistic + local components ───────────────
# spdep::lee() returns both the global L statistic (via lee.test /
# lee.mc for significance) AND per-location local Lee's L values.
zx <- scale(spatial_df$X)[, 1]
zy <- scale(spatial_df$Y)[, 1]

n_obs <- length(zx)
S2    <- Szero(lw)

lee_local <- lee(zx, zy, lw, n_obs, S2, zero.policy = TRUE)
global_L  <- lee_local$L                     # global Lee's L statistic
local_L   <- lee_local$localL                # per-borough local values

# Two-sided test, matching the ground truth script (bivariate_moran_all_years.R).
# spdep::lee.mc() defaults to alternative = "greater" (upper tail only), under
# which a significantly NEGATIVE L would print a p-value close to 1.
set.seed(1)
lee_perm <- lee.mc(zx, zy, lw, nsim = 999, zero.policy = TRUE,
                   alternative = "two.sided")
global_p <- lee_perm$p.value

cat(sprintf("\nGlobal Lee's L = %.4f, two-sided permutation p-value = %.4f\n",
            global_L, global_p))

# ── Bivariate cluster classification (GeoDa-style quadrants) ───
# X standardised at borough i vs. the SPATIAL LAG of standardised Y
# at borough i's neighbourhood -- the bivariate analogue of the
# univariate LISA quadrant logic used in the Task 1 script.
lag_zy <- lag.listw(lw, zy, zero.policy = TRUE)

# Conditional permutation test for LOCAL Lee's L, per borough. spdep
# only ships a permutation test for the GLOBAL statistic (lee.mc());
# this extends the same logic localmoran_perm() uses for Local Moran's I
# (Task 1) to the bivariate case: hold x_i fixed, randomly reshuffle
# y among the other locations nsim times, recompute the spatial lag
# and local_L_i under each permutation, and compare the observed value
# against that null distribution.
local_lee_perm <- function(zx, zy, lw, nsim = 999, seed = 1) {
  set.seed(seed)
  n <- length(zx)
  obs_L <- zx * lag.listw(lw, zy, zero.policy = TRUE)
  p_val <- numeric(n)
  for (i in seq_len(n)) {
    perm_L <- numeric(nsim)
    for (s in seq_len(nsim)) {
      zy_perm <- zy
      zy_perm[-i] <- sample(zy[-i])          # i's own value stays fixed
      lag_perm <- lag.listw(lw, zy_perm, zero.policy = TRUE)[i]
      perm_L[s] <- zx[i] * lag_perm
    }
    p_val[i] <- (sum(abs(perm_L) >= abs(obs_L[i])) + 1) / (nsim + 1)
  }
  list(local_L = obs_L, p_val = p_val)
}

p_thresh <- 0.05
perm_result <- local_lee_perm(zx, zy, lw, nsim = 999)

# Borough-level permutation significance is approximated here via the
# sign/magnitude of the local L value relative to its neighbourhood-
# size-adjusted expectation; for a rigorous per-borough p-value, loop
# lee.mc() with a leave-one-out local permutation, or use
# rgeoda::local_bimoran() if the rgeoda package is available.
p_thresh <- 0.05
perm_result <- local_lee_perm(zx, zy, lw, nsim = 999)
local_L  <- perm_result$local_L
local_p  <- perm_result$p_val
sig_mask <- local_p < p_thresh

# Conditional permutation test for LOCAL Lee's L, per borough. spdep
# only ships a permutation test for the GLOBAL statistic (lee.mc());
# this extends the same logic localmoran_perm() uses for Local Moran's I
# (Task 1) to the bivariate case: hold x_i fixed, randomly reshuffle
# y among the other locations nsim times, recompute the spatial lag
# and local_L_i under each permutation, and compare the observed value
# against that null distribution.
local_lee_perm <- function(zx, zy, lw, nsim = 999, seed = 1) {
  set.seed(seed)
  n <- length(zx)
  obs_L <- zx * lag.listw(lw, zy, zero.policy = TRUE)
  p_val <- numeric(n)
  for (i in seq_len(n)) {
    perm_L <- numeric(nsim)
    for (s in seq_len(nsim)) {
      zy_perm <- zy
      zy_perm[-i] <- sample(zy[-i])          # i's own value stays fixed
      lag_perm <- lag.listw(lw, zy_perm, zero.policy = TRUE)[i]
      perm_L[s] <- zx[i] * lag_perm
    }
    p_val[i] <- (sum(abs(perm_L) >= abs(obs_L[i])) + 1) / (nsim + 1)
  }
  list(local_L = obs_L, p_val = p_val)
}

cluster <- case_when(
  missing_mask        ~ "No data",
  !sig_mask           ~ "ns",
  zx > 0 & lag_zy > 0  ~ "HH",
  zx < 0 & lag_zy < 0  ~ "LL",
  zx > 0 & lag_zy < 0  ~ "HL",
  zx < 0 & lag_zy > 0  ~ "LH",
  TRUE                 ~ "ns"
)

spatial_df <- spatial_df %>%
  mutate(cluster = cluster, zx = zx, lag_zy = lag_zy, local_L = local_L)

cat("\nBivariate cluster counts:\n"); print(table(spatial_df$cluster))

# ── Shared palette & levels (consistent with Task 1 figure) ────
cluster_colors <- c(
  "HH" = "#D7191C", "LH" = "#2C7BB6", "LL" = "#08306B",
  "HL" = "#F46D43", "ns" = "#BDBDBD", "No data" = "#F5F0E8"
)
cluster_levels <- c("HH", "HL", "LH", "LL", "ns", "No data")

BASE <- 26   # master font knob, matches Task 1 figure

# ── Panel (a): bivariate cluster map ────────────────────────────
sf_data <- spatial_df %>% mutate(cluster = factor(cluster, levels = cluster_levels))
present <- levels(droplevels(sf_data$cluster))
cols    <- cluster_colors[present]

centroids <- sf_data %>%
  filter(cluster %in% c("HH", "LL", "HL", "LH")) %>%
  mutate(
    cx = st_coordinates(st_centroid(geometry))[, 1],
    cy = st_coordinates(st_centroid(geometry))[, 2]
  ) %>%
  st_drop_geometry()

map_a <- ggplot(sf_data) +
  geom_sf(aes(fill = cluster), color = "white", linewidth = 0.6) +
  coord_sf(expand = FALSE) +
  scale_fill_manual(
    values = cols, drop = FALSE, name = "Bivariate\ncluster",
    breaks = present,
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
  labs(tag = "a", subtitle = "Gas Consumption (X) vs. Waste Generation (Y), 2023") +
  theme_void(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    plot.subtitle = element_text(size = BASE + 3, color = "grey20", face = "italic",
                                  margin = margin(t = 4, b = 4)),
    legend.position = "right",
    legend.text = element_text(size = BASE + 3),
    legend.title = element_text(size = BASE + 3, face = "bold"),
    plot.margin = margin(0, 0, 0, 0)
  )

# ── Panel (b): the bivariate scatter Lee's L is built from ─────
# x-axis: standardised Gas Consumption at borough i
# y-axis: spatial LAG of standardised Waste Generation among i's
#         neighbours -- this cross-variable, cross-location pairing
#         is exactly what distinguishes Lee's L from a univariate
#         Moran scatterplot.
xlo <- min(zx, na.rm = TRUE) - 0.6
xhi <- max(zx, na.rm = TRUE) + 0.6
ylo <- min(lag_zy, na.rm = TRUE) - 0.35
yhi <- max(lag_zy, na.rm = TRUE) + 0.35

lm_fit <- lm(lag_zy ~ zx)

scatter_b <- ggplot(sf_data, aes(x = zx, y = lag_zy)) +
  coord_cartesian(xlim = c(xlo, xhi), ylim = c(ylo, yhi), expand = FALSE) +
  scale_x_continuous(oob = oob_keep) +
  scale_y_continuous(oob = oob_keep) +
  geom_hline(yintercept = 0, color = "grey55", linewidth = 0.7, linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.7, linetype = "dashed") +
  geom_point(aes(color = cluster), size = 4.5, alpha = 0.9) +
  geom_abline(aes(slope = coef(lm_fit)[2], intercept = coef(lm_fit)[1],
                  linetype = "Lee's L regression slope"),
              color = "#CC3333", linewidth = 1.6) +
  scale_linetype_manual(
    name = NULL, values = c("Lee's L regression slope" = "solid"),
    guide = guide_legend(override.aes = list(color = "#CC3333", linewidth = 1.4), order = 2)
  ) +
  scale_color_manual(values = cols, guide = "none") +
  geom_text_repel(
    data = filter(sf_data, cluster %in% c("HH", "LL", "HL", "LH")),
    aes(label = borough, color = cluster), size = BASE * 0.32, fontface = "bold",
    box.padding = 0.6, point.padding = 0.4, force = 2, force_pull = 0.5,
    min.segment.length = 0.2, max.overlaps = Inf,
    xlim = c(xlo, xhi), ylim = c(ylo, yhi), show.legend = FALSE, seed = 77
  ) +
  annotate("text", x = xlo + (xhi - xlo) * 0.02, y = yhi - (yhi - ylo) * 0.04,
           label = "LH", hjust = 0, vjust = 1, size = BASE * 0.5,
           color = "#2C7BB6", fontface = "bold") +
  annotate("text", x = xhi - (xhi - xlo) * 0.02, y = yhi - (yhi - ylo) * 0.04,
           label = "HH", hjust = 1, vjust = 1, size = BASE * 0.5,
           color = "#D7191C", fontface = "bold") +
  annotate("text", x = xlo + (xhi - xlo) * 0.02, y = ylo + (yhi - ylo) * 0.04,
           label = "LL", hjust = 0, vjust = 0, size = BASE * 0.5,
           color = "#08306B", fontface = "bold") +
  annotate("text", x = xhi - (xhi - xlo) * 0.02, y = ylo + (yhi - ylo) * 0.04,
           label = "HL", hjust = 1, vjust = 0, size = BASE * 0.5,
           color = "#F46D43", fontface = "bold") +
  labs(tag = "b", x = "Gas Consumption (standardised z-score)",
       y = "Spatial Lag of Waste Generation (standardised)") +
  theme_classic(base_size = BASE) +
  theme(
    plot.tag = element_text(size = BASE + 6, face = "bold", margin = margin(r = 4)),
    axis.title = element_text(size = BASE + 3, face = "bold"),
    axis.text = element_text(size = BASE + 1),
    legend.position = "bottom",
    legend.text = element_text(size = BASE + 3),
    legend.key.width = unit(1.8, "cm"),
    plot.margin = margin(6, 40, 4, 14)
  )

# ── Combine ───────────────────────────────────────────────────
combined <- (map_a | scatter_b) + plot_layout(widths = c(1.3, 1))

caption_text <- paste0(
  "(a) Bivariate cluster map: boroughs where Gas Consumption (X) is significantly linked to neighbouring Waste Generation (Y); No data = City of London.\n",
  "(b) Scatter: each borough's standardised Gas Consumption (x) vs. the mean standardised Waste Generation of its neighbours (y); red line = fitted slope.\n",
  "HH/LL = high/low X with high/low neighbouring Y; HL/LH = high X with low neighbouring Y, or low X with high neighbouring Y; ns = not significant.\n",
  sprintf("Global Lee's L = %.3f (999-permutation two-sided p = %.3f); local tests: 999 permutations, p < %.2f; n = %d boroughs.",
          global_L, global_p, p_thresh, sum(!missing_mask))
) %>% str_wrap(width = 200)

final_plot <- combined +
  plot_annotation(
    title = "Cross-Variable Clustering: Gas Consumption and Waste Generation Across London Boroughs (2023)",
    subtitle = "Lee's L Bivariate Spatial Association  \u00b7  Queen Contiguity Weights  \u00b7  999 Permutations",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = BASE + 11, face = "bold", hjust = 0.5,
                                 margin = margin(t = 4, b = 8)),
      plot.subtitle = element_text(size = BASE + 4, hjust = 0.5, color = "grey30",
                                    margin = margin(t = 0, b = 10)),
      plot.caption  = element_text(size = BASE - 2, hjust = 0, color = "grey15",
                                   lineheight = 1.4, margin = margin(t = 8)),
      plot.margin = margin(6, 12, 6, 12)
    )
  )

# ── Save ──────────────────────────────────────────────────────
ggsave(
  "figures/Figure_CrossVariableClustering_Gas_Waste.png",
  plot = final_plot, width = 32, height = 16, dpi = 300, bg = "white"
)

cat("\nSaved: figures/Figure_CrossVariableClustering_Gas_Waste.png\n")
