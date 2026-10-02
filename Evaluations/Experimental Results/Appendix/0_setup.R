packages <- c(
  "tidyverse",    # dplyr, ggplot2, tidyr, readr, stringr
  "readxl",       # read Excel files
  "sf",           # spatial features
  "spdep",        # Moran's I, spatial weights
  "tmap",         # choropleth maps
  "janitor",      # clean_names(), get_dupes()
  "zoo",          # na.approx() for interpolation
  "scales",       # axis label formatting
  "ggcorrplot",   # correlation matrix heatmap
  "patchwork",    # combine ggplot panels
  "fmsb",         # radar charts
  "viridis",      # colour scales
  "ggsci" ,        # D3 colour palette
  "readr"
)

installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(packages, library, character.only = TRUE))

# ── Global ggplot theme ──────────────────────────────────────
theme_dep <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    plot.caption     = element_text(colour = "grey55", size = 8),
    panel.grid.minor = element_blank(),
    legend.position  = "right"
  )
theme_set(theme_dep)

# ── Output directories ───────────────────────────────────────
dir.create("figures",    showWarnings = FALSE)
dir.create("data/clean", showWarnings = FALSE, recursive = TRUE)

cat("Setup complete.\n")