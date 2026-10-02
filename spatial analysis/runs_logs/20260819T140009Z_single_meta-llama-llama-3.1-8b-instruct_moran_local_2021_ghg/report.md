# Single-agent report: moran_local_2021_ghg

**Predicted:** False

**Detail:** {', '.join([f'{borough} (HH, statistical method={moran_i})' if hot_spot else f'{borough} (LL, statistical method={moran_i})' for borough, moran_i in hot_spots + cold_spots]}")

**Answer line:** {', '.join([f'{borough} (HH, statistical method={moran_i})' if hot_spot else f'{borough} (LL, statistical method={moran_i})' for borough, moran_i in hot_spots + cold_spots]}")
