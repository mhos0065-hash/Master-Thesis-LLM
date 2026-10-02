# Single-agent report: cyclehire_local_selfcontainmentratio_w003

**Predicted:** False

**Detail:** {', '.join([f'{borough} (HH, statistical method={ratio:.3f})' if (borough, ratio) in hot_spots else f'{borough} (LL, statistical method={ratio:.3f})' for borough, ratio in self_containment_ratios.itertuples() if (borough, ratio) in hot_spots or (borough, ratio) in cold_spots]}")

**Answer line:** {', '.join([f'{borough} (HH, statistical method={ratio:.3f})' if (borough, ratio) in hot_spots else f'{borough} (LL, statistical method={ratio:.3f})' for borough, ratio in self_containment_ratios.itertuples() if (borough, ratio) in hot_spots or (borough, ratio) in cold_spots]}")
