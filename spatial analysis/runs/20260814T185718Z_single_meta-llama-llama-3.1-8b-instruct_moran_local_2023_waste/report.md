# Single-agent report: moran_local_2023_waste

**Predicted:** False

**Detail:** {merged_df['borough'].unique()[np.argmax(hot_spots['Waste_value'])] if p_value_hot < 0.05 else None}, {merged_df['borough'].unique()[np.argmin(cold_spots['Waste_value'])] if p_value_cold < 0.05 else None}")

**Answer line:** {merged_df['borough'].unique()[np.argmax(hot_spots['Waste_value'])] if p_value_hot < 0.05 else None}, {merged_df['borough'].unique()[np.argmin(cold_spots['Waste_value'])] if p_value_cold < 0.05 else None}")
