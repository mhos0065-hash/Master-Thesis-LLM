# Single-agent report: moran_local_no2crime_no2_Q4_2021

**Predicted:** False

**Detail:** {', '.join([f'{borough} (HH, p = {p_values[boroughs.tolist().index(borough)})' if moran_i_values[borough] > 0.5 else f'{borough} (LL, p = {p_values[boroughs.tolist().index(borough)})' for borough, moran_i_value in moran_i_values.items() if moran_i_value > 0.5 or moran_i_value < -0.5])}")

**Answer line:** {', '.join([f'{borough} (HH, p = {p_values[boroughs.tolist().index(borough)})' if moran_i_values[borough] > 0.5 else f'{borough} (LL, p = {p_values[boroughs.tolist().index(borough)})' for borough, moran_i_value in moran_i_values.items() if moran_i_value > 0.5 or moran_i_value < -0.5])}")
