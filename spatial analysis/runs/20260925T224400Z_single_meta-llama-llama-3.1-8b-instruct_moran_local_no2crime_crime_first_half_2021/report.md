# Single-agent report: moran_local_no2crime_crime_first_half_2021

**Predicted:** False

**Detail:** {', '.join([f'{borough} ({'HH' if crime_rate > np.mean(neighbor_crime_rates) else 'LL'}' if (borough, crime_rate) in hotspots else (borough, crime_rate) in coldspots) for (borough, crime_rate) in hotspots + coldspots])}")

**Answer line:** {', '.join([f'{borough} ({'HH' if crime_rate > np.mean(neighbor_crime_rates) else 'LL'}' if (borough, crime_rate) in hotspots else (borough, crime_rate) in coldspots) for (borough, crime_rate) in hotspots + coldspots])}")
