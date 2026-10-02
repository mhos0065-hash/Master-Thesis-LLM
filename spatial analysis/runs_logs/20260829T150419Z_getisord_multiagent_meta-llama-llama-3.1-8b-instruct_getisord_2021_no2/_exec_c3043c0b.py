
import builtins
_real_import = builtins.__import__
_blocked_imports = ['ftplib', 'smtplib', 'pip']

def _guarded_import(name, *args, **kwargs):
    if any(name == b or name.startswith(b + '.') for b in _blocked_imports):
        raise ImportError(f"Import of '{name}' is blocked inside the sandbox.")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import

# Block actual network calls rather than blocking the imports themselves.
# Libraries like matplotlib and pandas import socket/urllib internally for
# harmless reasons (defining classes, string parsing) -- blocking the
# import breaks them. We only want to stop the sandbox code from actually
# reaching the network, so we disable the connecting functions instead.
def _blocked_network(*a, **kw):
    raise RuntimeError("Network access is blocked inside the sandbox.")

import socket as _socket
_socket.socket.connect = _blocked_network
_socket.create_connection = _blocked_network

try:
    import urllib.request as _urllib_request
    _urllib_request.urlopen = _blocked_network
except Exception:
    pass

try:
    import http.client as _http_client
    _http_client.HTTPConnection.connect = _blocked_network
except Exception:
    pass

try:
    import requests as _requests
    _requests.api.request = _blocked_network
except Exception:
    pass

import pandas as pd
import numpy as np
DATASET_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\getisord\\getisord_2021_no2.csv'
ARTIFACTS_DIR = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\runs\\20260829T150419Z_getisord_multiagent_meta-llama-llama-3.1-8b-instruct_getisord_2021_no2\\artifacts'
ADJACENCY_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\borough_adjacency.csv'


import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('chunks_csv/getisord/getisord_2021_no2.csv')
adjacency = pd.read_csv('chunks_csv/borough_adjacency.csv')

# Handle missing values in the NO2_value column
data['NO2_value'] = data['NO2_value'].fillna(data['NO2_value'].mean())

# Calculate combined NO2 values
data['combined_NO2'] = data['NO2_value']
for index, row in data.iterrows():
    neighbors = adjacency[adjacency['borough'] == row['borough']]
    data.loc[index, 'combined_NO2'] += neighbors['neighbor'].map(data.set_index('borough')['NO2_value']).sum()

# Calculate citywide average
citywide_average = data['combined_NO2'].mean()

# Apply Getis-Ord Gi* method
data['Gi*'] = (data['combined_NO2'] - citywide_average) / np.sqrt((data['combined_NO2'] - citywide_average).var() * (data['borough'].nunique() - 1))

# Identify hot and cold spots
hot_spots = data[data['Gi*'] > 1.96]
cold_spots = data[data['Gi*'] < -1.96]

# Print results
data_points_used = data['NO2_value'].notna().sum()
print(f'data_points_used = {data_points_used}')

results = []
for index, row in hot_spots.iterrows():
    results.append(f'{row["borough"]} (HOT, Gi*={row["Gi*"]})')
for index, row in cold_spots.iterrows():
    results.append(f'{row["borough"]} (COLD, Gi*={row["Gi*"]})')

if results:
    print(f'FINAL ANSWER: {", ".join(results)}')
else:
    print('FINAL ANSWER: none')
