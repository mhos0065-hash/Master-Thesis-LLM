
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
DATASET_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\changespillover\\changespillover_2022_2023_waste.csv'
ARTIFACTS_DIR = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\runs\\20260819T231326Z_changespillover_openai-gpt-4o-mini_changespillover_2022_2023_waste\\artifacts'
ADJACENCY_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\borough_adjacency.csv'


import pandas as pd
import numpy as np

# Load the datasets
waste_data = pd.read_csv('chunks_csv/changespillover/changespillover_2022_2023_waste.csv')
adjacency_data = pd.read_csv('chunks_csv/borough_adjacency.csv')

# Compute the change in waste from 2022 to 2023
waste_data['change'] = waste_data['Waste_2023'] - waste_data['Waste_2022']

# Create a mapping of boroughs to their changes
change_map = waste_data.set_index('borough')['change'].to_dict()

# Create a list to hold the changes for neighbors
neighbor_changes = []

# For each borough, find its neighbors and collect their changes
for borough in adjacency_data['borough'].unique():
    neighbors = adjacency_data[adjacency_data['borough'] == borough]['neighbor'].tolist()
    borough_change = change_map.get(borough, 0)
    for neighbor in neighbors:
        neighbor_change = change_map.get(neighbor, 0)
        neighbor_changes.append((borough_change, neighbor_change))

# Calculate the mean change and the variance of changes
mean_change = np.mean([change[0] for change in neighbor_changes])
mean_neighbor_change = np.mean([change[1] for change in neighbor_changes])

# Calculate the covariance and variance
covariance = np.cov([change[0] for change in neighbor_changes], [change[1] for change in neighbor_changes])[0][1]
variance = np.var([change[0] for change in neighbor_changes])

# Calculate the Moran's I statistic
n = len(neighbor_changes)
moran_i = (n / np.sum([1 for _ in neighbor_changes])) * (covariance / variance)

# Permutation test for significance
np.random.seed(0)
permuted_moran_i = []
for _ in range(500):
    permuted_changes = np.random.permutation([change[0] for change in neighbor_changes])
    perm_cov = np.cov(permuted_changes, [change[1] for change in neighbor_changes])[0][1]
    perm_moran_i = (n / np.sum([1 for _ in neighbor_changes])) * (perm_cov / variance)
    permuted_moran_i.append(perm_moran_i)

# Calculate p-value
p_value = np.sum(np.array(permuted_moran_i) >= moran_i) / 500

# Determine if clustered or not based on p-value
if p_value < 0.05:
    result = "CLUSTERED"
else:
    result = "NOT CLUSTERED"

# Print the final result
print(f"FINAL ANSWER: {result} (test statistic = {moran_i}, p = {p_value})")
