
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
DATASET_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\bivariate\\bivariate_2021_energy_ghg.csv'
ARTIFACTS_DIR = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\runs\\20260815T062122Z_bivariate_openai-gpt-4o-mini_bivariate_2021_energy_ghg\\artifacts'
ADJACENCY_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\Master Thesis\\Master-Thesis-LLM\\spatial analysis\\chunks_csv\\borough_adjacency.csv'


import pandas as pd
import numpy as np

# Load the datasets
data_path = 'chunks_csv/bivariate/bivariate_2021_energy_ghg.csv'
adjacency_path = ADJACENCY_PATH

df = pd.read_csv(data_path)
adjacency_df = pd.read_csv(adjacency_path)

# Prepare the data
# Create a mapping of boroughs to their Energy and GHG values
borough_values = df.set_index('borough')[['Energy_value', 'GHG_value']]

# Create a function to calculate the observed Lee's L statistic
def calculate_lees_L(df, borough_values):
    L = 0
    n = len(df)
    
    for index, row in df.iterrows():
        borough = row['borough']
        neighbors = row['neighbor'].split()  # Split neighbors by whitespace
        energy_value = borough_values.loc[borough, 'Energy_value']
        
        for neighbor in neighbors:
            if neighbor in borough_values.index:
                ghg_value = borough_values.loc[neighbor, 'GHG_value']
                L += energy_value * ghg_value
    
    return L / n  # Normalize by the number of boroughs

# Calculate the observed Lee's L statistic
observed_L = calculate_lees_L(adjacency_df, borough_values)

# Permutation test to assess significance
n_permutations = 500
permuted_L_values = []

for _ in range(n_permutations):
    # Shuffle GHG values
    shuffled_ghg = borough_values['GHG_value'].sample(frac=1, random_state=None).reset_index(drop=True)
    
    # Create a new DataFrame for the shuffled values
    shuffled_borough_values = borough_values.copy()
    shuffled_borough_values['GHG_value'] = shuffled_ghg
    
    # Calculate Lee's L for the shuffled data
    L_permuted = calculate_lees_L(adjacency_df, shuffled_borough_values)
    permuted_L_values.append(L_permuted)

# Calculate p-value
p_value = np.sum(np.array(permuted_L_values) >= observed_L) / n_permutations

# Determine the relationship
if p_value < 0.05:
    if observed_L > 0:
        result = "POSITIVE"
    else:
        result = "NEGATIVE"
else:
    result = "NOT SIGNIFICANT"

# Print the results
print(f"Observed L: {observed_L}, p-value: {p_value}")
