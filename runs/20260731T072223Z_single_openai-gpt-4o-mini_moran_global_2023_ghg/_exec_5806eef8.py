
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
DATASET_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\thesis_experiment_pipeline\\first run_new\\thesis_experiment_pipeline\\chunks_csv\\moran_global_2023_ghg.csv'
ARTIFACTS_DIR = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\thesis_experiment_pipeline\\first run_new\\thesis_experiment_pipeline\\runs\\20260731T072223Z_single_openai-gpt-4o-mini_moran_global_2023_ghg\\artifacts'
ADJACENCY_PATH = 'D:\\Australia\\Monash\\Semester 4\\Research stream\\thesis_experiment_pipeline\\first run_new\\thesis_experiment_pipeline\\chunks_csv\\borough_adjacency.csv'


import pandas as pd

# Load the greenhouse gas emissions data
DATASET_PATH = 'chunks_csv/moran_global_2023_ghg.csv'
df = pd.read_csv(DATASET_PATH)

# Display the first few rows of the dataframe to understand its structure
df.head()
