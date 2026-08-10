"""
A lightweight LOCAL sandbox for running code written by the LLM.

Honest note for your thesis: this is NOT a full security sandbox like
Docker or e2b.dev — it's a "reasonable first line of defence" you can run
with no extra signup. It enforces:

  1. Code runs in its own subprocess (separate from your main program), so
     a crash or infinite loop can't take down the whole pipeline.
  2. A hard timeout kills code that runs too long.
  3. Actual network calls (socket connections, urlopen, http requests) are
     disabled at the function level — not by blocking the imports, since
     libraries like pandas and matplotlib import socket/urllib internally
     for harmless reasons and blocking those imports breaks them. A few
     modules with no legitimate use in data analysis (ftplib, smtplib,
     pip) are still blocked at import time.
  4. Any files the code saves (e.g. matplotlib charts) are forced into this
     run's own artifacts/ folder — never anywhere else on your machine.

For your final thesis system, swap this file for a call to a real sandbox
service (e2b.dev has a free tier). Nothing else in the pipeline needs to
change — every other file just calls run_sandboxed_code().
"""
import subprocess
import sys
import re
import os
import uuid

BLOCKED_IMPORTS = ["ftplib", "smtplib", "pip"]

SANDBOX_PREAMBLE = """
import builtins
_real_import = builtins.__import__
_blocked_imports = {blocked!r}

def _guarded_import(name, *args, **kwargs):
    if any(name == b or name.startswith(b + '.') for b in _blocked_imports):
        raise ImportError(f"Import of '{{name}}' is blocked inside the sandbox.")
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
""".format(blocked=BLOCKED_IMPORTS)


def extract_code_blocks(text):
    """Pull ```python ... ``` blocks out of an LLM reply."""
    return re.findall(r"```python(.*?)```", text, re.DOTALL)


def run_sandboxed_code(code, run_dir, dataset_path, timeout_seconds=30, extra_vars=None):
    """
    Executes `code` in an isolated subprocess with restricted imports.
    The dataset path and an artifacts folder are pre-loaded as variables
    DATASET_PATH and ARTIFACTS_DIR for the generated code to use.

    extra_vars: optional dict of {VAR_NAME: value} to also inject as
    top-level variables in the sandboxed script -- e.g. for insight types
    that need more than one input file, such as spatial correlation needing
    both indicator values AND a borough adjacency list:
        extra_vars={"ADJACENCY_PATH": "/path/to/borough_adjacency.csv"}
    String values are treated as file paths and made absolute; other types
    (numbers, lists, dicts) are inserted as literal Python via repr().

    Returns captured stdout/stderr as a string.
    """
    artifacts_dir = os.path.join(os.path.abspath(run_dir), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    extra_lines = ""
    for var_name, value in (extra_vars or {}).items():
        if isinstance(value, str):
            extra_lines += f"{var_name} = {os.path.abspath(value)!r}\n"
        else:
            extra_lines += f"{var_name} = {value!r}\n"

    full_script = (
        SANDBOX_PREAMBLE + "\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        f"DATASET_PATH = {os.path.abspath(dataset_path)!r}\n"
        f"ARTIFACTS_DIR = {artifacts_dir!r}\n"
        + extra_lines +
        "\n"
        + code
    )

    tmp_file = os.path.join(run_dir, f"_exec_{uuid.uuid4().hex[:8]}.py")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(full_script)

    # Force UTF-8 everywhere for this subprocess. Without this, on Windows,
    # both writing the script file AND capturing its output default to the
    # system's local codepage (often cp1252), which crashes on ordinary
    # Unicode characters like the degree symbol (°) that a model might
    # reasonably use in a plot label or printed statistic.
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, tmp_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=child_env,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[STDERR]\n{result.stderr[-1500:]}"
        return output.strip() or "(code ran with no output — use print() to see results)"
    except subprocess.TimeoutExpired:
        return f"ERROR: code timed out after {timeout_seconds} seconds."
    finally:
        try:
            os.remove(tmp_file)
        except OSError:
            pass
