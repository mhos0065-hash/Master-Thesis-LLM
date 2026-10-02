import numpy as np
from scipy.spatial import Delaunay

rng = np.random.default_rng(42)

N_BOROUGHS = 33          # matches London borough count
N_TRIALS = 3000

def build_adjacency(n, rng):
    """Delaunay triangulation of random points -> planar contiguity graph,
    structurally comparable to real borough adjacency (avg degree ~5-6,
    similar to London's actual borough network)."""
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    adj = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[simplex[i], simplex[j]] = True
    np.fill_diagonal(adj, False)
    return adj

def row_standardize(adj):
    W = adj.astype(float)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs

def lee_l(x, y, W):
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    n = len(x)
    num = n * np.sum((W @ zx) * (W @ zy))
    den = np.sum(W.sum(axis=1) ** 2) * np.sqrt(np.sum(zx**2) * np.sum(zy**2))
    return num / den

def cross_moran(x, y, W):
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    n = len(x)
    S0 = W.sum()
    num = n * np.sum(zx * (W @ zy))
    den = S0 * np.sqrt(np.sum(zx**2) * np.sum(zy**2))
    return num / den

def spatially_smooth(noise, W, strength):
    """Applies (I - strength*W)^-1 style smoothing to induce realistic
    spatial autocorrelation, strength in [0, 0.9)."""
    n = len(noise)
    I = np.eye(n)
    return np.linalg.solve(I - strength * W, noise)

lee_diffs_rel, lee_diffs_abs = [], []
cm_diffs_rel, cm_diffs_abs = [], []
lee_binary_vals, cm_binary_vals = [], []

for trial in range(N_TRIALS):
    adj = build_adjacency(N_BOROUGHS, rng)
    W_row = row_standardize(adj)
    W_bin = adj.astype(float)

    # vary true correlation strength and spatial autocorrelation strength
    true_corr = rng.uniform(-0.6, 0.6)
    spatial_strength = rng.uniform(0.0, 0.6)

    noise_x = rng.normal(size=N_BOROUGHS)
    noise_y = true_corr * noise_x + np.sqrt(max(1 - true_corr**2, 0.01)) * rng.normal(size=N_BOROUGHS)

    x = spatially_smooth(noise_x, W_row, spatial_strength)
    y = spatially_smooth(noise_y, W_row, spatial_strength)

    L_row = lee_l(x, y, W_row)
    L_bin = lee_l(x, y, W_bin)
    CM_row = cross_moran(x, y, W_row)
    CM_bin = cross_moran(x, y, W_bin)

    if np.isfinite(L_row) and np.isfinite(L_bin) and abs(L_row) > 1e-6:
        lee_diffs_abs.append(abs(L_row - L_bin))
        lee_diffs_rel.append(abs(L_row - L_bin) / abs(L_row))
        lee_binary_vals.append(L_bin)

    if np.isfinite(CM_row) and np.isfinite(CM_bin) and abs(CM_row) > 1e-6:
        cm_diffs_abs.append(abs(CM_row - CM_bin))
        cm_diffs_rel.append(abs(CM_row - CM_bin) / abs(CM_row))
        cm_binary_vals.append(CM_bin)

lee_diffs_rel = np.array(lee_diffs_rel)
lee_diffs_abs = np.array(lee_diffs_abs)
cm_diffs_rel = np.array(cm_diffs_rel)
cm_diffs_abs = np.array(cm_diffs_abs)

print(f"N valid trials: Lee's L = {len(lee_diffs_rel)}, Cross-Moran = {len(cm_diffs_rel)}")
print()
print("=== Lee's L: row-standardized vs binary weights ===")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: relative diff = {np.percentile(lee_diffs_rel, p)*100:6.2f}%   "
          f"absolute diff = {np.percentile(lee_diffs_abs, p):.4f}")
print()
print("=== Cross-Moran's I: row-standardized vs binary weights ===")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: relative diff = {np.percentile(cm_diffs_rel, p)*100:6.2f}%   "
          f"absolute diff = {np.percentile(cm_diffs_abs, p):.4f}")
