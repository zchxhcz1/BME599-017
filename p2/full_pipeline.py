"""
End-to-end pipeline for 20D medium optimization (Batches 1–4 + final prediction).

Each batch has a clearly documented entry point:
- propose_batch1(): generate 40-point maximin LHD (runs 1–40).
- propose_batch2(): use Batch 1 results, select active factors via ARD GP, propose 20 EI points (runs 41–60).
- propose_batch3(): use Batch 1–2 results, fit GP in active space, hybrid global/local/replicate (runs 61–80).
- propose_batch4(): use Batch 1–3 results, local EI + exploit (remaining runs up to 96) (runs 81–96 by default).
- final_prediction(): use all results (Runs 1–4) to output a single predicted optimum (x01–x20).

I/O conventions:
- Template: FinalProjectRunTemplate.csv
- Results: results Run 1.csv, results Run 2.csv, results Run 3.csv, results Run 4.csv
- Outputs (written in template format with group name GPR_Forever):
  batch1_runs.csv, batch2_runs.csv, batch3_design.csv, Batch4_runs.csv, final_prediction.csv

Dependencies: numpy, scipy, scikit-learn (no pandas required).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ---------------------------------------------------------------------------
# Common configuration
# ---------------------------------------------------------------------------

TEMPLATE_PATH = "FinalProjectRunTemplate.csv"
GROUP_NAME = "GPR_Forever"
RESPONSE_COL = "response"
STD_EPS = 1e-6
RANDOM_SEED = 123

# Results file candidates (first existing path per run is used)
RUN_FILE_CANDIDATES = {
    1: ["results Run 1.csv", "results Run 1.csv", "results.csv", "results.csv"],
    2: ["results Run 2.csv", "results Run 2.csv"],
    3: ["results Run 3.csv", "results Run 3.csv"],
    4: ["results Run 4.csv", "results Run 4.csv"],
}


# ---------------------------------------------------------------------------
# CSV/template utilities
# ---------------------------------------------------------------------------

def resolve_run_files(run_ids: List[int]) -> List[str]:
    """Pick the first existing file for each requested run id."""
    resolved = []
    for rid in run_ids:
        found = None
        for path in RUN_FILE_CANDIDATES[rid]:
            if os.path.exists(path):
                found = path
                break
        if found is None:
            raise FileNotFoundError(f"None of these files were found for run {rid}: {RUN_FILE_CANDIDATES[rid]}")
        resolved.append(found)
    return resolved


def read_csv_matrix(path: str) -> Tuple[List[str], np.ndarray]:
    """Read CSV using built-in csv to avoid extra dependencies."""
    import csv

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty file: {path}")
    header = rows[0]
    data = []
    for r in rows[1:]:
        if len(r) != len(header):
            raise ValueError(f"Row length mismatch in {path}")
        data.append([float(x) for x in r])
    return header, np.array(data, dtype=float)


def load_all_results(files: List[str]) -> Tuple[List[str], np.ndarray]:
    """Return shared header and concatenated data matrix."""
    headers, datas = [], []
    for f in files:
        h, d = read_csv_matrix(f)
        headers.append(h)
        datas.append(d)
    base = headers[0]
    for h in headers[1:]:
        if h != base:
            raise ValueError("Headers differ across result files.")
    return base, np.vstack(datas)


def get_factor_columns(header: List[str]) -> List[str]:
    cols = [f"x{i:02d}" for i in range(1, 21)]
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")
    return cols


def load_template_rows(path: str) -> List[List[str]]:
    import csv

    with open(path, newline="") as f:
        return list(csv.reader(f))


def write_with_template(
    X_full: np.ndarray,
    template_rows: List[List[str]],
    factor_cols: List[str],
    output_path: str,
    start_run: int,
    group_name: str | None = None,
) -> None:
    """Fill a block of runs into the template, preserving formatting."""
    import csv

    rows = [r.copy() for r in template_rows]
    header = rows[1]
    factor_idx = {name: idx for idx, name in enumerate(header) if isinstance(name, str) and name.startswith("x")}
    if group_name:
        rows[0][0] = group_name
    start_idx = 1 + start_run
    if start_idx + X_full.shape[0] > len(rows):
        raise ValueError("Template does not have enough rows for these runs.")
    for i in range(X_full.shape[0]):
        row_idx = start_idx + i
        rows[row_idx][0] = str(start_run + i)
        for j, col in enumerate(factor_cols):
            rows[row_idx][factor_idx[col]] = f"{X_full[i, j]:.12f}"
    with open(output_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------------
# Design utilities (LHD, GP, EI, active selection)
# ---------------------------------------------------------------------------

def _latin_hypercube(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=(n_samples, dim))
    a, b = cut[:n_samples], cut[1:n_samples + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    samples = np.zeros_like(rdpoints)
    for j in range(dim):
        order = rng.permutation(n_samples)
        samples[:, j] = rdpoints[order, j]
    return samples


def _min_pairwise_distance(X: np.ndarray) -> float:
    n = X.shape[0]
    if n < 2:
        return np.inf
    min_dist = np.inf
    for i in range(n - 1):
        dists = np.linalg.norm(X[i + 1 :] - X[i], axis=1)
        min_dist = min(min_dist, float(dists.min()))
    return min_dist


def maximin_lhd(n_samples: int, dim: int, n_restarts: int, rng: np.random.Generator) -> np.ndarray:
    best, best_score = None, -np.inf
    for _ in range(max(1, n_restarts)):
        cand = _latin_hypercube(n_samples, dim, rng)
        score = _min_pairwise_distance(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best


def standardize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean, y_std = y.mean(), y.std()
    if y_std == 0:
        y_std = 1.0
    return (y - y_mean) / y_std, y_mean, y_std


def fit_gp(X: np.ndarray, y_scaled: np.ndarray) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-3
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=5,
        random_state=RANDOM_SEED,
    )
    gp.fit(X, y_scaled)
    return gp


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best) / sigma
    ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-9] = 0.0
    return ei


def select_diverse(points: np.ndarray, scores: np.ndarray, n_select: int, min_dist: float) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        if len(selected) >= n_select:
            break
        p = points[idx]
        if not selected:
            selected.append(p)
            continue
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.all(dists >= min_dist):
            selected.append(p)
    if len(selected) < n_select:  # pad if needed
        for idx in order:
            if len(selected) >= n_select:
                break
            if any(np.allclose(points[idx], s) for s in selected):
                continue
            selected.append(points[idx])
    return np.array(selected[:n_select])


def select_active_via_ard(X: np.ndarray, y: np.ndarray, factor_cols: List[str], k: int) -> List[str]:
    """
    Active set via ARD GP length scales + main-effect screening (from Batch 2 logic).
    """
    gp_full = fit_gp(X, (y - y.mean()) / (y.std() if y.std() > 0 else 1.0))
    rbf_kernel = gp_full.kernel_.k1 if hasattr(gp_full.kernel_, "k1") else gp_full.kernel_
    length_scales = np.asarray(rbf_kernel.length_scale, dtype=float)
    importance_ls = 1.0 / length_scales

    # main-effect amplitude at 0.25 vs 0.75 (others fixed at 0.5)
    effects = []
    base = np.full(X.shape[1], 0.5)
    for j in range(X.shape[1]):
        x_low = base.copy()
        x_high = base.copy()
        x_low[j] = 0.25
        x_high[j] = 0.75
        mu_low = float(gp_full.predict(x_low.reshape(1, -1)))
        mu_high = float(gp_full.predict(x_high.reshape(1, -1)))
        effects.append(abs(mu_high - mu_low))
    effects = np.array(effects)

    def _norm(arr: np.ndarray) -> np.ndarray:
        a_min, a_max = arr.min(), arr.max()
        return np.zeros_like(arr) if np.isclose(a_max, a_min) else (arr - a_min) / (a_max - a_min)

    score = (_norm(importance_ls) + _norm(effects)) / 2.0
    order = np.argsort(score)[::-1]
    return [factor_cols[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# Batch-specific pipelines
# ---------------------------------------------------------------------------

def propose_batch1(n_runs: int = 40, outfile: str = "batch1_runs.csv") -> None:
    """
    Batch 1 (global exploration):
    - 40-point maximin Latin Hypercube in full 20D.
    - Written to batch1_runs.csv in template format (runs 1–40).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X = maximin_lhd(n_runs, dim=20, n_restarts=50, rng=rng)
    template_rows = load_template_rows(TEMPLATE_PATH)
    factor_cols = [f"x{i:02d}" for i in range(1, 21)]
    write_with_template(X, template_rows, factor_cols, outfile, start_run=1, group_name=GROUP_NAME)
    print(f"Batch 1 design saved to {outfile} (runs 1–{n_runs}).")


def propose_batch2(outfile: str = "batch2_runs.csv") -> None:
    """
    Batch 2 (model-based global EI with ARD active set):
    - Load Batch 1 results.
    - Select top-6 active factors via ARD GP.
    - Propose 20 EI points with diversity in active space; inactive fixed at 0.5.
    - Written to batch2_runs.csv in template format (runs 41–60).
    """
    files = resolve_run_files([1])  # Batch 1 results only
    header, data = load_all_results(files)
    factor_cols = get_factor_columns(header)
    y = data[:, header.index(RESPONSE_COL)]
    X = data[:, [header.index(c) for c in factor_cols]]

    active_vars = select_active_via_ard(X, y, factor_cols, k=6)
    active_idx = [factor_cols.index(c) for c in active_vars]
    X_active = X[:, active_idx]

    y_scaled, y_mean, y_std = standardize_y(y)
    gp = fit_gp(X_active, y_scaled)
    y_best = float(y.max())

    rng = np.random.default_rng(RANDOM_SEED)
    candidates = rng.uniform(0, 1, size=(100_000, len(active_idx)))
    mu_s, sigma = gp.predict(candidates, return_std=True)
    mu = mu_s * y_std + y_mean
    ei = expected_improvement(mu, sigma, y_best)
    selected = select_diverse(candidates, ei, n_select=20, min_dist=0.1)

    # Map to full space with inactive=0.5
    X_full = np.full((selected.shape[0], 20), 0.5)
    X_full[:, active_idx] = selected

    template_rows = load_template_rows(TEMPLATE_PATH)
    write_with_template(X_full, template_rows, factor_cols, outfile, start_run=41, group_name=GROUP_NAME)
    print(f"Batch 2 design saved to {outfile} (runs 41–60). Active factors: {active_vars}")


def propose_batch3(outfile: str = "batch3_design.csv") -> None:
    """
    Batch 3 (hybrid global/local around best):
    - Load Batch 1–2 results.
    - Active set: same ARD method (top-6).
    - Inactive baselines: mean of top 25% yields.
    - 8 global EI (diverse) + 8 local perturbations + 4 replicate/near-replicate = 20 runs.
    - Written starting at run 61.
    """
    files = resolve_run_files([1, 2])
    header, data = load_all_results(files)
    factor_cols = get_factor_columns(header)
    y = data[:, header.index(RESPONSE_COL)]
    X = data[:, [header.index(c) for c in factor_cols]]

    active_vars = select_active_via_ard(X, y, factor_cols, k=6)
    active_idx = [factor_cols.index(c) for c in active_vars]
    X_active = X[:, active_idx]

    # Baseline for inactive = mean over top 25% yields
    threshold = np.quantile(y, 0.75)
    high_mask = y >= threshold
    inactive_baseline = {factor_cols[i]: float(X[high_mask, i].mean()) for i in range(20) if i not in active_idx}

    y_scaled, y_mean, y_std = standardize_y(y)
    gp = fit_gp(X_active, y_scaled)
    y_best = float(y.max())
    best_idx = int(np.argmax(y))
    x_best_active = X_active[best_idx]

    rng = np.random.default_rng(RANDOM_SEED)
    # Global EI (8 points)
    candidates = rng.uniform(0, 1, size=(80_000, len(active_idx)))
    mu_s, sigma = gp.predict(candidates, return_std=True)
    mu = mu_s * y_std + y_mean
    ei = expected_improvement(mu, sigma, y_best)
    global_pts = select_diverse(candidates, ei, n_select=8, min_dist=0.1)
    # Local (8 points)
    local_pts = np.clip(x_best_active + rng.uniform(-0.05, 0.05, size=(8, len(active_idx))), 0.0, 1.0)
    # Replicates/near (4 points)
    reps = []
    reps.append(x_best_active.copy())
    reps.append(x_best_active.copy())
    reps.append(np.clip(x_best_active + rng.uniform(-0.01, 0.01, size=len(active_idx)), 0, 1))
    reps.append(np.clip(x_best_active + rng.uniform(-0.01, 0.01, size=len(active_idx)), 0, 1))

    X_act_batch3 = np.vstack([global_pts, local_pts, np.vstack(reps)])  # 20 x |A|

    # Map to full
    X_full = np.zeros((X_act_batch3.shape[0], 20))
    for j, col in enumerate(factor_cols):
        if j in active_idx:
            X_full[:, j] = X_act_batch3[:, active_idx.index(j)]
        else:
            X_full[:, j] = inactive_baseline[col]

    template_rows = load_template_rows(TEMPLATE_PATH)
    write_with_template(X_full, template_rows, factor_cols, outfile, start_run=61, group_name=GROUP_NAME)
    print(f"Batch 3 design saved to {outfile} (runs 61–80). Active factors: {active_vars}")


def propose_batch4(outfile: str = "Batch4_runs.csv") -> None:
    """
    Batch 4 (final local EI + exploit):
    - Load Batch 1–3 results.
    - Active vars: top-4 by abs correlation (skip near-constant).
    - Trust region around best observed (±0.15 on active vars).
    - Split remaining budget into explore (EI-diverse) + exploit (replicates/jitters of top runs).
    - Written starting at run 81 using template format.
    """
    files = resolve_run_files([1, 2, 3])
    header, data = load_all_results(files)
    factor_cols = get_factor_columns(header)
    y = data[:, header.index(RESPONSE_COL)]
    X = data[:, [header.index(c) for c in factor_cols]]

    active_vars = select_active_variables = select_active_variables = None  # type: ignore

    # correlation-based active set (K=4)
    stds = np.std(X, axis=0)
    variable_mask = stds >= STD_EPS
    corrs = []
    for j, col in enumerate(factor_cols):
        if not variable_mask[j]:
            corrs.append((-np.inf, col))
            continue
        xj = X[:, j]
        cj = np.corrcoef(xj, y)[0, 1] if np.std(xj) > 0 and np.std(y) > 0 else 0.0
        corrs.append((abs(cj), col))
    active_vars = [c for _, c in sorted(corrs, key=lambda t: t[0], reverse=True)[:4]]
    active_idx = [factor_cols.index(c) for c in active_vars]

    X_active = X[:, active_idx]
    y_scaled, y_mean, y_std = standardize_y(y)
    gp = fit_gp(X_active, y_scaled)
    y_best = float(y.max())

    best_idx = int(np.argmax(y))
    x_best_full = X[best_idx]
    x_best_active = x_best_full[active_idx]
    lows = np.maximum(0.0, x_best_active - 0.15)
    highs = np.minimum(1.0, x_best_active + 0.15)

    n_candidates = max(5000, 500 * len(active_idx))
    rng = np.random.default_rng(RANDOM_SEED)
    candidates = rng.uniform(lows, highs, size=(n_candidates, len(active_idx)))
    mu_s, sigma = gp.predict(candidates, return_std=True)
    mu = mu_s * y_std + y_mean
    ei = expected_improvement(mu, sigma, y_best)

    # Budget
    n_so_far = X.shape[0]
    remaining = 96 - n_so_far
    top_m = min(3, n_so_far)
    while top_m > 0 and 2 * top_m > remaining:
        top_m -= 1
    n_exploit = 2 * top_m
    n_explore = max(remaining - n_exploit, 0)

    designs_full = []
    if n_explore > 0:
        sel = select_diverse(candidates, ei, n_explore, min_dist=0.05)
        for cand in sel:
            x_new = x_best_full.copy()
            x_new[active_idx] = cand
            designs_full.append(x_new)
    if n_exploit > 0:
        top_idx = np.argsort(y)[::-1][:top_m]
        for idx in top_idx:
            x_vec = X[idx]
            designs_full.append(x_vec.copy())
            jitter = rng.uniform(-0.03, 0.03, size=len(active_idx))
            x_jit = x_vec.copy()
            x_jit[active_idx] = np.clip(x_jit[active_idx] + jitter, 0, 1)
            designs_full.append(x_jit)
    while len(designs_full) < remaining:
        designs_full.append(x_best_full.copy())
    X_full = np.array(designs_full[:remaining])

    template_rows = load_template_rows(TEMPLATE_PATH)
    write_with_template(X_full, template_rows, factor_cols, outfile, start_run=81, group_name=GROUP_NAME)
    print(f"Batch 4 design saved to {outfile} (runs 81–96). Active factors: {active_vars}")


def final_prediction(outfile: str = "final_prediction.csv") -> None:
    """
    Final predicted optimum (no more experiments):
    - Load Runs 1–4.
    - Active set: top-4 by abs correlation (skip near-constant).
    - Fit GP on active vars (scaled y), maximize predictive mean over random candidates.
    - Combine optimal active values with inactive from best observed full run.
    - Write one-row template-style CSV.
    """
    files = resolve_run_files([1, 2, 3, 4])
    header, data = load_all_results(files)
    factor_cols = get_factor_columns(header)
    y = data[:, header.index(RESPONSE_COL)]
    X = data[:, [header.index(c) for c in factor_cols]]

    # correlation-based active set (K=4)
    stds = np.std(X, axis=0)
    variable_mask = stds >= STD_EPS
    corrs = []
    for j, col in enumerate(factor_cols):
        if not variable_mask[j]:
            corrs.append((-np.inf, col))
            continue
        xj = X[:, j]
        cj = np.corrcoef(xj, y)[0, 1] if np.std(xj) > 0 and np.std(y) > 0 else 0.0
        corrs.append((abs(cj), col))
    active_vars = [c for _, c in sorted(corrs, key=lambda t: t[0], reverse=True)[:4]]
    active_idx = [factor_cols.index(c) for c in active_vars]

    X_active = X[:, active_idx]
    y_scaled, y_mean, y_std = standardize_y(y)
    gp = fit_gp(X_active, y_scaled)

    rng = np.random.default_rng(RANDOM_SEED)
    n_candidates = max(50_000, 10_000 * len(active_idx))
    candidates = rng.uniform(0, 1, size=(n_candidates, len(active_idx)))
    mu_s, _sigma = gp.predict(candidates, return_std=True)
    mu = mu_s * y_std + y_mean
    best_idx_pred = int(np.argmax(mu))
    x_opt_active = candidates[best_idx_pred]
    mu_opt = float(mu[best_idx_pred])

    best_obs_idx = int(np.argmax(y))
    x_best_full = X[best_obs_idx]
    x_opt_full = x_best_full.copy()
    x_opt_full[active_idx] = x_opt_active

    template_rows = load_template_rows(TEMPLATE_PATH)
    write_with_template(
        x_opt_full.reshape(1, -1),
        template_rows,
        factor_cols,
        outfile,      # output_path
        start_run=1,
        group_name=GROUP_NAME,
    )

    print(f"Final prediction saved to {outfile}")
    print(f"Active vars: {active_vars}")
    print(f"Best observed response: {y.max():.6f}")
    print(f"Predicted max GP mean: {mu_opt:.6f}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uncomment the stages you need. Each stage writes its own CSV.
    # propose_batch1()
    # propose_batch2()
    # propose_batch3()
    # propose_batch4()
    # final_prediction()
    pass
