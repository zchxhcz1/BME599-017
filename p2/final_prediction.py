"""
Compute a final predicted optimum medium (x01–x20) using all completed runs.

Workflow:
- Load all four result CSVs (Runs 1–4).
- Select top-4 active variables by absolute correlation (skip near-constant cols).
- Fit a GP on active variables with scaled responses.
- Maximize GP predictive mean over random candidates in active space.
- Combine optimal active values with inactive values from the best observed run.
- Save one-row CSV final_prediction.csv with columns x01..x20.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_FILE_CANDIDATES: List[List[str]] = [
    ["results Run 1.csv", "p1/results Run 1.csv", "results.csv", "p1/results.csv"],
    ["results Run 2.csv", "p1/results Run 2.csv"],
    ["results Run 3.csv", "p1/results Run 3.csv"],
    ["results Run 4.csv", "p1/results Run 4.csv"],
]
RESPONSE_COL = "response"
K_ACTIVE = 4
STD_EPS = 1e-6
RANDOM_SEED = 123
TEMPLATE_PATH = "p1/FinalProjectRunTemplate.csv"
GROUP_NAME = "GPR_Forever"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_run_files(candidates: List[List[str]]) -> List[str]:
    """Pick the first existing file for each run from candidate lists."""
    resolved = []
    for opts in candidates:
        found = None
        for path in opts:
            if os.path.exists(path):
                found = path
                break
        if found is None:
            raise FileNotFoundError(f"None of these files were found for a run: {opts}")
        resolved.append(found)
    return resolved


def read_csv_matrix(path: str) -> Tuple[List[str], np.ndarray]:
    """Read CSV using built-in csv; returns (header, data matrix)."""
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
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
    headers = []
    datas = []
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


def select_active_variables(X: np.ndarray, y: np.ndarray, factor_cols: List[str], k: int) -> List[str]:
    """Top-k variables by absolute Pearson correlation, ignoring near-constant."""
    stds = np.std(X, axis=0)
    variable_mask = stds >= STD_EPS
    if not np.any(variable_mask):
        raise ValueError("All variables are near-constant; cannot select active variables.")
    corrs = []
    for j, col in enumerate(factor_cols):
        if not variable_mask[j]:
            corrs.append((-np.inf, col))
            continue
        xj = X[:, j]
        cj = np.corrcoef(xj, y)[0, 1] if np.std(xj) > 0 and np.std(y) > 0 else 0.0
        corrs.append((abs(cj), col))
    ranked = sorted(corrs, key=lambda t: t[0], reverse=True)
    return [c for _, c in ranked[: min(k, len(ranked))]]


def standardize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    return (y - y_mean) / y_std, y_mean, y_std


def fit_gp(X: np.ndarray, y_scaled: np.ndarray) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=5,
        random_state=RANDOM_SEED,
    )
    gp.fit(X, y_scaled)
    return gp


def load_template_rows(path: str) -> List[List[str]]:
    """Read template preserving structure."""
    import csv

    with open(path, newline="") as f:
        return list(csv.reader(f))


def write_with_template(
    x_full: np.ndarray,
    template_rows: List[List[str]],
    factor_cols: List[str],
    output_path: str,
    run_id: int = 1,
    group_name: str | None = None,
) -> None:
    """Write single-row design into template format."""
    import csv

    rows = [r.copy() for r in template_rows]
    header = rows[1]
    factor_idx = {name: idx for idx, name in enumerate(header) if isinstance(name, str) and name.startswith("x")}
    if group_name:
        rows[0][0] = group_name
    row_idx = 1 + run_id  # run 1 corresponds to index 2
    if row_idx >= len(rows):
        raise ValueError("Template too short for requested run_id.")
    rows[row_idx][0] = str(run_id)
    for j, col in enumerate(factor_cols):
        rows[row_idx][factor_idx[col]] = f"{x_full[j]:.12f}"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Load data
    run_files = resolve_run_files(RUN_FILE_CANDIDATES)
    header, data = load_all_results(run_files)
    factor_cols = get_factor_columns(header)
    y_idx = header.index(RESPONSE_COL)
    X_full = data[:, [header.index(c) for c in factor_cols]]
    y = data[:, y_idx]

    # Active variable selection
    active_vars = select_active_variables(X_full, y, factor_cols, K_ACTIVE)
    active_indices = [factor_cols.index(c) for c in active_vars]

    # GP fit on active subspace
    X_active = X_full[:, active_indices]
    y_scaled, y_mean, y_std = standardize_y(y)
    gp = fit_gp(X_active, y_scaled)

    # Candidate search for max predicted mean
    n_candidates = max(50000, 10000 * K_ACTIVE)
    candidates = rng.uniform(0.0, 1.0, size=(n_candidates, K_ACTIVE))
    mu_scaled, sigma = gp.predict(candidates, return_std=True)
    mu = mu_scaled * y_std + y_mean
    best_idx_pred = int(np.argmax(mu))
    x_opt_active = candidates[best_idx_pred]
    mu_opt = float(mu[best_idx_pred])

    # Best observed run
    best_obs_idx = int(np.argmax(y))
    x_best_full = X_full[best_obs_idx]

    # Build full optimal vector
    x_opt_full = x_best_full.copy()
    for i, idx in enumerate(active_indices):
        x_opt_full[idx] = x_opt_active[i]

    # Save one-row CSV in template format to match Batch4_runs.csv style
    out_path = os.path.join(os.getcwd(), "final_prediction.csv")
    template_rows = load_template_rows(TEMPLATE_PATH)
    write_with_template(
        x_opt_full,
        template_rows,
        factor_cols,
        output_path=out_path,
        run_id=1,
        group_name=GROUP_NAME,
    )

    # Summary
    print(f"Total runs loaded: {len(X_full)}")
    print(f"Active variables: {active_vars}")
    print(f"Best observed response: {y.max():.6f}")
    print(f"Predicted max GP mean: {mu_opt:.6f}")
    print("Best observed full vector (first 5 dims):", x_best_full[:5])
    print("Predicted optimal full vector (first 5 dims):", x_opt_full[:5])
    print(f"Saved final prediction to: {out_path}")


if __name__ == "__main__":
    main()
