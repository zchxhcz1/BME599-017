import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def plot_maxOD_vs_glucose(df, normalized=None):
    plt.figure(figsize=(8,5))
    if normalized is None:
        plt.scatter(df['FinalGlucoseConc'], df['MaxOD'], alpha=0.5, label='Raw')
        plt.ylabel('MaxOD')
    else:
        plt.scatter(df['FinalGlucoseConc'], normalized, alpha=0.5, label='Normalized')
        plt.ylabel('Normalized MaxOD')
    plt.xlabel('Final Glucose Concentration (mM)')
    plt.title('MaxOD vs Glucose Concentration')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_gp_model(df, mask):
    # mask: boolean array for selecting wells (blank or max glucose)
    X = df.loc[mask, ['Row', 'Col']].values
    y = df.loc[mask, 'MaxOD'].values
    # Kernel: RBF + WhiteKernel (noise), allow tuning
    kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X, y)
    return gp

def predict_on_plate(gp, rows, cols):
    # Predict for all (row, col) pairs
    grid = np.array([[r, c] for r in rows for c in cols])
    mu, std = gp.predict(grid, return_std=True)
    mu_grid = mu.reshape(len(rows), len(cols))
    std_grid = std.reshape(len(rows), len(cols))
    return mu_grid, std_grid

def plot_plate(mu_grid, title, vmin=None, vmax=None):
    plt.figure(figsize=(10,6))
    plt.imshow(mu_grid, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Predicted MaxOD')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def normalize_data(df, B_grid, G_grid):
    # Map each well to its predicted B and G
    norm = []
    for _, row in df.iterrows():
        r, c = int(row['Row']), int(row['Col'])
        b = B_grid[r, c]
        g = G_grid[r, c]
        norm.append((row['MaxOD'] - b) / g if g != 0 else np.nan)
    return np.array(norm)
