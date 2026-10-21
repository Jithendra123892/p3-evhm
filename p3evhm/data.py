# p3evhm.data: loaders for NASA PCoE .mat files, telemetry CSVs, and a synthetic cell generator
# so the framework runs with zero downloads.
import os
import numpy as np
import pandas as pd


def load_telemetry_csv(path):
    """Load a CSV with columns [cell, cycle, capacity, temp]. Returns DataFrame sorted per cell."""
    df = pd.read_csv(path)
    return df.sort_values(["cell", "cycle"]).reset_index(drop=True)


def load_nasa_mat(data_dir, cells=("B0005", "B0006", "B0007", "B0018")):
    """Parse NASA PCoE Battery Data Set .mat files into a telemetry DataFrame
    with columns [cell, cycle, capacity, temp] (mean discharge temperature).

    Raises FileNotFoundError with a helpful message if files are missing.
    """
    from scipy.io import loadmat
    rows = []
    for cell in cells:
        path = None
        for root, _dirs, files in os.walk(data_dir):
            if cell + ".mat" in files:
                path = os.path.join(root, cell + ".mat")
                break
        if path is None:
            raise FileNotFoundError(
                f"{cell}.mat not found under {data_dir}. Download the NASA PCoE Battery Data Set "
                f"(Saha & Goebel, 2007) or use p3evhm.data.synthetic_cell for a zero-download start."
            )
        m = loadmat(path)
        cyc = m[cell][0, 0]["cycle"][0]
        n = 0
        for c in cyc:
            if str(c["type"][0]) == "discharge":
                d = c["data"][0, 0]
                names = d.dtype.names
                if names and "Capacity" in names:
                    T = float(np.mean(d["Temperature_measured"])) if ("Temperature_measured" in names) else 25.0
                    rows.append((cell, n, float(d["Capacity"][0][0]), T))
                    n += 1
    return pd.DataFrame(rows, columns=["cell", "cycle", "capacity", "temp"])


def synthetic_cell(n_cycles=168, rated=2.0, seed=0, knee=0.72, noise=0.004):
    """Generate a physically plausible ageing trajectory (two-phase fade + knee + noise).

    Returns a DataFrame with columns [cell, cycle, capacity, temp] for cell 'SYN01'.
    Good enough to exercise the whole pipeline without any dataset download.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_cycles)
    base = 0.30 * t + 0.85 * np.exp(6.0 * (t - knee)) / (1.0 + np.exp(6.0 * (t - knee))) * 0.55
    cap = rated * 0.97 - base - rng.normal(0.0, noise, n_cycles).cumsum() * 0.12
    cap = np.clip(cap, 0.55 * rated, rated)
    temp = 24.0 + 6.0 * t + rng.normal(0.0, 0.8, n_cycles)
    return pd.DataFrame({"cell": "SYN01", "cycle": np.arange(n_cycles),
                         "capacity": cap, "temp": temp})
