# p3evhm.evaluate: the B1-B6 benchmark harness and grid runner (validation protocol of the paper).
import time
import numpy as np
import pandas as pd

from . import models


def run_variants(train_q, train_T, test_q, eol, seed=0, N=2000, epochs=200, hidden=32, device="auto"):
    """Run baselines B1-B6 on one (cell, start) configuration.

    Convention: test_q[i] = capacity at cycle (start + i); a crossing at index i means RUL = i.
    Returns dict: per-variant metrics {mae, rmse, rul_err, cover, interval, V}.
    """
    s = len(train_q); true_len = len(test_q)
    true_cross = np.where(np.asarray(test_q) <= eol)[0]
    out = {}
    w = 20; sl, ic = np.polyfit(np.arange(s - w, s), np.asarray(train_q)[-w:], 1); sl = min(sl, -1e-4)
    pred = ic + sl * np.arange(s, s + true_len); cr = np.where(pred <= eol)[0]
    out["B1_linear"] = dict(mae=float(np.abs(pred - test_q).mean()),
                            rmse=float(np.sqrt(((pred - test_q) ** 2).mean())),
                            rul_err=int(abs(cr[0] - true_cross[0])) if len(cr) and len(true_cross) else None,
                            V=models.violation_rate(pred))
    p2, t2 = models.pi_lstm_forecast(train_q, train_T, steps=true_len, lam1=0.0, lam2=0.0,
                                     hidden=hidden, epochs=epochs, seed=seed, device=device, return_train_time=True)
    p3, t3 = models.pi_lstm_forecast(train_q, train_T, steps=true_len, lam1=2.0, lam2=1.0,
                                     hidden=hidden, epochs=epochs, seed=seed, device=device, return_train_time=True)
    for nm, p in [("B2_lstm", p2), ("B3_pi_lstm", p3)]:
        cr = np.where(p <= eol)[0]
        out[nm] = dict(mae=float(np.abs(p - test_q).mean()), rmse=float(np.sqrt(((p - test_q) ** 2).mean())),
                       rul_err=int(abs(cr[0] - true_cross[0])) if len(cr) and len(true_cross) else None,
                       V=models.violation_rate(p))
    sl5 = abs((train_q[-40:][0] - train_q[-1]) / 40)
    sl4 = abs((p2[-1] - p2[0]) / max(len(p2) - 1, 1)); sl6 = abs((p3[-1] - p3[0]) / max(len(p3) - 1, 1))
    for nm, mu, sd in [("B4_lstm_pf", sl4, sl4 * 0.8 + 1e-4), ("B5_phys_pf", sl5, sl5 * 0.8 + 1e-4),
                       ("B6_p3evhm", sl6, sl6 * 0.8 + 1e-4)]:
        med, lo, hi, _ = models.pf_rul(train_q[-1], mu, sd, eol, true_len, N=N, seed=seed, obs0=test_q[0])
        t0 = int(true_cross[0]) if len(true_cross) else None
        out[nm] = dict(rul_err=int(abs(med - t0)) if (med is not None and t0 is not None) else None,
                       interval=[lo, hi],
                       cover=(float(lo <= t0 <= hi) if (med is not None and lo is not None and t0 is not None) else None))
    out["_train_times"] = {"B2": t2, "B3": t3}
    return out


def eol_cycle(q, rated, theta_pct):
    """First 1-indexed cycle whose capacity <= theta% of rated; None if never crossed."""
    below = np.where(np.asarray(q) <= theta_pct / 100.0 * rated)[0]
    return int(below[0] + 1) if len(below) else None


def run_grid(df, cells, thetas, fracs=(0.4, 0.5, 0.6, 0.7, 0.8), rated=2.0, N=2000, epochs=200,
             hidden=32, device="auto", progress=None):
    """Full validation grid: for each (cell, start fraction) run B1-B6.

    df: telemetry DataFrame [cell, cycle, capacity, temp]; thetas: dict cell -> EOL %.
    Returns a tidy DataFrame with one row per (cell, frac, variant).
    """
    rows = []
    items = list(cells)
    for i, cell in enumerate(items):
        d = df[df.cell == cell].sort_values("cycle")
        q = d.capacity.values; T = d.temp.fillna(25.0).values
        eolc = eol_cycle(q, rated, thetas[cell])
        if eolc is None:
            continue
        eol = thetas[cell] / 100.0 * rated
        for frac in fracs:
            start = max(int(round(eolc * frac)), 15); s = start - 1
            res = run_variants(q[:s], T[:s], q[s:], eol, seed=0, N=N, epochs=epochs, hidden=hidden, device=device)
            for var, m in res.items():
                if var.startswith("_"):
                    continue
                rows.append(dict(cell=cell, theta_pct=thetas[cell], true_eol_cycle=eolc, start=start,
                                 true_rul=(eolc - start), variant=var,
                                 mae=m.get("mae"), rmse=m.get("rmse"), rul_err=m.get("rul_err"),
                                 cover=m.get("cover"), V=m.get("V")))
            if progress:
                progress(i + 1, len(items), cell, start)
    return pd.DataFrame(rows)


def summarize(grid_df):
    """Aggregate a grid DataFrame: per-variant mean MAE/RMSE/RUL err, coverage, V, availability."""
    g = grid_df
    out = []
    for var, sub in g.groupby("variant"):
        rul = sub.rul_err.dropna()
        cov = sub.cover.dropna()
        out.append(dict(variant=var, n=len(sub), n_with_rul=len(rul),
                        mae_mean=None if sub.mae.isna().all() else round(float(sub.mae.mean()), 4),
                        rmse_mean=None if sub.rmse.isna().all() else round(float(sub.rmse.mean()), 4),
                        rul_err_mean=None if len(rul) == 0 else round(float(rul.mean()), 1),
                        cover_mean=None if len(cov) == 0 else round(float(cov.mean()), 2),
                        V_mean=None if sub.V.isna().all() else round(float(sub.V.mean()), 3)))
    return pd.DataFrame(out)
