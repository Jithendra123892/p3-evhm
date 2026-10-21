# P3-EVHM FULL VALIDATION PROTOCOL (local CPU) — breadth + repeatability + generalization + scaling.
# 4 cells x 5 start points x B1-B6; full Eq.(3) loss (data+mono+temperature prior); 3 seeds on primary config;
# cross-cell: train on B0005+B0006 pooled -> test B0007/B0018; CPU N-scaling timing.
# Incremental JSON saves; per-config try/except.
import os, time, json, traceback
import numpy as np, pandas as pd
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)
BASE = r"C:\Users\jithe\p3-evhm-paper"
RATED = 2.0
RES = {"meta": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                "threads": 4, "protocol": "4 cells x 5 starts x B1-B6, full Eq3 loss, seeds, cross-cell, N-scaling"}}

# ---------- telemetry (capacity + mean discharge temperature) ----------
from scipy.io import loadmat
rows = []
for cell in ["B0005", "B0006", "B0007", "B0018"]:
    m = loadmat(os.path.join(BASE, "data", cell + ".mat"))
    cyc = m[cell][0, 0]["cycle"][0]
    n = 0
    for c in cyc:
        if str(c["type"][0]) == "discharge":
            d = c["data"][0, 0]
            if d.dtype.names and "Capacity" in d.dtype.names:
                T = float(np.mean(d["Temperature_measured"])) if "Temperature_measured" in d.dtype.names else 25.0
                rows.append((cell, n, float(d["Capacity"][0][0]), T)); n += 1
df = pd.DataFrame(rows, columns=["cell", "cycle", "capacity", "temp"])
df.to_csv(os.path.join(BASE, "data", "telemetry.csv"), index=False)
print("TELEMETRY_OK", len(df), flush=True)

class LastState(nn.Module):
    def forward(self, o): out, _ = o; return out[:, -1, :]

def train_pi_lstm(train_q, train_T, W=8, hidden=32, epochs=200, lam1=2.0, lam2=1.0, lr=5e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    X = np.lib.stride_tricks.sliding_window_view(train_q, W + 1)
    Xt = torch.tensor(X[:, :W], dtype=torch.float32).unsqueeze(-1)
    Yt = torch.tensor(X[:, -1], dtype=torch.float32)
    last = torch.tensor(X[:, W - 1], dtype=torch.float32)
    Tt = torch.tensor(train_T[W - 1:-1], dtype=torch.float32)
    dT = -(train_q[1:] - train_q[:-1])
    A = float(np.polyfit((train_T[1:] - np.mean(train_T))[1:], dT[1:], 1)[0]) if len(dT) > 10 else 0.0
    g = torch.clamp(A * (Tt - Tt.mean()), min=0.0)
    net = nn.Sequential(nn.LSTM(1, hidden, batch_first=True), LastState(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr); mse = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad(); out = net(Xt).squeeze(-1)
        loss = mse(out, Yt) + lam1 * torch.relu(out - last).mean() + lam2 * ((out[:-1] - out[1:]) - g[1:]).pow(2).mean()
        loss.backward(); opt.step()
    return net

def forecast(net, hist, steps, W=8):
    hist = list(hist); preds = []
    with torch.no_grad():
        for _ in range(steps):
            xw = torch.tensor(np.array(hist[-W:]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = float(net(xw).squeeze()); preds.append(p); hist.append(p)
    return np.array(preds)

def pf_rul(q0, slope_mu, slope_sd, eol, steps, N=2000, proc=0.003, seed=0, obs0=None):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    qi = np.clip(rng.normal(q0, 0.01, N), eol, None)
    di = np.clip(np.abs(rng.normal(slope_mu, slope_sd, N)), 1e-5, None)
    if obs0 is not None:  # observation update at the start cycle, so crossing step k -> cycle start+k
        wts = np.exp(-((obs0 - qi) ** 2) / (2 * 0.01 ** 2))
        if wts.sum() > 0:
            qi = qi[rng.choice(N, N, p=wts / wts.sum())]
    cross = np.full(N, -1)
    qf = qi.copy()
    for k in range(1, steps + 1):
        qf = qf - di + rng.normal(0, proc, N)
        newly = (cross < 0) & (qf <= eol)
        cross[newly] = k
    valid = cross[cross > 0]; rul = valid[valid <= steps]
    dt = time.time() - t0
    if len(rul) == 0: return None, None, None, dt
    lo, med, hi = np.percentile(rul, [2.5, 50, 97.5])
    return int(med), float(lo), float(hi), dt

def run_variants(train_q, train_T, test_q, eol, seed=0):
    s = len(train_q); true_len = len(test_q)
    true_cross = np.where(test_q <= eol)[0]
    out = {}
    w = 20; sl, ic = np.polyfit(np.arange(s - w, s), train_q[-w:], 1); sl = min(sl, -1e-4)
    pred = ic + sl * np.arange(s, s + true_len); cr = np.where(pred <= eol)[0]
    out["B1_linear"] = dict(mae=float(np.abs(pred - test_q).mean()), rmse=float(np.sqrt(((pred - test_q) ** 2).mean())),
                            rul_err=int(abs(cr[0] - true_cross[0])) if len(cr) and len(true_cross) else None,
                            V=float((np.diff(pred) > 0).mean()))
    net2 = train_pi_lstm(train_q, train_T, lam1=0.0, lam2=0.0, seed=seed)
    net3 = train_pi_lstm(train_q, train_T, lam1=2.0, lam2=1.0, seed=seed)
    p2 = forecast(net2, list(train_q[-8:]), true_len); p3 = forecast(net3, list(train_q[-8:]), true_len)
    for nm, p in [("B2_lstm", p2), ("B3_pi_lstm", p3)]:
        cr = np.where(p <= eol)[0]
        out[nm] = dict(mae=float(np.abs(p - test_q).mean()), rmse=float(np.sqrt(((p - test_q) ** 2).mean())),
                       rul_err=int(abs(cr[0] - true_cross[0])) if len(cr) and len(true_cross) else None,
                       V=float((np.diff(p) > 0).mean()))
    sl5 = abs((train_q[-40:][0] - train_q[-1]) / 40)
    sl4 = abs((p2[-1] - p2[0]) / max(len(p2) - 1, 1)); sl6 = abs((p3[-1] - p3[0]) / max(len(p3) - 1, 1))
    for nm, mu, sd in [("B4_lstm_pf", sl4, sl4 * 0.8 + 1e-4), ("B5_phys_pf", sl5, sl5 * 0.8 + 1e-4), ("B6_p3evhm_cpu", sl6, sl6 * 0.8 + 1e-4)]:
        med, lo, hi, _ = pf_rul(train_q[-1], mu, sd, eol, true_len, seed=seed, obs0=test_q[0])
        t0 = true_cross[0] if len(true_cross) else None
        out[nm] = dict(rul_err=int(abs(med - t0)) if (med is not None and t0 is not None) else None,
                       interval=[lo, hi], cover=(float(lo <= t0 <= hi) if (med is not None and lo is not None and t0 is not None) else None))
    return out, p2, p3

# ---------- main grid ----------
CELLS = [("B0005", 70), ("B0006", 70), ("B0007", 80), ("B0018", 70)]
FRACS = [0.4, 0.5, 0.6, 0.7, 0.8]
for cell, theta_pct in CELLS:
    try:
        d = df[df.cell == cell].sort_values("cycle")
        q = d.capacity.values; T = d.temp.fillna(25).values
        eol = theta_pct / 100.0 * RATED
        below = np.where(q <= eol)[0]
        assert len(below), f"{cell} never crosses {theta_pct}%"
        true_eol = int(below[0] + 1)
        cellrec = {"theta_pct": theta_pct, "true_eol_cycle": true_eol, "n_cycles": len(q), "starts": {}}
        for frac in FRACS:
            try:
                start = max(int(round(true_eol * frac)), 15); s = start - 1
                seed = 0
                if cell == "B0005" and frac == 0.8:
                    seed_runs = []
                    for sd_ in [0, 1, 2]:
                        v, _, _ = run_variants(q[:s], T[:s], q[s:], eol, seed=sd_)
                        seed_runs.append(v)
                    agg = {}
                    for var in seed_runs[0]:
                        agg[var] = {}
                        for k in seed_runs[0][var]:
                            vals = [r[var][k] for r in seed_runs if isinstance(r[var][k], (int, float))]
                            if vals and all(isinstance(v2, (int, float)) for v2 in vals):
                                agg[var][k] = round(float(np.mean(vals)), 4)
                            else:
                                agg[var][k] = seed_runs[0][var][k]
                    rec = {"start": start, "true_rul": true_eol - start, "seeds": "0,1,2 (mean)", **agg}
                else:
                    v, p2, p3 = run_variants(q[:s], T[:s], q[s:], eol, seed=0)
                    rec = {"start": start, "true_rul": true_eol - start, **v}
                cellrec["starts"][str(frac)] = rec
                b3 = rec["B3_pi_lstm"]; b2 = rec["B2_lstm"]; b6 = rec["B6_p3evhm_cpu"]; b5 = rec["B5_phys_pf"]
                print(f"{cell} {theta_pct}% start {rec['start']}: B2 MAE {b2['mae']:.4f} V {b2['V']:.3f} | B3 MAE {b3['mae']:.4f} V {b3['V']:.3f} | B5 err {b5['rul_err']} cov {b5['cover']} | B6 err {b6['rul_err']} cov {b6['cover']}", flush=True)
            except Exception:
                cellrec["starts"][str(frac)] = {"error": traceback.format_exc()[-500:]}
                print(f"{cell} frac {frac}: ERROR (recorded)", flush=True)
        RES[cell] = cellrec
        json.dump(RES, open(os.path.join(BASE, "results", "full_results.json"), "w"), indent=2, default=str)
    except Exception:
        RES[cell] = {"error": traceback.format_exc()[-500:]}
        print(cell, "CELL ERROR", flush=True)

# ---------- cross-cell generalization: train B0005+B0006 pooled -> test B0007/B0018 ----------
try:
    RES["cross_cell"] = {}
    for tcell, tfrac in [("B0007", 0.6), ("B0018", 0.6)]:
        td = df[df.cell == tcell].sort_values("cycle"); tq = td.capacity.values
        teol_rec = {"B0007": 80, "B0018": 70}[tcell]
        teol = teol_rec / 100.0 * RATED
        tbelow = np.where(tq <= teol)[0]; teolc = int(tbelow[0] + 1)
        tstart = max(int(round(teolc * tfrac)), 15)
        pooled_q = np.concatenate([df[df.cell == c].sort_values("cycle").capacity.values for c in ["B0005", "B0006"]])
        pooled_T = np.concatenate([df[df.cell == c].sort_values("cycle").temp.fillna(25).values for c in ["B0005", "B0006"]])
        v, _, _ = run_variants(pooled_q, pooled_T, tq[tstart - 1:], teol, seed=0)
        RES["cross_cell"][tcell] = {"train_on": ["B0005", "B0006"], "theta_pct": teol_rec,
                                    "true_eol_cycle": teolc, "start": tstart, **v}
        print(f"cross-cell -> {tcell}: B2 V {v['B2_lstm']['V']:.3f} | B3 MAE {v['B3_pi_lstm']['mae']:.4f} V {v['B3_pi_lstm']['V']:.3f} | B6 err {v['B6_p3evhm_cpu']['rul_err']}", flush=True)
    json.dump(RES, open(os.path.join(BASE, "results", "full_results.json"), "w"), indent=2, default=str)
except Exception:
    RES["cross_cell"] = {"error": traceback.format_exc()[-500:]}

# ---------- CPU N-scaling ----------
try:
    d = df[df.cell == "B0005"].sort_values("cycle"); q = d.capacity.values
    eol = 0.7 * RATED; s = 99
    sl5 = abs((q[:s][-40:][0] - q[:s][-1]) / 40)
    scaling = {}
    for N in [1000, 5000, 20000, 100000, 200000]:
        t0 = time.time(); pf_rul(q[:s][-1], sl5, sl5 * 0.8 + 1e-4, eol, 68, N=N); tc = time.time() - t0
        scaling[str(N)] = {"cpu_s": round(tc, 4), "cpu_ms_per_cycle": round(tc / 68 * 1000, 3)}
        print("N", N, scaling[str(N)], flush=True)
    RES["scaling_cpu"] = scaling
    json.dump(RES, open(os.path.join(BASE, "results", "full_results.json"), "w"), indent=2, default=str)
except Exception:
    RES["scaling_cpu"] = {"error": traceback.format_exc()[-500:]}

json.dump(RES, open(os.path.join(BASE, "results", "full_results.json"), "w"), indent=2, default=str)
print("FULL_PROTOCOL_DONE", flush=True)
