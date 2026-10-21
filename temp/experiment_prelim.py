# P3-EVHM PRELIMINARY PROTOTYPE EXPERIMENT (CPU, real NASA PCoE data).
# Variants: B1 linear trend | B2 LSTM (data loss) | B3 PI-LSTM (+monotonicity loss)
#           B4 LSTM+PF | B5 physics+PF | B6 P3-EVHM (PI-LSTM+PF, CPU path).
# Metrics: capacity MAE/RMSE, RUL abs error, 95% interval coverage/width, violation rate V, runtime.
# Clearly PRELIMINARY: single-cell demos, CPU-only, prototype implementation.
import os, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
BASE = r"C:\Users\jithe\p3-evhm-paper"
RATED = 2.0  # Ah rated capacity of NASA PCoE 18650 cells

df = pd.read_csv(os.path.join(BASE, "data", "capacity.csv"))

def run_cell(cell, theta_pct, start_cycle, W=8, N=2000, lam1=2.0, epochs=300):
    c = df[df.cell == cell].sort_values("cycle")
    q = c.capacity.values  # 0-indexed cycle i -> q[i], cycle number = i+1
    ref = q[0]
    eol = theta_pct / 100.0 * RATED
    below = np.where(q <= eol)[0]
    true_eol_cycle = int(below[0] + 1) if len(below) else None  # 1-indexed
    true_rul = true_eol_cycle - start_cycle if true_eol_cycle else None
    s = start_cycle - 1  # 0-indexed start
    train_q, test_q = q[:s], q[s:]
    res = {}

    # ---------- B1: recent-trend linear ----------
    t0 = time.time()
    w = 20
    x = np.arange(s - w, s); y = train_q[-w:]
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0: slope = -1e-4
    fut = np.arange(s, len(q))
    pred_lin = intercept + slope * fut
    cross = np.where(pred_lin <= eol)[0]
    rul_lin = int(cross[0]) if len(cross) else None
    res["B1_linear"] = dict(mae=np.abs(pred_lin - test_q).mean(), rmse=np.sqrt(((pred_lin - test_q) ** 2).mean()),
                            rul_err=abs(rul_lin - true_rul) if rul_lin else None, cov=None, width=None,
                            V=float((np.diff(pred_lin) > 0).mean()), t=time.time() - t0)

    # ---------- B2/B3: LSTM and PI-LSTM (torch CPU) ----------
    def train_lstm(pi):
        torch.manual_seed(0)
        X = np.lib.stride_tricks.sliding_window_view(train_q, W + 1)
        Xt = torch.tensor(X[:, :W], dtype=torch.float32).unsqueeze(-1)  # (n, W, 1)
        Yt = torch.tensor(X[:, -1], dtype=torch.float32)                # next capacity
        last = torch.tensor(X[:, W - 1], dtype=torch.float32)           # last observed in window
        net = nn.Sequential(nn.LSTM(1, 32, batch_first=True), LastState(), nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        mse = nn.MSELoss()
        for ep in range(epochs):
            opt.zero_grad()
            out = net(Xt).squeeze(-1)
            loss = mse(out, Yt)
            if pi: loss = loss + lam1 * torch.relu(out - last).mean()   # Eq.(4): punish predicted increase
            loss.backward(); opt.step()
        # recursive multi-step forecast on test horizon
        with torch.no_grad():
            hist = list(train_q[-W:]); preds = []
            for _ in range(len(test_q)):
                xw = torch.tensor(np.array(hist[-W:]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                p = float(net(xw).squeeze()); preds.append(p); hist.append(p)
            return np.array(preds)
    class LastState(nn.Module):
        def forward(self, o):
            out, _ = o; return out[:, -1, :]
    t0 = time.time(); pred_b2 = train_lstm(False); t_b2 = time.time() - t0
    t0 = time.time(); pred_b3 = train_lstm(True);  t_b3 = time.time() - t0

    def lstm_rul(pred):
        cross = np.where(pred <= eol)[0]
        return int(cross[0]) if len(cross) else None
    for name, pred, tt in [("B2_lstm", pred_b2, t_b2), ("B3_pi_lstm", pred_b3, t_b3)]:
        rul = lstm_rul(pred)
        res[name] = dict(mae=np.abs(pred - test_q).mean(), rmse=np.sqrt(((pred - test_q) ** 2).mean()),
                         rul_err=abs(rul - true_rul) if rul else None, cov=None, width=None,
                         V=float((np.diff(pred) > 0).mean()), t=tt)

    # ---------- PF core ----------
    def run_pf(slope_mu, slope_sd, obs=None, obs_sigma=0.01, tag=""):
        t0 = time.time()
        # particle init at start: capacity around last observed, decline rate around slope posterior
        q0 = train_q[-1]
        qi = np.clip(np.random.normal(q0, 0.01, N), eol, None)
        di = np.clip(np.abs(np.random.normal(slope_mu, slope_sd, N)), 1e-5, None)
        wi = np.ones(N) / N
        # observation update on actual test measurements up to start (filtering)
        if obs is not None:
            for z in obs:
                wi *= np.exp(-(z - qi) ** 2 / (2 * obs_sigma ** 2))
                wi /= wi.sum()
                if 1.0 / np.sum(wi ** 2) < N / 2:   # Eq.(7) resampling
                    idx = np.random.choice(N, N, p=wi); qi, wi = qi[idx], np.ones(N) / N
        # forward propagation per particle, count threshold crossings
        cross_cycles = np.full(N, -1)
        qf = qi.copy()
        for k in range(1, len(test_q) + 1):
            qf = qf - di + np.random.normal(0, 0.003, N)   # physics-consistent decline + process noise
            newly = (cross_cycles < 0) & (qf <= eol)
            cross_cycles[newly] = k
        valid = cross_cycles[cross_cycles > 0]
        rul_dist = valid[valid <= len(test_q)]
        med = int(np.median(rul_dist)) if len(rul_dist) else None
        lo, hi = (np.percentile(rul_dist, 2.5), np.percentile(rul_dist, 97.5)) if len(rul_dist) else (None, None)
        cov = None
        if med is not None and true_rul is not None:
            cov = float((lo <= true_rul <= hi)); width = float(hi - lo)
        else: width = None
        med_traj = np.minimum.accumulate(q0 - np.sort(di)[N // 2] * np.arange(1, len(test_q) + 1))
        res[tag] = dict(mae=None, rmse=None, rul_err=abs(med - true_rul) if med is not None else None,
                        cov=cov, width=width, V=float((np.diff(med_traj) > 0).mean()), t=time.time() - t0,
                        rul_median=med, interval=[lo, hi])
        return med_traj

    # B5: physics+PF (linear-trend posterior from training tail, no deep learning)
    w = 40; xs = np.arange(s - w, s); sl, ic = np.polyfit(xs, train_q[-w:], 1)
    run_pf(abs(sl), abs(sl) * 0.8 + 1e-4, obs=None, tag="B5_phys_pf")
    # B4: LSTM+PF (decline prior from B2 forecast slope; LSTM forecast as soft observation prior not used for filtering truth)
    sl_l = abs((pred_b2[-1] - pred_b2[0]) / max(len(pred_b2) - 1, 1))
    run_pf(sl_l, sl_l * 0.8 + 1e-4, obs=None, tag="B4_lstm_pf")
    # B6: P3-EVHM (PI-LSTM decline prior + PF, CPU path)
    sl_p = abs((pred_b3[-1] - pred_b3[0]) / max(len(pred_b3) - 1, 1))
    med6 = run_pf(sl_p, sl_p * 0.8 + 1e-4, obs=None, tag="B6_p3evhm_cpu")
    return dict(cell=cell, theta_pct=theta_pct, start=start_cycle, ref_capacity=ref, eol_ah=eol,
                true_eol_cycle=true_eol_cycle, true_rul=true_rul, q=q, test_q=test_q,
                pred_lin=pred_lin, pred_b2=pred_b2, pred_b3=pred_b3, med6=med6, res=res)

out = {}
runs = {}
for cell, theta, start in [("B0005", 70, 100), ("B0007", 80, 60)]:
    r = run_cell(cell, theta, start)
    runs[cell] = r; out[cell] = {k: {kk: vv for kk, vv in v.items() if kk != "interval"} | 
                                 ({"interval": v.get("interval")} if "interval" in v else {})
                                 for k, v in r["res"].items()}
    print("== ", cell, " theta", str(theta) + "%", " EOL cycle:", r["true_eol_cycle"], " true RUL:", r["true_rul"])
    for k, v in r["res"].items():
        print("   %-14s MAE %s RMSE %s RULerr %s cov %s width %s V %.3f t %.2fs" % (
            k, "None" if v["mae"] is None else "%.4f" % v["mae"],
            "None" if v["rmse"] is None else "%.4f" % v["rmse"],
            "None" if v["rul_err"] is None else str(v["rul_err"]),
            v["cov"], "None" if v.get("width") is None else "%.0f" % v["width"], v["V"], v["t"]))

with open(os.path.join(BASE, "results", "preliminary_results.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("JSON_OK")

# ---------- arrays + results figure ----------
npz = {}
for cell, r in runs.items():
    npz[cell + "_q"] = r["q"]; npz[cell + "_test_q"] = r["test_q"]
    npz[cell + "_pred_lin"] = r["pred_lin"]; npz[cell + "_pred_b2"] = r["pred_b2"]
    npz[cell + "_pred_b3"] = r["pred_b3"]; npz[cell + "_med6"] = r["med6"]
    npz[cell + "_meta"] = np.array([r["theta_pct"], r["start"], r["true_eol_cycle"] or -1, r["true_rul"] or -1])
np.savez(os.path.join(BASE, "results", "preliminary_arrays.npz"), **npz)
print("NPZ_OK")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=200)
for ax, cell in zip(axes, ["B0005", "B0007"]):
    r = runs[cell]; q = r["q"]; theta = r["theta_pct"]; eol = theta / 100 * 2.0
    s = r["start"] - 1; n = len(q)
    ax.plot(range(1, n + 1), q, "k-", lw=1.4, label="measured capacity")
    ax.axhline(eol, color="crimson", ls="--", lw=1.0, label="EOL threshold (%d%% = %.2f Ah)" % (theta, eol))
    ax.plot(range(r["start"], n + 1), r["pred_lin"], color="tab:blue", lw=1.0, alpha=0.8, label="B1 linear")
    ax.plot(range(r["start"], n + 1), r["pred_b2"], color="tab:orange", lw=1.0, alpha=0.8, label="B2 LSTM")
    ax.plot(range(r["start"], n + 1), r["pred_b3"], color="tab:green", lw=1.0, alpha=0.8, label="B3 PI-LSTM")
    if r["med6"] is not None:
        ax.plot(range(r["start"], n + 1), r["med6"], color="tab:purple", lw=1.3, ls="-.", label="B6 P3-EVHM (PF median)")
    if r["true_eol_cycle"]:
        ax.axvline(r["true_eol_cycle"], color="gray", ls=":", lw=1.0, label="true EOL cycle %d" % r["true_eol_cycle"])
    ax.axvline(r["start"], color="gray", lw=0.8); ax.text(r["start"] + 1, q[0], "start", fontsize=7, color="gray")
    ax.set_xlabel("cycle"); ax.set_ylabel("capacity (Ah)")
    ax.set_title("%s, RUL start cycle %d (%s threshold)" % (cell, r["start"], str(theta) + "%"), fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left"); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results", "preliminary_fig.png"), bbox_inches="tight", facecolor="white")
print("FIG2_OK", os.path.getsize(os.path.join(BASE, "results", "preliminary_fig.png")))
