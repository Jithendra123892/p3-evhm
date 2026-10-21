# P3-EVHM live dashboard: pick a cell + prediction start, train PI-LSTM vs data-only LSTM, run the particle filter,
# and show SoH, RUL interval, health state, risk alerts, physics-violation rates, and CPU timing side by side.
# Cached per configuration: first run trains (~10-30 s on CPU), repeats are instant.
import os, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from p3evhm import models as P3

BASE = os.path.dirname(os.path.abspath(__file__))
RATED = 2.0  # Ah, NASA PCoE 18650 cells
torch.set_num_threads(4)

st.set_page_config(page_title="P3-EVHM Dashboard", page_icon=":battery:", layout="wide")

@st.cache_data(show_spinner=False)
def load_telemetry():
    return pd.read_csv(os.path.join(BASE, "data", "telemetry.csv"))

DF = load_telemetry()
CELLS = ["B0005", "B0006", "B0007", "B0018"]
EOL_CYCLE = {"B0005": 125, "B0006": 109, "B0007": 86, "B0018": 97}
THETA_DEFAULT = {"B0005": 70, "B0006": 70, "B0007": 80, "B0018": 70}

class LastState(nn.Module):
    def forward(self, o):
        out, _ = o
        return out[:, -1, :]

@st.cache_data(show_spinner=False)
def train_forecast(_q, _T, lam1, lam2, steps, hidden=24, epochs=80, W=8, lr=5e-3, seed=0):
    # canonical math lives in the p3evhm package; this wrapper only adds Streamlit caching
    q = np.asarray(_q, dtype=float); T = np.asarray(_T, dtype=float)
    preds, dt = P3.pi_lstm_forecast(q, T, steps=steps, lam1=lam1, lam2=lam2, W=W, hidden=hidden,
                                    epochs=epochs, lr=lr, seed=seed, return_train_time=True)
    return dt, preds

def pf_rul(q0, slope_mu, slope_sd, eol, steps, N=5000, proc=0.003, seed=0, obs0=None):
    med, lo, hi, dt, traj = P3.pf_rul(q0, slope_mu, slope_sd, eol, steps, N=N, proc=proc,
                                      seed=seed, obs0=obs0, return_traj=True)
    return med, lo, hi, dt, traj

PRETRAINED_PATH = os.path.join(BASE, "p3evhm", "pretrained", "pi_lstm_base_v1.pt")

@st.cache_resource(show_spinner=False)
def load_pretrained_net():
    return P3.load_pretrained(PRETRAINED_PATH, device="cpu")

@st.cache_data(show_spinner=True)
def run_pipeline(cell, theta_pct, start, epochs, N, seed=0, mode="transfer"):
    d = DF[DF.cell == cell].sort_values("cycle")
    q = d.capacity.values; T = d.temp.fillna(25).values
    eol = theta_pct / 100.0 * RATED
    s = start - 1
    train_q, test_q, train_T = q[:s], q[s:], T[:s]
    true_cross = np.where(test_q <= eol)[0]
    true_rul = int(true_cross[0]) if len(true_cross) else None  # cycles after start (test_q[i] = cycle start+i)
    out = {"cell": cell, "theta": theta_pct, "start": start, "eol_ah": eol,
           "q": q, "test_q": test_q, "true_rul": true_rul}
    # B1 linear
    w = 20; sl, ic = np.polyfit(np.arange(s - w, s), train_q[-w:], 1); sl = min(sl, -1e-4)
    out["pred_lin"] = ic + sl * np.arange(s, len(q))
    # B2 / B3 LSTMs
    if mode == "transfer" and os.path.exists(PRETRAINED_PATH):
        # pretrained fleet checkpoint -> zero-shot (reference) + frozen-body head fine-tune (PI model)
        net0, cfg = load_pretrained_net()
        import copy
        net0 = copy.deepcopy(net0)  # finetune mutates the net; never leak state across cells/runs
        W = int(cfg.get("W", 8))
        out["pred_b2"] = P3.forecast(net0, train_q, len(test_q), W=W)          # zero-shot
        net_ft, ft_s, _ = P3.finetune(net0, train_q, train_T, freeze_body=True,
                                      W=W, epochs=30, lr=1e-3, seed=seed)
        out["pred_b3"] = P3.forecast(net_ft, train_q, len(test_q), W=W)
        out["lstm_train_s"] = round(ft_s, 3)
        out["b2_label"] = "zero-shot (pretrained, unadapted)"
        out["b3_label"] = "B3 fine-tuned PI-LSTM (frozen body)"
    else:
        ts2, out["pred_b2"] = train_forecast(tuple(train_q), tuple(train_T), 0.0, 0.0, len(test_q), epochs=epochs, seed=seed)
        ts3, out["pred_b3"] = train_forecast(tuple(train_q), tuple(train_T), 2.0, 1.0, len(test_q), epochs=epochs, seed=seed)
        out["lstm_train_s"] = round(ts3, 2)
        out["b2_label"] = "B2 data-only LSTM"
        out["b3_label"] = "B3 PI-LSTM"
    # PF on the PI-LSTM decline prior (B6 live) and the physics prior (B5)
    sl5 = abs((train_q[-40:][0] - train_q[-1]) / 40)
    sl6 = abs((out["pred_b3"][-1] - out["pred_b3"][0]) / max(len(out["pred_b3"]) - 1, 1))
    med, lo, hi, pf_s, traj = pf_rul(train_q[-1], sl6, sl6 * 0.8 + 1e-4, eol, len(test_q), N=N, seed=seed, obs0=test_q[0])
    out.update(rul_med=med, rul_lo=lo, rul_hi=hi, pf_s=pf_s, pf_traj=traj)
    med5, lo5, hi5, _, traj5 = pf_rul(train_q[-1], sl5, sl5 * 0.8 + 1e-4, eol, len(test_q), N=N, seed=seed, obs0=test_q[0])
    out.update(rul5_med=med5, rul5_lo=lo5, rul5_hi=hi5, pf5_traj=traj5)
    # violation rates on test horizon
    out["V_b2"] = float((np.diff(out["pred_b2"]) > 0).mean())
    out["V_b3"] = float((np.diff(out["pred_b3"]) > 0).mean())
    out["soh_now"] = 100.0 * q[start - 1] / RATED
    return out

# ---------------- UI ----------------
st.title("P3-EVHM — Parallel Physics-Informed Prognostics for EV Battery Health Management")
st.caption("A. Vibhas Chowdhary · Maddala Jithendra · Guide: Dr. Kavitha R — Dept. of Computing Technologies, SRM IST  |  Data: NASA Ames PCoE Battery Data Set")

with st.sidebar:
    st.header("Controls")
    cell = st.selectbox("Battery cell (NASA PCoE)", CELLS)
    theta = st.radio("EOL threshold (% of rated 2.0 Ah)", [70, 80], index=0 if THETA_DEFAULT[cell] == 70 else 1)
    max_start = EOL_CYCLE[cell] - 2
    start = st.slider("Prediction start cycle", 15, max_start, max(int(round(EOL_CYCLE[cell] * 0.6)), 15))
    N = st.select_slider("Particles N", [1000, 5000, 20000, 100000], value=5000)
    mode = "transfer" if st.radio("Model mode", ["Pretrained + fine-tune (fast)", "From scratch"], index=0) \
        .startswith("Pretrained") else "scratch"
    if mode == "transfer" and not os.path.exists(PRETRAINED_PATH):
        st.warning("Pretrained checkpoint not found - falling back to from-scratch training.")
        mode = "scratch"
    epochs = st.select_slider("LSTM epochs (from-scratch mode only; fine-tune uses 30)", [40, 80, 200], value=200)
    st.caption("Pretrained mode loads the fleet checkpoint (3 NASA cells) and adapts only the output head to the "
               "selected cell in a fraction of a second. From-scratch trains the PI-LSTM live. "
               "The paper's GPU timings were measured on a Kaggle Tesla T4 (Table IV).")

tabs = st.tabs(["Start Here (read first)", "Live Prediction", "Validation Grid (40 runs)", "GPU Scaling (Tesla T4)", "Agent Architecture", "About"])

with tabs[0]:
    st.subheader("What this application does — in plain words")
    st.markdown(
        "A lithium-ion battery ages every time it charges and discharges. This app reads real NASA lab measurements "
        "(voltage, current, temperature, capacity, 168 charge-discharge cycles per cell), learns how the cell degrades, "
        "and answers two questions a service engineer cares about: **how healthy is the battery right now (SoH)** and "
        "**how many cycles are left before it must be replaced (RUL)** — with an honest uncertainty band, not a single guess. "
        "Two models do the work: a physics-informed LSTM (a neural network trained with a rule that forbids it from "
        "predicting capacity going UP) and a particle filter (thousands of simulated futures that give the uncertainty band). "
        "The neural network part runs on GPU on NVIDIA hardware; here on this laptop it runs on CPU and on Kaggle's Tesla T4 GPU it was measured at up to 16.5x faster."
    )
    st.image(os.path.join(BASE, "flow_dfd.png"), caption="Left: system flowchart. Right: Data Flow Diagram Level 1 (P1-P5 = the five agents; D1-D3 = data stores).")
    st.subheader("Mam's requirement checklist — where each item is demonstrated")
    st.table(pd.DataFrame({
        "Requirement": ["1. SDG + problem statistics", "2. Requirements + existing solutions", "3. Quantitative dataset (Kaggle/NASA)",
                        "4. Flowchart", "5. DFD Level 1", "6. Mathematical model", "7. Parallel model (GPU/CUDA)",
                        "8. Expected output", "9. Technology stack", "10. Hardware/software record", "11. Budget",
                        "12. Dashboard navigation", "13. Strong novelty", "14. Working application + validation"],
        "Where to show it": ["Paper Sec. I (SDG 7/9) + About tab", "Paper Sec. II + IX", "This app loads the real NASA PCoE data; Kaggle T4 run for GPU",
                             "Figure above, panel (a)", "Figure above, panel (b)", "Paper Sec. V, Eqs. (1)-(8)",
                             "GPU Scaling tab (measured on Tesla T4)", "Live Prediction tab: SoH, RUL interval, risk alert",
                             "Paper Sec. IX + VALIDATION_REPORT Sec. 2", "VALIDATION_REPORT Sec. 2 (both environments)",
                             "Paper Table II", "This dashboard, 6 tabs", "Paper Sec. II-E: verified unpublished gap",
                             "40 measured runs on 2 machines + this live demo"],
    }))
    st.subheader("How to drive the demo (60 seconds)")
    st.markdown(
        "1. Open **Live Prediction** — the four cards show SoH, RUL median, 95% interval, and the physics-violation rate. "
        "The coloured band is the maintenance alert.\n"
        "2. Drag **Prediction start cycle** — the alert logic reacts (that is the decision-support story).\n"
        "3. Switch the cell to **B0007** in the sidebar — threshold auto-switches to 80%, everything recomputes.\n"
        "4. **Validation Grid** tab = 40 measured runs on two machines. **GPU Scaling** tab = the measured Tesla T4 speedups.\n"
        "5. If asked for evidence: `VALIDATION_REPORT.md` (data hashes, both environments, reproduction commands)."
    )

with tabs[4]:
    st.subheader("The five agents — each one is a Perception -> Decision -> Action loop")
    st.table(pd.DataFrame({
        "Agent": ["A1 Ingest & Validate", "A2 Features & Windows", "A3 PI-LSTM SoH (GPU)",
                  "A4 Particle Filter Engine (GPU)", "A5 Decision & Alerts"],
        "Perceives (inputs)": ["raw .mat cycle records", "validated telemetry", "feature windows",
                               "SoH state + decline prior", "RUL interval, SoH, violation rate"],
        "Decides (logic)": ["schema + cycle-count assertions", "window construction, leakage-safe scaling",
                            "gradient training with physics loss (Eq. 3-5)", "propagate / weight / resample (Eq. 6-7)",
                            "threshold rules on interval lower bound"],
        "Acts (outputs)": ["validated telemetry to D1", "windows to P3", "SoH forecast + violation rate V (Eq. 8)",
                           "RUL distribution + 95% interval", "ALERT / CAUTION / OK + dashboard + logs to D3"],
    }))
    st.caption("Honest positioning (also stated in the paper): these are rule-based software agents - modular "
               "perception/decision/action loops communicating through logged records, in the style of tool-based PHM "
               "services (PHMForge, 2026) - NOT autonomous LLM agents and NOT a formal multi-agent-systems theory claim. "
               "A3 and A4 are the GPU-bound stages; A1, A2, A5 run on CPU.")

with tabs[5]:
    below = DF[(DF.cell == cell)].capacity.min() <= theta / 100.0 * RATED
    if not below:
        st.warning(f"{cell} never reaches the {theta}% threshold in this dataset — switch to 80% for this cell.")
    else:
        R = run_pipeline(cell, int(theta), int(start), int(epochs), int(N), mode=mode)
        c1, c2, c3, c4 = st.columns(4)
        soh = R["soh_now"]
        state = ("Healthy" if soh > 90 else "Ageing" if soh > 80 else "Replace soon" if soh > 70 else "End of life")
        c1.metric("SoH at start cycle", f"{soh:.1f}%", state)
        if R["rul_med"] is not None and R["true_rul"]:
            err = abs(R["rul_med"] - R["true_rul"])
            c2.metric("Predicted RUL (median)", f"{R['rul_med']} cyc",
                      f"true {R['true_rul']} · err {err} cyc", delta_color="off")
            c3.metric("95% prediction interval", f"[{R['rul_lo']:.0f}, {R['rul_hi']:.0f}] cyc",
                      f"coverage target 95%")
        else:
            c2.metric("Predicted RUL", "no crossing", "particles above EOL", delta_color="off")
            c3.metric("95% prediction interval", "—")
        c4.metric("Physics violation rate V", f"PI-LSTM {R['V_b3']:.3f}", f"{R.get('b2_label', 'data-only LSTM')} {R['V_b2']:.3f}", delta_color="off")
        # risk alert
        if R["rul_med"] is not None and R["true_rul"]:
            margin = R["rul_lo"]
            if margin <= 10:
                st.error(f"ALERT: lower 95% bound {margin:.0f} cycles — schedule replacement inspection within {margin:.0f} cycles.")
            elif margin <= 25:
                st.warning(f"CAUTION: lower 95% bound {margin:.0f} cycles — plan maintenance window.")
            else:
                st.success(f"OK: lower 95% bound {margin:.0f} cycles — routine monitoring.")
        eol_ah = R["eol_ah"]; s0 = R["start"]; n = len(R["q"])
        fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
        ax.plot(range(1, n + 1), R["q"], "k-", lw=1.6, label="measured capacity")
        ax.axhline(eol_ah, color="crimson", ls="--", lw=1.1, label=f"EOL threshold ({theta}% = {eol_ah:.2f} Ah)")
        ax.plot(range(s0, n + 1), R["pred_b2"], color="tab:orange", lw=1.1, alpha=0.85, label=R.get("b2_label", "B2 data-only LSTM"))
        ax.plot(range(s0, n + 1), R["pred_b3"], color="tab:green", lw=1.1, alpha=0.85, label=R.get("b3_label", "B3 PI-LSTM"))
        ax.plot(range(s0, n + 1), R["pred_lin"], color="tab:blue", lw=1.0, alpha=0.7, label="B1 linear")
        if R["rul_med"] is not None:
            ax.plot(range(s0, n + 1), R["pf_traj"], color="tab:purple", lw=1.5, ls="-.", label="PF median trajectory (B6 live)")
        if R["true_rul"] is not None:
            ax.axvline(s0 + R["true_rul"], color="gray", ls=":", lw=1.1, label=f"true EOL cycle {s0 + R['true_rul']}")
        ax.axvline(s0, color="gray", lw=0.8)
        ax.set_xlabel("cycle"); ax.set_ylabel("capacity (Ah)")
        ax.set_title(f"{cell} — prediction from cycle {s0 + 1} ({theta}% threshold, N={N}, {epochs} epochs)", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.25)
        st.pyplot(fig)
        m1, m2, m3 = st.columns(3)
        m1.metric("LSTM training time (this machine, CPU)", f"{R['lstm_train_s']} s", "PI-LSTM incl. physics loss")
        m2.metric("Particle filter time (N=%d)" % N, f"{R['pf_s'] * 1000:.1f} ms", "GPU (T4): 7.0-20.2 ms at 1k-500k")
        m3.metric("Physics+PF RUL (B5 prior)", (f"{R['rul5_med']} cyc" if R["rul5_med"] else "no crossing"),
                  (f"interval [{R['rul5_lo']:.0f}, {R['rul5_hi']:.0f}]" if R["rul5_med"] else ""))

with tabs[2]:
    st.subheader("Measured validation grid — 20 configurations per environment, 40 total")
    try:
        k = json.load(open(os.path.join(BASE, "results", "kernel_grid_stats.json")))
        import statistics
        rows = [
            ["Mean violation rate V — data-only LSTM (B2)", "%.3f" % statistics.mean(k["B2V"]), "%.3f" % statistics.mean(k["B3V"]), "PI constraint cuts implausible steps ~5x"],
            ["Configs where PI-LSTM reduced V", "—", "15 of 20 (equal 3)", "measured, per-config"],
            ["Mean capacity MAE B2 -> B3 (Ah)", "0.2854 -> 0.2569", "—", "improves in both environments"],
            ["B5 physics+PF RUL error mean (with RUL)", "18.7 cycles (20/20)", "—", "available in every config"],
            ["B5 95% interval coverage", "0.85", "—", "calibrated, slightly narrow"],
            ["B6 fused RUL availability", "9 of 20", "—", "weakness: decline prior too shallow early"],
        ]
        st.table(pd.DataFrame(rows, columns=["Metric", "E2 Tesla T4", "E1 local CPU", "Note"]))
    except Exception as e:
        st.error(f"results file not found: {e}")
    st.caption("Full per-config results: results/full_results.json (local), kernel_out/p3evhm_results.json (T4). VALIDATION_REPORT.md has provenance: two byte-verified data sources, SHA-256 logged.")

with tabs[3]:
    st.subheader("Measured GPU scaling of the particle filter (Tesla T4, kernel v3)")
    st.table(pd.DataFrame({
        "Particles N": ["1,000", "5,000", "20,000", "100,000", "500,000"],
        "CPU time (s)": ["0.0327*", "0.0076", "0.0169", "0.0614", "0.3333"],
        "GPU device time (ms)": ["7.0", "7.7", "7.4", "9.4", "20.2"],
        "End-to-end speedup": ["0.65x (vs warm CPU)", "0.98x", "2.3x", "6.5x", "16.5x"],
    }))
    st.caption("*T4-host N=1,000 CPU timing inflated by first-call warmup; warm CPU measures 4.5 ms, so the GPU is slower at very small N. Latency target <100 ms met at every N. Source: kernel_out/p3evhm_results.json.")

with tabs[5]:
    st.markdown(
        "**What this application demonstrates**\n\n"
        "- Live PI-LSTM (Eq. 3-5: data + monotonicity + temperature-prior loss) vs data-only LSTM, trained in front of you\n"
        "- Particle-filter RUL with a 95% prediction interval and configurable alerts (A4 agent)\n"
        "- Physics-constraint violation rate V (Eq. 8) measured per run (A5 agent)\n"
        "- Measured CPU timing here vs measured Tesla T4 GPU timings from the Kaggle kernel run\n\n"
        "**Research gap (verified by literature survey):** no published framework integrates physics-informed LSTM training, "
        "particle-filter RUL uncertainty, and GPU-parallel particle computation for battery health management. "
        "Closest fusions defer physics constraints to future work (TL-MLSTM-PF 2025; IEEE hybrid PF-NN 2021); "
        "GPU battery prognostics has not been revisited since PHM Society 2014 (pre-deep-learning).\n\n"
        "**Scope:** research decision-support prototype on NASA PCoE laboratory cells — not a certified BMS safety function."
    )
