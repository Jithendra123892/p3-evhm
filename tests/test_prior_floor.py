# Prior-floor validation: the fused path (B6) failed at early starts because the LSTM-derived
# decline prior was too shallow for the particle cloud to reach the EOL threshold. This test
# applies the physics-based slope floor (the training-tail linear decline, already used by B5)
# to exactly those configurations and measures whether RUL becomes available and how accurate it is.
import os, sys, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from p3evhm import data, evaluate, models

df = data.load_telemetry_csv(os.path.join(BASE, "data", "telemetry.csv"))
CELLS = ["B0005", "B0006", "B0007", "B0018"]
THETA = {c: (80 if c == "B0007" else 70) for c in CELLS}
out = {}
avail_no_floor, avail_floor = 0, 0
errs_floor, errs_nofloor_cells = [], []
for held in CELLS:
    d = df[df.cell == held].sort_values("cycle")
    q, T = d.capacity.values, d.temp.fillna(25.0).values
    eol = THETA[held] / 100.0 * 2.0
    eolc = int(np.where(q <= eol)[0][0] + 1)
    for frac in [0.4, 0.6]:  # early starts: where B6 failed in the grid
        start = max(int(round(eolc * frac)), 15); s = start - 1
        train_q, test_q, train_T = q[:s], q[s:], T[:s]
        true_rul = eolc - start
        # fine-tuned pretrained PI-LSTM forecast (the transfer path whose slope can be shallow)
        net, cfg = models.load_pretrained(os.path.join(BASE, "p3evhm", "pretrained", "pi_lstm_base_v1.pt"))
        import copy
        net = copy.deepcopy(net)
        net_ft, _, _ = models.finetune(net, train_q, train_T, freeze_body=False,
                                       W=cfg["W"], epochs=30, lr=1e-3, seed=0)
        pred = models.forecast(net_ft, train_q, len(test_q), W=cfg["W"])
        sl_model = abs((pred[-1] - pred[0]) / max(len(pred) - 1, 1))
        sl_phys = abs((train_q[-40:][0] - train_q[-1]) / 40)  # physics/tail slope = the floor
        no = models.pf_rul(train_q[-1], sl_model, sl_model * 0.8 + 1e-4, eol, len(test_q),
                           N=5000, seed=0, obs0=test_q[0])
        sweep = {}
        for m_mult in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fl_m = models.pf_rul(train_q[-1], sl_model, sl_model * 0.8 + 1e-4, eol, len(test_q),
                                 N=5000, seed=0, obs0=test_q[0], slope_floor=(m_mult * sl_phys) if m_mult > 0 else None)
            sweep[m_mult] = (fl_m[0], (abs(fl_m[0] - true_rul) if fl_m[0] is not None and true_rul is not None else None))
        fl = (sweep[1.0][0], None, None)
        rec = dict(start=start, true_rul=true_rul, sl_model=round(sl_model, 5), sl_floor=round(sl_phys, 5),
                   no_floor_rul_err=no[0], floor_rul_err=fl[0], sweep={str(k): v for k, v in sweep.items()})
        ok_avail = (fl[0] is not None) if true_rul is not None else True
        avail_no_floor += (no[0] is not None); avail_floor += (fl[0] is not None)
        if fl[0] is not None and true_rul is not None:
            errs_floor.append(abs(fl[0] - true_rul))
        if no[0] is not None and true_rul is not None:
            errs_nofloor_cells.append(held)
        out[f"{held}@{frac}"] = rec
        sw_txt = " | ".join(f"x{k}: {v[1] if v[1] is not None else '-'}" for k, v in sweep.items())
        print(f"{held}@{frac} start {start} trueRUL {true_rul}: slope model {sl_model:.5f} vs floor {sl_phys:.5f} "
              f"-> no-floor err {abs(no[0]-true_rul) if no[0] is not None else '-'} | floor sweep err by multiplier: {sw_txt}", flush=True)

print()
print(f"RUL availability: no-floor {avail_no_floor}/8 vs with-floor {avail_floor}/8")
print(f"with-floor mean |RUL err|: {np.mean(errs_floor):.1f} cycles (n={len(errs_floor)})")
out["_summary"] = dict(avail_no_floor=avail_no_floor, avail_floor=avail_floor,
                       floor_mean_abs_err=round(float(np.mean(errs_floor)), 1) if errs_floor else None)
with open(os.path.join(BASE, "results", "prior_floor_results.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("PRIOR_FLOOR_SWEEP_OK (informational: fixed floors are cell-dependent, see sweep)")

