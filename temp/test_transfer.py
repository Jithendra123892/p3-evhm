# Transfer-learning comparison on the held-out cell B0007 (start at 60% of life):
# (a) from scratch (200 epochs) | (b) pretrained zero-shot | (c) pretrained + head fine-tune (body FROZEN, 30 ep)
# (d) pretrained + full fine-tune (30 ep, lr 1e-3). Metrics: capacity MAE/RMSE, violation rate V, RUL err (LSTM + PF), adaptation time.
import os, sys, time, json
import numpy as np

BASE = r"C:\Users\jithe\p3-evhm-paper"
sys.path.insert(0, BASE)
from p3evhm import data, models

df = data.load_telemetry_csv(os.path.join(BASE, "data", "telemetry.csv"))
d = df[df.cell == "B0007"].sort_values("cycle")
q, T = d.capacity.values, d.temp.fillna(25.0).values
RATED, THETA = 2.0, 80
eol = THETA / 100.0 * RATED
eolc = int(np.where(q <= eol)[0][0] + 1)
start = max(int(round(eolc * 0.6)), 15); s = start - 1
true_cross = np.where(q[s:] <= eol)[0]
true_rul = int(true_cross[0]) if len(true_cross) else None
train_q, test_q, train_T = q[:s], q[s:], T[:s]
print(f"B0007 | 80% EOL={eol:.2f} Ah | true EOL cycle {eolc} | start {start} | true RUL {true_rul}")

def evaluate_pred(name, pred, adapt_s):
    cr = np.where(pred <= eol)[0]
    sl = abs((pred[-1] - pred[0]) / max(len(pred) - 1, 1))
    med, lo, hi, pf_s = models.pf_rul(train_q[-1], sl, sl * 0.8 + 1e-4, eol, len(test_q), N=5000, seed=0, obs0=test_q[0])
    rec = dict(mae=float(np.abs(pred - test_q).mean()), rmse=float(np.sqrt(((pred - test_q) ** 2).mean())),
               V=float((np.diff(pred) > 0).mean()), adapt_s=round(adapt_s, 3),
               rul_err=int(abs(cr[0] - true_rul)) if len(cr) and true_rul is not None else None,
               pf_rul_err=int(abs(med - true_rul)) if med is not None and true_rul is not None else None,
               pf_interval=[lo, hi])
    print(f"{name:<26} MAE {rec['mae']:.4f} RMSE {rec['rmse']:.4f} V {rec['V']:.3f} "
          f"LSTM-RULerr {rec['rul_err']} PF-RULerr {rec['pf_rul_err']} adapt {rec['adapt_s']}s")
    return rec

out = {}
t0 = time.time()
pred_scratch = models.pi_lstm_forecast(train_q, train_T, steps=len(test_q), lam1=2.0, lam2=1.0,
                                       W=8, hidden=32, epochs=200, seed=0, device="auto")
out["a_from_scratch_200ep"] = evaluate_pred("from scratch (200 ep)", pred_scratch, time.time() - t0)

net, cfg = models.load_pretrained(os.path.join(BASE, "p3evhm", "pretrained", "pi_lstm_base_v1.pt"))
pred_zs = models.forecast(net, train_q, len(test_q), W=cfg["W"])
out["b_zero_shot"] = evaluate_pred("pretrained zero-shot", pred_zs, 0.0)

net_ft, ft_s, n_train = models.finetune(net, train_q, train_T, freeze_body=True, W=cfg["W"],
                                        epochs=30, lr=1e-3, seed=0)
pred_ft = models.forecast(net_ft, train_q, len(test_q), W=cfg["W"])
out["c_head_finetune_30ep"] = evaluate_pred("head fine-tune (frozen)", pred_ft, ft_s)
out["c_trainable_params"] = int(n_train)

net2, cfg2 = models.load_pretrained(os.path.join(BASE, "p3evhm", "pretrained", "pi_lstm_base_v1.pt"))
net_ft2, ft_s2, n_train2 = models.finetune(net2, train_q, train_T, freeze_body=False, W=cfg2["W"],
                                           epochs=30, lr=1e-3, seed=0)
pred_ft2 = models.forecast(net_ft2, train_q, len(test_q), W=cfg2["W"])
out["d_full_finetune_30ep"] = evaluate_pred("full fine-tune (30 ep)", pred_ft2, ft_s2)
out["d_trainable_params"] = int(n_train2)
out["context"] = dict(cell="B0007", theta_pct=THETA, start=start, true_rul=true_rul, true_eol_cycle=eolc)

with open(os.path.join(BASE, "results", "transfer_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("TRANSFER_OK")
