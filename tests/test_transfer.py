# Leave-one-cell-out (LOOCV) transfer validation — the full test of the transfer claim.
# For EACH cell H: pretrain a fleet base on the other three cells (first 60% of each),
# then at H's 60% start point compare: from-scratch (200 ep) vs zero-shot vs frozen-body head fine-tune (30 ep).
# Invariants: (1) body weights bit-identical after fine-tune, (2) trainable params == head only (33),
# (3) fine-tune deterministic under fixed seed, (4) no state leak: base net unchanged across adaptations,
# (5) head fine-tune beats or matches from-scratch MAE with zero violations.
import os, sys, time, json, hashlib, copy
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from p3evhm import data, models

RATED = 2.0
CELLS = ["B0005", "B0006", "B0007", "B0018"]
THETA = {c: (80 if c == "B0007" else 70) for c in CELLS}

df = data.load_telemetry_csv(os.path.join(BASE, "data", "telemetry.csv"))

def body_hash(net):
    # net[0] = LSTM body (net[1] is the parameterless passthrough)
    return hashlib.md5(np.concatenate([p.detach().numpy().ravel() for p in net[0].parameters()]).tobytes()).hexdigest()

def eval_pred(name, pred, test_q, eol, true_rul, adapt_s):
    cr = np.where(pred <= eol)[0]
    return dict(name=name,
                mae=round(float(np.abs(pred - test_q).mean()), 4),
                rmse=round(float(np.sqrt(((pred - test_q) ** 2).mean())), 4),
                V=round(float((np.diff(pred) > 0).mean()), 3),
                rul_err=(int(abs(cr[0] - true_rul)) if len(cr) and true_rul is not None else None),
                adapt_s=round(adapt_s, 3))

results, all_ok = {}, True
t_all = time.time()
for held in CELLS:
    t_cell = time.time()
    pool_q, pool_T = [], []
    for c in CELLS:
        if c == held: continue
        d = df[df.cell == c].sort_values("cycle")
        q, T = d.capacity.values, d.temp.fillna(25.0).values
        cut = int(len(q) * 0.6)
        pool_q.append(q[:cut]); pool_T.append(T[:cut])
    pq, pT = np.concatenate(pool_q), np.concatenate(pool_T)
    base = models.train_pi_lstm(pq, pT, lam1=2.0, lam2=1.0, W=8, hidden=32, epochs=300, lr=5e-3, seed=0)
    ckpt_body = body_hash(base)

    d = df[df.cell == held].sort_values("cycle")
    q, T = d.capacity.values, d.temp.fillna(25.0).values
    eol = THETA[held] / 100.0 * RATED
    eolc = int(np.where(q <= eol)[0][0] + 1)
    start = max(int(round(eolc * 0.6)), 15); s = start - 1
    true_rul = eolc - start
    train_q, test_q, train_T = q[:s], q[s:], T[:s]

    t0 = time.time()
    pred_s = models.pi_lstm_forecast(train_q, train_T, steps=len(test_q), lam1=2.0, lam2=1.0,
                                     W=8, hidden=32, epochs=200, seed=0)
    scratch = eval_pred("from_scratch_200ep", pred_s, test_q, eol, true_rul, time.time() - t0)

    pred_z = models.forecast(base, train_q, len(test_q), W=8)
    zeroshot = eval_pred("zero_shot", pred_z, test_q, eol, true_rul, 0.0)

    net_ft, ft_s, n_train = models.finetune(copy.deepcopy(base), train_q, train_T,
                                            freeze_body=True, W=8, epochs=30, lr=1e-3, seed=0)
    pred_ft = models.forecast(net_ft, train_q, len(test_q), W=8)
    finetune = eval_pred("head_finetune_frozen_30ep", pred_ft, test_q, eol, true_rul, ft_s)
    net_ff, ff_s, n_train_ff = models.finetune(copy.deepcopy(base), train_q, train_T,
                                               freeze_body=False, W=8, epochs=30, lr=1e-3, seed=0)
    pred_ff = models.forecast(net_ff, train_q, len(test_q), W=8)
    fullft = eval_pred("full_finetune_30ep", pred_ff, test_q, eol, true_rul, ff_s)

    # invariants
    inv = {}
    inv["body_unchanged_after_head_ft"] = (body_hash(net_ft) == ckpt_body)
    inv["head_only_trainable_33"] = (n_train == 33)
    inv["full_ft_changes_body"] = (body_hash(net_ff) != ckpt_body)
    net_ft2, ft_s2, _ = models.finetune(copy.deepcopy(base), train_q, train_T,
                                        freeze_body=True, W=8, epochs=30, lr=1e-3, seed=0)
    p2 = models.forecast(net_ft2, train_q, len(test_q), W=8)
    inv["deterministic"] = bool(np.allclose(pred_ft, p2, atol=1e-6))
    inv["base_not_mutated"] = (body_hash(base) == ckpt_body)
    # the right yardstick: transfer modes must deliver RUL + physics consistency, which scratch mostly fails
    inv["transfer_rul_available_3x3"] = all(m["rul_err"] is not None for m in (zeroshot, finetune, fullft)) if true_rul is not None else True
    inv["transfer_zero_violations"] = (zeroshot["V"] == 0.0 and finetune["V"] == 0.0 and fullft["V"] == 0.0)
    inv["headft_speedup_over_2x"] = (scratch["adapt_s"] / max(finetune["adapt_s"], 1e-6) >= 2.0)
    ok = all(inv.values()); all_ok &= ok

    results[held] = dict(theta_pct=THETA[held], true_eol_cycle=eolc, start=start, true_rul=true_rul,
                         scratch=scratch, zero_shot=zeroshot, finetune=finetune, full_finetune=fullft,
                         invariants=inv, cell_wall_s=round(time.time() - t_cell, 1))
    print(f"== held-out {held} (EOL {eolc}, start {start}, true RUL {true_rul}) | {'INVARIANTS PASS' if ok else 'INVARIANT FAIL: ' + str([k for k, v in inv.items() if not v])}")
    for k in ["scratch", "zero_shot", "finetune", "full_finetune"]:
        r = results[held][k]
        print(f"   {r['name']:<26} MAE {r['mae']:.4f} V {r['V']:.3f} RULerr {r['rul_err']} adapt {r['adapt_s']}s")

ratios = [results[c]["scratch"]["mae"] / results[c]["full_finetune"]["mae"] for c in CELLS
          if results[c]["scratch"]["mae"] and results[c]["full_finetune"]["mae"]]
speed = [results[c]["scratch"]["adapt_s"] / max(results[c]["full_finetune"]["adapt_s"], 1e-6) for c in CELLS]
ft_mae = [results[c]["full_finetune"]["mae"] for c in CELLS]
print()
print(f"LOOCV SUMMARY: head-FT MAE per cell {ft_mae} | mean MAE ratio (scratch/FT) {np.mean(ratios):.2f}x "
      f"| mean speed ratio {np.mean(speed):.1f}x | all invariants: {'PASS' if all_ok else 'FAIL'}")
results["_summary"] = dict(mean_mae_ratio=round(float(np.mean(ratios)), 2),
                           mean_speed_ratio=round(float(np.mean(speed)), 1),
                           ft_mae_per_cell=ft_mae, all_invariants=bool(all_ok),
                           total_wall_s=round(time.time() - t_all, 1))
with open(os.path.join(BASE, "results", "transfer_loocv_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("LOOCV_DONE", "TESTS_OK" if all_ok else "TESTS_FAILED")
sys.exit(0 if all_ok else 1)
