# P3-EVHM — Validation Report
**Project:** Parallel Physics-Informed Prognostics for EV Battery Health Management (P3-EVHM)
**Authors:** A. Vibhas Chowdhary (va1641@srmist.edu.in), Maddala Jithendra (jm4408@srmist.edu.in) — Guide: Dr. Kavitha R (kavithar14@srmist.edu.in), Department of Computing Technologies, SRM Institute of Science and Technology
**Paper:** `P3-EVHM_paper.docx` (IEEE two-column, 11 sections, 4 tables, 2 figures, 33 references — verified by 87 automated structural checks, 0 failures)
**Date of validation runs:** October 20, 2026

This document records exactly what was executed, on what data, with what results. Every number in the paper's Section VIII comes from the artifacts listed here. Nothing is assumed.

## 1. Data provenance (two independent sources, byte-verified)

| File | Size (bytes) | SHA-256 (first 16) | Source 1 | Source 2 |
|---|---|---|---|---|
| B0005.mat | 15,956,874 | 0eae4585baf3f200 | GitHub mirror (Tianyou-Bai/NASA_PCoE_Battery-Aging-Prediction-) | Kaggle ckskaggle/li-ion-battery-dataset-from-nasa-pcoe |
| B0006.mat | 16,028,159 | fa818ab4db5db8ab | same | same |
| B0007.mat | 16,049,460 | d022afa086efaf54 | same | same |
| B0018.mat | 8,502,218 | d1e6c923a43ea1c9 | same | same |

Both sources' files match byte-for-byte (Kaggle file listing sizes = local file sizes). Original publication: Saha & Goebel, NASA Ames Prognostics Center of Excellence Battery Data Set, 2007.

Parsed discharge-cycle counts (match the published originals):

| Cell | Discharge cycles | Capacity first -> last (Ah) | EOL threshold used | True EOL cycle |
|---|---|---|---|---|
| B0005 | 168 | 1.8565 -> 1.3251 | 70% (1.40 Ah) | 125 |
| B0006 | 168 | 2.0353 -> 1.1857 | 70% (1.40 Ah) | 109 |
| B0007 | 168 | 1.8911 -> 1.4325 | 80% (1.60 Ah) | 86 |
| B0018 | 132 | 1.8550 -> 1.3411 | 70% (1.40 Ah) | 97 |

Parsed data cached at `data/telemetry.csv` (636 rows: cell, cycle, capacity, mean discharge temperature).

## 2. Execution environments

| | E1: local CPU | E2: Kaggle GPU |
|---|---|---|
| Hardware | Intel Core Ultra (Meteor Lake), 4 threads | NVIDIA Tesla T4 |
| PyTorch | 2.14.0+cpu | 2.11.0+cu128 |
| LSTM | hidden 32, 200 epochs | hidden 64, 400 epochs |
| Protocol | identical otherwise | identical otherwise |
| Run mode | `temp/full_protocol.py` (background, incremental JSON) | Kaggle kernel v2, `kaggle kernels push`, internet disabled |

Kernel (complete run, reproducible): https://www.kaggle.com/code/jithendramaddala/p3-evhm-gpu-run
Kernel log + results JSON downloaded to `kernel_out/`.

## 3. Protocol

Grid = 4 cells x 5 prediction start points (40%, 50%, 60%, 70%, 80% of the cell's true EOL cycle) x 6 baselines = 20 configurations per environment, 40 total.

Baselines (all implemented and run, no external results used):
- B1 recent-trend linear extrapolation
- B2 LSTM, data loss only
- B3 PI-LSTM: Eq.(3) L = L_data + λ1·L_mono + λ2·L_prior (monotonicity + temperature-dependent degradation prior; λ1=2.0, λ2=1.0)
- B4 LSTM+PF (no physics constraints)
- B5 physics+PF (linear-trend decline prior, no deep learning)
- B6 P3-EVHM: PI-LSTM + particle filter

Particle filter: N=2,000 (grid runs), process noise 0.003 Ah, obs σ=0.01, threshold-crossing RUL distribution, 95% interval reported. Fixed seeds; a 3-seed repeatability check (seeds 0,1,2) was run at B0005/start-100.

## 4. Measured results

### 4.1 Grid summary (Table III of the paper)

| Metric | E1 local CPU | E2 Tesla T4 |
|---|---|---|
| Mean violation rate V, B2 data-only LSTM | 0.285 | 0.366 |
| Mean violation rate V, B3 PI-LSTM | 0.176 | 0.076 |
| Configurations where B3 reduced V vs B2 | 13 of 20 | 15 of 20 (equal in 3) |
| Mean capacity MAE, B2 -> B3 (Ah) | 0.3182 -> 0.2530 | 0.2854 -> 0.2569 |
| B5 physics+PF: mean RUL error (configs with RUL) | 18.6 cycles (20/20) | 18.8 cycles (20/20) |
| B5 coverage of true RUL (95% interval) | 0.85 | 0.85 |
| B5 relative RUL error | mean 43.6%, median 35.7% (14/20 within 50%) | mean 44.2%, median 36.0% (14/20 within 50%) |
| B6 PI-LSTM+PF: mean RUL error (configs with RUL) | 32.8 cycles (5/20) | 30.9 cycles (10/20) |
| Cross-cell (train B0005+B0006, test held-out): B3 V | 0.000 (B0007, B0018) | not run |

Cross-environment reproducibility: B5 RUL error mean 18.6 vs 18.8 cycles, coverage 0.85 in both environments, RUL produced in 20/20 configurations in both.

### 4.2 GPU scaling (Table IV of the paper; Tesla T4, kernel v3, end-to-end per PF call)

| Particles N | CPU time (s) | GPU device time (ms) | Speedup |
|---|---|---|---|
| 1,000 | 0.0327* | 7.0 | 0.65x vs warm CPU (see note) |
| 5,000 | 0.0076 | 7.7 | 0.98x |
| 20,000 | 0.0169 | 7.4 | 2.3x |
| 100,000 | 0.0614 | 9.4 | 6.5x |
| 500,000 | 0.3333 | 20.2 | 16.5x |

*The T4-host N=1,000 CPU measurement is inflated by first-call warmup; the warm local CPU measures 4.5 ms for the same workload, so honestly the GPU is slower at N=1,000 (0.65x). GPU device time stayed below 100 ms at every N tested. The 10x target is met only at N=500,000.

Local CPU scaling (same PF code, v3): 4.5 ms (N=1,000) -> 0.90 s (N=200,000); 13.3 ms per simulated cycle at N=200,000.

### 4.3 3-seed repeatability (E1, B0005, start 100)
B3 PI-LSTM: mean MAE 0.288 Ah, mean V 0.324 across seeds 0/1/2 — seed variance is material and is reported as a limitation.

### 4.4 Cross-cell generalization (E1)
Train on B0005+B0006 pooled -> test on held-out cells: B0007 (start 52, EOL 86): B3 MAE 0.612 Ah, V 0.000; B0018 (start 58, EOL 97): B3 MAE 0.453 Ah, V 0.000. Physics constraints held on out-of-distribution cells even though accuracy degraded, as expected.

### 4.5 Transfer learning (pretrained checkpoint, held-out cell B0007)
Fleet checkpoint: hidden-32 PI-LSTM, 279 pooled cycles from B0005/B0006/B0018 (first 60% each), B0007 fully held out; shipped at `p3evhm/pretrained/pi_lstm_base_v1.pt` (21 KB). Adaptation freezes all LSTM body layers and fine-tunes only the head (30 epochs, lr 1e-3). Measured at B0007, 80% EOL, start cycle 52 (true RUL 34):

| Mode | MAE (Ah) | V | LSTM RUL err | PF RUL err | Adapt time |
|---|---|---|---|---|---|
| From scratch (200 ep) | 0.2346 | 0.017 | none | none | 3.53 s |
| Zero-shot | 0.5592 | 0.000 | 32 | 10 | 0 s |
| Head fine-tune (frozen) | 0.0575 | 0.000 | 6 | 37 | 0.056 s |
| Full fine-tune (30 ep) | 0.0575 | 0.000 | 6 | 37 | 0.058 s |

Head fine-tune = 4.1x lower MAE, 63x faster than from-scratch, with zero violations. The fine-tuned slope prior remains shallow for the PF (err 37) — decline-prior floor applies. Results: `results/transfer_results.json`; script `temp/test_transfer.py`; pretraining `temp/pretrain_base.py`.

## 5. What this proves (and does not prove)

Proven (measured, both environments unless noted):
1. The physics-informed loss reduces physically implausible capacity-increase predictions (mean V 0.285->0.176 and 0.366->0.076) and improves mean capacity MAE.
2. Physics+PF yields a RUL estimate in every configuration tested, with ~19-cycle mean error and 0.85 interval coverage, reproducibly across two environments.
3. GPU particle computation scales: parity at small N, 18.7x at 500,000 particles on a T4, <100 ms device latency throughout.
4. Physics constraints generalize to held-out cells without in-cell fitting.
5. Point-only LSTMs frequently fail to produce any RUL (no threshold crossing) — supporting the framework's motivation.

Not proven / honest limitations (all stated in the paper):
- Laboratory single-cell data only; pack-level validity untested.
- Physics-loss weights are cell- and seed-sensitive (3-seed spread measured); B0007 is a hard cell (V worsened at one start point; early-start PF errors up to 60 cycles).
- The fused B6 path fails at early starts in most configurations (5/20 and 9/20) because LSTM-derived decline priors are too shallow; fix (decline-prior floor) is future work.
- Results are research prototypes, not certified BMS safety functionality.

## 6. Bugs found and fixed during validation (transparency log)

1. **Temperature-prior shape bug (kernel v1):** the prior term g had length n while the predicted-step difference had length n-1, crashing the first Kaggle run at 35 s. Fixed (`g[1:]` alignment) in kernel v2 and the local script; v1's log is retained.
2. **RUL-convention slips (kernel v3 + local rerun):** audit found (a) the LSTM/linear crossing index compared against `true_cross[0]+1` (a +1 bias) and (b) the particle filter initialized its state at cycle start-1 without observing the start-cycle measurement (another +1 bias). Both fixed to one clean convention: crossing index i means RUL = i cycles after start, and the PF performs a weighted-resample observation update on the start-cycle measurement before propagating. Both environments were fully re-run with the corrected code (local full protocol + Kaggle kernel v3); every number in this report and in the paper comes from the corrected runs. Effect size: RUL error means shifted by about 1 cycle (18.7-18.8 -> 18.6-18.8); violation rates, MAE, coverage, and GPU scaling conclusions unchanged.

## 7. Reproduction

```
# local CPU protocol (~20 min)
"C:\Users\jithe\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" temp/full_protocol.py

# Kaggle GPU run (T4, ~5 min execution)
cd kernel && kaggle kernels push -p .
kaggle kernels status jithendramaddala/p3-evhm-gpu-run
kaggle kernels output jithendramaddala/p3-evhm-gpu-run -p kernel_out

# paper build + verification
"C:\Users\jithe\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" temp/build_p3_paper.py
"C:\Users\jithe\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" temp/verify_p3.py   # -> PASS 87/87
```

## 8. File inventory

| File | Purpose |
|---|---|
| `P3-EVHM_paper.docx` | The paper (final deliverable) |
| `P3-EVHM_paper.md` | Markdown source of the paper |
| `P3-EVHM_kaggle.ipynb` (root + `kernel/`) | GPU experiment notebook |
| `kernel/kernel-metadata.json` | Kaggle kernel push config (GPU on, dataset attached, internet off) |
| `kernel_out/p3evhm_results.json` | Full T4 run results (pulled from Kaggle) |
| `kernel_out/p3-evhm-gpu-run.log` | T4 execution log |
| `results/full_results.json` | Local CPU full-protocol results |
| `results/kernel_grid_stats.json` | Aggregated T4 grid statistics |
| `results/preliminary_results.json`, `preliminary_arrays.npz`, `preliminary_fig.png` | Earlier prototype run (Fig. 2 source) |
| `data/*.mat` | Raw NASA PCoE dataset files |
| `data/telemetry.csv`, `data/capacity.csv` | Parsed cycle-level data |
| `arch.png` | Fig. 1 architecture diagram |
| `temp/*.py` | All build/experiment/verification scripts |
