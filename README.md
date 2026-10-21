# P3-EVHM — Parallel Physics-Informed Prognostics for EV Battery Health Management

A small, installable framework for lithium-ion battery **State of Health (SoH)** estimation and
**Remaining Useful Life (RUL)** prediction that combines three things the literature so far
publishes only separately:

1. **Physics-informed LSTM** — training loss = data error + monotonicity penalty + temperature-dependent degradation prior (Eq. 3–5 of the paper),
2. **Particle-filter RUL** — thousands of simulated futures → a full RUL *distribution* with a 95% prediction interval, not a single guess (Eq. 6–8),
3. **GPU parallelism** — the particle engine runs on CPU (NumPy) or CUDA (PyTorch); measured up to **16.5×** end-to-end speedup at 500,000 particles on a Tesla T4.

Verified on the real NASA Ames PCoE battery dataset: 40 measured configurations across two
compute environments — physics constraints cut physically implausible predictions by ~5×
(mean violation rate 0.366 → 0.076), and the physics+PF pipeline produced a RUL estimate in
**every one of the 20 configurations on both machines** (mean error ≈ 19 cycles, 95% interval
coverage 0.85). Full provenance: [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

## 30-second start (no data, no downloads)

```bash
pip install -e .
python -m p3evhm.demo                      # synthetic ageing cell, full B1-B6 benchmark
python -m tests.test_smoke                 # framework self-test
```

## With the real NASA PCoE data

```bash
# put B0005.mat ... B0018.mat anywhere under ./data (NASA PCoE Battery Data Set, Saha & Goebel 2007)
python -m p3evhm.demo --source nasa --cell B0005 --epochs 200
python -m p3evhm.demo --source nasa --cell B0007 --theta 80 --out results/my_run.json
streamlit run app.py                       # the live dashboard (6 tabs, self-explaining)
```

## Use it as a library

```python
from p3evhm import data, models, evaluate

df = data.synthetic_cell()                             # or data.load_nasa_mat("data")
q, T = df.capacity.values, df.temp.values
pred, train_s = models.pi_lstm_forecast(q[:100], T[:100], steps=60, lam1=2.0, lam2=1.0,
                                        epochs=200, return_train_time=True)
rul, lo, hi, pf_s = models.pf_rul(q[99], slope_mu=0.004, slope_sd=0.004,
                                  eol=1.4, steps=60, N=5000, obs0=q[100], device="auto")
grid = evaluate.run_grid(df, cells=["B0005"], thetas={"B0005": 70})
print(evaluate.summarize(grid))
```

`device="auto"` uses CUDA when available; `device="cuda"` forces GPU particle computation.

## Live dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Jithendra123892/p3-evhm&branch=main&mainModule=app.py)

Or locally: `streamlit run app.py`. The dashboard is self-explaining: tab 1 maps every
requirement (flowchart, DFD, math model, GPU parallel model, outputs, budget) to where it is
demonstrated, and includes the 5-agent architecture table (perception → decision → action).

## Repository layout

```
p3evhm/            the installable framework (data loaders, models, evaluate, demo CLI)
app.py             the 6-tab live dashboard (Streamlit)
tests/test_smoke.py  end-to-end self-test on synthetic data
VALIDATION_REPORT.md  data provenance (SHA-256), both environments, reproduction commands
P3-EVHM_paper.docx    the paper (IEEE two-column, 4 tables, 2 figures, 33 references)
kernel/               Kaggle GPU kernel (kernel-metadata.json + notebook)
```

## Honest scope

- Laboratory single-cell data (NASA PCoE); pack-level validity untested.
- The fused PI-LSTM+PF path fails at early prediction starts without a decline-prior floor (documented; future work).
- Research decision-support prototype — not a certified BMS safety function.

## Citation

If you use this framework, cite the paper: *Parallel Physics-Informed Prognostics for EV
Battery Health Management (P3-EVHM)* — A. V. Chowdhary, M. Jithendra, K. R (SRM IST, 2026).
GPU measurements: [Kaggle kernel](https://www.kaggle.com/code/jithendramaddala/p3-evhm-gpu-run).

## License

MIT — see [LICENSE](LICENSE).
