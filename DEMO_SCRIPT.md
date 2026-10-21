# P3-EVHM — Demo Script for Mam (5-minute working-application walkthrough)

## Start the application (do this before she arrives)

Open Command Prompt (or the terminal you use) and run:

```
cd C:\Users\jithe\p3-evhm-paper
C:\Users\jithe\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m streamlit run app.py --server.port 8599
```

A browser tab opens at http://localhost:8599.

**Important:** the very first load takes 1-2 minutes (the AI libraries import and the first model trains live on CPU). Every action after that is instant because results are cached. If you want it already open when mam arrives, start it 3 minutes early and click through one prediction yourself first.

## The 5-minute walkthrough

### Step 1 — Live Prediction tab (opens by default)
Point at the four metric cards on top:
- **SoH at start cycle** — current State of Health (%) and its state label (Healthy / Ageing / Replace soon)
- **Predicted RUL (median)** — remaining useful life in cycles, with the true value shown for comparison
- **95% prediction interval** — the uncertainty band (this is the particle filter's output — we predict a distribution, not one number)
- **Physics violation rate V** — fraction of predicted steps that break physics (capacity going UP). Say: "the data-only LSTM violates physics; our physics-informed version cuts this about 5x."

The colour band under the cards is the live alert: red ALERT / yellow CAUTION / green OK based on the lower bound of the RUL interval. Move the **Prediction start cycle** slider and watch the alert logic react — this is the maintenance-decision story.

The chart below: black = measured capacity, red dashed = EOL threshold (70% = 1.40 Ah), green = PI-LSTM, orange = plain LSTM, purple dashed = particle-filter median trajectory.

### Step 2 — Change the battery cell
In the left sidebar switch the cell to **B0007** — the app automatically switches to the 80% threshold (this cell never reaches 70%), retrains, and shows a different RUL/interval. Say: "the pipeline adapts per cell; all four NASA cells are validated."

### Step 3 — Validation Grid tab
40 measured runs (20 on this laptop's CPU + 20 on an NVIDIA Tesla T4 GPU on Kaggle). Key line: physics constraint cut implausible predictions from 0.366 to 0.076 mean on the T4 and produced RUL estimates in 20/20 configurations on both machines. Say: "we ran the same protocol on two different computers and the result reproduced — 18.6 vs 18.8 cycles mean RUL error."

### Step 4 — GPU Scaling tab
Measured speedup table: no speedup at small particle counts (0.98x — honest), rising to **16.5x at 500,000 particles** (20 ms per prediction window). Say: "the GPU is not decoration — we measured where it helps and where it doesn't, and it matches the published literature on kernel-launch overhead."

### Step 5 — About tab
The research gap in one sentence, with the citation-backed claim: no published framework combines physics-informed LSTM + particle filter + GPU-parallel particle computation; the two closest papers explicitly list physics constraints as future work they did not do.

## Likely questions and short answers

- **"What is the accuracy?"** — RUL median error ~35% relative (best near EOL: 2-9 cycles; early starts worse). Capacity MAE improves 0.29 -> 0.26 Ah with the physics loss. Published specialists with richer features report better raw accuracy; our contribution is the integration (physics consistency + uncertainty + GPU scaling), which none of them combine.
- **"Why is the linear baseline so good?"** — capacity fade is locally near-linear, so a 20-point trend is genuinely strong; that's why we report it as a baseline instead of hiding it.
- **"Does it work in a real car?"** — the behaviour (physics constraints, intervals, ms-latency) transfers; accuracy would degrade on drive-cycle pack data until we add incremental-capacity features and re-fit priors. It is a research decision-support prototype, not a certified BMS — stated in the paper.
- **"Where did the data come from?"** — NASA Ames PCoE Battery Data Set, downloaded from two independent sources and verified byte-for-byte (SHA-256 logged in VALIDATION_REPORT.md).
- **"Is the GPU result real?"** — yes: Kaggle kernel `jithendramaddala/p3-evhm-gpu-run`, Tesla T4, results JSON pulled back and included in the paper's Table IV.

## If something looks stuck
- First load slow → wait; the spinner says "Running train_forecast(...)". Cached afterwards.
- If the browser tab errors → close it, re-run the start command (the cache survives within the session).
- Everything is reproducible: VALIDATION_REPORT.md lists every command.

## Files to have open on the laptop
1. The dashboard (browser tab)
2. `P3-EVHM_paper.docx` — the paper (Section VIII has all measured results, Tables III-IV, Figs. 1-2)
3. `VALIDATION_REPORT.md` — provenance and reproduction commands
