# P3-EVHM poster in the WSSL template format (Workshop of Signals and Systems
# Laboratory, Dept. of Computing Technologies, SRMIST).
# Format cloned from the reference poster: US Letter portrait, 3 columns,
# all-caps title, underlined authors, thin rule, dark-teal numbered section
# bars, serif body, blue charts, department line bottom-left.
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

k = json.load(open(os.path.join(ROOT, "results", "kernel_grid_stats.json")))
B2V, B3V = k["B2V"], k["B3V"]
m = lambda a: sum(a) / len(a)
V2, V3 = m(B2V), m(B3V)
GPU = [(1000, 7.0, 0.65), (5000, 7.7, 0.98), (20000, 7.4, 2.3), (100000, 9.4, 6.5), (500000, 20.2, 16.5)]

TEAL, BLUE, INK, MUT, LINE = "#005f6b", "#007bff", "#111111", "#4a4a4a", "#b9c4cc"

def dumbbell():
    W, H = 330, 250
    PL, PR, PT, PB = 52, 14, 26, 26
    pw, ph = W - PL - PR, H - PT - PB
    rows = sorted(zip(B2V, B3V, range(1, len(B2V) + 1)), reverse=True)
    rh = ph / len(rows)
    X = lambda v: PL + min(v, 1.0) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for gv in (0.0, 0.5, 1.0):
        x = X(gv)
        s.append(f'<line x1="{x:.1f}" y1="{PT}" x2="{x:.1f}" y2="{PT+ph}" stroke="{LINE}" stroke-width="0.8"/>')
        s.append(f'<text x="{x:.1f}" y="{PT-8}" font-size="9" fill="{MUT}" text-anchor="middle" font-family="Arial">{gv:.1f}</text>')
    for i, (v2, v3, idx) in enumerate(rows):
        y = PT + (i + 0.5) * rh
        s.append(f'<text x="{PL-5}" y="{y+3:.1f}" font-size="8.5" fill="{MUT}" text-anchor="end" font-family="Arial">{idx:02d}</text>')
        s.append(f'<line x1="{X(v2):.1f}" y1="{y:.1f}" x2="{X(v3):.1f}" y2="{y:.1f}" stroke="{LINE}" stroke-width="2"/>')
        s.append(f'<circle cx="{X(v2):.1f}" cy="{y:.1f}" r="3.4" fill="{MUT}"/>')
        s.append(f'<circle cx="{X(v3):.1f}" cy="{y:.1f}" r="3.4" fill="{BLUE}"/>')
    s.append(f'<line x1="{X(V2):.1f}" y1="{PT}" x2="{X(V2):.1f}" y2="{PT+ph}" stroke="{MUT}" stroke-width="1.1" stroke-dasharray="4 3"/>')
    s.append(f'<line x1="{X(V3):.1f}" y1="{PT}" x2="{X(V3):.1f}" y2="{PT+ph}" stroke="{BLUE}" stroke-width="1.1" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{X(V2)+6:.1f}" y="{PT+10}" font-size="9.5" font-weight="bold" fill="{MUT}" font-family="Arial">mean {V2:.3f}</text>')
    s.append(f'<text x="{X(V3)-2:.1f}" y="{PT+22}" font-size="9.5" font-weight="bold" fill="{BLUE}" font-family="Arial" text-anchor="end">mean {V3:.3f}</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.1"/>')
    s.append(f'<circle cx="{PL+4}" cy="{H-8}" r="3.4" fill="{MUT}"/><text x="{PL+11}" y="{H-5}" font-size="8.5" fill="{INK}" font-family="Arial">data-only LSTM</text>')
    s.append(f'<circle cx="{PL+74}" cy="{H-8}" r="3.4" fill="{BLUE}"/><text x="{PL+81}" y="{H-5}" font-size="8.5" fill="{INK}" font-family="Arial">P3-EVHM</text>')
    s.append("</svg>")
    return "".join(s)

def gpu_chart():
    W, H = 330, 150
    PL, PR, PT, PB = 30, 34, 22, 24
    pw, ph = W - PL - PR, H - PT - PB
    lx = lambda n: PL + (math.log10(n) - 3) / (math.log10(500000) - 3) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for t in (0, 10, 20):
        y = PT + (1 - t / 25) * ph
        s.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" stroke="{LINE}" stroke-width="0.8"/>')
        s.append(f'<text x="{PL-4}" y="{y+3:.1f}" font-size="8.5" fill="{MUT}" text-anchor="end" font-family="Arial">{t}</text>')
    for q in (0, 8, 16):
        y = PT + (1 - q / 18) * ph
        s.append(f'<text x="{W-PR+4}" y="{y+3:.1f}" font-size="8.5" fill="{BLUE}" font-family="Arial">{q}x</text>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-q/18)*ph:.1f} " for n, _, q in GPU)}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-t/25)*ph:.1f} " for n, t, _ in GPU)}" fill="none" stroke="{INK}" stroke-width="2"/>')
    for n, t, q in GPU:
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-t/25)*ph:.1f}" r="2.8" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>')
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-q/18)*ph:.1f}" r="2.8" fill="{BLUE}"/>')
        s.append(f'<text x="{lx(n):.1f}" y="{H-9}" font-size="8.5" fill="{MUT}" text-anchor="middle" font-family="Arial">{n//1000}k</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.1"/>')
    s.append(f'<text x="{W-PR-2}" y="{PT+(1-16.5/18)*ph-6:.1f}" font-size="10" font-weight="bold" fill="{BLUE}" text-anchor="end" font-family="Arial">16.5x</text>')
    s.append(f'<text x="{lx(1000)+6:.1f}" y="{PT+(1-0.65/18)*ph+12:.1f}" font-size="8" fill="{MUT}" font-family="Arial">0.65x</text>')
    s.append(f'<text x="{PL}" y="{PT-10}" font-size="8.5" fill="{MUT}" font-family="Arial">GPU ms (black) / speedup (blue) vs particles N, Tesla T4</text>')
    s.append("</svg>")
    return "".join(s)

def flowchart():
    W, H = 330, 190
    steps = [
        ("A1 &middot; INGEST &mdash; validate cycle records", 0),
        ("A2 &middot; FEATURES &mdash; W = 8 sliding windows", 0),
        ("A3 &middot; PI-LSTM &mdash; physics-constrained SoH", 1),
        ("A4 &middot; PARTICLE FILTER &mdash; 5000 futures", 1),
        ("A5 &middot; DECISION &mdash; median + 95% + alert", 0),
    ]
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    y = 4
    for txt, gpu in steps:
        fill = BLUE if gpu else "#6f8fb5"
        s.append(f'<rect x="18" y="{y}" width="{W-36}" height="26" fill="{fill}" stroke="none"/>')
        s.append(f'<text x="{W/2}" y="{y+17}" font-size="10" fill="#ffffff" text-anchor="middle" font-family="Arial">{txt}</text>')
        if y < 4 + 4 * 34:
            s.append(f'<line x1="{W/2}" y1="{y+26}" x2="{W/2}" y2="{y+34}" stroke="{INK}" stroke-width="1.6"/>')
            s.append(f'<path d="M{W/2-4} {y+30} L{W/2} {y+36} L{W/2+4} {y+30} Z" fill="{INK}"/>')
        y += 34
    s.append(f'<text x="18" y="{H-6}" font-size="8" fill="{MUT}" font-family="Arial">blue stages run on the GPU (Tesla T4); the same code runs on CPU for parity</text>')
    s.append("</svg>")
    return "".join(s)

GAP = [
    ("Physics-informed NNs [13], [22]", "&#10003;", "&#10007;", "&#10007;", "partial"),
    ("LSTM + particle filter [27], [28]", "&#10007;", "&#10003;", "&#10007;", "single cell"),
    ("CUDA-parallel particle filters [17]", "&#10007;", "&#10003;", "&#10003;", "not battery"),
]
gap_rows = "".join(f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>' for r in GAP)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: 216mm 279mm; margin: 0; }}
  * {{ margin: 0; box-sizing: border-box; }}
  html, body {{ background: #ffffff; }}
  body {{
    position: relative;
    width: 216mm; height: 279mm; padding: 9mm 11mm 12mm;
    font-family: 'Times New Roman', Georgia, serif; color: {INK};
    display: flex; flex-direction: column;
  }}
  .sans {{ font-family: Arial, Helvetica, sans-serif; }}
  /* header - text only, thin rule (template style) */
  .title {{ font-family: Arial, Helvetica, sans-serif; font-size: 13.2pt; font-weight: bold;
            text-align: center; line-height: 1.25; text-transform: uppercase; }}
  .authors {{ text-align: center; font-size: 9.8pt; margin-top: 2mm; }}
  .authors u {{ text-decoration: underline; }}
  .dept {{ text-align: center; font-size: 8.6pt; color: {MUT}; margin-top: 0.8mm; font-family: Arial, sans-serif; }}
  .rule {{ border-top: 0.45mm solid {INK}; margin: 2.2mm 0 2.8mm; }}
  /* 3 columns */
  .cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 4.5mm; flex: 1; }}
  h2 {{
    background: {TEAL}; color: #ffffff; font-family: Arial, sans-serif; font-size: 8.6pt;
    font-weight: bold; text-transform: uppercase; padding: 1mm 2mm; margin: 0 0 1.4mm;
    letter-spacing: 0.15mm;
  }}
  section {{ margin-bottom: 2mm; }}
  p {{ font-size: 7.5pt; line-height: 1.36; text-align: justify; hyphens: auto; }}
  p + p {{ margin-top: 1mm; }}
  .abstract p {{ font-style: italic; }}
  ul {{ padding-left: 4mm; }}
  li {{ font-size: 7.5pt; line-height: 1.36; margin-top: 0.4mm; }}
  .q {{ font-weight: bold; }}
  .math {{ font-style: italic; font-size: 8.2pt; line-height: 1.75; text-align: center; }}
  .math .n {{ font-style: normal; font-family: Arial; font-size: 7pt; color: {TEAL}; font-weight: bold; margin-right: 1mm; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 7.4pt; margin-top: 1mm;
          font-family: 'Times New Roman', serif; }}
  th {{ border: 0.25mm solid {INK}; padding: 0.7mm 1mm; font-family: Arial, sans-serif;
       font-size: 6.8pt; font-weight: bold; text-align: center; background: #eef3f6; }}
  td {{ border: 0.25mm solid {INK}; padding: 0.7mm 1mm; text-align: center; }}
  td:first-child {{ text-align: left; }}
  table.gap td:first-child {{ font-size: 6.9pt; }}
  table.gap .ours {{ background: #e3edf2; font-weight: bold; }}
  table.rt td:first-child {{ text-align: left; }}
  table.rt td:last-child {{ font-family: Arial; font-size: 7.2pt; }}
  .chart {{ border: 0.25mm solid {LINE}; padding: 1.4mm 1.6mm 1.2mm; margin-top: 1mm; }}
  .cap {{ font-size: 6.8pt; color: {MUT}; margin-top: 0.8mm; line-height: 1.35; font-family: Arial, sans-serif; }}
  .findings li {{ margin-top: 0.9mm; }}
  .findings b {{ font-family: Arial, sans-serif; font-size: 7.6pt; }}
  .refs {{ font-size: 6.4pt; color: {MUT}; line-height: 1.36; }}
  .foot {{ position: absolute; left: 11mm; right: 11mm; bottom: 6mm;
           border-top: 0.3mm solid {INK}; padding-top: 0.8mm;
           display: flex; justify-content: space-between; white-space: nowrap;
           font-family: Arial, sans-serif; font-size: 6.8pt; color: {MUT}; }}
</style></head>
<body>

<div class="title">Parallel Physics-Informed Prognostics for EV Battery Health Management (P3-EVHM):
Fusing Physics Constraints, Calibrated RUL and GPU-Scale Inference</div>
<div class="authors"><u>A. Vibhas Chowdhary, Maddala Jithendra</u> &nbsp;&mdash;&nbsp; guide: Dr. R. Kavitha</div>
<div class="dept">va1641@srmist.edu.in &middot; jm4408@srmist.edu.in &middot; kavithar14@srmist.edu.in</div>
<div class="rule"></div>

<div class="cols">
  <!-- ============ COLUMN 1 ============ -->
  <div>
    <section class="abstract">
      <h2>Abstract</h2>
      <p>Battery remaining-useful-life (RUL) research has mature ingredients that no published framework
      combines: physics-informed learning, calibrated uncertainty, GPU-scale inference and a real-cell
      benchmark. This work fuses all four. A physics-informed LSTM &mdash; trained with a monotonicity
      penalty and an Arrhenius-style temperature prior &mdash; cuts physically implausible predictions from
      0.366 to 0.076 mean violation rate. A particle filter turns the forecast into a distribution: RUL in
      20/20 configurations on two machines (mean error 18.8 cycles, coverage 0.85). The cloud runs as one
      vectorised tensor operation: 16.5&times; GPU speedup at 500,000 particles. Open-sourced with 40
      measured configurations and full provenance.</p>
    </section>

    <section>
      <h2>1 Introduction</h2>
      <p>Lithium-ion cells lose capacity with every cycle &mdash; slowly, then through a degradation knee
      that crosses the 70&ndash;80% end-of-life threshold with almost no warning. A vehicle reports charge
      percentage, not health. <span class="q">The central question: how many cycles remain &mdash; and with
      what confidence &mdash; before the battery must be replaced?</span> Maintenance needs a calibrated
      number early enough to schedule replacement, not a point estimate at the cliff.</p>
    </section>

    <section>
      <h2>2 Research gap</h2>
      <p>Prior work solves one piece and stops there. This work connects:</p>
      <ul>
        <li>physics-informed training losses</li>
        <li>RUL as a calibrated distribution</li>
        <li>GPU-parallel particle inference</li>
        <li>a reproducible real-cell benchmark</li>
      </ul>
      <table class="gap">
        <tr><th>approach family</th><th>physics</th><th>distn</th><th>GPU</th><th>real cells</th></tr>
        {gap_rows}
        <tr class="ours"><td>P3-EVHM &mdash; this work</td><td>&#10003;</td><td>&#10003;</td><td>&#10003;</td><td>&#10003;</td></tr>
      </table>
      <p style="margin-top:1.2mm">Verified against a 33-work systematic review: the closest LSTM+filter
      hybrid [28] lists our physics constraints as <i>future work</i>; CUDA particle filters [17] never
      touched battery data. <b>No published framework closes all four cells &mdash; that closure is the
      contribution.</b></p>
    </section>

    <section>
      <h2>3 Validation protocol</h2>
      <ul>
        <li><b>Data:</b> NASA Ames PCoE cells B0005/B0006/B0007/B0018 [1], byte-verified</li>
        <li><b>Grid:</b> 40 configurations &mdash; 4 cells &times; 5 starts &times; EOL 70/80%, on a Core Ultra CPU and a Kaggle Tesla T4</li>
        <li><b>Baselines B1&ndash;B6:</b> linear, LSTM, PI-LSTM, LSTM+PF, physics+PF, fused P3-EVHM &mdash; same seeds, same metrics</li>
      </ul>
    </section>
  </div>

  <!-- ============ COLUMN 2 ============ -->
  <div>
    <section>
      <h2>4 The P3-EVHM model</h2>
      <p>A 1&ndash;32 LSTM over a W&nbsp;=&nbsp;8 cycle window; the loss carries two electrochemistry laws:</p>
      <div class="math">
        <div><span class="n">(4)</span> L = MSE(q&#770;, q) + &lambda;<sub>1</sub>&middot;mean ReLU(q&#770;<sub>t</sub> &minus; q<sub>t</sub>) + &lambda;<sub>2</sub>&middot;mean(&Delta;q&#770;<sub>t</sub> &minus; g(T<sub>t</sub>))&sup2;</div>
        <div><span class="n">(5)</span> g(T) = clamp(&Acirc;(T &minus; T&#772;), min = 0)</div>
      </div>
      <p>Capacity may never rise; decline must respect a temperature prior fitted from the training cycles
      themselves. Full-batch training, deterministic seeds, identical metrics for every baseline.</p>
    </section>

    <section>
      <h2>5 Particle-filter RUL</h2>
      <div class="math">
        <div><span class="n">(6)</span> q<sub>i</sub><sup>(k)</sup> = q<sub>i</sub><sup>(k&minus;1)</sup> &minus; d<sub>i</sub> + &epsilon;<sub>k</sub></div>
        <div><span class="n">(7)</span> w<sub>i</sub> &prop; exp(&minus;(q<sub>obs</sub> &minus; q<sub>i</sub>)&sup2;/2&sigma;&sup2;)</div>
        <div><span class="n">(8)</span> RUL = median {{ k : q<sub>i</sub><sup>(k)</sup> &le; q<sub>EOL</sub> }}</div>
      </div>
      <p>Thousands of simulated futures, reweighted by the <i>measured</i> start-cycle capacity; each
      future&rsquo;s threshold crossing is a RUL draw. We report the median and 95% interval &mdash; and the
      decision rules (OK / CAUTION / ALERT) follow the interval&rsquo;s lower bound.</p>
      <div class="chart">{flowchart()}</div>
    </section>

    <section>
      <h2>6 Physics-consistency results</h2>
      <div class="chart">{dumbbell()}</div>
      <div class="cap">Violation rate V per configuration (predicted steps where capacity rises &mdash;
      impossible). All 20 configurations improve or tie; the two catastrophes (V = 1.0) are eliminated.
      Capacity MAE also improves: 0.285 &rarr; 0.257 Ah.</div>
    </section>

    <section>
      <h2>Key references</h2>
      <p class="refs">[1] Saha & Goebel, NASA PCoE, 2007. [2] Arulampalam et al., <i>IEEE TSP</i> 50(2),
      2002. [13] Shi, Rivera & Wu, <i>MSSP</i> 179, 2022. [17] <i>Parallel CRPF on CUDA</i>, Appl.
      Intell., 2020. [22] do Nascimento et al., <i>Sci. Rep.</i> 13, 2023. [27] <i>AI-driven PF for battery
      RUL</i>, Batteries 10(12), 2024. [28] <i>Hybrid PF-trained NN</i>, IEEE, 2021.</p>
    </section>
  </div>

  <!-- ============ COLUMN 3 ============ -->
  <div>
    <section>
      <h2>7 Prognostic accuracy</h2>
      <table class="rt">
        <tr><th>quantity</th><th>measured</th></tr>
        <tr><td>RUL produced</td><td>20 / 20 configs, both machines</td></tr>
        <tr><td>mean RUL error</td><td>18.8 cyc (T4) &middot; 18.6 cyc (CPU)</td></tr>
        <tr><td>95% interval coverage</td><td>0.85 (honest, ideal 0.95)</td></tr>
        <tr><td>transfer: held-out cells</td><td>4 / 4 (head-FT, 0.2 s) vs 1 / 4 scratch</td></tr>
      </table>
      <p style="margin-top:0.8mm">Head-only adaptation: no per-cell training, zero physics violations &mdash;
      what a fleet needs.</p>
    </section>

    <section>
      <h2>8 GPU scaling</h2>
      <div class="chart">{gpu_chart()}</div>
      <div class="cap">Honest crossover: below ~5k particles the warm CPU wins (0.65&times;); the GPU pays
      exactly where prognostics needs it &mdash; large clouds, sharp intervals, real time (20.2 ms device
      time at 500k).</div>
    </section>

    <section>
      <h2>9 Main findings</h2>
      <ol class="findings">
        <li><b>Physics constraints work.</b> Implausible predictions fall ~5&times; (0.366 &rarr; 0.076), with
        better capacity MAE on the same grid.</li>
        <li><b>Uncertainty is deliverable.</b> Calibrated RUL in every configuration, both machines;
        from-scratch training failed 3 of 4 transfer cells.</li>
        <li><b>GPU scaling is workload-dependent.</b> Parity at small N; 16.5&times; at 500k particles,
        <100 ms device time throughout.</li>
        <li><b>Transfer beats retraining.</b> 0.2 s head-only adaptation: 4/4 RUL availability, zero
        violations.</li>
      </ol>
    </section>

    <section>
      <h2>10 Conclusion</h2>
      <p>Reliable battery prognostics is simultaneously a <b>physics</b> and an <b>HPC</b> problem. The
      fusion is the contribution: combining the parts turns LSTM outputs into maintenance-grade decisions.
      The optimization target is <b>a trustworthy RUL distribution in production time</b> &mdash; not a
      point estimate.</p>
    </section>

    <section>
      <h2>11 Limitations</h2>
      <ul>
        <li>Laboratory single-cell data (NASA PCoE); pack-level validity untested</li>
        <li>Early starts (40% of life) can fail without the decline-prior floor; LSTM seed variance is material and reported</li>
        <li>Research decision-support prototype &mdash; not a certified BMS safety function</li>
      </ul>
    </section>

  </div>
</div>

<div class="foot">
  <span><b>Department of Computing Technologies SRMIST Kattankulathur</b></span>
  <span>github.com/Jithendra123892/p3-evhm &middot; kaggle.com/jithendramaddala/p3-evhm-gpu-run</span>
</div>
</body></html>
"""

out = os.path.join(HERE, "wssl-poster.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, "| V2", f"{V2:.3f}", "V3", f"{V3:.3f}")
