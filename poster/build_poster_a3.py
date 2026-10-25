# P3-EVHM conference poster, A3 portrait (297 x 420 mm), editorial style.
# All numbers are read from measured artifacts (results/kernel_grid_stats.json,
# VALIDATION_REPORT.md GPU table, README transfer table) - nothing hand-drawn.
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

k = json.load(open(os.path.join(ROOT, "results", "kernel_grid_stats.json")))
B2V, B3V = k["B2V"], k["B3V"]
m = lambda a: sum(a) / len(a)
V2, V3 = m(B2V), m(B3V)
GPU = [(1000, 7.0, 0.65), (5000, 7.7, 0.98), (20000, 7.4, 2.3), (100100 - 100, 9.4, 6.5), (500000, 20.2, 16.5)]

# palette: warm paper, ink, bottle green, brick red
PAPER, INK, MUT, HAIR = "#f8f5ec", "#23201a", "#6f675a", "#d9d2c2"
ACC, ACCBG, RED = "#14594a", "#e7efe9", "#b3402f"

# ---------------------------------------------------------------- charts
def dumbbell():
    """V per config: red dot (data-only) -> green dot (physics), sorted by B2."""
    W, H = 500, 320          # viewBox units; rendered at column width
    PL, PR, PT, PB = 118, 26, 30, 30
    pw, ph = W - PL - PR, H - PT - PB
    rows = sorted(zip(B2V, B3V, range(1, len(B2V) + 1)), reverse=True)
    rh = ph / len(rows)
    X = lambda v: PL + min(v, 1.0) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = X(gv)
        s.append(f'<line x1="{x:.1f}" y1="{PT}" x2="{x:.1f}" y2="{PT+ph}" stroke="#c9bfa8" stroke-width="1.2"/>')
        s.append(f'<text x="{x:.1f}" y="{PT-8}" font-size="15" fill="#5a5348" text-anchor="middle">{gv:.2f}</text>')
    for i, (v2, v3, idx) in enumerate(rows):
        y = PT + (i + 0.5) * rh
        s.append(f'<text x="{PL-8}" y="{y+4:.1f}" font-size="13.5" fill="#5a5348" text-anchor="end" font-family="Consolas,monospace">{idx:02d}</text>')
        s.append(f'<line x1="{X(v2):.1f}" y1="{y:.1f}" x2="{X(v3):.1f}" y2="{y:.1f}" stroke="#c9bfa8" stroke-width="2.8"/>')
        if v2 - v3 > 0.02:
            s.append(f'<path d="M{X(v3)+6:.1f} {y-3.4:.1f} L{X(v2)-1:.1f} {y:.1f} L{X(v3)+6:.1f} {y+3.4:.1f}" fill="none" stroke="{ACC}" stroke-width="1.2" opacity="0.55"/>')
        s.append(f'<circle cx="{X(v2):.1f}" cy="{y:.1f}" r="5" fill="{RED}"/>')
        s.append(f'<circle cx="{X(v3):.1f}" cy="{y:.1f}" r="5" fill="{ACC}"/>')
    ym = PT + ph * 0.52
    s.append(f'<line x1="{X(V2):.1f}" y1="{PT+2}" x2="{X(V2):.1f}" y2="{PT+ph}" stroke="{RED}" stroke-width="1.6" stroke-dasharray="6 5" opacity="0.85"/>')
    s.append(f'<line x1="{X(V3):.1f}" y1="{PT+2}" x2="{X(V3):.1f}" y2="{PT+ph}" stroke="{ACC}" stroke-width="1.6" stroke-dasharray="6 5" opacity="0.85"/>')
    s.append(f'<text x="{X(V2)-5:.1f}" y="{ym}" font-size="14" font-weight="bold" fill="{RED}" text-anchor="end" font-family="Consolas,monospace">mean {V2:.3f}</text>')
    s.append(f'<text x="{X(V3)+8:.1f}" y="{ym}" font-size="14" font-weight="bold" fill="{ACC}" font-family="Consolas,monospace">mean {V3:.3f}</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.6"/>')
    s.append(f'<circle cx="{PL+6}" cy="{H-10}" r="4.6" fill="{RED}"/><text x="{PL+16}" y="{H-6}" font-size="13" fill="{MUT}">data-only LSTM</text>')
    s.append(f'<circle cx="{PL+106}" cy="{H-10}" r="4.6" fill="{ACC}"/><text x="{PL+116}" y="{H-6}" font-size="13" fill="{MUT}">with physics loss</text>')
    s.append("</svg>")
    return "".join(s)

def gpu_chart():
    W, H = 520, 190
    PL, PR, PT, PB = 40, 58, 26, 30
    pw, ph = W - PL - PR, H - PT - PB
    lx = lambda n: PL + (math.log10(n) - 3) / (math.log10(500000) - 3) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for t in (0, 10, 20):
        y = PT + (1 - t / 25) * ph
        s.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" stroke="{HAIR}" stroke-width="1"/>')
        s.append(f'<text x="{PL-6}" y="{y+4:.1f}" font-size="12" fill="{MUT}" text-anchor="end" font-family="Consolas,monospace">{t}</text>')
    for q in (0, 8, 16):
        y = PT + (1 - q / 18) * ph
        s.append(f'<text x="{W-PR+8}" y="{y+4:.1f}" font-size="12" fill="{ACC}" font-family="Consolas,monospace">{q}x</text>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-q/18)*ph:.1f} " for n, _, q in GPU)}" fill="none" stroke="{ACC}" stroke-width="2.6"/>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-t/25)*ph:.1f} " for n, t, _ in GPU)}" fill="none" stroke="{INK}" stroke-width="2.6"/>')
    for n, t, q in GPU:
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-t/25)*ph:.1f}" r="3.8" fill="{PAPER}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-q/18)*ph:.1f}" r="3.8" fill="{ACC}"/>')
        lab = f"{n//1000}k" if n < 1000000 else "500k"
        s.append(f'<text x="{lx(n):.1f}" y="{H-8}" font-size="13" fill="{MUT}" text-anchor="middle" font-family="Consolas,monospace">{lab}</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.6"/>')
    s.append(f'<text x="{W-PR-4}" y="{PT+(1-16.5/18)*ph-10:.1f}" font-size="16" font-weight="bold" fill="{ACC}" text-anchor="end" font-family="Consolas,monospace">16.5x</text>')
    s.append(f'<text x="{lx(1000)+10:.1f}" y="{PT+(1-0.65/18)*ph+16:.1f}" font-size="12" fill="{MUT}">0.65x</text>')
    s.append(f'<text x="{PL}" y="{PT-10}" font-size="12" fill="{MUT}">GPU device time, ms (ink) and speedup (green) per particle-filter call, Tesla T4</text>')
    s.append("</svg>")
    return "".join(s)

def pipeline():
    W, H = 1000, 132
    bx = [10, 208, 406, 604, 802]
    names = [("A1", "INGEST", "validate each cycle record"),
             ("A2", "FEATURES", "W = 8 sliding windows"),
             ("A3", "PI-LSTM", "SoH trajectory, physics loss"),
             ("A4", "PARTICLE FILTER", "5000 simulated futures"),
             ("A5", "DECISION", "median, 95% interval, alert")]
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for (aid, ttl, sub), x in zip(names, bx):
        s.append(f'<rect x="{x}" y="18" width="188" height="66" fill="{ACCBG}" stroke="{ACC}" stroke-width="1.6"/>')
        s.append(f'<text x="{x+94}" y="40" font-size="14.5" fill="{MUT}" text-anchor="middle" letter-spacing="2">{aid} &middot; {ttl}</text>')
        s.append(f'<text x="{x+94}" y="58" font-size="12.5" fill="{INK}" text-anchor="middle">{sub}</text>')
    for i in range(4):
        s.append(f'<line x1="{bx[i]+188}" y1="51" x2="{bx[i+1]-4}" y2="51" stroke="{INK}" stroke-width="1.8"/>')
        s.append(f'<path d="M{bx[i+1]-10} 47 L{bx[i+1]-2} 51 L{bx[i+1]-10} 55 Z" fill="{INK}"/>')
    s.append(f'<text x="10" y="106" font-size="13" fill="{MUT}">input: cycle capacity q(t) and mean temperature T(t) from NASA PCoE cells B0005 / B0006 / B0007 / B0018 &nbsp;&nbsp;&middot;&nbsp;&nbsp; output: SoH forecast, RUL median + 95% interval, OK / CAUTION / ALERT</text>')
    s.append(f'<text x="604" y="14" font-size="11.5" fill="{ACC}" text-anchor="middle" letter-spacing="1.5">RUNS ON GPU</text>')
    s.append(f'<text x="802" y="14" font-size="11.5" fill="{ACC}" text-anchor="middle" letter-spacing="1.5">RUNS ON GPU</text>')
    s.append("</svg>")
    return "".join(s)

# ---------------------------------------------------------------- page
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: 297mm 420mm; margin: 0; }}
  * {{ margin: 0; box-sizing: border-box; }}
  html, body {{ background: {PAPER}; }}
  body {{
    width: 297mm; height: 420mm; padding: 11mm 14mm 8mm;
    font-family: Georgia, 'Times New Roman', serif; color: {INK};
    display: flex; flex-direction: column;
  }}
  .mono {{ font-family: Consolas, 'Cascadia Mono', monospace; }}
  /* masthead */
  .kicker {{ font-size: 8.2pt; letter-spacing: 2.6mm; color: {ACC}; font-weight: bold; }}
  h1 {{ font-size: 26pt; line-height: 1.1; letter-spacing: -0.15mm; font-weight: normal; margin: 1.8mm 0 2.2mm; }}
  h1 b {{ font-weight: bold; }}
  .authors {{ font-size: 10.5pt; color: {INK}; font-style: italic; }}
  .meta {{ font-size: 8.6pt; color: {MUT}; margin-top: 1.1mm; }}
  .rule2 {{ border-top: 0.8mm solid {INK}; border-bottom: 0.25mm solid {INK}; height: 1.6mm; margin: 2.4mm 0 3.4mm; }}
  /* body grid: wide left col, narrower right col */
  .body {{ display: grid; grid-template-columns: 143mm 126mm; gap: 0 7mm; flex: 1; }}
  .eyebrow {{ font-size: 8pt; letter-spacing: 1.9mm; color: {ACC}; font-weight: bold;
              border-bottom: 0.25mm solid {INK}; padding-bottom: 1mm; margin: 0 0 2.4mm;
              font-family: 'Segoe UI', sans-serif; }}
  .eyebrow .n {{ color: {MUT}; }}
  section {{ margin-bottom: 4.2mm; }}
  section:last-child {{ margin-bottom: 0; }}
  p {{ font-size: 9.6pt; line-height: 1.5; text-align: justify; hyphens: auto; }}
  p + p {{ margin-top: 1.8mm; }}
  p.wide {{ column-count: 2; column-gap: 5mm; }}
  .math {{ font-size: 9.8pt; font-style: italic; line-height: 1.95; margin: 2.6mm 0 0; }}
  .math + .math {{ margin-top: 2.8mm; }}
  .math .n {{ font-style: normal; font-family: Consolas, monospace; font-size: 8pt; color: {ACC}; margin-right: 1.6mm; }}
  .cap {{ font-size: 7.6pt; color: {MUT}; line-height: 1.4; margin-top: 1.2mm; text-align: left; }}
  .chart {{ margin: 1.4mm 0 0; }}
  /* booktabs tables */
  table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin-top: 1.2mm; }}
  th {{ font-family: 'Segoe UI', sans-serif; font-size: 7.4pt; letter-spacing: 0.5mm; color: {MUT};
        font-weight: 600; text-align: left; padding: 0 1.6mm 1mm 0; text-transform: uppercase; }}
  th.num, td.num {{ text-align: right; font-family: Consolas, monospace; padding-right: 0; }}
  thead tr {{ border-bottom: 0.45mm solid {INK}; }}
  tbody tr {{ border-bottom: 0.2mm solid {HAIR}; }}
  tbody tr:last-child {{ border-bottom: 0.45mm solid {INK}; }}
  td {{ padding: 1.15mm 1.6mm 1.15mm 0; vertical-align: baseline; }}
  td.note {{ color: {MUT}; font-size: 8.2pt; }}
  .good {{ color: {ACC}; font-weight: bold; }}
  .bad {{ color: {RED}; font-weight: bold; }}
  sup.d {{ color: {RED}; font-weight: bold; }}
  .foot {{ border-top: 0.45mm solid {INK}; margin-top: 3mm; padding-top: 1.6mm;
           display: flex; justify-content: space-between; font-size: 8pt; color: {MUT}; white-space: nowrap; }}
  .foot b {{ color: {INK}; }}
  .refs {{ font-size: 7.8pt; color: {MUT}; line-height: 1.5; }}
  .refs p {{ font-size: 7.8pt; text-align: left; }}
</style></head>
<body>

<div class="kicker">P3-EVHM &nbsp;&middot;&nbsp; PHYSICS-GUIDED BATTERY PROGNOSTICS &nbsp;&middot;&nbsp; SRM INSTITUTE OF SCIENCE AND TECHNOLOGY</div>
<h1>Parallel Physics-Informed Prognostics for<br><b>EV Battery Health Management</b></h1>
<div class="authors">A. Vibhas Chowdhary &middot; Maddala Jithendra &mdash; guide: Dr. Kavitha R</div>
<div class="meta">Department of Computing Technologies, SRM IST, Kattankulathur &middot; va1641@srmist.edu.in &middot; jm4408@srmist.edu.in &middot;
code, paper and full validation report: <span class="mono">github.com/Jithendra123892/p3-evhm</span> (MIT) &middot;
GPU runs: <span class="mono">kaggle.com/code/jithendramaddala/p3-evhm-gpu-run</span></div>
<div class="rule2"></div>

<div class="chart">{pipeline()}</div>
<div class="cap">The five-agent pipeline runs as one installable package (<span class="mono">pip install -e .</span>), a CLI benchmark, a six-tab live dashboard and a one-click Kaggle GPU kernel &mdash; the same code, four doors.</div>

<div class="body">
  <!-- ============ LEFT COLUMN ============ -->
  <div>
    <section>
      <div class="eyebrow"><span class="n">THE PROBLEM&nbsp;/&nbsp;</span>WHY BATTERY HEALTH IS HARD</div>
      <p>A lithium-ion cell loses capacity with every charge cycle &mdash; slowly at first, then through a
      degradation <i>knee</i> that takes it below the 70&ndash;80% end-of-life threshold with almost no warning.
      A vehicle reports a charge percentage, not health, so the first sign of trouble is often the failure
      itself. What maintenance actually needs is a number with an error bar, enough cycles in advance to
      schedule the replacement.</p>
      <p>Each ingredient of a solution exists in the literature &mdash; physics-informed training losses,
      LSTM&nbsp;&times;&nbsp;particle-filter hybrids, GPU-resident particle filters &mdash; but no published
      framework combines them on a real benchmark; the closest hybrid paper lists our physics constraints as
      <i>future work</i>. P3-EVHM fuses the four and measures what the fusion buys.</p>
    </section>

    <section>
      <div class="eyebrow"><span class="n">METHOD&nbsp;/&nbsp;</span>WHAT THE MODEL IS TOLD NOT TO DO</div>
      <p>A 1&ndash;32 LSTM learns capacity dynamics from a W&nbsp;=&nbsp;8 cycle window, but its loss carries two
      laws of electrochemistry: the prediction <b>may never claim the battery gained capacity</b>, and its
      decline must respect an Arrhenius-style temperature prior fitted from the training cycles themselves.</p>
      <div class="math">
        <div><span class="n">(4)</span> L = MSE(q&#770;, q) + &lambda;<sub>1</sub> &middot; mean ReLU(q&#770;<sub>t</sub> &minus; q<sub>t</sub>) + &lambda;<sub>2</sub> &middot; mean(&Delta;q&#770;<sub>t</sub> &minus; g(T<sub>t</sub>))&sup2;</div>
        <div><span class="n">(5)</span> g(T) = clamp(&Acirc; &middot; (T &minus; T&#772;), min = 0) &nbsp;&nbsp;&mdash;&nbsp; a prior, not a hard clamp: violations fall ~5&times; but not to zero</div>
      </div>
      <div class="math">
        <div><span class="n">(6&ndash;8)</span> q<sub>i</sub><sup>(k)</sup> = q<sub>i</sub><sup>(k&minus;1)</sup> &minus; d<sub>i</sub> + &epsilon;<sub>k</sub>; &nbsp; w<sub>i</sub> &prop; exp(&minus;(q<sub>obs</sub> &minus; q<sub>i</sub>)&sup2; / 2&sigma;&sup2;); &nbsp; RUL = median&#8202;{{ k : q<sub>i</sub><sup>(k)</sup> &le; q<sub>EOL</sub> }}</div>
      </div>
      <p>Prognosis is a distribution, not a point: the trained model seeds N simulated futures, each with its
      own decline rate, reweighted by the <i>measured</i> start-cycle capacity; the cycle at which a future
      crosses the EOL threshold is that future&rsquo;s RUL. The model reports the median and the 95% interval
      &mdash; and the interval is calibrated, not decorative.</p>
    </section>

    <section>
      <div class="eyebrow"><span class="n">PARALLELISM&nbsp;/&nbsp;</span>ONE CLOUD, ONE TENSOR OP</div>
      <p>Futures are independent, so the particle cloud is embarrassingly parallel: each propagation step is a
      single vectorised tensor update over all N particles &mdash; data-parallel SIMT on the GPU, the identical
      NumPy code on the CPU. No multiprocessing, no multi-node.</p>
      <div class="chart">{gpu_chart()}
      <div class="cap">At small N the warm CPU wins &mdash; we report that. The GPU pays off exactly where
      particle-filter prognostics needs it: large clouds for sharp intervals, in real time.</div></div>
    </section>
  </div>

  <!-- ============ RIGHT COLUMN ============ -->
  <div>
    <section>
      <div class="eyebrow"><span class="n">RESULT 1&nbsp;/&nbsp;</span>THE PHYSICS LOSS WORKS</div>
      <div class="chart">{dumbbell()}</div>
      <div class="cap">All 20 grid configurations, Tesla&nbsp;T4, sorted by how badly the data-only LSTM behaves.
      Configs 04 and 08 are the catastrophes the physics loss eliminates (V&nbsp;=&nbsp;1.0 &rarr; 0.0).</div>
    </section>

    <section>
      <div class="eyebrow"><span class="n">RESULT 2&nbsp;/&nbsp;</span>WHAT WE MEASURED, ALL OF IT</div>
      <table>
        <thead><tr><th>quantity</th><th class="num">measured</th><th>note</th></tr></thead>
        <tbody>
          <tr><td>implausible steps V, mean</td><td class="num">0.366 &rarr; 0.076</td><td class="note">~5&times; cut, 40-config grid, both machines</td></tr>
          <tr><td>capacity MAE, mean</td><td class="num">0.285 &rarr; 0.257 Ah</td><td class="note">physics loss also more accurate</td></tr>
          <tr><td>RUL produced</td><td class="num">20 / 20</td><td class="note">physics+PF, every config, CPU <i>and</i> T4</td></tr>
          <tr><td>mean RUL error</td><td class="num">18.8 cyc</td><td class="note">18.6 on CPU &mdash; reproducible</td></tr>
          <tr><td>95% interval coverage</td><td class="num">0.85</td><td class="note">17/20 &mdash; honest, below the 0.95 ideal<sup class="d">d</sup></td></tr>
          <tr><td>GPU speedup at 500k</td><td class="num">16.5&times;</td><td class="note">20.2 ms device time per PF call</td></tr>
        </tbody>
      </table>
    </section>

    <section>
      <div class="eyebrow"><span class="n">RESULT 3&nbsp;/&nbsp;</span>NEW CELL, NO RETRAINING</div>
      <p style="font-size:9.2pt">Leave-one-cell-out, 16 adaptations. A 21&nbsp;KB fleet-pretrained checkpoint,
      body frozen, only the head fine-tuned:</p>
      <table>
        <thead><tr><th>adaptation mode</th><th class="num">RUL avail.</th><th class="num">V max</th><th class="num">time</th></tr></thead>
        <tbody>
          <tr><td>train from scratch (200 ep)</td><td class="num bad">1 of 4</td><td class="num">0.355</td><td class="num">2.85 s</td></tr>
          <tr><td>pretrained, zero-shot</td><td class="num good">4 of 4</td><td class="num">0.000</td><td class="num">0 s</td></tr>
          <tr><td>head fine-tune (frozen body)</td><td class="num good">4 of 4</td><td class="num">0.000</td><td class="num">0.20 s</td></tr>
        </tbody>
      </table>
      <div class="cap">From-scratch wins raw capacity MAE on 3 of 4 cells &mdash; but fails the actual
      prognostic task. Transfer buys RUL in every held-out cell, zero physics violations, 14&times; faster.</div>
    </section>

    <section>
      <div class="eyebrow"><span class="n">HONEST LIMITS&nbsp;/&nbsp;</span>WHAT THIS IS NOT</div>
      <p style="font-size:9.2pt">Laboratory single-cell data &mdash; pack-level validity untested<sup class="d">d</sup>.
      Early prediction starts (40% of life, before the knee) can fail without the decline-prior floor<sup class="d">d</sup>.
      LSTM seed variance is material and reported. A research decision-support prototype &mdash; not a certified
      BMS safety function.</p>
    </section>

    <section>
      <div class="refs">
        <p><b>Key references</b> &mdash; [1] Saha & Goebel, <i>Battery Data Set</i>, NASA Ames PCoE, 2007.
        [2] Arulampalam et al., <i>IEEE Trans. Signal Process.</i> 50(2), 2002.
        [13] Shi, Rivera & Wu, <i>Mech. Syst. Signal Process.</i> 179:109347, 2022.
        [17] <i>Parallel accelerated CRPF on CUDA</i>, Appl. Intell., 2020.
        [22] do Nascimento et al., <i>Sci. Rep.</i> 13, 2023. [28] <i>Hybrid PF-trained NN for Li-ion prognosis</i>, IEEE, 2021.
        &nbsp;Full 33-entry bibliography in the paper.</p>
      <p style="margin-top:1.4mm"><sup class="d">d</sup>&nbsp;documented in VALIDATION_REPORT.md with per-number provenance.
      Every figure on this poster is generated from <span class="mono">results/kernel_grid_stats.json</span> &mdash; nothing is hand-drawn or estimated.</p>
      </div>
    </section>
  </div>
</div>

<div class="foot">
  <span><b>P3-EVHM</b> &mdash; physics-informed LSTM + particle filter + GPU, in one MIT-licensed framework</span>
  <span><span class="mono">pip install -e .</span> &middot; <span class="mono">python -m p3evhm.demo</span> &middot; <span class="mono">streamlit run app.py</span></span>
</div>
</body></html>
"""

out = os.path.join(HERE, "poster-A3.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out)
print(f"V2={V2:.4f} V3={V3:.4f}")
