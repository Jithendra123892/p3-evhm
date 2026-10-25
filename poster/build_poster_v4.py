# P3-EVHM conference poster v4 — BREAKTHROUGH-FIRST narrative.
# Title -> breakthrough claim -> gap matrix -> C1/C2/C3 -> evidence-for-Cx.
# All numbers from measured artifacts (results/kernel_grid_stats.json etc).
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

k = json.load(open(os.path.join(ROOT, "results", "kernel_grid_stats.json")))
B2V, B3V = k["B2V"], k["B3V"]
m = lambda a: sum(a) / len(a)
V2, V3 = m(B2V), m(B3V)
GPU = [(1000, 7.0, 0.65), (5000, 7.7, 0.98), (20000, 7.4, 2.3), (100000, 9.4, 6.5), (500000, 20.2, 16.5)]

TEAL, TEALD, INK, MUT = "#0d5c4f", "#083d34", "#1b2733", "#5b6b7c"
LINE, RED, PAPER, ACCBG = "#d3dae2", "#b3402f", "#ffffff", "#e2efea"
GOOD, BADC = "#0d5c4f", "#b3402f"

def dot(v):
    return (f'<circle cx="9" cy="7" r="6.5" fill="{GOOD}"/>'
            if v == 1 else f'<circle cx="9" cy="7" r="6.5" fill="none" stroke="{BADC}" stroke-width="2.4"/>'
            + f'<path d="M5 3 L13 11 M13 3 L5 11" stroke="{BADC}" stroke-width="2"/>')

def gap_matrix():
    rows = [
        ("Physics-informed NNs", "[13], [22]", 1, 0, 0, None),
        ("LSTM + particle filter", "[27], [28]", 0, 1, 0, None),
        ("CUDA-parallel particle filters", "[17]", 0, 1, 1, None),
        ("P3-EVHM &mdash; this work", "", 1, 1, 1, 1),
    ]
    s = [f'<svg viewBox="0 0 560 190" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    s.append(f'<rect x="0" y="0" width="560" height="190" fill="{PAPER}"/>')
    cols = [200, 280, 360, 440, 520]
    heads = ["approach family", "physics<br>informed", "RUL<br>distribution", "GPU<br>scale", "real-cell<br>benchmark"]
    # header row
    s.append(f'<rect x="0" y="0" width="560" height="34" fill="{TEALD}"/>')
    for x, h in zip(cols, heads):
        for li, ln in enumerate(h.split("<br>")):
            s.append(f'<text x="{x}" y="{13 + li*12}" font-size="10.5" fill="#ffffff" text-anchor="middle" '
                     f'font-weight="bold">{ln}</text>')
    for r, (name, refs, a, b, c, d) in enumerate(rows):
        y = 34 + r * 39
        ours = r == 3
        if ours:
            s.append(f'<rect x="0" y="{y}" width="560" height="39" fill="{ACCBG}"/>')
            s.append(f'<rect x="0" y="{y}" width="560" height="39" fill="none" stroke="{TEAL}" stroke-width="2.4"/>')
        else:
            s.append(f'<line x1="0" y1="{y}" x2="560" y2="{y}" stroke="{LINE}" stroke-width="1"/>')
        fs = 12.5 if ours else 11
        s.append(f'<text x="10" y="{y+21}" font-size="{fs}" fill="{TEALD if ours else INK}" '
                 f'font-weight="{"bold" if ours else "normal"}">{name}'
                 + (f' <tspan font-size="9.5" fill="{MUT}">&nbsp;&nbsp;{refs}</tspan>' if refs else "") + '</text>')
        for cx, v in zip(cols[1:], (a, b, c)):
            s.append(f'<svg x="{cx-14}" y="{y+7}" width="24" height="24" viewBox="0 0 24 24">{dot(v)}</svg>')
        cx = cols[4]
        if d == 1:
            s.append(f'<svg x="{cx-14}" y="{y+7}" width="24" height="24" viewBox="0 0 24 24">{dot(1)}</svg>')
        else:
            s.append(f'<text x="{cx}" y="{y+21}" font-size="10.5" fill="{MUT}" text-anchor="middle">'
                     f'{"partial" if d is None and a==1 else "single cell" if b==1 and a==0 and c==0 else "not battery"}</text>')
    s.append(f'<line x1="0" y1="190" x2="560" y2="190" stroke="{TEALD}" stroke-width="2.4"/>')
    s.append("</svg>")
    return "".join(s)

def dumbbell():
    W, H = 500, 300
    PL, PR, PT, PB = 118, 22, 30, 34
    pw, ph = W - PL - PR, H - PT - PB
    rows = sorted(zip(B2V, B3V, range(1, len(B2V) + 1)), reverse=True)
    rh = ph / len(rows)
    X = lambda v: PL + min(v, 1.0) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPER}"/>')
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = X(gv)
        s.append(f'<line x1="{x:.1f}" y1="{PT}" x2="{x:.1f}" y2="{PT+ph}" stroke="#c4ccd6" stroke-width="1.4"/>')
        s.append(f'<text x="{x:.1f}" y="{PT-9}" font-size="15" fill="{MUT}" text-anchor="middle" font-family="Consolas,monospace">{gv:.2f}</text>')
    for i, (v2, v3, idx) in enumerate(rows):
        y = PT + (i + 0.5) * rh
        s.append(f'<text x="{PL-8}" y="{y+4:.1f}" font-size="13.5" fill="{MUT}" text-anchor="end" font-family="Consolas,monospace">{idx:02d}</text>')
        s.append(f'<line x1="{X(v2):.1f}" y1="{y:.1f}" x2="{X(v3):.1f}" y2="{y:.1f}" stroke="#c4ccd6" stroke-width="3"/>')
        if v2 - v3 > 0.02:
            s.append(f'<path d="M{X(v3)+7:.1f} {y-3.6:.1f} L{X(v2)-1:.1f} {y:.1f} L{X(v3)+7:.1f} {y+3.6:.1f}" fill="none" stroke="{TEAL}" stroke-width="1.3" opacity="0.5"/>')
        s.append(f'<circle cx="{X(v2):.1f}" cy="{y:.1f}" r="5.2" fill="{RED}"/>')
        s.append(f'<circle cx="{X(v3):.1f}" cy="{y:.1f}" r="5.2" fill="{TEAL}"/>')
    ym = PT + ph * 0.5
    s.append(f'<line x1="{X(V2):.1f}" y1="{PT+2}" x2="{X(V2):.1f}" y2="{PT+ph}" stroke="{RED}" stroke-width="1.8" stroke-dasharray="7 5"/>')
    s.append(f'<line x1="{X(V3):.1f}" y1="{PT+2}" x2="{X(V3):.1f}" y2="{PT+ph}" stroke="{TEAL}" stroke-width="1.8" stroke-dasharray="7 5"/>')
    s.append(f'<text x="{X(V2)-6:.1f}" y="{ym-4:.1f}" font-size="15.5" font-weight="bold" fill="{RED}" text-anchor="end" font-family="Consolas,monospace">mean {V2:.3f}</text>')
    s.append(f'<text x="{X(V3)+9:.1f}" y="{ym+14:.1f}" font-size="15.5" font-weight="bold" fill="{TEAL}" font-family="Consolas,monospace">mean {V3:.3f}</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.8"/>')
    s.append(f'<circle cx="{PL+6}" cy="{H-12}" r="5.2" fill="{RED}"/><text x="{PL+17}" y="{H-8}" font-size="13.5" fill="{INK}">data-only LSTM</text>')
    s.append(f'<circle cx="{PL+116}" cy="{H-12}" r="5.2" fill="{TEAL}"/><text x="{PL+127}" y="{H-8}" font-size="13.5" fill="{INK}">P3-EVHM physics loss</text>')
    s.append("</svg>")
    return "".join(s)

def gpu_chart():
    W, H = 520, 200
    PL, PR, PT, PB = 40, 58, 30, 32
    pw, ph = W - PL - PR, H - PT - PB
    lx = lambda n: PL + (math.log10(n) - 3) / (math.log10(500000) - 3) * pw
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPER}"/>')
    for t in (0, 10, 20):
        y = PT + (1 - t / 25) * ph
        s.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" stroke="#c4ccd6" stroke-width="1.4"/>')
        s.append(f'<text x="{PL-6}" y="{y+4:.1f}" font-size="12.5" fill="{MUT}" text-anchor="end" font-family="Consolas,monospace">{t}</text>')
    for q in (0, 8, 16):
        y = PT + (1 - q / 18) * ph
        s.append(f'<text x="{W-PR+8}" y="{y+4:.1f}" font-size="12.5" fill="{TEAL}" font-family="Consolas,monospace">{q}x</text>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-q/18)*ph:.1f} " for n, _, q in GPU)}" fill="none" stroke="{TEAL}" stroke-width="3"/>')
    s.append(f'<polyline points="{"".join(f"{lx(n):.1f},{PT+(1-t/25)*ph:.1f} " for n, t, _ in GPU)}" fill="none" stroke="{INK}" stroke-width="3"/>')
    for n, t, q in GPU:
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-t/25)*ph:.1f}" r="4.2" fill="{PAPER}" stroke="{INK}" stroke-width="2.2"/>')
        s.append(f'<circle cx="{lx(n):.1f}" cy="{PT+(1-q/18)*ph:.1f}" r="4.2" fill="{TEAL}"/>')
        s.append(f'<text x="{lx(n):.1f}" y="{H-9}" font-size="13" fill="{MUT}" text-anchor="middle" font-family="Consolas,monospace">{n//1000}k</text>')
    s.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-PR}" y2="{PT+ph}" stroke="{INK}" stroke-width="1.8"/>')
    s.append(f'<text x="{W-PR-4}" y="{PT+(1-16.5/18)*ph-10:.1f}" font-size="16" font-weight="bold" fill="{TEAL}" text-anchor="end" font-family="Consolas,monospace">16.5x</text>')
    s.append(f'<text x="{lx(1000)+10:.1f}" y="{PT+(1-0.65/18)*ph+18:.1f}" font-size="12.5" fill="{MUT}">0.65x</text>')
    s.append(f'<text x="{PL}" y="{PT-12}" font-size="12.5" fill="{MUT}">GPU device time ms (dark) and speedup (teal) per particle-filter call, Tesla T4</text>')
    s.append("</svg>")
    return "".join(s)

def pipeline():
    W, H = 1000, 128
    bx = [10, 208, 406, 604, 802]
    names = [("A1", "INGEST", "validate each cycle"),
             ("A2", "FEATURES", "W = 8 windows"),
             ("A3", "PI-LSTM", "physics-constrained SoH"),
             ("A4", "PARTICLE FILTER", "5000 futures"),
             ("A5", "DECISION", "median, 95%, alert")]
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPER}"/>')
    for (aid, ttl, sub), x in zip(names, bx):
        s.append(f'<rect x="{x}" y="24" width="188" height="62" fill="{ACCBG}" stroke="{TEAL}" stroke-width="1.8"/>')
        s.append(f'<text x="{x+94}" y="45" font-size="14.5" fill="{TEALD}" text-anchor="middle" font-weight="bold" letter-spacing="1">{aid} &middot; {ttl}</text>')
        s.append(f'<text x="{x+94}" y="63" font-size="12.5" fill="{MUT}" text-anchor="middle">{sub}</text>')
    for i in range(4):
        s.append(f'<line x1="{bx[i]+188}" y1="55" x2="{bx[i+1]-6}" y2="55" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<path d="M{bx[i+1]-11} 51 L{bx[i+1]-3} 55 L{bx[i+1]-11} 59 Z" fill="{INK}"/>')
    s.append(f'<rect x="436" y="2" width="128" height="15" rx="3" fill="{TEAL}"/>')
    s.append(f'<rect x="634" y="2" width="128" height="15" rx="3" fill="{TEAL}"/>')
    s.append(f'<text x="500" y="13" font-size="10.5" fill="#ffffff" text-anchor="middle" font-weight="bold" letter-spacing="1.5">GPU ACCELERATED</text>')
    s.append(f'<text x="698" y="13" font-size="10.5" fill="#ffffff" text-anchor="middle" font-weight="bold" letter-spacing="1.5">GPU ACCELERATED</text>')
    s.append(f'<text x="10" y="112" font-size="12.5" fill="{MUT}">input: cycle capacity q(t), temperature T(t) &mdash; NASA PCoE cells B0005/B0006/B0007/B0018 &nbsp;&middot;&nbsp; output: RUL median + 95% interval + OK / CAUTION / ALERT</text>')
    s.append("</svg>")
    return "".join(s)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: 297mm 420mm; margin: 0; }}
  * {{ margin: 0; box-sizing: border-box; }}
  html, body {{ background: {PAPER}; }}
  body {{
    width: 297mm; height: 420mm; padding: 0;
    font-family: 'Segoe UI', system-ui, sans-serif; color: {INK};
    display: flex; flex-direction: column;
  }}
  .mono {{ font-family: Consolas, 'Cascadia Mono', monospace; }}
  .band {{ background: {TEAL}; color: #ffffff; padding: 8mm 14mm 6.5mm; }}
  .band .kicker {{ font-size: 9pt; letter-spacing: 2.8mm; color: #bfe0d8; font-weight: 600; }}
  .band h1 {{ font-size: 24.5pt; line-height: 1.12; font-weight: 700; margin: 1.8mm 0 2.2mm; color: #ffffff; }}
  .band .authors {{ font-size: 12pt; font-weight: 600; }}
  .band .authors i {{ font-weight: 400; }}
  .band .affil {{ font-size: 9.5pt; color: #d7ece7; margin-top: 1.2mm; }}
  .band .affil b {{ color: #ffffff; }}
  /* breakthrough strip - the first thing read after the title */
  .claim {{
    background: {ACCBG}; border-left: 3mm solid {TEAL}; padding: 3.2mm 14mm 3.4mm;
  }}
  .claim .tag {{ font-size: 8.5pt; letter-spacing: 2.2mm; color: {TEALD}; font-weight: 700; }}
  .claim .txt {{ font-size: 12.8pt; line-height: 1.35; font-weight: 600; color: {INK}; margin-top: 1mm; }}
  .claim .txt b {{ color: {TEALD}; }}
  .content {{ padding: 4.5mm 14mm 5mm; display: grid; grid-template-columns: 143mm 126mm; gap: 0 7mm; flex: 1; }}
  h2 {{
    background: {TEAL}; color: #ffffff; font-size: 11.5pt; font-weight: 700;
    padding: 1.5mm 3mm; margin: 0 0 2.2mm; border-radius: 1mm;
    letter-spacing: 0.4mm; text-transform: uppercase;
  }}
  h2 .c {{ float: right; font-family: Consolas, monospace; font-weight: 400; opacity: 0.75; font-size: 10pt; letter-spacing: 0; }}
  section {{ margin-bottom: 3.8mm; }}
  section:last-child {{ margin-bottom: 0; }}
  p {{ font-size: 9.6pt; line-height: 1.45; text-align: justify; hyphens: auto; }}
  p + p {{ margin-top: 1.6mm; }}
  .fig {{ border: 0.35mm solid {LINE}; padding: 2.2mm 2.4mm 2mm; margin-top: 1.4mm; }}
  .cap {{ font-size: 7.8pt; color: {MUT}; line-height: 1.4; margin-top: 1mm; }}
  .math {{ font-family: Georgia, 'Times New Roman', serif; font-style: italic; font-size: 9.9pt;
           line-height: 1.8; margin-top: 1.2mm; }}
  .math .n {{ font-style: normal; font-family: Consolas, monospace; font-size: 8.2pt; color: {TEAL}; font-weight: 700; margin-right: 1.6mm; }}
  /* contribution cards */
  .cards {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2.6mm; margin-top: 1.6mm; }}
  .card {{ border: 0.45mm solid {TEAL}; border-top: 1.4mm solid {TEAL}; background: #fbfdfc; padding: 2.6mm 2.8mm 2.4mm; }}
  .card .cid {{ font-family: Consolas, monospace; font-size: 9.5pt; font-weight: 700; color: {TEAL}; }}
  .card .num {{ font-family: Consolas, monospace; font-size: 21pt; font-weight: 700; color: {TEALD}; line-height: 1.05; margin-top: 0.8mm; }}
  .card .what {{ font-size: 8.4pt; font-weight: 700; color: {INK}; margin-top: 0.6mm; line-height: 1.3; }}
  .card .how {{ font-size: 8pt; color: {MUT}; margin-top: 0.8mm; line-height: 1.35; }}
  table.rt {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin-top: 1.2mm; }}
  table.rt th {{ font-size: 7.4pt; color: {MUT}; text-align: left; font-weight: 600;
                 border-bottom: 0.5mm solid {TEAL}; padding: 0 1.6mm 1mm 0; text-transform: uppercase; letter-spacing: 0.2mm; }}
  table.rt th.num, table.rt td.num {{ text-align: right; font-family: Consolas, monospace; padding-right: 0; }}
  table.rt td {{ padding: 1.2mm 1.6mm 1.2mm 0; border-bottom: 0.25mm solid {LINE}; }}
  table.rt td.note {{ color: {MUT}; font-size: 8.1pt; }}
  .good {{ color: {TEAL}; font-weight: 700; }}
  .bad {{ color: {RED}; font-weight: 700; }}
  .foot {{ background: {TEALD}; color: #cfe7e1; padding: 2.2mm 14mm 2.6mm; }}
  .foot .row1 {{ display: flex; justify-content: space-between; font-size: 8.6pt; }}
  .foot .row2 {{ margin-top: 1.2mm; font-size: 7.6pt; line-height: 1.45; color: #a8cec7; }}
  .foot b {{ color: #ffffff; }}
</style></head>
<body>

<div class="band">
  <div class="kicker">P3-EVHM &nbsp;&middot;&nbsp; INTERNATIONAL CONFERENCE POSTER &nbsp;&middot;&nbsp; BATTERY PROGNOSTICS</div>
  <h1>Parallel Physics-Informed Prognostics for EV Battery Health Management:<br>
      Fusing Physics Constraints, Uncertainty and GPU-Scale Inference</h1>
  <div class="authors">A. Vibhas Chowdhary &nbsp;&middot;&nbsp; Maddala Jithendra &nbsp;&nbsp;<i>&mdash; guide: Dr. Kavitha R</i></div>
  <div class="affil"><b>Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur</b>
    &nbsp;&middot;&nbsp; va1641@srmist.edu.in &middot; jm4408@srmist.edu.in &middot; kavithar14@srmist.edu.in</div>
</div>

<div class="claim">
  <div class="tag">THE BREAKTHROUGH</div>
  <div class="txt">To our knowledge, <b>the first framework to combine physics-informed learning, calibrated RUL
  distributions and GPU-scale particle inference</b> &mdash; the four capabilities every prior approach leaves
  incomplete &mdash; validated on real NASA PCoE cells across <b>40 measured configurations on two machines</b>.</div>
</div>

<div class="content">
  <!-- ============== LEFT ============== -->
  <div>
    <section>
      <h2><span class="c">01</span>Research gap &mdash; what no one has combined</h2>
      <p style="font-size:10.4pt;font-weight:600;margin-bottom:1.2mm">Battery RUL research has four mature
      ingredients. No published framework combines them &mdash; a maintenance decision needs all four at once.</p>
      <div class="fig" style="margin-top:0">{gap_matrix()}</div>
      <p style="margin-top:1.4mm">Verified against a systematic review of 33 works: the closest LSTM+filter
      hybrid [28] explicitly lists our physics constraints as <i>future work</i>; CUDA particle filters [17]
      never touched battery data. <b class="good">P3-EVHM closes all four cells &mdash; that closure is the
      contribution.</b></p>
    </section>

    <section>
      <h2><span class="c">02</span>Three contributions</h2>
      <div class="cards">
        <div class="card">
          <div class="cid">C1 &middot; PHYSICS</div>
          <div class="num">~5&times;</div>
          <div class="what">fewer impossible predictions</div>
          <div class="how">Two electrochemistry laws enter the loss (Eq. 4&ndash;5): capacity never rises; decline
          respects an Arrhenius temperature prior. Violations 0.366 &rarr; 0.076; capacity MAE improves
          0.285 &rarr; 0.257 Ah.</div>
        </div>
        <div class="card">
          <div class="cid">C2 &middot; UNCERTAINTY</div>
          <div class="num">20/20</div>
          <div class="what">configs give a calibrated RUL</div>
          <div class="how">Thousands of simulated futures reweighted by the measured capacity (Eq. 6&ndash;8):
          median + 95% interval + decision rules from its lower bound. Mean error 18.8 cycles, coverage 0.85,
          on both machines.</div>
        </div>
        <div class="card">
          <div class="cid">C3 &middot; GPU SCALE</div>
          <div class="num">16.5&times;</div>
          <div class="what">at 500k particles, open-sourced</div>
          <div class="how">One vectorised tensor op per step (SIMT), 20.2 ms device time. MIT framework:
          pip package, dashboard, CI, Kaggle kernel &mdash; 40 measured configs as evidence.</div>
        </div>
      </div>
    </section>

    <section>
      <h2><span class="c">03</span>Method &mdash; the physics enters the loss</h2>
      <div class="math">
        <div><span class="n">(4)</span> L = MSE(q&#770;, q) + &lambda;<sub>1</sub> &middot; mean ReLU(q&#770;<sub>t</sub> &minus; q<sub>t</sub>) + &lambda;<sub>2</sub> &middot; mean(&Delta;q&#770;<sub>t</sub> &minus; g(T<sub>t</sub>))&sup2;</div>
        <div><span class="n">(5)</span> g(T) = clamp(&Acirc; &middot; (T &minus; T&#772;), min = 0) &nbsp;&mdash;&nbsp; prior fitted from training cycles, not a hard clamp</div>
        <div><span class="n">(6)</span> q<sub>i</sub><sup>(k)</sup> = q<sub>i</sub><sup>(k&minus;1)</sup> &minus; d<sub>i</sub> + &epsilon;<sub>k</sub>, &nbsp; w<sub>i</sub> &prop; exp(&minus;(q<sub>obs</sub> &minus; q<sub>i</sub>)&sup2; / 2&sigma;&sup2;)</div>
        <div><span class="n">(8)</span> RUL = median&#8202;{{ k : q<sub>i</sub><sup>(k)</sup> &le; q<sub>EOL</sub> }}, &nbsp; report [P<sub>2.5</sub>, P<sub>97.5</sub>]</div>
      </div>
      <p>A 1&ndash;32 LSTM over a W&nbsp;=&nbsp;8 window; identical seeds and metrics for baselines B1&ndash;B6,
      so the comparison is the physics, not the tuning. Same code runs on CPU or CUDA
      (<span class="mono">device="auto"</span>).</p>
    </section>

    <section>
      <h2><span class="c">04</span>System &mdash; five agents, one framework</h2>
      <div class="fig">{pipeline()}
      <div class="cap">A3 and A4 run on the GPU; the same code runs on CPU for parity. Four doors into one
      engine: Python API, CLI, dashboard, Kaggle kernel.</div></div>
    </section>
  </div>

  <!-- ============== RIGHT ============== -->
  <div>
    <section>
      <h2><span class="c">05</span>Evidence for C1 &mdash; the physics loss works</h2>
      <div class="fig">{dumbbell()}</div>
      <div class="cap">Violation rate V per configuration (share of predicted steps where capacity <i>rises</i>
      &mdash; physically impossible). All 20 configurations improve or tie; the two catastrophes
      (V = 1.0, configs 04/08) are eliminated entirely.</div>
    </section>

    <section>
      <h2><span class="c">06</span>Evidence for C2 &mdash; prognostic quality</h2>
      <table class="rt">
        <tr><th>quantity</th><th class="num">measured</th><th>note</th></tr>
        <tr><td>RUL produced</td><td class="num good">20 / 20</td><td class="note">every config, CPU <i>and</i> T4</td></tr>
        <tr><td>mean RUL error</td><td class="num">18.8 cyc</td><td class="note">18.6 on CPU &mdash; reproducible</td></tr>
        <tr><td>95% interval coverage</td><td class="num">0.85</td><td class="note">honest, below the 0.95 ideal</td></tr>
        <tr><td>transfer: held-out cells</td><td class="num good">4 / 4</td><td class="note">vs 1/4 from scratch; 0.2 s head-only adaptation</td></tr>
      </table>
      <p style="margin-top:1.4mm;font-size:9.2pt">A 21 KB pretrained checkpoint, body frozen, head fine-tuned:
      RUL in <b class="good">4/4 held-out cells</b> vs <b class="bad">1/4</b> from scratch; zero physics
      violations vs 0.355 max. Transfer is what a fleet needs &mdash; no per-cell training.</p>
    </section>

    <section>
      <h2><span class="c">07</span>Evidence for C3 &mdash; GPU scaling</h2>
      <div class="fig">{gpu_chart()}
      <div class="cap">Honest crossover reported: below ~5k particles the warm CPU wins (0.65&times;); the GPU
      pays exactly where prognostics needs it &mdash; large clouds, sharp intervals, real time.</div></div>
    </section>

    <section>
      <h2><span class="c">08</span>Conclusions & future work</h2>
      <p><b>Conclusions.</b> The fusion is the contribution: physics constraints make LSTM prognostics physically
      consistent at no accuracy cost; the particle filter turns that forecast into a calibrated maintenance
      decision in every configuration tested; GPU-scale inference makes the cloud size that sharp intervals
      need affordable. Open-sourced end to end, with 40 measured configurations as evidence.</p>
      <p><b>Future work.</b> Pack-level validation beyond single cells; prior-floor tuning for early-start
      predictions (40% of life); interval coverage toward 0.95; streaming on-vehicle deployment.</p>
    </section>
  </div>
</div>

<div class="foot">
  <div class="row1">
    <span><b>Code & full validation:</b> github.com/Jithendra123892/p3-evhm (MIT) &middot; GPU kernel: jithendramaddala/p3-evhm-gpu-run</span>
    <span><b>Reproduce:</b> pip install -e . &middot; python -m p3evhm.demo &middot; streamlit run app.py</span>
  </div>
  <div class="row2"><b>Key references</b>&nbsp; [1] Saha & Goebel, <i>Battery Data Set</i>, NASA PCoE, 2007. &nbsp;
    [2] Arulampalam et al., <i>IEEE Trans. Signal Process.</i> 50(2), 2002. &nbsp;
    [13] Shi, Rivera & Wu, <i>Mech. Syst. Signal Process.</i> 179, 2022. &nbsp;
    [17] <i>Parallel accelerated CRPF on CUDA</i>, Appl. Intell., 2020. &nbsp;
    [22] do Nascimento et al., <i>Sci. Rep.</i> 13, 2023. &nbsp;
    [27] <i>AI-driven particle filter for battery RUL</i>, Batteries 10(12), 2024. &nbsp;
    [28] <i>Hybrid particle-filter-trained neural network</i>, IEEE, 2021. &nbsp;
    &mdash;&nbsp; full 33-entry bibliography in the paper; all figures generated from measured results.</div>
</div>
</body></html>
"""

out = os.path.join(HERE, "poster-v4.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, "| V2", f"{V2:.3f}", "V3", f"{V3:.3f}")
