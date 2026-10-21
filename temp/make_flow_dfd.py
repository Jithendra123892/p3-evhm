# Generate mam-ready figures: (1) system flowchart, (2) DFD Level 1 with external entities, processes, data stores.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

BASE = r"C:\Users\jithe\p3-evhm-paper"
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=200)

# ---- Panel A: flowchart ----
ax = axes[0]; ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
steps = [
    (8,  52, "START: load NASA PCoE\ncell data (.mat)"),
    (8,  40, "A1: validate cycles\n(missing/non-finite checks)"),
    (8,  28, "A2: build features\n+ sliding windows (W=8)"),
    (8,  16, "A3: train PI-LSTM\n(data + physics loss)"),
    (38, 16, "A4: particle filter\n(propagate/weight/resample)"),
    (38, 28, "SoH estimate +\nRUL distribution"),
    (38, 40, "A5: decision rules\n(ALERT / CAUTION / OK)"),
    (38, 52, "END: dashboard +\nmaintenance alert"),
]
for x, y, t in steps:
    ax.add_patch(FancyBboxPatch((x, y - 4), 22, 8.5, boxstyle="round,pad=0.5",
                                fc="#eef2f7", ec="#2b3a55", lw=1.1))
    ax.text(x + 11, y, t, ha="center", va="center", fontsize=7.0, color="#1a2433")
arrow_pairs = [((19, 52), (19, 48.5)), ((19, 36), (19, 32.5)), ((19, 24), (19, 20.5)),
               ((30, 16), (38, 16)), ((49, 20.5), (49, 24)), ((49, 32), (49, 36)), ((49, 44), (49, 48))]
for (x0, y0), (x1, y1) in arrow_pairs:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#2b3a55"))
# decision diamond for physics-violation check
ax.add_patch(plt.Polygon([[63, 14], [75, 18], [87, 14], [75, 10]], fc="#fff7e6", ec="#b7791f", lw=1.1))
ax.text(75, 14, "violations\nV low?", ha="center", va="center", fontsize=6.6)
ax.add_patch(FancyArrowPatch((60, 16), (63, 14), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#2b3a55"))
ax.text(70, 6, "yes -> trust RUL interval", fontsize=6.4, color="#4a5568")
ax.add_patch(FancyArrowPatch((87, 14), (94, 14), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#b7791f"))
ax.text(88.5, 17, "no -> flag", fontsize=6.4, color="#b7791f")
ax.set_title("(a) System flowchart", fontsize=9.5, weight="bold")

# ---- Panel B: DFD Level 1 ----
ax = axes[1]; ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
def entity(x, y, t):
    ax.add_patch(Rectangle((x, y - 4), 17, 8.5, fc="#fdf2f2", ec="#8b2f2f", lw=1.2))
    ax.text(x + 8.5, y, t, ha="center", va="center", fontsize=6.8, color="#5f2120")
def process(x, y, n, t):
    ax.add_patch(FancyBboxPatch((x, y - 4.5), 16, 9.5, boxstyle="round,pad=0.4", fc="#eef2f7", ec="#2b3a55", lw=1.1))
    ax.text(x + 2.0, y + 2.6, n, ha="center", va="center", fontsize=7.2, weight="bold", color="#2b3a55")
    ax.text(x + 8.5, y - 1.2, t, ha="center", va="center", fontsize=6.4, color="#1a2433")
def store(x, y, n, t):
    ax.add_patch(Rectangle((x, y - 3.5), 20, 7.5, fc="#f0fff4", ec="#276749", lw=1.1))
    ax.plot([x, x], [y - 3.5, y + 4.0], color="#276749", lw=1.1)
    ax.text(x + 2.0, y + 0.5, n, ha="center", va="center", fontsize=7.0, weight="bold", color="#276749")
    ax.text(x + 11.5, y, t, ha="center", va="center", fontsize=6.3, color="#1a2433")
def flow(x0, y0, x1, y1, t="", col="#2b3a55"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=col))
    if t: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, t, ha="center", fontsize=5.8, color=col)

entity(2, 52, "Battery /\nsensors")
process(30, 52, "P1", "Ingest &\nvalidate")
process(58, 52, "P2", "Feature &\nwindows")
process(58, 30, "P3", "SoH est.\nPI-LSTM")
process(30, 30, "P4", "RUL particle\nfilter")
process(30, 10, "P5", "Alerts &\ndashboard")
entity(2, 10, "Maintenance\nteam")
store(80, 52, "D1", "Telemetry")
store(80, 30, "D2", "Models")
store(80, 10, "D3", "Results log")
flow(19, 52, 30, 52, "raw V/I/T")
flow(46, 52, 58, 52, "valid cycles")
flow(66, 47.5, 80, 55.5); flow(80, 48.5, 74, 47.5)  # P2 <-> D1
flow(66, 52, 74, 55.5)
flow(66, 47.5, 66, 34.5, "windows")
flow(74, 30, 80, 32.5); flow(80, 27.5, 74, 27.5)     # P3 <-> D2
flow(58, 30, 46, 30, "SoH state")
flow(38, 25.5, 38, 14.5, "RUL interval")
flow(30, 25.5, 22, 38, "SoH/V")                       # P4 reads D2? simplify: P3->D2 handled
flow(38, 10, 19, 10, "alerts")
flow(46, 10, 80, 12.5); flow(80, 7.5, 46, 8.5)        # P5 <-> D3
ax.set_title("(b) DFD Level 1 (context: agents = processes)", fontsize=9.5, weight="bold")

fig.tight_layout()
out = os.path.join(BASE, "flow_dfd.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("FIG_OK", os.path.getsize(out))
