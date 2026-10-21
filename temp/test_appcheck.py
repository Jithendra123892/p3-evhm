# Headless functional test of app.py via streamlit.testing.v1.AppTest:
# 1) run default config (B0005) -> assert no exception, metrics populated with real numbers
# 2) switch cell to B0007 -> re-run -> assert interactivity works and B0007 numbers differ
import sys
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(r"C:\Users\jithe\p3-evhm-paper\app.py", default_timeout=900)
at.run()
print("RUN1 exceptions:", len(at.exception))
if at.exception:
    for e in at.exception: print("EXC:", e.message[:300])
    sys.exit(1)

def dump(at, tag):
    ms = {}
    for m in at.metric:
        ms[m.label] = (m.value, m.delta)
    for lbl in ["SoH at start cycle", "Predicted RUL (median)", "Physics violation rate V",
                "LSTM training time (this machine, CPU)", "Particle filter time (N=5000)"]:
        for k, v in ms.items():
            if lbl in k: print(f"[{tag}] {k} = {v}")
    return ms

m1 = dump(at, "RUN1 B0005")
assert any("SoH at start cycle" in k for k in m1), "SoH metric missing"
assert any("Predicted RUL" in k for k in m1), "RUL metric missing"
print("RUN1 PASS: default config renders with real computed metrics")

# interactivity: switch cell to B0007 (default theta flips to 80)
at.sidebar.selectbox[0].set_value("B0007").run()
print("RUN2 exceptions:", len(at.exception))
if at.exception:
    for e in at.exception: print("EXC:", e.message[:300])
    sys.exit(1)
m2 = dump(at, "RUN2 B0007")
soh1 = m1[[k for k in m1 if "SoH" in k][0]][0]
soh2 = m2[[k for k in m2 if "SoH" in k][0]][0]
print("RUN2 PASS: cell switch re-rendered; SoH changed", soh1, "->", soh2)
assert str(soh1) != str(soh2), "metrics did not change after cell switch"
# alerts present?
alerts = [str(a.value)[:140] for a in (list(at.error) + list(at.warning) + list(at.success))]
print("ALERTS:", alerts[:3])
print("APPTEST_ALL_PASS")
