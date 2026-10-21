# Parse NASA PCoE .mat files -> capacity.csv (cell, cycle_index, capacity). Progress printed per cell.
import numpy as np, pandas as pd, time
from scipy.io import loadmat
import os
BASE = r"C:\Users\jithe\p3-evhm-paper"
rows = []
for cell in ["B0005", "B0006", "B0007", "B0018"]:
    t0 = time.time()
    m = loadmat(os.path.join(BASE, "data", cell + ".mat"))
    cyc = m[cell][0, 0]["cycle"][0]
    n = 0
    for ci, c in enumerate(cyc):
        if str(c["type"][0]) == "discharge":
            d = c["data"][0, 0]
            names = d.dtype.names
            if names is not None and "Capacity" in names:
                cap = float(d["Capacity"][0][0])
                rows.append((cell, n, cap)); n += 1
    print(cell, "discharge cycles:", n, "elapsed", round(time.time() - t0, 1), "s", flush=True)
df = pd.DataFrame(rows, columns=["cell", "cycle", "capacity"])
df.to_csv(os.path.join(BASE, "data", "capacity.csv"), index=False)
print("CSV_OK", len(df), "rows")
print(df.groupby("cell")["capacity"].agg(["count", "first", "last", "min"]))
