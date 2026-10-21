# Pretrain the fleet base PI-LSTM on B0005+B0006+B0018 (first 60% of each cell; B0007 fully held out)
# and save the checkpoint into the package so users get instant predictions + fast fine-tuning.
import os
import sys
import numpy as np
import pandas as pd

BASE = r"C:\Users\jithe\p3-evhm-paper"
sys.path.insert(0, BASE)
from p3evhm import models

df = pd.read_csv(os.path.join(BASE, "data", "telemetry.csv")).sort_values(["cell", "cycle"])
pool_q, pool_T = [], []
for cell in ["B0005", "B0006", "B0018"]:
    d = df[df.cell == cell]
    q, T = d.capacity.values, d.temp.fillna(25.0).values
    cut = int(len(q) * 0.6)
    pool_q.append(q[:cut]); pool_T.append(T[:cut])
pq = np.concatenate(pool_q); pT = np.concatenate(pool_T)
print("pretrain pool:", len(pq), "cycles from 3 cells (B0007 held out)")

net = models.train_pi_lstm(pq, pT, lam1=2.0, lam2=1.0, W=8, hidden=32, epochs=300, lr=5e-3, seed=0)
path = models.save_pretrained(net, os.path.join(BASE, "p3evhm", "pretrained", "pi_lstm_base_v1.pt"),
                              config={"hidden": 32, "W": 8, "cells": ["B0005", "B0006", "B0018"],
                                      "pool_cycles": len(pq), "epochs": 300, "lam1": 2.0, "lam2": 1.0})
print("PRETRAINED_SAVED", path, os.path.getsize(path), "bytes")

# quick sanity: zero-shot forecast on the held-out cell's early segment
d7 = df[df.cell == "B0007"]
q7, T7 = d7.capacity.values, d7.temp.fillna(25.0).values
cut7 = int(len(q7) * 0.6)
preds = models.forecast(net, q7[:cut7], 20, W=8)
print("zero-shot sanity: first 5 forecasts", np.round(preds[:5], 4), "| last measured train", round(q7[cut7 - 1], 4))
