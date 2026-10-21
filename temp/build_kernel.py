# Rebuild P3-EVHM Kaggle notebook (clean, validated) + kernel-metadata.json for `kaggle kernels push` with GPU.
import json, os
BASE = r"C:\Users\jithe\p3-evhm-paper"
KDIR = os.path.join(BASE, "kernel")
os.makedirs(KDIR, exist_ok=True)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = [
md("""# P3-EVHM — Parallel Physics-Informed Prognostics for EV Battery Health Management
**GPU run.** Dataset: ckskaggle/li-ion-battery-dataset-from-nasa-pcoe (NASA PCoE .mat files, byte-identical to the NASA Prognostics Data Repository originals).
Pipeline: PI-LSTM (data + monotonicity + temperature-prior loss, Eq. 3-5 of the paper) + particle filter (Eq. 6-8), baselines B1-B6, GPU N-scaling timing.
All validation checks print inline; results land in `/kaggle/working/p3evhm_results.json`."""),
code("""import os, glob, time, json, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
torch.backends.cudnn.deterministic = True
SEED = 0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
print('device:', device, '| torch', torch.__version__, '| GPU:', GPU, flush=True)
RATED = 2.0  # Ah, NASA PCoE 18650 cells

# ---- Locate dataset ----
mats = {os.path.basename(p).replace('.mat',''): p for p in glob.glob('/kaggle/input/**/*.mat', recursive=True)}
need = ['B0005','B0006','B0007','B0018']
missing = [c for c in need if c not in mats]
assert not missing, f'MISSING .mat files: {missing}'
print('dataset files found:', sorted(mats.keys()), flush=True)

def parse_cell(cell):
    from scipy.io import loadmat
    m = loadmat(mats[cell]); cyc = m[cell][0,0]['cycle'][0]
    rows = []
    for c in cyc:
        if str(c['type'][0]) == 'discharge':
            d = c['data'][0,0]; names = d.dtype.names
            if names and 'Capacity' in names:
                T = float(np.mean(d['Temperature_measured'])) if ('Temperature_measured' in names) else 25.0
                rows.append((float(d['Capacity'][0][0]), T))
    return pd.DataFrame(rows, columns=['capacity','temp'])

data = {c: parse_cell(c) for c in need}
# validation: cycle counts vs NASA originals (B0005 168, B0006 168, B0007 168, B0018 132)
expect = {'B0005':168,'B0006':168,'B0007':168,'B0018':132}
for c in need:
    ok = len(data[c]) == expect[c]
    print(f'{c}: {len(data[c])} discharge cycles (expect {expect[c]}) {"OK" if ok else "MISMATCH"} | cap {data[c].capacity.iloc[0]:.4f} -> {data[c].capacity.iloc[-1]:.4f} Ah', flush=True)
    assert ok, f'cycle count mismatch for {c}'
print('DATA_VALIDATION_PASS', flush=True)"""),
code("""class LastState(nn.Module):
    def forward(self, o): out,_ = o; return out[:,-1,:]

def train_pi_lstm(train_q, train_T, W=8, hidden=64, epochs=400, lam1=2.0, lam2=1.0, lr=5e-3, dev=device):
    # Eq. (3): L = L_data + lam1*L_mono + lam2*L_prior ; Eq. (4) monotonicity; Eq. (5) temperature prior
    X = np.lib.stride_tricks.sliding_window_view(train_q, W+1)
    Xt = torch.tensor(X[:,:W], dtype=torch.float32, device=dev).unsqueeze(-1)
    Yt = torch.tensor(X[:,-1], dtype=torch.float32, device=dev)
    last = torch.tensor(X[:,W-1], dtype=torch.float32, device=dev)
    Tt = torch.tensor(train_T[W-1:-1], dtype=torch.float32, device=dev)
    dT = -(train_q[1:] - train_q[:-1])
    A = float(np.polyfit((train_T[1:] - np.mean(train_T))[1:], dT[1:], 1)[0]) if len(dT) > 10 else 0.0
    g = torch.clamp(A * (Tt - Tt.mean()), min=0.0)
    net = nn.Sequential(nn.LSTM(1, hidden, batch_first=True), LastState(), nn.Linear(hidden,1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); mse = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad(); out = net(Xt).squeeze(-1)
        loss = mse(out, Yt) + lam1*torch.relu(out - last).mean() + lam2*((out[:-1]-out[1:]) - g[1:]).pow(2).mean()
        loss.backward(); opt.step()
    return net

def forecast(net, hist, steps, W=8, dev=device):
    hist = list(hist); preds = []
    with torch.no_grad():
        for _ in range(steps):
            xw = torch.tensor(np.array(hist[-W:]), dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
            p = float(net(xw).squeeze()); preds.append(p); hist.append(p)
    return np.array(preds)

def pf_rul(q0, slope_mu, slope_sd, eol, steps, N=20000, proc=0.003, dev=torch.device('cpu'), Nn=None, obs0=None):
    t0 = time.time()
    qi = torch.clamp(torch.normal(q0, 0.01, (N,), device=dev), min=eol)
    di = torch.clamp(torch.normal(slope_mu, slope_sd, (N,), device=dev).abs(), min=1e-5)
    if obs0 is not None:  # observation update at the start cycle, so crossing step k -> cycle start+k
        wts = torch.exp(-((obs0 - qi) ** 2) / (2 * 0.01 ** 2))
        if wts.sum() > 0:
            idx = torch.multinomial(wts / wts.sum(), N, replacement=True)
            qi = qi[idx]
    cross = torch.full((N,), -1.0, device=dev)
    for k in range(1, steps+1):
        qi = qi - di + torch.randn(N, device=dev)*proc
        newly = (cross < 0) & (qi <= eol)
        cross[newly] = float(k)
    valid = cross[cross > 0]; rul = valid[valid <= steps]
    dt = time.time() - t0
    if len(rul) == 0: return None, None, None, dt
    lo, med, hi = np.percentile(rul.detach().cpu().numpy(), [2.5, 50, 97.5])
    return int(med), float(lo), float(hi), dt"""),
code("""# ---- Full experiment: 4 cells, thresholds 70%/80%, start points, baselines B1-B6 ----
RESULTS = {'meta': {'gpu': GPU, 'torch': torch.__version__, 'seed': SEED, 'dataset': 'ckskaggle/li-ion-battery-dataset-from-nasa-pcoe'}}
CFG = [('B0005', 70), ('B0006', 70), ('B0007', 80), ('B0018', 70)]
for cell, theta_pct in CFG:
    d = data[cell]; q = d.capacity.values; T = d.temp.fillna(25).values
    eol = theta_pct/100.0*RATED
    below = np.where(q <= eol)[0]
    true_eol = int(below[0]+1) if len(below) else None
    assert true_eol, f'{cell} never crosses {theta_pct}% threshold'
    R = {'theta_pct': theta_pct, 'true_eol_cycle': true_eol, 'starts': {}}
    for frac in [0.4, 0.5, 0.6, 0.7, 0.8]:
        start = max(int(round(true_eol*frac)), 15); s = start-1
        train_q, test_q, train_T = q[:s], q[s:], T[:s]
        true_rul = true_eol - start
        rec = {'start': start, 'true_rul': true_rul}
        # B1 linear
        w = 20; sl, ic = np.polyfit(np.arange(s-w, s), train_q[-w:], 1); sl = min(sl, -1e-4)
        pred = ic + sl*np.arange(s, len(q)); cr = np.where(pred <= eol)[0]
        rec['B1_linear'] = dict(mae=float(np.abs(pred-test_q).mean()), rmse=float(np.sqrt(((pred-test_q)**2).mean())),
                                rul_err=int(abs(cr[0]-true_rul)) if len(cr) else None, V=float((np.diff(pred)>0).mean()))
        # B2 LSTM / B3 PI-LSTM
        net2 = train_pi_lstm(train_q, train_T, lam1=0.0, lam2=0.0)
        net3 = train_pi_lstm(train_q, train_T, lam1=2.0, lam2=1.0)
        p2 = forecast(net2, list(train_q[-8:]), len(test_q)); p3 = forecast(net3, list(train_q[-8:]), len(test_q))
        for nm, p in [('B2_lstm', p2), ('B3_pi_lstm', p3)]:
            cr = np.where(p <= eol)[0]
            rec[nm] = dict(mae=float(np.abs(p-test_q).mean()), rmse=float(np.sqrt(((p-test_q)**2).mean())),
                           rul_err=int(abs(cr[0]-true_rul)) if len(cr) else None, V=float((np.diff(p)>0).mean()))
        # B4/B5/B6 particle-filter variants
        sl5 = abs((train_q[-40:][0]-train_q[-1])/40)
        sl4 = abs((p2[-1]-p2[0])/max(len(p2)-1,1)); sl6 = abs((p3[-1]-p3[0])/max(len(p3)-1,1))
        for nm, mu, sd in [('B4_lstm_pf', sl4, sl4*0.8+1e-4), ('B5_phys_pf', sl5, sl5*0.8+1e-4), ('B6_p3evhm_gpu', sl6, sl6*0.8+1e-4)]:
            med, lo, hi, _ = pf_rul(train_q[-1], mu, sd, eol, len(test_q), N=20000, dev=device, obs0=test_q[0])
            rec[nm] = dict(rul_err=int(abs(med-true_rul)) if med is not None else None,
                           interval=[lo, hi], cover=(float(lo <= true_rul <= hi) if med is not None and lo is not None else None))
        R['starts'][str(frac)] = rec
        print(f'{cell} start {start}: B3 rul_err {rec["B3_pi_lstm"]["rul_err"]} | B5 {rec["B5_phys_pf"]["rul_err"]} | B6 {rec["B6_p3evhm_gpu"]["rul_err"]} | B2 V {rec["B2_lstm"]["V"]:.3f} | B3 V {rec["B3_pi_lstm"]["V"]:.3f}', flush=True)
    RESULTS[cell] = R
json.dump(RESULTS, open('/kaggle/working/p3evhm_results.json','w'), indent=2, default=str)
print('FULL_EXPERIMENT_DONE', flush=True)"""),
code("""# ---- GPU N-scaling timing (device-time measured with CUDA events) ----
d = data['B0005']; q = d.capacity.values; eol = 0.7*RATED; s = 99
sl5 = abs((q[:s][-40:][0]-q[:s][-1])/40)
scaling = {}
for N in [1000, 5000, 20000, 100000, 500000]:
    t0 = time.time(); pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cpu')); tc = time.time()-t0
    tg = None; sp = None
    if device.type == 'cuda':
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); tw0 = time.time()
        ev0.record(); pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cuda')); ev1.record()
        torch.cuda.synchronize(); tw = time.time()-tw0
        tg = ev0.elapsed_time(ev1)/1000.0; sp = tc/tg
        scaling[N] = {'cpu_s': round(tc,4), 'gpu_dev_ms': round(tg*1000,2), 'gpu_wall_s': round(tw,4), 'speedup': round(sp,2)}
    else:
        scaling[N] = {'cpu_s': round(tc,4), 'gpu_dev_ms': None, 'gpu_wall_s': None, 'speedup': None}
    print(N, scaling[N], flush=True)
RESULTS['scaling'] = scaling
json.dump(RESULTS, open('/kaggle/working/p3evhm_results.json','w'), indent=2, default=str)
print('ALL_DONE device', device.type, flush=True)"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 4}
nb_path = os.path.join(KDIR, "P3-EVHM_kaggle.ipynb")
with open(nb_path, "w", encoding="utf-8") as f: json.dump(nb, f, indent=1)

meta = {"id": "jithendramaddala/p3-evhm-gpu-run", "title": "P3 EVHM GPU Run",
        "code_file": "P3-EVHM_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "dataset_sources": ["ckskaggle/li-ion-battery-dataset-from-nasa-pcoe"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f: json.dump(meta, f, indent=2)
print("NB_OK", os.path.getsize(nb_path))
print("META_OK", os.path.getsize(os.path.join(KDIR, "kernel-metadata.json")))
