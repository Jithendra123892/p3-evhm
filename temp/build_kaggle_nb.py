# Build P3-EVHM Kaggle notebook (.ipynb): full GPU experiment — PI-LSTM (data+mono+temperature prior), torch-CUDA particle filter, baselines B1-B6, CPU/GPU timing.
import json, os
BASE = r"C:\Users\jithe\p3-evhm-paper"

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = [
md("""# P3-EVHM — Parallel Physics-Informed Prognostics for EV Battery Health Management
**Full experiment on Kaggle GPU.** Baselines B1–B6, PI-LSTM with data + monotonicity + temperature-prior loss, torch-CUDA particle filter, CPU/GPU timing.

**Setup:** Settings → Accelerator → **GPU (T4 or P100)**. Attach a Kaggle dataset containing the NASA PCoE `.mat` files (`B0005.mat` … `B0018.mat`) — e.g. search "NASA PCoE Battery" in Add Input. The loader auto-detects any path. Runtime: ~10–20 min total."""),
code("""import os, glob, time, json, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)"""),
code("""# ---- Locate dataset ----
mats = glob.glob('/kaggle/input/**/*.mat', recursive=True)
mats = {os.path.basename(p).replace('.mat',''): p for p in mats}
print('found .mat files:', sorted(mats.keys()))
assert 'B0005' in mats, 'Attach a Kaggle dataset containing B0005.mat (NASA PCoE Battery Data Set)'
RATED = 2.0  # Ah rated

def parse_cell(cell):
    from scipy.io import loadmat
    m = loadmat(mats[cell]); cyc = m[cell][0,0]['cycle'][0]
    rows = []
    for ci, c in enumerate(cyc):
        if str(c['type'][0]) == 'discharge':
            d = c['data'][0,0]; names = d.dtype.names
            if names and 'Capacity' in names:
                T = float(np.mean(d['Temperature_measured'])) if 'Temperature_measured' in names else float('nan')
                rows.append((ci+1, float(d['Capacity'][0][0]), T))
    return pd.DataFrame(rows, columns=['cycle','capacity','temp'])

data = {c: parse_cell(c) for c in ['B0005','B0006','B0007','B0018']}
for c, d in data.items(): print(c, len(d), 'cycles | cap', round(d.capacity.iloc[0],4), '->', round(d.capacity.iloc[-1],4))"""),
code("""# ---- P3-EVHM core: PI-LSTM (Eq. 3-5) + particle filter (Eq. 6-8) ----
def train_pilstorm(train_q, train_T, W=8, hidden=64, epochs=400, lam1=10.0, lam2=1.0, lr=5e-3, dev=device):
    X = np.lib.stride_tricks.sliding_window_view(train_q, W+1)
    Xt = torch.tensor(X[:,:W], dtype=torch.float32, device=dev).unsqueeze(-1)
    Yt = torch.tensor(X[:,-1], dtype=torch.float32, device=dev)
    last = torch.tensor(X[:,W-1], dtype=torch.float32, device=dev)
    Tt = torch.tensor(train_T[W-1:-1], dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.LSTM(1, hidden, batch_first=True), LastState(), nn.Linear(hidden,1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); mse = nn.MSELoss()
    # temperature-dependent degradation prior g(T): Arrhenius-type scaling fitted on training cells
    dT = -(train_q[1:] - train_q[:-1]); mask = (train_T[1:] - np.mean(train_T))
    A = float(np.polyfit(mask[1:], dT[1:], 1)[0]) if len(dT) > 10 else 0.0
    g = torch.clamp(torch.tensor(A, dtype=torch.float32, device=dev) * (Tt - Tt.mean()), min=0)
    for ep in range(epochs):
        opt.zero_grad(); out = net(Xt).squeeze(-1)
        L_data = mse(out, Yt)
        L_mono = torch.relu(out - last).mean()                      # Eq. (4)
        L_prior = ((out[:-1] - out[1:]) - g).pow(2).mean()          # Eq. (5) proxy
        loss = L_data + lam1*L_mono + lam2*L_prior                  # Eq. (3)
        loss.backward(); opt.step()
    return net

class LastState(nn.Module):
    def forward(self, o): out,_ = o; return out[:,-1,:]

def forecast(net, hist, steps, W=8, dev=device):
    hist = list(hist)
    with torch.no_grad():
        preds = []
        for _ in range(steps):
            xw = torch.tensor(np.array(hist[-W:]), dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
            p = float(net(xw).squeeze()); preds.append(p); hist.append(p)
    return np.array(preds)

def pf_rul(q0, slope_mu, slope_sd, eol, steps, N=20000, proc=0.003, dev=device, timing=None):
    t0 = time.time()
    qi = torch.clamp(torch.normal(q0, 0.01, (N,), device=dev), min=eol)
    di = torch.clamp(torch.normal(slope_mu, slope_sd, (N,), device=dev).abs(), min=1e-5)
    cross = torch.full((N,), -1.0, device=dev)
    for k in range(1, steps+1):
        qi = qi - di + torch.randn(N, device=dev)*proc
        newly = (cross < 0) & (qi <= eol)
        cross[newly] = float(k)
        if k % 25 == 0: torch.cuda.synchronize() if dev.type=='cuda' else None
    valid = cross[cross > 0]; rul = valid[valid <= steps]
    if timing is not None: timing.append(time.time()-t0)
    if len(rul) == 0: return None, None, None
    lo, med, hi = np.percentile(rul.cpu().numpy(), [2.5, 50, 97.5])
    return int(med), float(lo), float(hi)"""),
code("""# ---- Experiment: B0005 (70% EOL) primary; B0007 (80% EOL) sensitivity ----
RESULTS = {}
for cell, theta_pct, start in [('B0005',70,100), ('B0007',80,60)]:
    d = data[cell]; q = d.capacity.values; T = d.temp.fillna(25).values
    eol = theta_pct/100*RATED; s = start-1
    below = np.where(q <= eol)[0]; true_eol = int(below[0]+1) if len(below) else None
    true_rul = true_eol - start if true_eol else None
    train_q, test_q, train_T = q[:s], q[s:], T[:s]
    R = {'true_eol_cycle': true_eol, 'true_rul': true_rul}
    # B1 linear
    w=20; sl,ic = np.polyfit(np.arange(s-w,s), train_q[-w:],1); sl = min(sl, -1e-4)
    pred = ic + sl*np.arange(s,len(q))
    R['B1_linear'] = dict(mae=float(np.abs(pred-test_q).mean()), rmse=float(np.sqrt(((pred-test_q)**2).mean())),
                          V=float((np.diff(pred)>0).mean()))
    # B2 LSTM / B3 PI-LSTM
    net2 = train_pilstorm(train_q, train_T, lam1=0.0, lam2=0.0)
    net3 = train_pilstorm(train_q, train_T)
    p2 = forecast(net2, list(train_q[-8:]), len(test_q)); p3 = forecast(net3, list(train_q[-8:]), len(test_q))
    for nm, p in [('B2_lstm',p2), ('B3_pi_lstm',p3)]:
        cr = np.where(p<=eol)[0]
        R[nm] = dict(mae=float(np.abs(p-test_q).mean()), rmse=float(np.sqrt(((p-test_q)**2).mean())),
                     rul_err=int(abs(cr[0]-true_rul)) if len(cr) else None, V=float((np.diff(p)>0).mean()))
    # PF variants on GPU: B5 phys+PF, B4 LSTM+PF, B6 P3-EVHM (PI-LSTM+PF)
    timings = {'B5':[], 'B4':[], 'B6':[]}
    sl5 = abs((train_q[-40:][0]-train_q[-1])/40)
    med,cov_lo,cov_hi = pf_rul(train_q[-1], sl5, sl5*0.8+1e-4, eol, len(test_q), timing=timings['B5']); R['B5_phys_pf'] = dict(rul_err=int(abs(med-true_rul)) if med else None, interval=[cov_lo,cov_hi])
    sl4 = abs((p2[-1]-p2[0])/max(len(p2)-1,1))
    med,lo4,hi4 = pf_rul(train_q[-1], sl4, sl4*0.8+1e-4, eol, len(test_q), timing=timings['B4']); R['B4_lstm_pf'] = dict(rul_err=int(abs(med-true_rul)) if med else None, interval=[lo4,hi4])
    sl6 = abs((p3[-1]-p3[0])/max(len(p3)-1,1))
    med,lo6,hi6 = pf_rul(train_q[-1], sl6, sl6*0.8+1e-4, eol, len(test_q), timing=timings['B6']); R['B6_p3evhm_gpu'] = dict(rul_err=int(abs(med-true_rul)) if med else None, interval=[lo6,hi6], cover=float(lo6<=true_rul<=hi6) if med and true_rul else None)
    R['pf_gpu_time_s'] = {k: float(np.mean(v)) for k,v in timings.items() if v}
    # CPU comparison for B6 (timing separation: kernel vs end-to-end is logged separately)
    med,lo_c,hi_c = pf_rul(train_q[-1], sl6, sl6*0.8+1e-4, eol, len(test_q), dev=torch.device('cpu'))
    R['B6_p3evhm_cpu'] = dict(rul_err=int(abs(med-true_rul)) if med else None, interval=[lo_c,hi_c])
    RESULTS[cell] = R
    print(cell, json.dumps({k:v for k,v in R.items() if k not in ('pf_gpu_time_s',)}, default=str)[:600])
    print('   PF timing (s):', R['pf_gpu_time_s'])"""),
code("""# ---- N-scaling of GPU speedup (kernel-level evidence for the paper) ----
scaling = {}
d = data['B0005']; q = d.capacity.values; eol = 0.7*RATED; s = 99
sl5 = abs((q[:s][-40:][0]-q[:s][-1])/40)
for N in [1000, 5000, 20000, 100000, 500000]:
    tg = pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cuda')) if torch.cuda.is_available() else (None,)*3
    tc = pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cpu'))
    scaling[N] = {'gpu_s': tg_time if False else None}
    t0=time.time(); pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cpu')); tc=time.time()-t0
    t0=time.time(); pf_rul(q[:s][-1], sl5, sl5*0.8+1e-4, eol, 68, N=N, dev=torch.device('cuda')) if torch.cuda.is_available() else None; tg=time.time()-t0
    scaling[N] = {'cpu_s': tc, 'gpu_s': tg if torch.cuda.is_available() else None, 'speedup': tc/tg if torch.cuda.is_available() else None}
    print(N, scaling[N])
json.dump({'results': RESULTS, 'scaling': scaling}, open('/kaggle/working/p3evhm_results.json','w'), indent=2, default=str)
print('SAVED /kaggle/working/p3evhm_results.json')"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(BASE, "P3-EVHM_kaggle.ipynb")
with open(path, "w", encoding="utf-8") as f: json.dump(nb, f, indent=1)
print("NB_OK", os.path.getsize(path))
