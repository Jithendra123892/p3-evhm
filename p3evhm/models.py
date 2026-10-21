# p3evhm.models: canonical PI-LSTM training/inference and particle-filter RUL engine.
# Conventions (fixed in validation v3): arrays index cycles 1..n via q[i] = capacity at cycle i+1;
# a prediction crossing at index i means RUL = i cycles after the start cycle; the particle filter
# observes the start-cycle measurement (obs0) before propagating, so crossing step k = RUL k.
import time
import numpy as np
import torch
import torch.nn as nn


class _LastState(nn.Module):
    def forward(self, o):
        out, _ = o
        return out[:, -1, :]


def _resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_pi_lstm(train_q, train_T=None, lam1=2.0, lam2=1.0, W=8, hidden=32, epochs=200,
                  lr=5e-3, seed=0, device="auto"):
    """Train the physics-informed LSTM (Eq. 3-5): data loss + monotonicity + temperature prior.

    train_q: 1-D array of capacities for the training cycles (float, length >= W+2).
    train_T: 1-D array of mean temperatures aligned with train_q (optional; prior needs it).
    Returns the trained nn.Module.
    """
    q = np.asarray(train_q, dtype=float)
    dev = _resolve_device(device)
    if train_T is None:
        train_T = np.full(len(q), 25.0)
    T = np.asarray(train_T, dtype=float)
    if len(q) < W + 2:
        raise ValueError(f"need at least W+2={W+2} training cycles, got {len(q)}")
    torch.manual_seed(seed); np.random.seed(seed)
    X = np.lib.stride_tricks.sliding_window_view(q, W + 1)
    Xt = torch.tensor(X[:, :W], dtype=torch.float32, device=dev).unsqueeze(-1)
    Yt = torch.tensor(X[:, -1], dtype=torch.float32, device=dev)
    last = torch.tensor(X[:, W - 1], dtype=torch.float32, device=dev)
    Tt = torch.tensor(T[W - 1:-1], dtype=torch.float32, device=dev)
    dT = -(q[1:] - q[:-1])
    A = float(np.polyfit((T[1:] - np.mean(T))[1:], dT[1:], 1)[0]) if len(dT) > 10 else 0.0
    g = torch.clamp(A * (Tt - Tt.mean()), min=0.0)
    net = nn.Sequential(nn.LSTM(1, hidden, batch_first=True), _LastState(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(Xt).squeeze(-1)
        loss = mse(out, Yt) + lam1 * torch.relu(out - last).mean() + lam2 * ((out[:-1] - out[1:]) - g[1:]).pow(2).mean()
        loss.backward(); opt.step()
    return net


def forecast(net, train_q, steps, W=8):
    """Recursive multi-step forecast; returns `steps` capacities starting at the cycle after train_q."""
    hist = list(np.asarray(train_q, dtype=float)[-W:]); preds = []
    with torch.no_grad():
        for _ in range(steps):
            xw = torch.tensor(np.array(hist[-W:]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = float(net(xw).squeeze()); preds.append(p); hist.append(p)
    return np.array(preds)


def pi_lstm_forecast(train_q, train_T=None, steps=60, lam1=2.0, lam2=1.0, W=8, hidden=32,
                     epochs=200, lr=5e-3, seed=0, device="auto", return_train_time=False):
    """Convenience: train + recursive forecast in one call."""
    t0 = time.time()
    net = train_pi_lstm(train_q, train_T, lam1=lam1, lam2=lam2, W=W, hidden=hidden,
                        epochs=epochs, lr=lr, seed=seed, device=device)
    dt = time.time() - t0
    preds = forecast(net, train_q, steps, W=W)
    return (preds, dt) if return_train_time else preds


def pf_rul(q0, slope_mu, slope_sd, eol, steps, N=2000, proc=0.003, seed=0, obs0=None,
           obs_sigma=0.01, device="numpy", return_traj=False):
    """Particle-filter RUL estimation (Eq. 6-8).

    q0: capacity at the start cycle; obs0: measured capacity at the start cycle (recommended);
    slope_mu/slope_sd: decline-rate prior (Ah/cycle); eol: EOL capacity threshold;
    steps: horizon length; N: particle count.
    Returns (rul_median, rul_lo, rul_hi, elapsed_s[, median_trajectory]) or (None, None, None, dt) if no crossing.
    device="cuda" runs the particle math on the GPU via PyTorch CUDA operations.
    """
    t0 = time.time()
    if device == "numpy":
        rng = np.random.default_rng(seed)
        qi = np.clip(rng.normal(q0, 0.01, N), eol, None)
        di = np.clip(np.abs(rng.normal(slope_mu, slope_sd, N)), 1e-5, None)
        if obs0 is not None:
            wts = np.exp(-((obs0 - qi) ** 2) / (2 * obs_sigma ** 2))
            if wts.sum() > 0:
                qi = qi[rng.choice(N, N, p=wts / wts.sum())]
        cross = np.full(N, -1)
        qf = qi.copy()
        traj = np.zeros(steps)
        for k in range(1, steps + 1):
            qf = qf - di + rng.normal(0, proc, N)
            newly = (cross < 0) & (qf <= eol)
            cross[newly] = k
            traj[k - 1] = float(np.median(qf))
        valid = cross[cross > 0]; rul = valid[valid <= steps]
        dt = time.time() - t0
        if len(rul) == 0:
            return (None, None, None, dt, traj) if return_traj else (None, None, None, dt)
        lo, med, hi = np.percentile(rul, [2.5, 50, 97.5])
        out = (int(med), float(lo), float(hi), dt)
        return out + (traj,) if return_traj else out
    # torch path (CPU or CUDA)
    dev = _resolve_device(device)
    gen = torch.Generator(device=dev.type).manual_seed(seed)
    qi = torch.clamp(torch.normal(q0, 0.01, (N,), device=dev, generator=gen), min=eol)
    di = torch.clamp(torch.normal(slope_mu, slope_sd, (N,), device=dev, generator=gen).abs(), min=1e-5)
    if obs0 is not None:
        wts = torch.exp(-((float(obs0) - qi) ** 2) / (2 * obs_sigma ** 2))
        if wts.sum() > 0:
            qi = qi[torch.multinomial(wts / wts.sum(), N, replacement=True)]
    cross = torch.full((N,), -1.0, device=dev)
    for k in range(1, steps + 1):
        qi = qi - di + torch.randn(N, device=dev, generator=gen) * proc
        newly = (cross < 0) & (qi <= eol)
        cross[newly] = float(k)
    valid = cross[cross > 0]; rul = valid[valid <= steps]
    dt = time.time() - t0
    if len(rul) == 0:
        return (None, None, None, dt)
    lo, med, hi = np.percentile(rul.detach().cpu().numpy(), [2.5, 50, 97.5])
    return (int(med), float(lo), float(hi), dt)


def save_pretrained(net, path, config=None):
    """Save a pretrained PI-LSTM checkpoint (state_dict + architecture config)."""
    import os
    cfg = config or {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": cfg}, path)
    return path


def load_pretrained(path, device="auto"):
    """Load a checkpoint saved by save_pretrained. Returns (net, config).

    The checkpoint's config supplies W/hidden so the architecture always matches the weights.
    """
    dev = _resolve_device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    hidden = int(cfg.get("hidden", 32)); W = int(cfg.get("W", 8))
    net = nn.Sequential(nn.LSTM(1, hidden, batch_first=True), _LastState(), nn.Linear(hidden, 1)).to(dev)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    cfg["device"] = str(dev)
    return net, cfg


def finetune(net, train_q, train_T=None, freeze_body=True, W=8, epochs=30, lr=1e-3,
             lam1=2.0, lam2=1.0, seed=0, device="auto"):
    """Adapt a pretrained net to a new cell WITHOUT training from scratch.

    freeze_body=True freezes every LSTM body layer (requires_grad=False) and adapts only the
    output head on the new cell's early cycles - seconds instead of a full training run, and it
    protects the fleet-learned temporal features from catastrophic forgetting.
    Returns (net, elapsed_s, n_trainable_params).
    """
    q = np.asarray(train_q, dtype=float)
    dev = _resolve_device(device)
    if train_T is None:
        train_T = np.full(len(q), 25.0)
    T = np.asarray(train_T, dtype=float)
    if len(q) < W + 2:
        raise ValueError(f"need at least W+2={W+2} training cycles, got {len(q)}")
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.to(dev)
    if freeze_body:
        for p in net[1].parameters():
            p.requires_grad = False
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    X = np.lib.stride_tricks.sliding_window_view(q, W + 1)
    Xt = torch.tensor(X[:, :W], dtype=torch.float32, device=dev).unsqueeze(-1)
    Yt = torch.tensor(X[:, -1], dtype=torch.float32, device=dev)
    last = torch.tensor(X[:, W - 1], dtype=torch.float32, device=dev)
    Tt = torch.tensor(T[W - 1:-1], dtype=torch.float32, device=dev)
    dT = -(q[1:] - q[:-1])
    A = float(np.polyfit((T[1:] - np.mean(T))[1:], dT[1:], 1)[0]) if len(dT) > 10 else 0.0
    g = torch.clamp(A * (Tt - Tt.mean()), min=0.0)
    opt = torch.optim.Adam(trainable, lr=lr)
    mse = nn.MSELoss()
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(Xt).squeeze(-1)
        loss = mse(out, Yt) + lam1 * torch.relu(out - last).mean() + lam2 * ((out[:-1] - out[1:]) - g[1:]).pow(2).mean()
        loss.backward(); opt.step()
    dt = time.time() - t0
    net.eval()
    return net, dt, n_train


def violation_rate(pred):
    """Eq. (8): fraction of predicted steps that increase capacity (physically implausible)."""
    p = np.asarray(pred, dtype=float)
    return float((np.diff(p) > 0).mean())
