# p3evhm.demo: command-line quickstart. Works with zero data (synthetic), NASA .mat files, or a telemetry CSV.
# Usage:
#   python -m p3evhm.demo                          # synthetic cell, fast demo
#   python -m p3evhm.demo --source nasa --data-dir data --cell B0005
#   python -m p3evhm.demo --source csv --csv data/telemetry.csv --cell B0007
import argparse
import json
import os
import sys

from . import data, evaluate, models


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p3evhm.demo", description="P3-EVHM quickstart: SoH + RUL with a physics-informed LSTM and a particle filter.")
    ap.add_argument("--source", choices=["synthetic", "nasa", "csv"], default="synthetic")
    ap.add_argument("--data-dir", default="data", help="folder containing NASA .mat files (source=nasa)")
    ap.add_argument("--csv", default="data/telemetry.csv", help="telemetry CSV path (source=csv)")
    ap.add_argument("--cell", default=None, help="cell id, e.g. B0005 (nasa/csv sources)")
    ap.add_argument("--theta", type=int, default=None, help="EOL threshold %% of rated (default: 70, or 80 for B0007)")
    ap.add_argument("--frac", type=float, default=0.6, help="prediction start as fraction of life (0-1)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--particles", type=int, default=5000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "numpy"], help="auto=GPU if available")
    ap.add_argument("--rated", type=float, default=2.0, help="rated capacity in Ah")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    if args.source == "synthetic":
        df = data.synthetic_cell(seed=0); cell = "SYN01"
        thetas = {cell: 70}
    elif args.source == "nasa":
        df = data.load_nasa_mat(args.data_dir)
        cell = args.cell or "B0005"
        if cell not in set(df.cell):
            raise SystemExit(f"cell {cell} not found; available: {sorted(set(df.cell))}")
        thetas = {c: (80 if c == "B0007" else 70) for c in set(df.cell)}
    else:
        df = data.load_telemetry_csv(args.csv)
        cell = args.cell or sorted(set(df.cell))[0]
        if cell not in set(df.cell):
            raise SystemExit(f"cell {cell} not found; available: {sorted(set(df.cell))}")
        thetas = {c: (80 if c == "B0007" else 70) for c in set(df.cell)}
    theta = args.theta or thetas[cell]

    d = df[df.cell == cell].sort_values("cycle")
    q = d.capacity.values; T = d.temp.fillna(25.0).values
    eolc = evaluate.eol_cycle(q, args.rated, theta)
    if eolc is None:
        raise SystemExit(f"{cell} never reaches {theta}% of rated capacity; try --theta 80")
    start = max(int(round(eolc * args.frac)), 15); s = start - 1
    eol = theta / 100.0 * args.rated
    print(f"P3-EVHM demo | cell {cell} | rated {args.rated} Ah | EOL {theta}% = {eol:.2f} Ah "
          f"| true EOL cycle {eolc} | predicting from cycle {start} | device {args.device}")
    res = evaluate.run_variants(q[:s], T[:s], q[s:], eol, seed=0, N=args.particles,
                                epochs=args.epochs, device=args.device)
    soh = 100.0 * q[s] / args.rated
    state = "Healthy" if soh > 90 else "Ageing" if soh > 80 else "Replace soon" if soh > 70 else "End of life"
    print(f"SoH at start: {soh:.1f}% ({state}) | true RUL: {eolc - start} cycles")
    hdr = f"{'variant':<14}{'MAE':>10}{'RMSE':>10}{'RULerr':>8}{'cov':>7}{'V':>8}"
    print(hdr); print("-" * len(hdr))
    for var in ["B1_linear", "B2_lstm", "B3_pi_lstm", "B4_lstm_pf", "B5_phys_pf", "B6_p3evhm"]:
        m = res[var]
        fmt = lambda x: ("%10.4f" % x) if isinstance(x, float) else ("%10s" % str(x)) if x is not None else ("%10s" % "-")
        print(f"{var:<14}{fmt(m.get('mae'))}{fmt(m.get('rmse'))}"
              f"{('%8d' % m['rul_err']) if m.get('rul_err') is not None else '%8s' % '-'}"
              f"{('%7.2f' % m['cover']) if m.get('cover') is not None else '%7s' % '-'}"
              f"{('%8.3f' % m['V']) if m.get('V') is not None else '%8s' % '-'}")
    payload = dict(cell=cell, theta_pct=theta, start=start, true_eol_cycle=eolc, soh=round(soh, 2),
                   state=state, results=res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print("saved:", args.out)
    return payload


if __name__ == "__main__":
    main(sys.argv[1:])
