# Framework smoke test: exercises the whole pipeline on synthetic data in seconds. Run: python tests/test_smoke.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3evhm import data, evaluate, models


def main():
    df = data.synthetic_cell(n_cycles=120, seed=0)
    assert len(df) == 120 and df.capacity.min() > 1.0, "synthetic cell malformed"
    q = df.capacity.values; T = df.temp.values
    eolc = evaluate.eol_cycle(q, 2.0, 70)
    assert eolc is not None, "synthetic cell never crosses 70%"
    start = max(int(round(eolc * 0.6)), 15); s = start - 1
    eol = 0.7 * 2.0
    res = evaluate.run_variants(q[:s], T[:s], q[s:], eol, seed=0, N=1000, epochs=40, hidden=16)
    for var in ["B1_linear", "B2_lstm", "B3_pi_lstm", "B4_lstm_pf", "B5_phys_pf", "B6_p3evhm"]:
        assert var in res, f"missing variant {var}"
    for var in ["B1_linear", "B2_lstm", "B3_pi_lstm"]:
        assert 0.0 <= res[var]["V"] <= 1.0, f"V out of range for {var}"
        assert res[var]["mae"] is not None and res[var]["mae"] < 1.0, f"MAE implausible for {var}"
    print("smoke test PASS | true RUL:", eolc - start)
    for var in ["B1_linear", "B2_lstm", "B3_pi_lstm", "B4_lstm_pf", "B5_phys_pf", "B6_p3evhm"]:
        m = res[var]
        print(f"  {var:<12} MAE={m.get('mae')} RULerr={m.get('rul_err')} cov={m.get('cover')} V={m.get('V')}")
    # PF determinism with a fixed seed
    a = models.pf_rul(q[s - 1], 0.004, 0.004, eol, 30, N=500, seed=7, obs0=q[s])
    b = models.pf_rul(q[s - 1], 0.004, 0.004, eol, 30, N=500, seed=7, obs0=q[s])
    assert a[:3] == b[:3], "PF not deterministic under fixed seed"
    print("PF determinism PASS")
    print("TESTS_OK")


if __name__ == "__main__":
    main()
