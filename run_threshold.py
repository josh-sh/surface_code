"""
Monte-Carlo threshold study for the rotated surface code under code-capacity
(independent X error, perfect syndrome) noise, decoded with the from-scratch
MWPM decoder in surface_code.py.

Below the threshold p_th, increasing code distance d suppresses the logical
error rate; above it, larger codes are worse. The curves for different d
therefore cross at p_th. For code-capacity independent noise with MWPM the
accepted value is ~10.3%.

Outputs:
  threshold.png  - logical error rate vs physical error rate, one curve per d
  threshold.csv  - the raw numbers
"""

import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surface_code import RotatedSurfaceCode


def run(d, p, shots, rng):
    code = RotatedSurfaceCode(d)
    fails = 0
    fallbacks = 0
    qubit_ids = np.arange(code.n)
    for _ in range(shots):
        err = rng.random(code.n) < p
        error_set = set(qubit_ids[err].tolist())
        if not error_set:
            continue
        bits = err.astype(int)
        defects = code.syndrome(bits)
        correction, fb = code.decode(defects) if defects else (set(), False)
        fallbacks += int(fb)
        if code.logical_failure(error_set, correction):
            fails += 1
    return fails / shots, fallbacks / shots


def main():
    rng = np.random.default_rng(12345)
    distances = [3, 5, 7]
    ps = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16]
    shots = 4000

    t0 = time.time()
    results = {d: [] for d in distances}
    print(f"{'d':>3} {'p':>6} {'p_logical':>10} {'fallback':>9}")
    for d in distances:
        for p in ps:
            ler, fb = run(d, p, shots, rng)
            results[d].append(ler)
            print(f"{d:>3} {p:>6.3f} {ler:>10.4f} {fb:>9.4f}")
    print(f"elapsed {time.time() - t0:.1f}s")

    # plot
    plt.figure(figsize=(7, 5))
    for d in distances:
        plt.plot(ps, results[d], marker="o", label=f"d = {d}")
    plt.axvline(0.103, ls="--", color="gray", lw=1,
                label="accepted p$_{th}$ ≈ 10.3%")
    plt.xlabel("physical error rate  p")
    plt.ylabel("logical error rate  $p_L$")
    plt.title("Rotated surface code — MWPM threshold (code capacity)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold.png", dpi=150)

    with open("threshold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p"] + [f"d={d}" for d in distances])
        for i, p in enumerate(ps):
            w.writerow([p] + [results[d][i] for d in distances])
    print("wrote threshold.png, threshold.csv")


if __name__ == "__main__":
    main()
