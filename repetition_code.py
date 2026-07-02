"""
Distance-d repetition code the 1D warm-up decoder.

The repetition code protects against bit flips only. Its d-1 parity checks
compare neighbouring bits; a bit-flip chain lights the two checks at its ends,
so decoding is 1D minimum-weight matching, solved *exactly and trivially*:
walk left to right and, whenever the running parity is odd, flip the next bit.
A logical failure occurs when the residual flips more than half the register.

Under the code-capacity (perfect-measurement) model the crossing sits at
p = 0.5, the known repetition-code threshold a clean sanity check that the
Monte Carlo harness and logical bookkeeping are correct before trusting the
2D surface-code numbers.
"""

import numpy as np


def logical_error_rate(d, p, shots, rng):
    fails = 0
    for _ in range(shots):
        err = rng.random(d) < p                       # bit-flip errors
        # syndrome = parity between neighbours; decode by exact 1D matching
        corr = np.zeros(d, dtype=bool)
        parity = False
        for i in range(d - 1):
            parity ^= bool(err[i])
            if parity:                                # unmatched defect -> flip forward
                corr[i + 1] = True
                # note: flipping bit i+1 also flips its left check parity
        residual = err ^ corr
        # logical failure: majority of bits flipped (value decodes to 1)
        if residual.sum() * 2 > d:
            fails += 1
    return fails / shots


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Repetition code (code-capacity) logical error rate:")
    for d in (3, 5, 7):
        row = [f"d={d}:"]
        for p in (0.1, 0.3, 0.5, 0.7):
            row.append(f"p={p}:{logical_error_rate(d, p, 4000, rng):.3f}")
        print("  " + "  ".join(row))
