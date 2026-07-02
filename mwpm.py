"""
Minimum-weight perfect matching (MWPM) for surface-code decoding.

The decoding problem after syndrome extraction is: given a set of "defects"
(lit stabilizers), pair them up — or match them to the code boundary — so that
the total weight (number of physical errors that must have occurred) is minimal.
This is exactly minimum-weight perfect matching on the complete graph of defects,
augmented with the option for any defect to terminate on a boundary.

We solve it *exactly* with a bitmask dynamic program (an O(2^n * n) exact matcher
that is equivalent to Edmonds' blossom result on these small dense instances),
and fall back to a greedy matcher only for the rare high-defect shots above a
cap. The exact matcher is unit-tested against brute force in `_self_test`.

No external libraries are used (no NetworkX / SciPy / PyMatching) — the matching
is implemented from scratch.
"""

from functools import lru_cache
from itertools import combinations
import math


def match_exact(defects, dist, boundary_dist):
    """Exact minimum-weight perfect matching with a boundary.

    Args:
        defects: list of defect ids (indices 0..n-1 referencing the rows/cols
                 of `dist`).
        dist: dist[i][j] = graph distance between defect i and defect j.
        boundary_dist: boundary_dist[i] = distance from defect i to the nearest
                       boundary (cost of terminating a chain on the boundary).

    Returns:
        (total_weight, pairs, boundary_matched)
        pairs: list of (i, j) defect-index pairs matched to each other.
        boundary_matched: list of defect indices matched to the boundary.
    """
    n = len(defects)
    if n == 0:
        return 0.0, [], []

    NEG = None

    # memo[mask] -> (cost, choice) where choice encodes how the lowest set bit
    # of `mask` was resolved.
    memo = {}

    def solve(mask):
        if mask == 0:
            return 0.0, None
        if mask in memo:
            return memo[mask]
        # lowest set bit = first unmatched defect i
        i = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << i)

        # Option A: match i to the boundary.
        best_cost, best_choice = boundary_dist[i] + solve(rest)[0], ("B", i)

        # Option B: match i to another unmatched defect j.
        m = rest
        while m:
            j = (m & -m).bit_length() - 1
            m &= ~(1 << j)
            c = dist[i][j] + solve(rest & ~(1 << j))[0]
            if c < best_cost:
                best_cost, best_choice = c, ("P", i, j)

        memo[mask] = (best_cost, best_choice)
        return memo[mask]

    full = (1 << n) - 1
    total, _ = solve(full)

    # Reconstruct the matching by replaying the recorded choices.
    pairs, boundary_matched = [], []
    mask = full
    while mask:
        _, choice = memo[mask]
        if choice[0] == "B":
            _, i = choice
            boundary_matched.append(i)
            mask &= ~(1 << i)
        else:
            _, i, j = choice
            pairs.append((i, j))
            mask &= ~(1 << i)
            mask &= ~(1 << j)
    return total, pairs, boundary_matched


def match_greedy(defects, dist, boundary_dist):
    """Greedy nearest-neighbour fallback for large defect counts."""
    remaining = set(range(len(defects)))
    pairs, boundary_matched = [], []
    total = 0.0
    while remaining:
        i = min(remaining)
        remaining.discard(i)
        # best partner among remaining, or the boundary
        best_j, best_c = None, boundary_dist[i]
        for j in remaining:
            if dist[i][j] < best_c:
                best_c, best_j = dist[i][j], j
        if best_j is None:
            boundary_matched.append(i)
        else:
            remaining.discard(best_j)
            pairs.append((i, best_j))
        total += best_c
    return total, pairs, boundary_matched


def match(defects, dist, boundary_dist, exact_cap=14):
    """Dispatch to the exact matcher, or greedy above `exact_cap` defects."""
    if len(defects) <= exact_cap:
        return match_exact(defects, dist, boundary_dist), False
    return match_greedy(defects, dist, boundary_dist), True


# ---------------------------------------------------------------------------
def _brute_force(n, dist, boundary_dist):
    """Reference exact matcher by exhaustive enumeration (small n only)."""
    best = [math.inf]

    def rec(remaining, acc):
        if not remaining:
            best[0] = min(best[0], acc)
            return
        i = remaining[0]
        rest = remaining[1:]
        # boundary
        rec(rest, acc + boundary_dist[i])
        # pair with each other
        for k, j in enumerate(rest):
            rec(rest[:k] + rest[k + 1:], acc + dist[i][j])

    rec(list(range(n)), 0.0)
    return best[0]


def _self_test(trials=300, seed=0):
    """Validate the exact DP matcher against brute force on random instances."""
    import random
    rng = random.Random(seed)
    for _ in range(trials):
        n = rng.randint(0, 7)
        dist = [[0] * n for _ in range(n)]
        for i, j in combinations(range(n), 2):
            w = rng.uniform(0.5, 5.0)
            dist[i][j] = dist[j][i] = w
        boundary = [rng.uniform(0.5, 5.0) for _ in range(n)]
        (dp_cost, _, _), _ = match(list(range(n)), dist, boundary)
        bf_cost = _brute_force(n, dist, boundary) if n else 0.0
        assert abs(dp_cost - bf_cost) < 1e-9, (n, dp_cost, bf_cost)
    return True


if __name__ == "__main__":
    print("MWPM self-test:", "PASS" if _self_test() else "FAIL")
