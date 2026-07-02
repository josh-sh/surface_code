"""
Rotated surface code (one sector) + MWPM decoder, from scratch.

We build the Z-type stabilizers of a distance-d rotated surface code and decode
independent X errors under the *code-capacity* noise model (perfect syndrome
measurement). By the code's symmetry the Z-error / X-stabilizer sector is
identical, so a single sector suffices to locate the threshold.

Pipeline
--------
  errors (X on data qubits)
      -> syndrome (which Z-stabilizers are lit)
      -> defect graph (defects = lit stabilizers, weights = shortest error path)
      -> minimum-weight perfect matching  (mwpm.py)
      -> correction (XOR of qubits along matched paths)
      -> logical check via a homological cut (coboundary of a boundary region)

Everything is validated in `self_test()`:
  * stabilizer count == (d^2 - 1) / 2
  * every single-qubit error lights 1 or 2 stabilizers
  * a straight qubit chain across the code is a weight-d logical (empty syndrome,
    odd overlap with the observable cut)
"""

from collections import deque
import numpy as np

from mwpm import match


class RotatedSurfaceCode:
    def __init__(self, d):
        assert d % 2 == 1 and d >= 3, "distance d must be odd, >= 3"
        self.d = d
        self.qubits = [(x, y) for y in range(d) for x in range(d)]
        self.qindex = {q: i for i, q in enumerate(self.qubits)}
        self.n = len(self.qubits)

        self._build_stabilizers()
        self._build_error_graph()
        self._build_observable()

    # -- construction --------------------------------------------------------
    def _build_stabilizers(self):
        d = self.d
        stabs = []  # each stabilizer = sorted tuple of data-qubit indices
        centers = []
        for iy in range(-1, d):
            for ix in range(-1, d):
                cx, cy = ix + 0.5, iy + 0.5
                if (ix + iy) % 2 != 0:
                    continue  # keep only one colour -> Z-type checks
                corners = []
                for dx in (-0.5, 0.5):
                    for dy in (-0.5, 0.5):
                        x, y = int(cx + dx), int(cy + dy)
                        if 0 <= x < d and 0 <= y < d:
                            corners.append(self.qindex[(x, y)])
                if len(corners) == 4:
                    stabs.append(tuple(sorted(corners)))
                    centers.append((cx, cy))
                elif len(corners) == 2:
                    # weight-2 boundary check: keep Z-checks on top/bottom only
                    if cy < 0 or cy > d - 1:
                        stabs.append(tuple(sorted(corners)))
                        centers.append((cx, cy))
        self.stabs = stabs
        self.stab_centers = centers
        self.num_stab = len(stabs)
        # which stabilizers touch each qubit
        self.qubit_to_stabs = [[] for _ in range(self.n)]
        for s, qs in enumerate(stabs):
            for q in qs:
                self.qubit_to_stabs[q].append(s)

    def _build_error_graph(self):
        """Nodes = stabilizers (0..S-1) plus boundaries bL=S, bR=S+1.
        The rough boundaries of this sector are left/right, so degree-1
        boundary qubits attach to bL or bR by their x-coordinate.
        Edges carry the data-qubit id that connects them."""
        S = self.num_stab
        self.bL, self.bR = S, S + 1
        self.node_count = S + 2
        adj = [[] for _ in range(self.node_count)]  # (neighbor, qubit_id)
        d = self.d
        for q in range(self.n):
            ss = self.qubit_to_stabs[q]
            if len(ss) == 2:
                a, b = ss
                adj[a].append((b, q))
                adj[b].append((a, q))
            elif len(ss) == 1:
                a = ss[0]
                qx, _ = self.qubits[q]
                b = self.bR if qx > (d - 1) / 2 else self.bL
                adj[a].append((b, q))
                adj[b].append((a, q))
            # len 0: qubit lights no Z-stabilizer (pure boundary freedom) -> skip
        self.adj = adj

    def _build_observable(self):
        """Homological cut separating the two boundaries. Region A = left
        boundary + stabilizers in the left half. The observable is the set of
        qubits crossing the cut (coboundary of A); a residual chain flips the
        logical iff it overlaps this set an odd number of times."""
        d = self.d
        A = set()
        A.add(self.bL)
        for s, (cx, cy) in enumerate(self.stab_centers):
            if cx < (d - 1) / 2:
                A.add(s)
        obs = set()
        for q in range(self.n):
            ss = list(self.qubit_to_stabs[q])
            # reconstruct this qubit's two endpoints in the error graph
            if len(ss) == 2:
                a, b = ss
            elif len(ss) == 1:
                a = ss[0]
                qx, _ = self.qubits[q]
                b = self.bR if qx > (d - 1) / 2 else self.bL
            else:
                continue
            if (a in A) != (b in A):
                obs.add(q)
        self.observable = obs

    # -- BFS on the error graph ---------------------------------------------
    def _bfs(self, source):
        """Return dist[node] and pred_qubit[node] for path reconstruction."""
        dist = [-1] * self.node_count
        pred_edge = [None] * self.node_count  # (prev_node, qubit_id)
        dist[source] = 0
        dq = deque([source])
        while dq:
            u = dq.popleft()
            for v, q in self.adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    pred_edge[v] = (u, q)
                    dq.append(v)
        return dist, pred_edge

    def _path_qubits(self, pred_edge, target):
        qs = []
        node = target
        while pred_edge[node] is not None:
            prev, q = pred_edge[node]
            qs.append(q)
            node = prev
        return qs

    # -- syndrome / decode ---------------------------------------------------
    def syndrome(self, error_bits):
        """error_bits: length-n 0/1 array of X errors. Returns lit stabilizers."""
        lit = []
        for s, qs in enumerate(self.stabs):
            if sum(error_bits[q] for q in qs) % 2:
                lit.append(s)
        return lit

    def decode(self, defects):
        """Return the set of qubits to flip as a correction."""
        if not defects:
            return set()
        bfs = [self._bfs(dfc) for dfc in defects]
        dists = [b[0] for b in bfs]
        m = len(defects)
        dist = [[0] * m for _ in range(m)]
        bdist = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                dist[i][j] = dist[j][i] = dists[i][defects[j]]
            bdist[i] = min(dists[i][self.bL], dists[i][self.bR])
        (total, pairs, boundary_matched), fell_back = match(
            list(range(m)), dist, bdist)

        correction = set()
        for i, j in pairs:
            _, pred = bfs[i]
            for q in self._path_qubits(pred, defects[j]):
                correction ^= {q}
        for i in boundary_matched:
            di, pred = bfs[i]
            tgt = self.bL if di[self.bL] <= di[self.bR] else self.bR
            for q in self._path_qubits(pred, tgt):
                correction ^= {q}
        return correction, fell_back

    def logical_failure(self, error_set, correction):
        residual = error_set ^ correction
        return len(residual & self.observable) % 2 == 1

    # -- validation ----------------------------------------------------------
    def self_test(self):
        d = self.d
        assert self.num_stab == (d * d - 1) // 2, (
            self.num_stab, (d * d - 1) // 2)
        # single-qubit errors light 1 or 2 Z-stabilizers
        for q in range(self.n):
            w = len(self.qubit_to_stabs[q])
            assert w in (1, 2), (q, w)
        # a straight horizontal qubit chain is a logical: empty syndrome, odd cut
        found_logical = False
        for y0 in range(d):
            chain = {self.qindex[(x, y0)] for x in range(d)}
            bits = np.zeros(self.n, dtype=int)
            for q in chain:
                bits[q] = 1
            if len(self.syndrome(bits)) == 0:
                assert len(chain & self.observable) % 2 == 1, y0
                found_logical = True
        assert found_logical, "no logical chain found"
        return True


if __name__ == "__main__":
    for d in (3, 5, 7):
        code = RotatedSurfaceCode(d)
        ok = code.self_test()
        print(f"d={d}: stabilizers={code.num_stab}, self-test={'PASS' if ok else 'FAIL'}")
