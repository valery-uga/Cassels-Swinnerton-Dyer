#!/usr/bin/env python3
"""Exact, offline verification of the determinantal witness over F_101.

This checks only the nonemptiness calculation used in the paper. It does not
verify the surface input or replace the symbolic proofs of descent. No external
packages or network access are required. Run with Python 3.10 or later.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

P = 101
Poly = dict[tuple[int, ...], int]
Matrix = list[list[int]]


def add(*polys: Poly) -> Poly:
    out: Poly = {}
    for f in polys:
        for e, a in f.items():
            out[e] = (out.get(e, 0) + a) % P
    return {e: a for e, a in out.items() if a}


def scale(f: Poly, a: int) -> Poly:
    return {e: (a * b) % P for e, b in f.items() if (a * b) % P}


def mul(f: Poly, g: Poly) -> Poly:
    out: Poly = {}
    for e, a in f.items():
        for d, b in g.items():
            ed = tuple(x + y for x, y in zip(e, d))
            out[ed] = (out.get(ed, 0) + a * b) % P
    return {e: a for e, a in out.items() if a}


def linear(coefficients: list[int]) -> Poly:
    n = len(coefficients)
    return {tuple(int(i == j) for i in range(n)): a % P
            for j, a in enumerate(coefficients) if a % P}


def monomials(n: int, degree: int) -> list[tuple[int, ...]]:
    """Descending lexicographic order."""
    if n == 1:
        return [(degree,)]
    return [(a,) + e for a in range(degree, -1, -1)
            for e in monomials(n - 1, degree - a)]


def derivative(f: Poly, i: int) -> Poly:
    out: Poly = {}
    for e, a in f.items():
        if e[i]:
            d = list(e)
            d[i] -= 1
            out[tuple(d)] = a * e[i] % P
    return out


def evaluate(f: Poly, v: list[int]) -> int:
    total = 0
    for e, a in f.items():
        for x, m in zip(v, e):
            a = a * pow(x, m, P) % P
        total += a
    return total % P


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions")
    return [[sum(x*y for x, y in zip(row, col)) % P
             for col in zip(*b)] for row in a]


def matadd(a: Matrix, b: Matrix, factor: int = 1) -> Matrix:
    return [[(x + factor*y) % P for x, y in zip(ra, rb)]
            for ra, rb in zip(a, b)]


def identity(n: int) -> Matrix:
    return [[int(i == j) for j in range(n)] for i in range(n)]


def eliminate(a: Matrix) -> tuple[int, list[int], Matrix]:
    """Row-echelon form, retaining indices of independent original rows."""
    b = [[x % P for x in row] for row in a]
    rows = list(range(len(b)))
    r = 0
    for c in range(len(b[0])):
        q = next((i for i in range(r, len(b)) if b[i][c]), None)
        if q is None:
            continue
        b[r], b[q] = b[q], b[r]
        rows[r], rows[q] = rows[q], rows[r]
        inv = pow(b[r][c], -1, P)
        b[r] = [x*inv % P for x in b[r]]
        for i in range(r+1, len(b)):
            t = b[i][c]
            if t:
                b[i] = [(x-t*y) % P for x, y in zip(b[i], b[r])]
        r += 1
        if r == len(b):
            break
    return r, rows[:r], b


def rank(a: Matrix) -> int:
    return eliminate(a)[0]


def determinant(a: Matrix) -> int:
    if len(a) != len(a[0]):
        raise ValueError("Determinant requires a square matrix")
    b = [[x % P for x in row] for row in a]
    ans = 1
    n = len(b)
    for c in range(n):
        q = next((i for i in range(c, n) if b[i][c]), None)
        if q is None:
            return 0
        if q != c:
            b[c], b[q] = b[q], b[c]
            ans = -ans
        pivot = b[c][c]
        ans = ans * pivot % P
        inv = pow(pivot, -1, P)
        for i in range(c+1, n):
            t = b[i][c] * inv % P
            for j in range(c+1, n):
                b[i][j] = (b[i][j] - t*b[c][j]) % P
            b[i][c] = 0
    return ans % P


def inverse(a: Matrix) -> Matrix:
    n = len(a)
    b = [[x % P for x in row] + identity(n)[i] for i, row in enumerate(a)]
    for c in range(n):
        q = next((i for i in range(c, n) if b[i][c]), None)
        if q is None:
            raise ValueError("Singular matrix")
        b[c], b[q] = b[q], b[c]
        inv = pow(b[c][c], -1, P)
        b[c] = [x*inv % P for x in b[c]]
        for i in range(n):
            if i != c:
                t = b[i][c]
                b[i] = [(x-t*y) % P for x, y in zip(b[i], b[c])]
    return [row[n:] for row in b]


def kernel(a: Matrix) -> list[list[int]]:
    b = [[x % P for x in row] for row in a]
    pivots = []
    r = 0
    for c in range(len(b[0])):
        q = next((i for i in range(r, len(b)) if b[i][c]), None)
        if q is None:
            continue
        b[r], b[q] = b[q], b[r]
        inv = pow(b[r][c], -1, P)
        b[r] = [x*inv % P for x in b[r]]
        for i in range(len(b)):
            if i != r:
                t = b[i][c]
                b[i] = [(x-t*y) % P for x, y in zip(b[i], b[r])]
        pivots.append(c)
        r += 1
        if r == len(b):
            break
    basis = []
    for f in range(len(b[0])):
        if f not in pivots:
            v = [0]*len(b[0]); v[f] = 1
            for i, c in enumerate(pivots):
                v[c] = -b[i][f] % P
            basis.append(v)
    return basis


def minors3(rows: list[list[Poly]]) -> list[Poly]:
    result = []
    for indices in itertools.combinations(range(5), 3):
        terms = []
        for perm in itertools.permutations(range(3)):
            sign = (-1)**sum(perm[i] > perm[j] for i in range(3) for j in range(i+1,3))
            terms.append(scale(mul(mul(rows[indices[0]][perm[0]],
                                       rows[indices[1]][perm[1]]),
                                   rows[indices[2]][perm[2]]), sign))
        result.append(add(*terms))
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    alpha = [0, 1, 2, 3, 4]
    a = [[0,2,3,-1,1], [1,0,2,1,-1], [3,3,0,0,1],
         [1,3,-2,0,1], [0,-2,1,0,0]]
    b = [[0 if i == j else a[i][j]*pow((alpha[i]-alpha[j]) % P, -1, P) % P
          for j in range(5)] for i in range(5)]
    d = [[alpha[i] if i == j else 0 for j in range(5)] for i in range(5)]
    ell = matmul(matmul(inverse(b), d), b)
    dl = matmul(d, ell)
    dpl = matadd(d, ell)
    require(determinant(a) == 20 and determinant(b) == 29, "Determinant mismatch")
    require(matadd(matmul(d,b),matmul(b,d),-1) == [[x%P for x in row] for row in a],
            "Commutator identity failed")
    u = [[13,11,93], [32,85,76], [24,73,17], [48,35,76], [5,10,81]]
    require(rank(u) == 3, "The chosen plane is not a plane")
    pu, qu = matmul(dpl,u), matmul(dl,u)
    plane_minors = minors3([[linear(u[i]), linear(pu[i]), linear(qu[i])] for i in range(5)])
    basis5 = monomials(3,5)
    rows = []
    for f in plane_minors:
        for e in monomials(3,2):
            g = mul(f,{e:1})
            rows.append([g.get(m,0) for m in basis5])
    plane_rank, selected, _ = eliminate(rows)
    require(plane_rank == 21, "Plane ideal does not contain all degree-five forms")
    plane_det = determinant([rows[i] for i in selected])
    require(plane_det != 0, "Selected plane certificate minor is zero")

    all_minors = minors3([[linear(identity(5)[i]), linear(dpl[i]), linear(dl[i])]
                         for i in range(5)])
    points = [[76,72,96,1,1], [31,43,95,1,1], [93,1,29,85,1],
              [1,67,22,1,1], [77,24,94,1,1]]
    kernels = []
    line_records = []
    for i, h in enumerate(points):
        di = matadd(d,identity(5),-alpha[i])
        li = matadd(ell,identity(5),-alpha[i])
        w = matmul(di,li)
        basis_w = kernel(w)
        require(len(basis_w) == 2, "Exceptional vector space has wrong dimension")
        require(all(x[0] == 0 for x in matmul(w,[[v] for v in h])), "Point not on line")
        nh1, nh2 = matmul(dpl,[[v] for v in h]), matmul(dl,[[v] for v in h])
        nh = [[h[j],nh1[j][0],nh2[j][0]] for j in range(5)]
        jac = [[evaluate(derivative(f,j),h) for j in range(5)] for f in all_minors]
        require(rank(nh) == 2 and rank(jac) == 3, "Line smoothness test failed")
        kernels.append(basis_w)
        line_records.append({"index":i,"point":h,"matrix_rank":rank(nh),
                             "jacobian_rank":rank(jac),"kernel_basis":basis_w})
    pairwise = []
    for i,j in itertools.combinations(range(5),2):
        r = rank(kernels[i]+kernels[j])
        require(r == 4,"Exceptional lines are not disjoint")
        pairwise.append({"pair":[i,j],"rank":r})

    result = {"field_prime":P,"alpha":alpha,"A":a,"B":b,"L":ell,
              "det_A":determinant(a),"det_B":determinant(b),"plane_matrix":u,
              "plane_coefficient_shape":[60,21],"plane_coefficient_rank":plane_rank,
              "ordering":"row triples lex; monomials descending lex; minor first, multiplier second",
              "plane_minor_rows_zero_based":selected,"plane_minor_determinant":plane_det,
              "lines":line_records,"pairwise_line_checks":pairwise,
              "scope":"Nonemptiness witness only; no surface arithmetic input is verified."}
    out = Path(__file__).resolve().parent / "determinantal_certificate.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Exact verification over F_101: PASS")
    print(f"det A = {determinant(a)}; det B = {determinant(b)}")
    print(f"Plane coefficient matrix: 60 x 21, rank {plane_rank}")
    print(f"Selected zero-based rows: {selected}")
    print(f"Selected 21 x 21 determinant: {plane_det}")
    print("Five line points: rank N = 2; Jacobian rank = 3 (each)")
    print("Ten pairs of exceptional lines: concatenated rank = 4 (each)")
    print("The certificate proves a nonempty genericity open, not the surface input.")


if __name__ == "__main__":
    main()
