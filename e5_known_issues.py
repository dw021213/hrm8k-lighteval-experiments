"""E5: re-check the KSM rows reported as data errors (rows are 0-based, revision c360cab)."""
import itertools
import math

from datasets import load_dataset

REV = "c360cabf8d733a82455565358b3dc965aab9ba8d"
ksm = load_dataset("HAERAE-HUB/HRM8K", "KSM", split="test", revision=REV)
for i in (3, 82, 109, 201, 323, 1254, 1262, 1396, 1420, 1427):
    print(f"KSM[{i}] answer={ksm[i]['answer']!r} original_answer={ksm[i]['original_answer']!r}")


def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


X, W = gauss_legendre(60)


def integrate(f, a, b):
    m, h = (a + b) / 2, (b - a) / 2
    return h * sum(w * f(m + h * x) for x, w in zip(X, W))


# Row 3: sum of three iterated integrals; expected (pi - 3) / 4
f = lambda x, y: x * x / (x * x + y * y) ** 1.5
i1 = integrate(lambda y: integrate(lambda x: f(x, y) - math.floor(x + y), math.sqrt(1 - y * y), y), 1 / math.sqrt(2), 1)
i2 = integrate(lambda y: integrate(lambda x: f(x, y), 0, y), 1, math.sqrt(2))
i3 = integrate(lambda y: integrate(lambda x: f(x, y), 0, math.sqrt(4 - y * y)), math.sqrt(2), 2)
print(f"row 3: numeric = {i1 + i2 + i3:.6f}, (pi-3)/4 = {(math.pi - 3) / 4:.6f}")

# Row 201: P(1 is not first and 4 is not fourth) over permutations of 1..5; expected 13/20
perms = list(itertools.permutations(range(1, 6)))
good = sum(1 for p in perms if p[0] != 1 and p[3] != 4)
print(f"row 201: {good}/{len(perms)} = {good / len(perms)} (13/20 = {13 / 20})")

# Row 109: take a helix with curvature 1 and torsion tau; beta(s) = integral of its principal normal.
# For the helix, N(s) = (-cos(s/c), -sin(s/c), 0), so beta is a planar circle of radius c = 1/sqrt(1+tau^2):
# curvature 1/c = sqrt(1+tau^2), torsion 0. Check numerically with finite differences.
tau = 0.75
a, b = 1 / (1 + tau**2), tau / (1 + tau**2)
c = math.sqrt(a * a + b * b)
N = lambda s: (-math.cos(s / c), -math.sin(s / c), 0.0)
beta = lambda s: (-c * math.sin(s / c), c * math.cos(s / c) - c, 0.0)  # antiderivative of N with beta(0)=0
h, s0 = 1e-3, 0.4
d1 = [(p - q) / (2 * h) for p, q in zip(beta(s0 + h), beta(s0 - h))]
d2 = [(p - 2 * m + q) / (h * h) for p, m, q in zip(beta(s0 + h), beta(s0), beta(s0 - h))]
assert all(abs(u - v) < 1e-6 for u, v in zip(d1, N(s0))), "beta' should equal N"
kappa_beta = math.sqrt(sum(v * v for v in d2))
print(f"row 109: tau={tau}: kappa_beta = {kappa_beta:.6f}, sqrt(1+tau^2) = {math.sqrt(1 + tau**2):.6f}, "
      f"beta stays in the plane z=0 -> torsion 0")

# Row 1427: A needs 3 successes (p = 1/4 per turn), B needs 2 (p = 1/6), A throws first. Expected 40/81.
from fractions import Fraction as Fr

PA, PB = Fr(1, 4), Fr(1, 6)


def a_to_move(a, b):
    if a == 3:
        return Fr(1)
    if b == 2:
        return Fr(0)
    # x = PA*y(a+1,b) + (1-PA)*(PB*x(a,b+1) + (1-PB)*x)  ->  solve for x
    return (PA * b_to_move(a + 1, b) + (1 - PA) * PB * a_to_move(a, b + 1)) / (1 - (1 - PA) * (1 - PB))


def b_to_move(a, b):
    if a == 3:
        return Fr(1)
    if b == 2:
        return Fr(0)
    return PB * a_to_move(a, b + 1) + (1 - PB) * a_to_move(a, b)


print(f"row 1427: P(A wins) = {a_to_move(0, 0)} = {float(a_to_move(0, 0)):.6f}")


# Row 1262: max of f_AB + f_BC + f_CD + f_DA with n copies of each letter; brute force for small n.
def pairs(s, x, y):
    total, seen_y = 0, 0
    for ch in reversed(s):
        if ch == y:
            seen_y += 1
        elif ch == x:
            total += seen_y
    return total


assert [pairs("ACBDBACDCBAD", x, y) for x, y in ("AB", "BC", "CD", "DA")] == [4, 4, 6, 3]  # example in the problem
for n in (1, 2, 3):
    letters = "A" * n + "B" * n + "C" * n + "D" * n
    best = max(sum(pairs(s, x, y) for x, y in ("AB", "BC", "CD", "DA")) for s in set(itertools.permutations(letters)))
    print(f"row 1262: n={n} max={best} (3n^2={3 * n * n})")
print(f"row 1262: if the pattern holds, n=2020 gives {3 * 2020 * 2020}")

# Scan: KSM answers with a variable letter that does not occur in the question
import re

CMDS = r"\\(?:frac|dfrac|tfrac|sqrt|pi|ln|log|cdot|times|left|right|pm|mp|emptyset|mathbb|pmod|bmod|infty|cos|sin|tan|theta|tau|alpha|beta|gamma|lambda|mu|sigma|omega|phi|varphi|text|mathrm|operatorname|le|ge|leq|geq|cup|cap|in|setminus)\b"
with_letters, missing_rows = 0, []
for i, r in enumerate(ksm):
    body = re.sub(r"\\[A-Za-z]+", " ", re.sub(CMDS, " ", r["answer"]))
    letters = set(re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", body)) - {"e", "i"}
    if letters:
        with_letters += 1
        if any(not re.search(rf"(?<![A-Za-z\\]){l}(?![A-Za-z])", r["question"]) for l in letters):
            missing_rows.append((i, r["answer"]))
print(f"scan: {with_letters} answers contain a variable letter; letter absent from question in {len(missing_rows)}: {missing_rows}")
