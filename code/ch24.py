"""Chapter 24 (Credible Government Policies I): exercises 24.1-24.3.

24.1  finitely repeated black-box economies (Benoit-Krishna): checks by enumeration.
24.2  Stokey's example without a one-period Nash equilibrium: worst/best SPE, the stick sequence v_j, and the
      lowest discount factor that sustains Ramsey (with and without public randomization).
24.3  Kydland-Prescott: worst and best SPE values by the APS operator on intervals, the four-equation system
      (24.13.5)-(24.13.8) for delta = .08, and recursive SPEs that attain the worst value.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from rmtlib import FIG

# ============================================================== 24.1
print("=== 24.1")
# payoffs r[y][x] = u(x, x, y); C = diagonal
r2 = {"yM": {"xM": 10, "xH": 20}, "yH": {"xM": 4, "xH": 15}}
r3 = {"yL": {"xL": 3, "xM": 7, "xH": 9}, "yM": {"xL": 1, "xM": 10, "xH": 20}, "yH": {"xL": 0, "xM": 4, "xH": 15}}
for name, r in (("2x2", r2), ("3x3", r3)):
    ys = list(r); xs = list(r[ys[0]])
    H = {x: max(ys, key=lambda y: r[y][x]) for x in xs}
    C = [(x, "y" + x[1]) for x in xs]
    NE = [(x, y) for (x, y) in C if H[x] == y]
    R = max(C, key=lambda c: r[c[1]][c[0]])
    print(f"  {name}: best responses H = {H}; Nash equilibria {NE} values {[r[y][x] for x, y in NE]}; Ramsey {R} value {r[R[1]][R[0]]}")
# (e) twice repeated, reward NE M (10), punish NE L (3): 15 + 10 d >= 20 + 3 d
print(f"  (e) Ramsey in period 1 of the twice-repeated economy iff 15 + 10d >= 20 + 3d, i.e. d >= 5/7 = {5/7:.5f}")
d = .8
on = 15 + 15 * d + 10 * d ** 2
print(f"  (f) d = .8: on path (H,H),(H,H),(M,M): {on:.2f};  period-2 IC: 15 + 10d = {15+10*d:.2f} >= 20 + 3d = {20+3*d:.2f}")
conts = {"(L,L)": (3, 3), "(L,M)": (3, 10), "(M,L)": (10, 3), "(M,M)": (10, 10), "((H,H),(M,M))": (15, 10)}
for k, (w2, w3) in conts.items():
    dev = 20 + d * w2 + d ** 2 * w3
    print(f"      period-1 deviation followed by {k:14s}: {dev:6.2f} -> deters: {dev <= on}")
print(f"      period-1 IC with the harshest punishment holds for d >= {(-12 + np.sqrt(144 + 140)) / 14:.4f}")

# ============================================================== 24.2
print("\n=== 24.2")
d = .1 ** (1 / 20)
print(f"  delta = .1^(1/20) = {d:.6f}; delta^20 = {d**20:.6f}")
vR, dev_R, vlo = 10., 20., 1.
print(f"  worst value = min_y max_eta r(h(y), eta) = min(1, 20) = 1;  Ramsey IC: 10 >= (1-d)20 + d*1 = {(1-d)*20 + d:.4f}")
vj = d ** -np.arange(0, 21)
print("  v_j = delta^(-j):", np.round(vj, 5))
print("  stick periods before Ramsey for sigma_j: 20 - j;  IC at stick periods v_j >= (1-d)*1 + d*1 = 1:", bool(np.all(vj >= 1 - 1e-12)))
val = (1 - d) * np.sum(d ** np.arange(20) * 0) + d ** 20 * 10
print(f"  value of 20 sticks then Ramsey: {val:.10f}")
# lowest delta: with public randomisation 10 >= 20(1-d) + d  -> d >= 10/19
print(f"  (l) with public randomisation: delta >= 10/19 = {10/19:.6f}")
dstar = brentq(lambda x: x + x ** 2 + x ** 3 - 1, .3, .9)
print(f"      pure strategies: paths are L^k H^inf for delta < {(-9 + np.sqrt(481)) / 20:.4f}; Ramsey sustainable iff some k has .1 <= delta^k <= 2 - 1/delta")


def pure_ok(x):
    ks = np.arange(1, 200)
    dk = x ** ks
    return np.any((dk >= .1 - 1e-15) & (dk <= 2 - 1 / x + 1e-15))


grid = np.linspace(.50, .60, 100001)
okg = np.array([pure_ok(x) for x in grid])
print(f"      smallest delta on a fine grid with a valid k: {grid[np.argmax(okg)]:.5f};  root of d + d^2 + d^3 = 1: {dstar:.6f};"
      f" all larger grid points ok: {bool(okg[np.argmax(okg):].all())}")
print(f"      at delta*: worst SPE = 3 sticks then Ramsey, value 10 delta*^3 = {10*dstar**3:.5f} = 20 - 10/delta* = {20 - 10/dstar:.5f}")

# ============================================================== 24.3
print("\n=== 24.3  (U* = 5, theta = 1, y in [0, 10])")
Us, th = 5., 1.
Hf = lambda x: np.clip((th * Us + th ** 2 * x) / (1 + th ** 2), 0, 10)
r = lambda x, y: -.5 * ((Us - th * (y - x)) ** 2 + y ** 2)
yN = th * Us
vN, vR = r(yN, yN), r(0., 0.)
print(f"  Nash y = {yN}, vN = {vN};  Ramsey y = 0, vR = {vR};  temptation at Ramsey r(0,H(0)) = {r(0., Hf(0.)):.4f}")
Y = np.linspace(0, 10, 20001)
rc, rd = r(Y, Y), r(Y, Hf(Y))                         # r(h(y), y) and r(h(y), H(h(y)))


def aps(delta, tol=1e-12, maxit=20000):
    """Iterate the APS operator on intervals [a, b] starting from [min_C r, max_C r]."""
    a, b = rc.min(), rc.max()
    for it in range(maxit):
        ok_b = (1 - delta) * rc + delta * b >= (1 - delta) * rd + delta * a - 1e-14
        bn = np.max(np.where(ok_b, (1 - delta) * rc + delta * b, -np.inf))
        w1 = np.maximum(a, ((1 - delta) * (rd - rc) + delta * a) / delta)
        ok_a = w1 <= b + 1e-14
        an = np.min(np.where(ok_a, (1 - delta) * rc + delta * w1, np.inf))
        if abs(an - a) < tol and abs(bn - b) < tol:
            return an, bn, it
        a, b = an, bn
    return a, b, maxit


d = .95
a, b, it = aps(d)
ysharp = 10.
v1 = (rd.min() - (1 - d) * r(ysharp, ysharp)) / d
abreu = (1 - d) * r(ysharp, ysharp) + d * vR
print(f"  (i) delta = .95: APS interval [{a:.6f}, {b:.6f}] after {it} iterations;  min_y r(y,H(y)) = {rd.min():.4f} at y = {Y[np.argmin(rd)]:.1f}")
print(f"      continuation after the stick y=10: v1 = {v1:.6f} (inside [{a:.4f},{b:.4f}]);  values: Ramsey {vR}, Nash {vN}, Abreu stick-and-carrot {abreu:.4f}, worst {rd.min():.4f}")
print(f"      Abreu IC: {abreu:.4f} >= (1-d) r(10,H(10)) + d*abreu = {(1-d)*r(10., Hf(10.)) + d*abreu:.4f}")
# (j) method 3 and method 1
yt = np.sqrt(-2 * v1 - Us ** 2)
ic3 = (1 - d) * r(yt, Hf(yt)) + d * rd.min()
print(f"  (j) method 3: y=10 once, then y~ = {yt:.6f} forever (r(y~,y~) = {r(yt, yt):.6f} = v1); IC at y~: {v1:.4f} >= {ic3:.4f}: {v1 >= ic3}")
vs = [rd.min()]
while True:
    nxt = (vs[-1] - (1 - d) * r(10., 10.)) / d
    if nxt > vR: break
    vs.append(nxt)
J = len(vs) - 1
rt = (vs[-1] - d * vR) / (1 - d); ytr = np.sqrt(-2 * rt - Us ** 2)
ict = (1 - d) * r(ytr, Hf(ytr)) + d * rd.min()
print(f"      method 1: y = 10 at states v_0..v_{J-1} ({J} periods, v_0 = {vs[0]:.4f}), then at v_{J} = {vs[-1]:.4f} y = {ytr:.4f} once"
      f" (IC {vs[-1]:.4f} >= {ict:.4f}), then Ramsey")
# (k), (l)
dc = (vR - r(0., Hf(0.))) / (vN - r(0., Hf(0.)))
dtc = (vR - r(0., Hf(0.))) / (rd.min() - r(0., Hf(0.)))
print(f"  (k) delta_c = {dc:.6f};  (l) delta~_c = {dtc:.6f}; v1 needed at delta~_c: {(rd.min() - (1-dtc)*r(10.,10.))/dtc:.6f} (= vR)")
a_l, b_l, _ = aps(dtc)
print(f"      APS at delta~_c: [{a_l:.6f}, {b_l:.6f}]")


# (m) delta = .08: four equations
def four(z, delta):
    ylo, yhi = z
    vlo = r(ylo, Hf(ylo)); vhi = r(yhi, yhi)
    return [vhi - ((1 - delta) / delta * (vlo - r(ylo, ylo)) + vlo),
            vhi - ((1 - delta) * r(yhi, Hf(yhi)) + delta * vlo)]


d = .08
ylo, yhi = fsolve(four, [8., 2.], args=(d,), xtol=1e-12)
vlo, vhi = r(ylo, Hf(ylo)), r(yhi, yhi)
a8, b8, it8 = aps(d)
print(f"  (m) delta = .08: y_lo = {ylo:.6f}, y_hi = {yhi:.6f}, worst = {vlo:.6f}, best = {vhi:.6f};  APS: [{a8:.6f}, {b8:.6f}] ({it8} iterations)")
print(f"      residuals {four([ylo, yhi], d)}")

# figure: best and worst SPE values against delta
ds = np.r_[np.linspace(.01, .3, 59), np.linspace(.31, .99, 69)]
lo, hi = [], []
for x in ds:
    aa, bb, _ = aps(x, tol=1e-10)
    lo.append(aa); hi.append(bb)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(ds, hi, "k", lw=1.5, label="best SPE value")
ax.plot(ds, lo, "k--", lw=1.5, label="worst SPE value")
ax.axhline(vR, color="grey", lw=.7, ls=":"); ax.text(.99, vR + .8, "Ramsey", ha="right", fontsize=8)
ax.axhline(vN, color="grey", lw=.7, ls=":"); ax.text(.99, vN + .8, "Nash", ha="right", fontsize=8)
ax.axvline(dtc, color="grey", lw=.5); ax.text(dtc + .01, -47, r"$\tilde\delta_c=1/8$", fontsize=8)
ax.axvline(dc, color="grey", lw=.5); ax.text(dc + .01, -47, r"$\delta_c=1/3$", fontsize=8)
ax.set_ylim(-60, -8); ax.set_xlabel(r"$\delta$"); ax.set_ylabel("value"); ax.legend(fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(FIG + "ch24_03.pdf"); plt.close()
print("  figure data: worst value at delta = .05, .1, .125, .2:", [round(lo[np.argmin(abs(ds - x))], 4) for x in (.05, .1, .125, .2)])
