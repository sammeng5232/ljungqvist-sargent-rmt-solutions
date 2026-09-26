"""Chapter 23 (Optimal Unemployment Insurance): exercise 23.1, Hopenhayn and Nicolini's calibration.

beta = .999 (weeks), u(c) = c^(1-sigma)/(1-sigma) with sigma = .5, i.e. u(c) = 2 sqrt(c), w = 100, p(a) = 1 - exp(-r a),
r chosen so that the autarky hazard p(a*) = .1.
(a) autarky: closed form for r, confirmed by the iterative algorithm of section 23.2.1.
(b) C(V) with unobservable effort: Howard policy iteration on the Bellman equation (23.2.11) on 2001 nodes, with a vectorized
    golden-section search over V^u and linear interpolation of C (policy evaluation is an exact sparse linear solve);
    then time paths from three initial V, compared with full information and with the best constant benefit.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from rmtlib import FIG

beta, w = .999, 100.
u = lambda c: 2 * np.sqrt(c)
uinv = lambda x: (np.maximum(x, 0) / 2) ** 2
Ve = u(w) / (1 - beta)

# ------------------------------------------------------------------ (a) autarky and calibration of r
# p(a*) = .1 => exp(-r a*) = .9.  FOC beta r .9 (Ve - Vu) = 1 and Vu (1 - .9 beta) = -a* + .1 beta Ve  =>  r u(w) = (1-.9b)/(.9b) - ln(10/9)
r = ((1 - .9 * beta) / (.9 * beta) - np.log(10 / 9)) / u(w)
p = lambda a: 1 - np.exp(-r * a)
astar = np.log(10 / 9) / r
Vaut = Ve - 1 / (.9 * beta * r)
# the book's iterative algorithm: given Vu_j, effort from the FOC, then update Vu
Vj = 0.
for j in range(200000):
    aj = max(0., np.log(r * beta * (Ve - Vj)) / r)
    Vn = u(0.) - aj + beta * (p(aj) * Ve + (1 - p(aj)) * Vj)
    if abs(Vn - Vj) < 1e-10:
        break
    Vj = Vn
print("=== 23.1(a) autarky")
print(f"  Ve = {Ve:.4f}; calibrated r = {r:.6e}; a* = {astar:.4f}, p(a*) = {p(astar):.6f}, Vaut = {Vaut:.4f}")
print(f"  iterative algorithm: converged in {j} iterations to Vaut = {Vn:.4f}, a = {aj:.4f}, hazard {p(aj):.6f}; expected duration {1 / p(aj):.2f} weeks")
Vmax = Ve - 1 / (beta * r)                                            # V^u below this keeps search effort positive
print(f"  continuation values lie in [Vaut, Ve - 1/(beta r)] = [{Vaut:.4f}, {Vmax:.4f}]")

# ------------------------------------------------------------------ (b) the Bellman equation C(V) = min_{Vu} c + beta (1 - p(a)) C(Vu)
aof = lambda Vu: np.maximum(0., np.log(r * beta * (Ve - Vu)) / r)
def objective(V, Vu, Cspl):
    a = aof(Vu); pa = p(a)
    arg = V + a - beta * (pa * Ve + (1 - pa) * Vu)                    # u(c) from promise keeping
    return uinv(arg) + beta * (1 - pa) * Cspl(Vu) + 1e6 * np.maximum(-arg, 0)
gr = (np.sqrt(5) - 1) / 2
def golden(V, Cspl, lo, hi, it=60):
    lo = np.full_like(V, lo); hi = np.full_like(V, hi)
    x1 = hi - gr * (hi - lo); x2 = lo + gr * (hi - lo)
    f1 = objective(V, x1, Cspl); f2 = objective(V, x2, Cspl)
    for _ in range(it):
        left = f1 < f2
        hi = np.where(left, x2, hi); lo = np.where(left, lo, x1)
        x2n = np.where(left, x1, lo + gr * (hi - lo)); x1n = np.where(left, hi - gr * (hi - lo), x2)
        x1, x2 = x1n, x2n
        f1 = objective(V, x1, Cspl); f2 = objective(V, x2, Cspl)
    x = (lo + hi) / 2
    return x, objective(V, x, Cspl)
import scipy.sparse as sp
import scipy.sparse.linalg as spla
N = 2001
Vg = Vaut + (Vmax - Vaut) * (1 - np.cos(np.linspace(0, np.pi, N))) / 2   # clustered at both ends
def evaluate(pol):
    """Howard step: cost of following V^u = pol(V) forever, C = c + beta (1 - p) W C, W = linear interpolation weights."""
    a = aof(pol); pa = p(a)
    c = uinv(Vg + a - beta * (pa * Ve + (1 - pa) * pol))
    j = np.clip(np.searchsorted(Vg, pol, side="right") - 1, 0, N - 2)
    wt = (Vg[j + 1] - pol) / (Vg[j + 1] - Vg[j])
    rows = np.r_[np.arange(N), np.arange(N)]; cols = np.r_[j, j + 1]
    W = sp.csr_matrix((np.r_[wt, 1 - wt], (rows, cols)), shape=(N, N))
    return spla.spsolve((sp.identity(N, format="csr") - sp.diags(beta * (1 - pa)) @ W).tocsc(), c)
lin = lambda C: (lambda x: np.interp(x, Vg, C))
pol = np.full(N, Vaut)                                                # start: continuation value = autarky
C = evaluate(pol)
for it in range(500):
    pol_new, _ = golden(Vg, lin(C), Vaut, Vmax, it=80)
    C_new = evaluate(pol_new)
    dC, dpol = np.max(np.abs(C_new - C)), np.max(np.abs(pol_new - pol))
    C, pol = C_new, pol_new
    if dC < 1e-9 and dpol < 1e-7:
        break
Cs = CubicSpline(Vg, C)
bell = golden(Vg, lin(C), Vaut, Vmax, it=80)[1]
print(f"\n=== 23.1(b) C(V) with unobservable effort: policy iteration converged in {it} steps (last changes: C {dC:.1e}, policy {dpol:.1e});"
      f" Bellman residual {np.max(np.abs(bell - C)):.1e}")
print(f"  C increasing: {np.all(np.diff(C) > 0)}; C convex (second differences on the nonuniform grid): "
      f"{np.all(np.diff(np.diff(C) / np.diff(Vg)) > -1e-9)}; grid points with V^u(V) >= V: {np.where(pol[1:] >= Vg[1:])[0] + 1} of {N}")
# check the first-order relation C'(V) = C'(Vu) + C(Vu)/(Ve - Vu) = sqrt(c)  (from (23.2.8) with p = 1 - exp(-ra))
k = np.arange(100, N - 100, 100)
a_k = aof(pol[k]); c_k = uinv(Vg[k] + a_k - beta * (p(a_k) * Ve + (1 - p(a_k)) * pol[k]))
lhs = Cs(Vg[k], 1); rhs = Cs(pol[k], 1) + Cs(pol[k]) / (Ve - pol[k])
print(f"  envelope check C'(V) = 1/u'(c) = sqrt(c): max rel. error {np.max(np.abs(lhs - np.sqrt(c_k)) / np.sqrt(c_k)):.2e};"
      f" Euler check C'(V) = C'(Vu) + C(Vu)/(Ve-Vu): max rel. error {np.max(np.abs(lhs - rhs) / lhs):.2e}")

# full-information cost and the best constant-benefit scheme (worker chooses effort) delivering the same V
def C_full(V):
    def cost(a):
        pa = p(a); arg = V * (1 - beta * (1 - pa)) + a - beta * pa * Ve
        return uinv(arg) / (1 - beta * (1 - pa)) + 1e6 * max(-arg, 0)
    res = minimize_scalar(cost, bounds=(0, 3 * astar), method="bounded", options={"xatol": 1e-10})
    return res.fun, res.x
def constant_benefit(V0):
    def value(b):                                                     # worker's value with benefit b forever, effort optimal
        Vu = Vaut
        for _ in range(100000):
            a = max(0., np.log(r * beta * (Ve - Vu)) / r)
            Vn = u(b) - a + beta * (p(a) * Ve + (1 - p(a)) * Vu)
            if abs(Vn - Vu) < 1e-9:
                return Vn, a
            Vu = Vn
    b = brentq(lambda b: value(b)[0] - V0, 1e-9, w)
    Vb, a = value(b)
    return b, a, b / (1 - beta * (1 - p(a)))

def simulate(V0, T=50):
    V = V0; rows = []
    for t in range(1, T + 1):
        Vu = golden(np.array([V]), lin(C), Vaut, Vmax, it=80)[0][0]
        a = aof(Vu); c = uinv(V + a - beta * (p(a) * Ve + (1 - p(a)) * Vu))
        rows.append((t, V, c / w, a, p(a))); V = Vu
    return np.array(rows)

paths = {}
print("\n  Table (version of Hopenhayn and Nicolini's table 1): replacement ratio c/w by week of unemployment")
print("  V0       C(V0)    C_full   const b/w  C_const  saving   " + "  ".join(f"wk{t:>2}" for t in (1, 5, 10, 20, 30, 40, 50)) + "   a(1)   a(50)")
for V0 in (16850., 16942., 17000.):
    sim = simulate(V0); paths[V0] = sim
    Cf, af = C_full(V0); b, ab, Cb = constant_benefit(V0)
    rr = [sim[t - 1, 2] for t in (1, 5, 10, 20, 30, 40, 50)]
    CV0 = np.interp(V0, Vg, C)
    print(f"  {V0:.0f}  {CV0:8.2f} {Cf:8.2f}  {b / w:8.4f} {Cb:8.2f}  {1 - CV0 / Cb:6.2%}  " + "  ".join(f"{x:.3f}" for x in rr)
          + f"  {sim[0, 3]:6.1f} {sim[-1, 3]:6.1f}")
    print(f"        hazard week 1 {sim[0, 4]:.4f}, week 50 {sim[-1, 4]:.4f}; full information: constant c/w = "
          f"{uinv(V0 * (1 - beta * (1 - p(af))) + af - beta * p(af) * Ve) / w:.4f}, a = {af:.1f}, hazard {p(af):.4f};"
          f" constant benefit: effort {ab:.1f}, hazard {p(ab):.4f}")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
for V0, ls in zip(paths, ("-", "--", ":")):
    sim = paths[V0]
    ax[0].plot(sim[:, 0], sim[:, 2], "k" + ls, label=f"$V_0$ = {V0:,.0f}")
    ax[1].plot(sim[:, 0], sim[:, 3], "k" + ls, label=f"$V_0$ = {V0:,.0f}")
ax[1].axhline(astar, color="grey", lw=.6); ax[1].text(2, astar + 4, "autarky effort", fontsize=7)
ax[0].set_xlabel("duration (weeks)"); ax[0].set_ylabel("replacement ratio $c/w$"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("duration (weeks)"); ax[1].set_ylabel("search effort $a$"); ax[1].legend(fontsize=7)
ax[0].set_title("23.1: replacement ratio", fontsize=9); ax[1].set_title("23.1: search effort", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch23_01.pdf"); plt.close()
