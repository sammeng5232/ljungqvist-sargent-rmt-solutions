"""Chapter 21 (Incentives and Insurance): numerical parts of exercises 21.2-21.5.

21.2/21.3  village moneylender = Thomas-Worrall labor contract (same mathematics, same parameters):
           closed-form contract of section 21.3.3, profits (21.3.28), cross-section c.d.f.s, bank balance,
           zero-profit initial promise; the closed-form P(v) is checked to be a fixed point of the Bellman
           equation (21.3.4)-(21.3.8) by solving the right-hand side numerically at many v.
21.4       Thomas-Worrall with private information, Phelan-Townsend lotteries: value iteration where each
           Bellman step is a linear program.
21.5       IMF contract: closed-form optimal sustainable contract for a quadratic loss, checked by simulation.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq, linprog
from scipy.sparse import lil_matrix, csr_matrix
from rmtlib import FIG

np.set_printoptions(precision=5, suppress=True, linewidth=130)

# ============================================================== 21.2 / 21.3
beta, S, gam, lam = .5, 20, 2., .95
s = np.arange(1, S + 1)
Pi = (1 - lam) / (1 - lam ** S) * lam ** (s - 1)
y = s + 5.
u = lambda c: c ** (1 - gam) / (1 - gam)
uinv = lambda x: ((1 - gam) * x) ** (1 / (1 - gam))
up = lambda c: c ** (-gam)
vaut = Pi @ u(y) / (1 - beta)
cumPi = np.cumsum(Pi)
# (21.3.25) and (21.3.24)
uc = np.array([u(y[j]) - beta * np.sum(Pi[:j + 1] * (u(y[j]) - u(y[:j + 1]))) for j in range(S)])
c = uinv(uc)
A = u(y) + beta * vaut                                   # u(c_j) + beta w_j on the participation constraint
w = (A - uc) / beta
w_alt = np.array([(uc[j] * cumPi[j] + np.sum(Pi[j + 1:] * A[j + 1:])) / (1 - beta * cumPi[j]) for j in range(S)])
print("=== 21.2 / 21.3   (beta, S, gamma, lambda) = (.5, 20, 2, .95)")
print(f"  E y = {Pi @ y:.6f};  v_aut = {vaut:.6f};  check (21.3.24) vs (21.3.22a): max diff {np.max(np.abs(w - w_alt)):.1e};"
      f"  sum Pi (u(c)+beta w) - v_aut = {Pi @ (uc + beta * w) - vaut:.1e}")
print("  s   y_s     Pi_s      c_s        w_s       y_s - c_s")
for j in range(S):
    print(f" {j+1:2d} {y[j]:5.1f}  {Pi[j]:.5f}  {c[j]:9.5f}  {w[j]:9.5f}  {y[j]-c[j]:8.5f}")
# (21.3.28) profits P(w_k), backward from k = S
Pw = np.zeros(S)
Pw[S - 1] = Pi @ (y - c[S - 1]) / (1 - beta)
for k in range(S - 2, -1, -1):
    num = np.sum(Pi[:k + 1] * (y[:k + 1] - c[k])) + np.sum(Pi[k + 1:] * (y[k + 1:] - c[k + 1:])) + beta * np.sum(Pi[k + 1:] * Pw[k + 1:])
    Pw[k] = num / (1 - beta * cumPi[k])
print(f"  P(w_k), k = 1..S: {np.round(Pw, 5)}")
print(f"  (b) expected profits from offering v_aut: P(v_aut) = P(w_1) = {Pw[0]:.6f}")


def contract_P(v):
    """Closed-form value P(v) and consumption c~(v) for v >= v_aut (sections 21.3.3-21.3.5)."""
    if v >= w[-1] - 1e-14:
        ct = uinv((1 - beta) * v)
        return (Pi @ y - ct) / (1 - beta), ct
    k = np.searchsorted(w, v, side="right")               # v in [w_{k-1}, w_k), 1-based k = index + 1
    q = cumPi[k - 1]                                      # sum_{j<k} Pi_j (1-based)
    Phi = ((1 - beta * q) * v - np.sum(Pi[k:] * (uc[k:] + beta * w[k:]))) / q
    ct = uinv(Phi)
    P = (np.sum(Pi[:k] * (y[:k] - ct)) + np.sum(Pi[k:] * (y[k:] - c[k:] + beta * Pw[k:]))) / (1 - beta * q)
    return P, ct


# the closed form reproduces P(w_k) at the kinks
print("  closed-form P at w_k matches the backward recursion:", np.allclose([contract_P(x)[0] for x in w], Pw, atol=1e-12))

# Bellman check: solve the right side of (21.3.4) numerically at several v with P = closed form
Pfun = lambda x: contract_P(x)[0]
vbar = w[-1] + 5.0


def bellman_rhs(v):
    x0 = np.r_[np.maximum(c, contract_P(v)[1]), np.maximum(w, v)]
    obj = lambda z: -(Pi @ (y - z[:S]) + beta * Pi @ np.array([Pfun(t) for t in z[S:]]))
    cons = [{"type": "ineq", "fun": lambda z: Pi @ (u(z[:S]) + beta * z[S:]) - v}]
    cons += [{"type": "ineq", "fun": (lambda z, j=j: u(z[j]) + beta * z[S + j] - A[j])} for j in range(S)]
    bnds = [(1., 60.)] * S + [(vaut, vbar)] * S
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-13, "maxiter": 500})
    # also try from a perturbed start to make sure the maximum is not local
    res2 = minimize(obj, x0 * 1.01, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-13, "maxiter": 500})
    return max(-res.fun, -res2.fun)


vs_check = np.r_[vaut, np.linspace(vaut, w[-1], 9)[1:], w[-1] + .01]
errs = [abs(bellman_rhs(v) - Pfun(v)) for v in vs_check]
print(f"  Bellman fixed-point check at {len(vs_check)} values of v: max |T(P)(v) - P(v)| = {max(errs):.2e}")

# (c) cross-section c.d.f.s for v0 = v_aut: Prob(c_t <= c_j) = (cumPi_j)^(t+1)
ts = (0, 5, 10, 500)
F = {t: cumPi ** (t + 1) for t in ts}
for t in ts:
    print(f"  (c) t = {t:3d}: Prob(c_t = c_1) = {F[t][0]:.5f}, Prob(c_t <= c_10) = {F[t][9]:.5f}, Prob(c_t = c_S) = {1 - F[t][S-2]:.5f}, mean c = {np.diff(np.r_[0, F[t]]) @ c:.5f}")


def balances(ctilde_k, P0, T=101, Tlong=3000):
    """Bank balance (21.3.40) for a contract (k, c~) with v0 in [w_{k-1}, w_k); k = 1 means v0 = v_aut.
    Iterating B_t = B_{t-1}/beta + (Ey - Ec_t) multiplies rounding errors by 2^t when beta = .5, so use the
    equivalent B_t = beta^{-t} P(v0) - sum_{s>t} beta^{s-t} (Ey - Ec_s), whose tail sum is numerically stable."""
    k, ct = ctilde_k
    cc = c.copy()
    if k > 1:
        cc[:k - 1] = ct
    Ec = np.array([np.diff(np.r_[0, cumPi ** (t + 1)]) @ cc for t in range(Tlong)])
    flow = Pi @ y - Ec
    B = np.array([beta ** -t * P0 - np.sum(beta ** np.arange(1, Tlong - t) * flow[t + 1:]) for t in range(T)])
    Bit = np.zeros(T); prev = 0.                             # naive recursion, for comparison at small t
    for t in range(T):
        prev = prev / beta + flow[t]; Bit[t] = prev
    return B, Ec[:T], np.max(np.abs(B[:30] - Bit[:30]))


B0, Ec0, dB0 = balances((1, None), Pw[0])
print(f"  (d) bank balance, v0 = v_aut: B_0..B_5 = {np.round(B0[:6], 4)}, B_20 = {B0[20]:.4e}, B_100 = {B0[100]:.4e} (recursion agrees to {dB0:.1e} for t<30)")
print(f"      check (21.3.42): beta^t (B_t + beta P(w_S)) at t = 100: {beta**100 * (B0[100] + beta * Pw[-1]):.6f} = P(v_aut)")

# (e) zero-profit initial promise
print(f"  (e) P(w_S) = {Pw[-1]:.6f} (E y - c_S = {Pi @ y - c[-1]:.6f})")
kz = next(k for k in range(1, S + 1) if Pw[k - 1] <= 0)   # first k (1-based) with P(w_k) <= 0
q = cumPi[kz - 2]
ct0 = (np.sum(Pi[:kz - 1] * y[:kz - 1]) + np.sum(Pi[kz - 1:] * (y[kz - 1:] - c[kz - 1:] + beta * Pw[kz - 1:]))) / q
v0 = (q * u(ct0) + np.sum(Pi[kz - 1:] * (uc[kz - 1:] + beta * w[kz - 1:]))) / (1 - beta * q)
print(f"      k = {kz}: P(w_{kz-1}) = {Pw[kz-2]:.5f} > 0 >= P(w_{kz}) = {Pw[kz-1]:.5f};  c~ = {ct0:.6f}, v0 = {v0:.6f}"
      f" in [w_{kz-1}, w_{kz}) = [{w[kz-2]:.5f}, {w[kz-1]:.5f}); check P(v0) = {Pfun(v0):.2e}")
v0root = brentq(Pfun, vaut, w[-1])
print(f"      root of the closed-form P on [v_aut, w_S]: {v0root:.6f};  pooling value (not relevant since P(w_S) < 0): {u(Pi @ y)/(1-beta):.6f}")
B1, Ec1, dB1 = balances((kz, ct0), 0.)
print(f"      bank balance with P(v0) = 0: B_0..B_5 = {np.round(B1[:6], 4)}, B_20 = {B1[20]:.6f}, B_100 = {B1[100]:.6f},"
      f" limit (c_S - E y)/r = {(c[-1] - Pi @ y)/(1/beta - 1):.6f};  max B_t at t = {np.argmax(B1)}: {B1.max():.5f}")
# Monte Carlo check of P(v_aut) and of the zero-profit contract
rng = np.random.default_rng(0)
N, T = 200000, 40
draws = rng.choice(S, size=(N, T), p=Pi)
rec = np.maximum.accumulate(draws, axis=1)
prof_aut = ((y[draws] - c[rec]) * beta ** np.arange(T)).sum(1) + beta ** T * Pw[rec[:, -1]]
cc1 = c.copy(); cc1[:kz - 1] = ct0
prof_0 = ((y[draws] - cc1[rec]) * beta ** np.arange(T)).sum(1) + beta ** T * np.where(rec[:, -1] >= kz - 1, Pw[rec[:, -1]], Pfun(v0))
print(f"  Monte Carlo ({N} households): P(v_aut) = {prof_aut.mean():.5f} +- {prof_aut.std()/np.sqrt(N):.5f};  P(v0) = {prof_0.mean():.5f} +- {prof_0.std()/np.sqrt(N):.5f}")
# 21.3(d): expected wage-tenure profiles
print("  21.3(d) expected wage by tenure (v0 = v_spot):", np.round(Ec0[[0, 1, 2, 5, 10, 20, 50]], 4))
print("                            (zero profit v0):   ", np.round(Ec1[[0, 1, 2, 5, 10, 20, 50]], 4))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for t, ls in zip(ts, ("-", "--", "-.", ":")):
    ax[0].step(np.r_[c[0] - .5, c], np.r_[0, F[t]], where="post", ls=ls, color="k", label=f"t = {t}")
ax[0].set_xlabel("consumption (wage)"); ax[0].set_title(r"$F_t$, contract with $v_0=v_{aut}$", fontsize=9); ax[0].legend(fontsize=7)
ax[1].plot(np.arange(101), B0, "k", label=r"$v_0=v_{aut}$"); ax[1].plot(np.arange(101), B1, "b--", label=r"$P(v_0)=0$")
ax[1].set_yscale("symlog", linthresh=1); ax[1].set_xlabel("t"); ax[1].set_title("moneylender's bank balance $B_t$", fontsize=9); ax[1].legend(fontsize=7)
ax[2].plot(np.arange(51), Ec0[:51], "k.-", ms=3, label=r"$v_0=v_{spot}$"); ax[2].plot(np.arange(51), Ec1[:51], "b.--", ms=3, label=r"$P(v_0)=0$")
ax[2].axhline(Pi @ y, color="grey", lw=.6); ax[2].set_xlabel("tenure t"); ax[2].set_title("expected wage (consumption) by tenure", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch21_02.pdf"); plt.close()


# ============================================================== 21.5 (IMF): quadratic loss illustration
print("\n=== 21.5 illustration: W(T) = T^2, g in {.1,.2,.3,.4} equally likely, beta = .9")
bI = .9; g = np.array([.1, .2, .3, .4]); pg = np.full(4, .25)
Wl = lambda T: T ** 2
vautI = pg @ Wl(g) / (1 - bI)
# record state j (lowest g so far = g_j):  W(T_j) = W(g_j) + beta sum_{k>=j} pi_k [W(g_k) - W(g_j)]
WT = np.array([Wl(g[j]) + bI * np.sum(pg[j:] * (Wl(g[j:]) - Wl(g[j]))) for j in range(4)])
T_ = np.sqrt(WT)
Aim = Wl(g) + bI * vautI
wI = (Aim - WT) / bI
print(f"  v_aut = {vautI:.5f};  T_j = {np.round(T_, 5)} (g_j = {g});  promised losses w_j = {np.round(wI, 5)}")
print(f"  check: promise keeping sum pi (W(T)+beta w) = {pg @ (WT + bI * wI):.6f} = v_aut;  w_1 = W(T_1)/(1-beta)? {np.isclose(wI[0], WT[0]/(1-bI))};"
      f"  w_S = v_aut? {np.isclose(wI[-1], vautI)}")
# IMF profits from v_aut: simulate
rng = np.random.default_rng(1)
N, T = 200000, 80
dI = rng.choice(4, size=(N, T), p=pg)
recI = np.minimum.accumulate(dI, axis=1)                  # index of the lowest g so far
surplus = T_[recI] - g[dI]
PV = (surplus * bI ** np.arange(T)).sum(1)
print(f"  IMF expected PV of surpluses P(v_aut) = {PV.mean():.5f} +- {PV.std()/np.sqrt(N):.5f} (> 0)")
print("  average tax by tenure t = 0,1,2,5,10,30:", np.round(T_[recI].mean(0)[[0, 1, 2, 5, 10, 30]], 5), " E g =", pg @ g)
print("  average surplus by tenure:", np.round(surplus.mean(0)[[0, 1, 2, 5, 10, 30]], 5))
# exact value by the formula analogous to (21.3.28)
PI = np.zeros(4); cum = np.cumsum(pg[::-1])[::-1]         # cum[j] = sum_{k>=j} pi_k
PI[0] = (T_[0] - pg @ g) / (1 - bI)
for j in range(1, 4):
    PI[j] = (np.sum(pg[j:] * (T_[j] - g[j:])) + np.sum(pg[:j] * (T_[:j] - g[:j])) + bI * np.sum(pg[:j] * PI[:j])) / (1 - bI * cum[j])
print(f"  exact P at the record states: {np.round(PI, 5)};  P(v_aut) = P(w_S) = {PI[-1]:.5f}")
cum_ext = np.r_[cum, 0.]
ET = np.array([np.sum((cum_ext[:4] ** (t + 1) - cum_ext[1:] ** (t + 1)) * T_) for t in range(2000)])
print(f"  exact PV from the distribution of the record: {np.sum(bI ** np.arange(2000) * (ET - pg @ g)):.6f};"
      f"  exact average tax by tenure 0,1,2,5,10,30: {np.round(ET[[0, 1, 2, 5, 10, 30]], 5)}")




# ============================================================== 21.4 Thomas-Worrall with lotteries (Phelan-Townsend)
import time
print("\n=== 21.4  beta = .94, a = 5, gamma = 3, Y = {6,...,15} uniform, NB = NW = 25")
bT, aT, gT = .94, 5., 3.
Yg = np.arange(6., 16.); Sy = len(Yg); piY = np.full(Sy, 1 / Sy)
uT = lambda cc: (cc - aT) ** (1 - gT) / (1 - gT)
NB = 25
bmin = aT - Yg.max() + .33                                # the book prints 1 - ymax + .33 (see the remark in Ch21.tex)
Bg = np.linspace(bmin, Yg.max() - Yg.min(), NB)
vautT = piY @ uT(Yg) / (1 - bT)
bstar = Bg[Bg > aT - Yg.min()].min()                      # smallest transfer that keeps the lowest type above a
vlo = piY @ uT(Yg + bstar) / (1 - bT)                     # lowest deliverable value: b = bstar for all types forever
print(f"  B = [{bmin:.2f}, {Bg[-1]:.2f}] (spacing {Bg[1]-Bg[0]:.4f}); autarky value {vautT:.4f}; smallest usable transfer for y = 6: {bstar:.4f};"
      f" lowest deliverable value E u(y + {bstar:.3f})/(1-beta) = {vlo:.4f}")


def solve_pt(Wg, tag):
    """Value iteration with one LP per promised value (Bellman equation (1)-(5)), then policy iteration."""
    NW = len(Wg)
    idx = [(iy, ib, iw) for iy in range(Sy) for ib in range(NB) if Yg[iy] + Bg[ib] > aT + 1e-12 for iw in range(NW)]
    nv = len(idx)
    iyv = np.array([k[0] for k in idx]); ibv = np.array([k[1] for k in idx]); iwv = np.array([k[2] for k in idx])
    util = uT(Yg[iyv] + Bg[ibv]) + bT * Wg[iwv]             # u(y+b) + beta w for the truthful type
    Aeq = lil_matrix((1 + Sy, nv)); Aeq[0, :] = util
    for iy in range(Sy):
        Aeq[1 + iy, np.where(iyv == iy)[0]] = 1.
    Aeq = csr_matrix(Aeq)
    rows = []
    for iy in range(Sy):                                   # true type iy reports jy
        for jy in range(Sy):
            if iy == jy: continue
            r = np.zeros(nv); own = iyv == iy
            r[own] -= util[own] / piY[iy]
            oth = np.where(iyv == jy)[0]
            cl = Yg[iy] + Bg[ibv[oth]]
            ul = np.where(cl > aT + 1e-12, uT(np.maximum(cl, aT + 1e-9)), -1e4)   # u = -infinity below a
            r[oth] += (ul + bT * Wg[iwv[oth]]) / piY[jy]
            rows.append(r)
    Aub = csr_matrix(np.array(rows)); bub = np.zeros(len(rows))
    print(f"  [{tag}] W = [{Wg[0]:.4f}, {Wg[-1]:.4f}], {NW} points; LP: {nv} variables, {Aeq.shape[0]} equalities, {Aub.shape[0]} truth-telling inequalities")

    def lp(cost, v, bnds):
        for meth in ("highs", "highs-ipm", "highs-ds"):
            res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.r_[v, piY], bounds=bnds, method=meth)
            if res.status in (0, 2):
                return res
        raise RuntimeError(f"LP failed at v = {v}: {res.message}")

    def step(P, feas):
        newP = np.full(NW, -np.inf); pols = [None] * NW
        cost = np.where(feas[iwv], Bg[ibv] - bT * np.where(feas, P, 0.)[iwv], 0.)   # minimise -(-b + beta P(w))
        bnds = np.c_[np.zeros(nv), np.where(feas[iwv], np.inf, 0.)]
        for iv in np.where(feas)[0]:                          # the deliverable set can only shrink
            res = lp(cost, Wg[iv], bnds)
            if res.status == 0:
                newP[iv] = -res.fun; pols[iv] = res.x
        return newP, pols

    def evaluate(pols, feas):
        ids = np.where(feas)[0]; n = len(ids); pos_ = -np.ones(NW, int); pos_[ids] = np.arange(n)
        M = np.zeros((n, n)); r = np.zeros(n)
        for a_, iv in enumerate(ids):
            np.add.at(M[a_], pos_[iwv], pols[iv]); r[a_] = -(Bg[ibv] @ pols[iv])
        out = np.full(NW, -np.inf); out[ids] = np.linalg.solve(np.eye(n) - bT * M, r)
        return out

    t0 = time.time(); feas = np.ones(NW, bool); P = np.zeros(NW)
    for it in range(1, 3001):                               # stage 1: value iteration
        Pn, pols = step(P, feas)
        fn = np.isfinite(Pn)
        d = np.max(np.abs(Pn[fn] - P[fn])) if np.array_equal(fn, feas) else np.inf
        if not np.array_equal(fn, feas):
            print(f"    iteration {it:4d}: lowest deliverable grid value now {Wg[fn][0]:.4f}")
        feas = fn; P = Pn
        if d < 1e-4:
            print(f"    iteration {it:4d}: value iteration step {d:.1e}   ({time.time()-t0:.0f} s)")
            break
    for it2 in range(1, 101):                               # stage 2: policy iteration (LP step + exact evaluation)
        Pe = evaluate(pols, feas)
        Pn, pols = step(Pe, feas)
        if not np.array_equal(np.isfinite(Pn), feas):
            raise RuntimeError("feasible set changed during policy iteration")
        d = np.max(np.abs(Pn[feas] - Pe[feas])); P = Pn
        if d < 1e-10:
            print(f"    policy iteration converged after {it2} step(s): max|T(P) - P| = {d:.1e}   ({time.time()-t0:.0f} s)")
            break
    keep = np.where(feas)[0]
    X = np.array([pols[i] for i in keep])
    assert np.all(X[:, ~np.isin(iwv, keep)] < 1e-9)
    mw = -np.ones(NW, int); mw[keep] = np.arange(len(keep))
    sel = mw[iwv] >= 0
    X = X[:, sel]; iyv, ibv, iwv, util = iyv[sel], ibv[sel], mw[iwv[sel]], util[sel]
    W = Wg[keep]; P = P[keep]; n = len(keep)
    Mw = np.zeros((n, n)); rT = np.zeros(n)
    for iv in range(n):
        np.add.at(Mw[iv], iwv, X[iv]); rT[iv] = -(Bg[ibv] @ X[iv])
    Peval = np.linalg.solve(np.eye(n) - bT * Mw, rT)
    ic = Aub[:, np.where(sel)[0]] @ X.T
    print(f"    deliverable grid values: {n} of {NW}, from {W[0]:.4f} to {W[-1]:.4f}")
    print(f"    checks: policy evaluation reproduces P to {np.max(np.abs(Peval - P)):.1e}; promise keeping {np.max(np.abs(X @ util - W)):.1e};"
          f" largest truth-telling violation {ic.max():.1e}; rows of Prob(w'|w) sum to one: {np.allclose(Mw.sum(1), 1)}")
    print("    P(w):", np.round(P, 3))
    cvals = Yg[iyv] + Bg[ibv]
    cgrid = np.unique(np.round(cvals, 10))
    pos = np.searchsorted(cgrid, np.round(cvals, 10))
    Qc = np.zeros((n, len(cgrid)))
    for iv in range(n):
        np.add.at(Qc[iv], pos, X[iv])
    i0 = np.argmin(np.abs(W + 2))
    p = np.zeros(n); p[i0] = 1.; Ft = {}; stats = {}
    for t in range(1, 1001):
        if t in (1, 5, 10, 100, 1000):
            dens = p @ Qc; Ft[t] = np.cumsum(dens)
            m = dens @ cgrid; stats[t] = (m, np.sqrt(dens @ cgrid ** 2 - m ** 2), p @ W)
        p = p @ Mw
    print(f"    w0 = {W[i0]:.4f} (grid point closest to -2)")
    for t in (1, 5, 10, 100, 1000):
        print(f"     t = {t:4d}: mean c = {stats[t][0]:.4f}, std c = {stats[t][1]:.4f}, mean promised value = {stats[t][2]:.4f}")
    # a few features of the contract at w0: consumption and continuation value by reported y
    ew = np.array([(X[i0][iyv == iy] @ W[iwv[iyv == iy]]) / piY[iy] for iy in range(Sy)])
    ec = np.array([(X[i0][iyv == iy] @ cvals[iyv == iy]) / piY[iy] for iy in range(Sy)])
    print(f"    at w0: E[c | y] = {np.round(ec, 3)}\n           E[w' | y] = {np.round(ew, 3)}")
    return dict(W=W, P=P, Ft=Ft, cgrid=cgrid, stats=stats, w0=W[i0])


NW = 25
wmin = uT(Yg.min()) / (1 - bT); wmax = wmin / 20
resA = solve_pt(np.linspace(wmin, wmax, NW), "book's grid")
resB = solve_pt(np.linspace(vlo + 1e-6, wmax, NW), "grid on the deliverable range")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for k_, (res, tt) in enumerate(((resA, "book's W grid (7 deliverable points)"), (resB, "25 points on the deliverable range"))):
    for t, ls in zip((1, 5, 10, 100), ("-", "--", "-.", ":")):
        ax[k_].step(res["cgrid"], res["Ft"][t], where="post", ls=ls, color="k", label=f"t = {t}")
    ax[k_].set_xlim(5, 16); ax[k_].set_xlabel("consumption c"); ax[k_].legend(fontsize=7, loc="lower right")
    ax[k_].set_title(r"$F_t(c)$, " + tt, fontsize=9)
ax[2].plot(resA["W"], resA["P"], "ko-", ms=4, label="book's grid"); ax[2].plot(resB["W"], resB["P"], "b.-", label="refined grid")
ax[2].set_xlabel("promised value w"); ax[2].set_title("planner's value P(w)", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch21_04.pdf"); plt.close()
