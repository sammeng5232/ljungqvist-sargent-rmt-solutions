"""Chapter 18 (Incomplete Markets Models): computations for exercises 18.1, 18.2, 18.3 and 18.6.

18.1  three coupled Bellman equations (random discount factor), solved by the endogenous grid method; illustration only.
18.2  Bertola's mobility-cost model with the book's parameters: value iteration on a grid for A in [0, 3], the Markov chain on (A, w)
      and its invariant distribution.
18.3  reservation wage and stationary unemployment for an example f(U) = 1 - U, F uniform on [0, 1] (illustration of part d).
18.6  Huggett (1993) parameters: the Ea(r) curve, r_H < 0, the Bewley equilibrium with valued fiat money at r = 0, the rank-preserving
      transfers of part f and the welfare comparison of part g.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=4, suppress=True, linewidth=130)


# ------------------------------------------------------------------ generic tools: EGM, Young's lottery, policy evaluation
def egm(r, y, Q, betas, sig, grid, tol=1e-10, maxit=20000):
    """Consumption/savings with exogenous states z (income y[z], discount betas[z], transition Q), a' >= grid[0].
    Returns next-period assets a'(a, z) and consumption c(a, z) on the grid (shape N x Z)."""
    N, Z = len(grid), len(y)
    R = 1 + r
    c = np.maximum(R * grid[:, None] + y[None, :] - grid[0], 1e-3) * np.ones((N, Z))   # guess: eat everything above the limit
    for it in range(maxit):
        Emu = c ** (-sig) @ Q.T                                    # E[u'(c(a', z')) | z] for a' on the grid, shape N x Z
        cstar = (betas[None, :] * R * Emu) ** (-1 / sig)
        astar = (cstar + grid[:, None] - y[None, :]) / R           # beginning-of-period assets that choose a' = grid
        ap = np.empty((N, Z))
        for z in range(Z):
            ap[:, z] = np.interp(grid, astar[:, z], grid)          # constrained region: a' = grid[0] (np.interp clamps)
            ap[grid < astar[0, z], z] = grid[0]
        ap = np.clip(ap, grid[0], grid[-1])
        cn = R * grid[:, None] + y[None, :] - ap
        if np.max(np.abs(cn - c)) < tol:
            return ap, cn
        c = cn
    raise RuntimeError("EGM did not converge")


def young_matrix(ap, grid, Q):
    """Sparse transition matrix on (a, z) pairs (index a_i * Z + z) from policy a'(a, z) with lottery weights and exogenous Q."""
    N, Z = ap.shape
    j = np.clip(np.searchsorted(grid, ap, side="right") - 1, 0, N - 2)
    wgt = (grid[j + 1] - ap) / (grid[j + 1] - grid[j])                 # weight on grid[j]
    rows, cols, vals = [], [], []
    for z in range(Z):
        for z2 in range(Z):
            if Q[z, z2] == 0:
                continue
            src = np.arange(N) * Z + z
            rows += [src, src]; cols += [j[:, z] * Z + z2, (j[:, z] + 1) * Z + z2]
            vals += [wgt[:, z] * Q[z, z2], (1 - wgt[:, z]) * Q[z, z2]]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * Z, N * Z))


def stationary(T, tol=1e-13, maxit=200000):
    lam = np.full(T.shape[0], 1 / T.shape[0]); TT = T.T.tocsr()
    for it in range(maxit):
        new = TT @ lam
        if np.max(np.abs(new - lam)) < tol:
            return new / new.sum()
        lam = new
    raise RuntimeError("no convergence of the stationary distribution")


def values(ap, c, grid, Q, betas, sig):
    """Exact value of the (discretized) policy: v = u(c) + beta(z) T v, solved as a sparse linear system."""
    N, Z = c.shape
    T = young_matrix(ap, grid, Q)
    u = (c ** (1 - sig) / (1 - sig)).reshape(-1)
    B = sp.diags(np.tile(betas, N))
    return spla.spsolve((sp.identity(N * Z, format="csr") - B @ T).tocsc(), u).reshape(N, Z)


# ================================================================== 18.1 random discount factor (illustration)
print("=== 18.1  random discount factor: three coupled Bellman equations, EGM on (beta, s)")
bet = np.array([.94, .96, .98]); PB = np.array([[.90, .10, 0], [.05, .90, .05], [0, .10, .90]])
s1 = np.array([.3, 1.]); PS = np.array([[.5, .5], [.05, .95]])
Q1 = np.kron(PB, PS); y1 = np.tile(s1, 3); b1 = np.repeat(bet, 2)
grid1 = np.linspace(0, 1, 600) ** 2 * 60
ap1, c1 = egm(.02, y1, Q1, b1, 2., grid1)
lam1 = stationary(young_matrix(ap1, grid1, Q1)).reshape(len(grid1), 6)
mass = lam1.sum(0).reshape(3, 2).sum(1)
mean_by_beta = (lam1 * grid1[:, None]).sum(0).reshape(3, 2).sum(1) / mass
print(f"  r = .02, u = c^(1-2)/(1-2), beta in {bet}: mass by beta {mass}, mean assets by beta {mean_by_beta},"
      f" share of wealth held by the most patient {mean_by_beta[2]*mass[2]/(mean_by_beta@mass):.3f}; mass at a = 0: {lam1[0].sum():.3f};"
      f" mass above a = 50: {lam1[grid1 > 50].sum():.2e}")

# ================================================================== 18.2 Bertola: mobility costs
print("\n=== 18.2  mobility costs: m = .9, p = .8, R = 1.02, beta = .95, wg = 1.4, wb = 1, sigma = 4, A in [0, 3]")
m, p, R, beta, wg, wb, sig = .9, .8, 1.02, .95, 1.4, 1., 4.
def bertola(Amax, n):
    A = np.linspace(0, Amax, n); NA = len(A)
    u = lambda c: np.where(c > 0, np.maximum(c, 1e-12) ** (1 - sig) / (1 - sig), -1e12)
    Pw = np.array([[p, 1 - p], [1 - p, p]])                         # index 0 = good wage, 1 = bad wage
    wage = np.array([wg, wb])
    # flow utilities: stay in state w (income RA + w), or move (income RA + wg - m, next wage drawn from the good row)
    C_stay = [R * A[:, None] + wage[i] - A[None, :] for i in range(2)]
    C_move = R * A[:, None] + wg - m - A[None, :]
    U_stay = [u(C) for C in C_stay]; U_move = u(C_move)
    v = np.zeros((NA, 2))
    for it in range(5000):
        EV = v @ Pw.T                                                 # EV[a', i] = E[v(a', w') | w = i]
        stay = [U_stay[i] + beta * EV[None, :, i] for i in range(2)]
        move = U_move + beta * EV[None, :, 0]
        vs = np.column_stack([s.max(1) for s in stay]); vm = move.max(1)
        vn = np.column_stack([np.maximum(vs[:, 0], vm), np.maximum(vs[:, 1], vm)])
        if np.max(np.abs(vn - v)) < 1e-10:
            break
        v = vn
    v = vn
    mv = np.column_stack([vm > vs[:, 0], vm > vs[:, 1]])             # move decision by (A, w)
    pol = np.empty((NA, 2), dtype=int)
    for i in range(2):
        pol[:, i] = np.where(mv[:, i], move.argmax(1), stay[i].argmax(1))
    return A, NA, v, vs, vm, mv, pol, move, stay, it


def bertola_chain(A, mv, pol):
    """Sparse transition matrix of (A, w) (index 2k + i) and its stationary distribution (NA x 2)."""
    NA = len(A); rows, cols, vals = [], [], []
    for i in range(2):
        for k in range(NA):
            src = 2 * k + i; row = 0 if (mv[k, i] or i == 0) else 1   # movers draw next wage from the good row
            for j in range(2):
                rows.append(src); cols.append(2 * pol[k, i] + j); vals.append(Pw[row, j])
    T = sp.csr_matrix((vals, (rows, cols)), shape=(2 * NA, 2 * NA))
    return T, stationary(T).reshape(NA, 2)


Pw = np.array([[p, 1 - p], [1 - p, p]]); wage = np.array([wg, wb])
A, NA, v, vs, vm, mv, pol, move, stay, it = bertola(3., 601)
print(f"  value iteration converged in {it} iterations; movers in the good state: {mv[:, 0].sum()} grid points (never optimal)")
thr = A[np.argmax(mv[:, 1])] if mv[:, 1].any() else np.nan
print(f"  bad-wage state: move iff A >= {thr:.3f} (monotone: {np.all(mv[:, 1] == (A >= thr))})")
cons = np.where(mv, R * A[:, None] + wg - m - A[pol], R * A[:, None] + wage[None, :] - A[pol])
for a0 in (0, .5, 1, thr, 2, 3):
    k = np.argmin(np.abs(A - a0))
    print(f"   A = {A[k]:.3f}: good -> A' = {A[pol[k,0]]:.3f}, c = {cons[k,0]:.3f};  bad -> move = {mv[k,1]}, A' = {A[pol[k,1]]:.3f}, c = {cons[k,1]:.3f}")
# (c) Markov chain on (A, w): next w drawn from the good row if the worker moves (or is in the good state)
T2, lam2 = bertola_chain(A, mv, pol)
supp = A[lam2.sum(1) > 1e-10]
print(f"  invariant distribution: support A in [{supp.min():.3f}, {supp.max():.3f}] ({len(supp)} grid points); mass in bad state {lam2[:, 1].sum():.4f};"
      f" mean assets {A @ lam2.sum(1):.4f}; mass at A = 0 {lam2[0].sum():.4f}; mass at A >= 2.5 {lam2[A >= 2.5].sum():.4f}, at A = 3 {lam2[-1].sum():.4f}")
movers = (lam2 * mv).sum()
print(f"  fraction of workers who move each period {movers:.4f}; fraction of bad-state workers who move {movers / lam2[:, 1].sum():.4f}")
# uniqueness: count unit eigenvalues of the whole chain (closed classes), and the spectral gap on the recurrent support
evall = np.sort(np.abs(np.linalg.eigvals(T2.toarray())))[::-1]
idx = np.where(lam2.reshape(-1) > 1e-12)[0]
print(f"  eigenvalues of the full chain with |lambda| > 1 - 1e-9: {(evall > 1 - 1e-9).sum()}; recurrent support {len(idx)} states;"
      f" second largest |eigenvalue| {evall[1]:.4f}")
cdf = np.cumsum(lam2.sum(1))
print("  wealth quantiles (.1,.25,.5,.75,.9):", [round(A[np.searchsorted(cdf, q)], 3) for q in (.1, .25, .5, .75, .9)])
# robustness: the upper end A = 3 of the book's grid binds in the good state; rerun on [0, 8]
A8, NA8, _, _, _, mv8, pol8, _, _, _ = bertola(8., 1601)
T8, lam8 = bertola_chain(A8, mv8, pol8)
cdf8 = np.cumsum(lam8.sum(1))
print(f"  grid [0, 8]: move threshold {A8[np.argmax(mv8[:, 1])]:.3f}; mean assets {A8 @ lam8.sum(1):.4f}; mass at 0 {lam8[0].sum():.4f};"
      f" mass above 3 {lam8[A8 > 3].sum():.4f}; max of support {A8[lam8.sum(1) > 1e-10].max():.3f};"
      f" moving rate {(lam8 * mv8).sum():.4f}; quantiles", [round(A8[np.searchsorted(cdf8, q)], 3) for q in (.1, .25, .5, .75, .9)])
k8 = np.searchsorted(A8, 3.0)
print(f"  grid [0, 8]: good-state policy at A = 3: A' = {A8[pol8[k8, 0]]:.3f}; fixed point of the good-state rule: "
      f"{A8[np.argmax(A8[pol8[:, 0]] <= A8)]:.3f}")

# ================================================================== 18.3 unemployment: an example for part (d)
print("\n=== 18.3  example: f(U) = 1 - U, F uniform on [0,1], beta = .95, lambda = .05")
b3, l3 = .95, .05
k3 = b3 / (1 - b3 * (1 - l3))                                   # wbar = k3 mu int_{wbar} (w - wbar) dF, jobs start next period
def wbar(mu):                                                     # wbar = k3 mu (1 - wbar)^2 / 2  (uniform F)
    return brentq(lambda w: w - k3 * mu * (1 - w) ** 2 / 2, 0, 1) if mu > 0 else 0.
Hf = lambda U: (1 - U) * (1 - wbar(1 - U)) * U - l3 * (1 - U)
Us = np.linspace(1e-4, 1 - 1e-4, 20001); Hv = np.array([Hf(U) for U in Us])
roots = [brentq(Hf, Us[i], Us[i + 1]) for i in np.where(np.sign(Hv[:-1]) != np.sign(Hv[1:]))[0]]
for U in roots:
    print(f"  interior stationary equilibrium: U = {U:.6f}, mu = {1-U:.6f}, reservation wage {wbar(1-U):.6f}")
print(f"  U = 1 is always a (degenerate) stationary point: H(1) = f(1)(...) - lambda*0 = 0")

# ================================================================== 18.6 Huggett and Bewley
print("\n=== 18.6  Huggett (1993) parameters: sigma = 1.5, beta = .99322, s in {1, .1}, P = [[.925,.075],[.5,.5]], phi = 2")
sig6, beta6 = 1.5, .99322
s6 = np.array([1., .1]); P6 = np.array([[.925, .075], [.5, .5]])
phi = 2.
grid6 = -phi + (np.linspace(0, 1, 1500) ** 2) * (24 + phi)
b6 = np.full(2, beta6)
def Ea(r):
    ap, c = egm(r, s6, P6, b6, sig6, grid6)
    lam = stationary(young_matrix(ap, grid6, P6)).reshape(len(grid6), 2)
    return (lam * grid6[:, None]).sum(), lam, ap, c
rho = 1 / beta6 - 1
rs = np.linspace(-.02, .0055, 18)
Eas = np.array([Ea(r)[0] for r in rs])
print("  Ea(r) on a grid of r:", np.round(np.c_[rs, Eas], 4).tolist())
rH = brentq(lambda r: Ea(r)[0], -.02, .0)
EaH, lamH, apH, cH = Ea(rH)
Ea0, lamB, apB, cB = Ea(0.)
print(f"  rho = {rho:.6f} per period; Huggett equilibrium r_H = {rH:.6f} per period ({(1 + rH) ** 6 - 1:.4%} per year), Ea(r_H) = {EaH:.2e}")
print(f"  Bewley equilibrium with fiat money: r = 0, M0/p = Ea(0) = {Ea0:.6f}; mass at the debt limit: Huggett {lamH[0].sum():.4f}, Bewley {lamB[0].sum():.4f};"
      f" mass above a = 20: {lamB[grid6 > 20].sum():.2e}")
vH = values(apH, cH, grid6, P6, b6, sig6); vB = values(apB, cB, grid6, P6, b6, sig6)
# (f) rank-preserving transfers.  Time-0 wealth in Huggett is (1 + r_H) a; in Bewley (r = 0) it is a.
def quantile(lam_marg, xs, uq):
    cdf = np.cumsum(lam_marg); return xs[np.minimum(np.searchsorted(cdf, uq), len(xs) - 1)]
uq = (np.arange(20000) + .5) / 20000
aF = quantile(lamH.sum(1), (1 + rH) * grid6, uq); aG = quantile(lamB.sum(1), grid6, uq)
Ttr = aG - aF
print(f"  (f) transfers T(u) = G^-1(u) - F^-1(u): mean {Ttr.mean():.6f} (= M0/p), min {Ttr.min():.4f}, max {Ttr.max():.4f},"
      f" share of agents with T < 0: {(Ttr < -1e-9).mean():.4f}, with T = 0: {(np.abs(Ttr) <= 1e-9).mean():.4f}")
# (g) welfare: an agent of overall rank u and state s compares vB(aG(u), s) with vH(aF(u), s).  Draw s from Lambda_H given the rank.
def interp_v(vv, xs, a):                                         # value at beginning-of-period assets a (grid xs)
    return np.column_stack([np.interp(a, xs, vv[:, z]) for z in range(2)])
# Huggett: agent with beginning-of-period assets a has time-0 wealth (1 + rH) a; value vH is a function of a
aH_beg = aF / (1 + rH)
VH = interp_v(vH, grid6, aH_beg); VB = interp_v(vB, grid6, aG)
# probability of s given the Huggett asset rank: from lamH on the grid
cdfH = np.cumsum(lamH.sum(1)); jH = np.minimum(np.searchsorted(cdfH, uq), len(grid6) - 1)
ps_given = lamH[jH] / lamH[jH].sum(1, keepdims=True)
gain = VB - VH
ce = (VB / VH) ** (1 / (1 - sig6)) - 1                            # consumption-equivalent gain of moving to Bewley
share_pref = (ps_given * (gain > 0)).sum(1).mean()
print(f"  (g) overall-rank scheme: fraction of agents who prefer Bewley {share_pref:.4f}; CE gain: s = high: min {ce[:,0].min():.4%}, max {ce[:,0].max():.4%};"
      f" s = low: min {ce[:,1].min():.4%}, max {ce[:,1].max():.4%}")
losers = uq[(gain < 0).any(1)]
if len(losers):
    print(f"      agents who prefer Huggett are at ranks u in [{losers.min():.4f}, {losers.max():.4f}]")
# alternative scheme: ranks preserved within each employment state (maps Lambda_H exactly into Lambda_B)
share_s, ce_within = [], []
for z in range(2):
    aFz = quantile(lamH[:, z] / lamH[:, z].sum(), (1 + rH) * grid6, uq); aGz = quantile(lamB[:, z] / lamB[:, z].sum(), grid6, uq)
    VBz, VHz = np.interp(aGz, grid6, vB[:, z]), np.interp(aFz / (1 + rH), grid6, vH[:, z])
    g = VBz - VHz; cez = (VBz / VHz) ** (1 / (1 - sig6)) - 1
    share_s.append((g > 0).mean()); ce_within.append(cez)
    lose = uq[g < 0]
    print(f"      within-state scheme, s = {s6[z]}: transfers from {np.min(aGz - aFz):.4f} to {np.max(aGz - aFz):.4f};"
          f" fraction preferring Bewley {(g > 0).mean():.4f}; CE gain from {cez.min():.4%} to {cez.max():.4%}"
          + (f"; losers: within-state ranks <= {lose.max():.4f}, Huggett wealth <= {aFz[g < 0].max():.4f}" if len(lose) else ""))
pi_s = lamH.sum(0)
print(f"      within-state scheme overall fraction preferring Bewley {pi_s @ np.array(share_s):.4f}")
# joint distribution check: does the overall-rank scheme reproduce Lambda_B?  compare P(s = low | a-quantile) in both
cdfB = np.cumsum(lamB.sum(1)); jB = np.minimum(np.searchsorted(cdfB, uq), len(grid6) - 1)
psB = lamB[jB] / lamB[jB].sum(1, keepdims=True)
print(f"      max |P_H(s=low | rank) - P_B(s=low | rank)| over ranks: {np.max(np.abs(ps_given[:,1] - psB[:,1])):.4f}")

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].plot(A, mv[:, 1], "k-", lw=1.2, label="move? (bad wage)")
ax[0].plot(A, A[pol[:, 0]], "k--", lw=.9, label="$A'$, good wage")
ax[0].plot(A, A[pol[:, 1]], color="0.5", lw=.9, label="$A'$, bad wage")
ax[0].plot(A, A, ":", color="0.6", lw=.7)
ax[0].set_xlabel("assets $A$"); ax[0].legend(fontsize=7); ax[0].set_title("18.2: decision rules", fontsize=9)
ax[1].step(A, np.cumsum(lam2.sum(1)), "k", where="post", label="all workers")
ax[1].step(A, np.cumsum(lam2[:, 0]) / lam2[:, 0].sum(), "--", color="0.4", where="post", label="good wage")
ax[1].step(A, np.cumsum(lam2[:, 1]) / lam2[:, 1].sum(), ":", color="0.4", where="post", label="bad wage")
ax[1].axvline(thr, color="grey", lw=.6); ax[1].legend(fontsize=7, loc="lower right")
ax[1].set_xlabel("assets $A$"); ax[1].set_ylabel("CDF"); ax[1].set_title("18.2: invariant distribution of wealth", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch18_02.pdf"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
rr = np.linspace(-.02, .0060, 60); ee = np.array([Ea(r)[0] for r in rr])
ax[0].plot(ee, rr, "k"); ax[0].axvline(0, color="grey", lw=.6); ax[0].axhline(rho, color="grey", lw=.6, ls=":")
ax[0].plot([0], [rH], "ko", ms=4); ax[0].plot([Ea0], [0], "ks", ms=4)
ax[0].annotate(r"Huggett: $Ea(r_H)=0$", (0, rH), (.8, rH - .004), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.6))
ax[0].annotate(r"Bewley: $r=0$, $Ea(0)=M_0/p$", (Ea0, 0), (Ea0 + .6, -.006), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.6))
ax[0].annotate(r"$\rho=1/\beta-1$", (4.5, rho), (-0.3, rho - .0013), fontsize=8)
ax[0].set_xlabel("average assets $Ea(r)$"); ax[0].set_ylabel("interest rate $r$ (per period)")
ax[0].set_title(r"18.6(c): $Ea(r)$, Huggett parameters, $\phi=2$", fontsize=9)
ax[1].plot(uq, 100 * ce_within[0], "k", label=r"$s$ = high"); ax[1].plot(uq, 100 * ce_within[1], color="0.5", label=r"$s$ = low")
ax[1].axhline(0, color="grey", lw=.6); ax[1].set_xlabel("wealth rank within the employment state")
ax[1].set_ylabel("consumption-equivalent gain, %"); ax[1].legend(fontsize=7)
ax[1].set_title("18.6(g): gain from moving to the Bewley equilibrium", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch18_06.pdf"); plt.close()
