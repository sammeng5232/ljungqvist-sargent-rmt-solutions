"""Chapter 14 (Asset Pricing Empirics): numerical parts of the exercises, with cross-checks."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import log_ndtr, ndtr
from scipy.stats import lognorm
from rmtlib import FIG

np.set_printoptions(precision=6, suppress=True, linewidth=140)
rng = np.random.default_rng(14)
LN10 = np.log(10.0)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60)      # Gauss-Hermite nodes for E f(eps), eps ~ N(0,1)
gh_w = gh_w / gh_w.sum()
EN = lambda f: np.sum(gh_w * f(gh_x))                     # E f(eps) by quadrature

# =====================================================================================================
# 14.1  Hansen-Jagannathan bound from the Mehra-Prescott returns; CRRA stochastic discount factors
# =====================================================================================================
mu_R = np.array([1.07, 1.02])                             # as printed in the exercise
mu_R_tab = np.array([1.07, 1.010])                         # Table 14.3.1 (same data, bills 1.010)
Sig = np.array([[.0274, .00104], [.00104, .00308]])
Si = np.linalg.inv(Sig); one = np.ones(2)


def hj(v, mu=mu_R):
    """Lower bound on sigma(y) for E(y)=v: b = cov(x,x)^{-1}(1 - v Ex), bound = sqrt(b'cov b)  (14.5.8)-(14.5.9)."""
    b = Si @ (one - v * mu)
    return np.sqrt(b @ Sig @ b)


def parabola(mu):
    A, B, C = mu @ Si @ mu, one @ Si @ mu, one @ Si @ one   # bound^2 = A v^2 - 2 B v + C
    return A, B, C


for lab, mu in (("exercise (bills 1.02)", mu_R), ("Table 14.3.1 (bills 1.010)", mu_R_tab)):
    A, B, C = parabola(mu)
    vmin, smin = B / A, np.sqrt(C - B ** 2 / A)
    mpr_min, v_mpr = np.sqrt(A - B ** 2 / C), C / B          # min over v of bound(v)/v, attained at v = C/B
    print(f"14.1 {lab}: A,B,C = {A:.4f},{B:.4f},{C:.4f}; min bound {smin:.4f} at E(y)={vmin:.5f};"
          f" min market price of risk {mpr_min:.4f} at E(y)={v_mpr:.5f}")
    print("     bound at E(y) = .90 .92 .94 .96 .98 1.00 1.02:", np.round([hj(v, mu) for v in (.9, .92, .94, .96, .98, 1., 1.02)], 4))
# cross-check 1: parabola formula and pricing of both returns by y* = v + (x-Ex)'b
A, B, C = parabola(mu_R)
for v in (.9, .95, 1.0):
    b = Si @ (one - v * mu_R)
    Eyx = v * mu_R + Sig @ b                                # E[y x] = E y E x + cov(x, y)
    assert np.allclose(Eyx, 1) and abs(hj(v) ** 2 - (A * v * v - 2 * B * v + C)) < 1e-10
# cross-check 2: bound = E(y) x (maximal Sharpe ratio against R_f = 1/E(y)), maximized numerically over portfolios


def max_sharpe(v, mu=mu_R):
    Rf = 1 / v
    f = lambda w: -(w @ (mu - Rf)) / np.sqrt(w @ Sig @ w)
    best = min((minimize(f, w0, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
                for w0 in ([1., 0.], [0., 1.], [1., -1.], [-1., 1.], [-1., -1.])), key=lambda r: r.fun)
    return -best.fun


print("14.1 check (Sharpe-ratio route): max |v*SR_max(v) - bound(v)| over v in {.9,.95,.99,1.02} =",
      f"{max(abs(v * max_sharpe(v) - hj(v)) for v in (.9, .95, .99, 1.02)):.2e}")

# CRRA m = beta G^{-gamma}; consumption growth moments of the Mehra-Prescott data (Table 14.3.1): mean 1.018, var .00127
gbar, gvar, beta1 = 1.018, .00127, .99
s2g = np.log(1 + gvar / gbar ** 2); mg = np.log(gbar) - s2g / 2      # log G ~ N(mg, s2g) matching both moments
print(f"14.1 log growth: mean {mg:.6f}, std {np.sqrt(s2g):.6f}")
G2 = gbar + np.array([-1, 1]) * np.sqrt(gvar)                        # symmetric two-point distribution (Mehra-Prescott style)


def crra_moments(gam, beta=beta1):
    Em = beta * np.exp(-gam * mg + gam ** 2 * s2g / 2)
    sd = Em * np.sqrt(np.expm1(gam ** 2 * s2g))
    m2 = beta * G2 ** (-gam)
    return Em, sd, m2.mean(), m2.std()


print("14.1  gamma   E(m) logn   std(m) logn   E(m) 2pt   std(m) 2pt   HJ bound at E(m) (bills 1.02 / 1.010)   delta-method E, std")
crra_pts = {}
for gam in (0, 5, 10):
    Em, sd, Em2, sd2 = crra_moments(gam)
    crra_pts[gam] = (Em, sd, Em2, sd2)
    Ed = beta1 * gbar ** (-gam) * (1 + gam * (gam + 1) / 2 * gvar / gbar ** 2)   # second-order Taylor approximation
    sdd = beta1 * gam * gbar ** (-gam - 1) * np.sqrt(gvar)
    print(f"      {gam:3d}    {Em:.4f}      {sd:.4f}       {Em2:.4f}     {sd2:.4f}       {hj(Em):.4f} / {hj(Em, mu_R_tab):.4f}"
          f"                    {Ed:.4f}, {sdd:.4f}")
# is any gamma in [0,100] admissible with beta = .99?  (E(m) falls until gamma = mg/s2g, then rises again)
gg = np.linspace(0, 100, 10001)
for lab, mu, col in (("lognormal, bills 1.02", mu_R, 0), ("lognormal, bills 1.010", mu_R_tab, 0),
                     ("two-point, bills 1.02", mu_R, 2), ("two-point, bills 1.010", mu_R_tab, 2)):
    adm = [g for g in gg if crra_moments(g)[col + 1] >= hj(crra_moments(g)[col], mu)]
    print(f"14.1 beta=.99, {lab}: smallest gamma in [0,100] satisfying the bound:", "none" if not adm else f"{adm[0]:.2f}",
          f"(E(m), sigma(m) there: {crra_moments(adm[0])[col]:.4f}, {crra_moments(adm[0])[col+1]:.4f})" if adm else "")
print(f"14.1 lognormal E(m) is smallest at gamma = mg/s2g = {mg/s2g:.2f}, where E(m) = {crra_moments(mg/s2g)[0]:.4f}")
print("14.1 implied net risk-free rate 1/E(m)-1 for gamma = 0, 5, 10:", [f"{1/crra_moments(g)[0]-1:.4f}" for g in (0, 5, 10)])
# Kocherlakota: smallest gamma that can reach the bound when beta is free, and the beta it needs
for lab, mu in (("bills 1.02", mu_R), ("bills 1.010", mu_R_tab)):
    A_, B_, C_ = parabola(mu)
    mpr_min, v_star = np.sqrt(A_ - B_ ** 2 / C_), C_ / B_
    g_need = np.sqrt(np.log(1 + mpr_min ** 2) / s2g)
    b_need = v_star / np.exp(-g_need * mg + g_need ** 2 * s2g / 2)
    print(f"14.1 ({lab}) smallest admissible gamma = {g_need:.3f}, needs beta = {b_need:.4f} (E(m) = {v_star:.4f})")
    if lab == "bills 1.02":
        mpr14, vstar14 = mpr_min, v_star

fig, ax = plt.subplots(figsize=(6.2, 4.0))
vv = np.linspace(.87, 1.02, 400)
ax.plot(vv, [hj(v) for v in vv], "k-", lw=1.6, label="HJ bound (bills 1.02, as in the exercise)")
ax.plot(vv, [hj(v, mu_R_tab) for v in vv], "k--", lw=1.0, label="HJ bound (bills 1.010, Table 14.3.1)")
glist = np.linspace(0, 28, 400)
loc = np.array([crra_moments(g)[:2] for g in glist])
ax.plot(loc[:, 0], loc[:, 1], ":", color="0.4", lw=1.2, label=r"CRRA locus (lognormal), $\gamma\in[0,28]$, $\beta=.99$")
for gam, (Em, sd, Em2, sd2) in crra_pts.items():
    ax.plot(Em, sd, "ko", ms=5); ax.plot(Em2, sd2, "o", mfc="none", mec="0.35", ms=7)
    ax.annotate(rf"$\gamma={gam}$", (Em, sd), textcoords="offset points", xytext=(4, 7), fontsize=9)
for gam in (15, 20, 25):
    Em, sd = crra_moments(gam)[:2]
    ax.plot(Em, sd, "k+", ms=6); ax.annotate(rf"${gam}$", (Em, sd), textcoords="offset points", xytext=(4, -9), fontsize=7.5)
ax.axvspan(.9, 1.02, color="0.93", zorder=0)
ax.set_xlim(.87, 1.02); ax.set_ylim(0, 2.0)
ax.set_xlabel(r"$E(m)$"); ax.set_ylabel(r"$\sigma(m)$"); ax.legend(fontsize=7.5, loc="upper right", frameon=False)
plt.tight_layout(); plt.savefig(FIG + "ch14_01.pdf"); plt.close()

# =====================================================================================================
# 14.2  Regime switching and the term structure
# =====================================================================================================
rng = np.random.default_rng(1402)
def ts_regime(beta, gam, a0, a1, mu, tau, p0, p1, nmax):
    """Per-period log yields r_n(s), s = 0 (good), 1 (bad): r_n = rbar - (1/n) log h_n, h_n = (DP)^{n-1} D 1."""
    P = np.array([[p0, 1 - p0], [1 - p1, p1]]); k = gam * a1
    D = np.diag([1.0, np.exp(k)])
    rbar = -np.log(beta) + gam * (a0 + mu) - gam ** 2 * tau ** 2 / 2
    h = D @ np.ones(2); out = [rbar - np.log(h)]
    for n in range(2, nmax + 1):
        h = D @ P @ h; out.append(rbar - np.log(h) / n)
    rinf = rbar - np.log(np.max(np.abs(np.linalg.eigvals(D @ P))))
    return np.array(out), rinf, rbar, k, P


pars2 = dict(beta=.99, gam=2.0, a0=.03, a1=.02, mu=.005, tau=.02)
print("14.2 illustrative parameters:", pars2)
for (p0, p1) in ((.9, .8), (.6, .4), (.3, .2)):
    r, rinf, rbar, k, P = ts_regime(**pars2, p0=p0, p1=p1, nmax=200)
    amp2 = r[1, 0] - r[1, 1]
    lam = p0 + p1 - 1
    pred = .5 * (k + np.log(p1 * np.exp(k) + 1 - p1) - np.log((1 - p0) * np.exp(k) + p0))
    print(f"14.2 pi(0)={p0}, pi(1)={p1} (autocorr {lam:+.1f}): r_1 = {100*r[0,0]:.3f}/{100*r[0,1]:.3f}%,"
          f" r_2 = {100*r[1,0]:.3f}/{100*r[1,1]:.3f}%, r_10 = {100*r[9,0]:.3f}/{100*r[9,1]:.3f}%,"
          f" r_inf = {100*rinf:.3f}%  (good/bad)")
    print(f"     amplitude r_n(0)-r_n(1), n=1,2,5,10,40: {np.round(100*(r[[0,1,4,9,39],0]-r[[0,1,4,9,39],1]),4)} (% points);"
          f" two-period amplitude/short amplitude = {amp2/k:.4f} (formula {pred/k:.4f})")
    # Monte Carlo check of the two- and five-period bond prices from state s (antithetic in the epsilon shocks)
    if (p0, p1) == (.9, .8):
        g_, t_ = pars2["gam"], pars2["tau"]
        for s0 in (0, 1):
            N = 4000000; s = np.full(N, s0); lreg = np.zeros(N); S = np.zeros(N)
            for j in range(5):
                lreg += np.log(pars2["beta"]) - g_ * (pars2["a0"] - pars2["a1"] * s + pars2["mu"])
                S += rng.standard_normal(N)
                if j == 1: q2 = np.mean(np.exp(lreg) * np.cosh(g_ * t_ * S))
                s = np.where(rng.random(N) < P[s, 1], 1, 0)
            q5 = np.mean(np.exp(lreg) * np.cosh(g_ * t_ * S))
            print(f"     MC check, s={s0}: r_2 {-np.log(q2)/2:.6f} vs {r[1, s0]:.6f};  r_5 {-np.log(q5)/5:.6f} vs {r[4, s0]:.6f}")
# grid check of the claim h_n(1) >= h_n(0) (all maturities procyclical) and of the slope signs
viol = 0
for p0 in np.linspace(.02, .98, 25):
    for p1 in np.linspace(.02, .98, 25):
        for k in (.01, .1, 1., 3.):
            P = np.array([[p0, 1 - p0], [1 - p1, p1]]); D = np.diag([1, np.exp(k)]); h = D @ np.ones(2)
            for n in range(1, 120):
                if h[1] < h[0] * (1 - 1e-12) or h[0] < 1 - 1e-12 or h[1] > np.exp(n * k) * (1 + 1e-12): viol += 1
                h = D @ P @ h
print("14.2 grid check (25x25 chains x 4 k x 119 maturities): violations of 1 <= h_n(0) <= h_n(1) <= e^{nk}:", viol)

# =====================================================================================================
# 14.3  Growth slowdowns and stock prices
# =====================================================================================================
beta3, nu3, pi3 = .96, 1.02, .95
print(f"14.3 beta={beta3}, nu={nu3}, pi={pi3}")
print("     gamma   w_G (growing)   w_S (stopped)   p_T/p_(T-1)-1   R_T (gross)   Rf growing   Rf stopped   w_G by direct sum")
for gam in (.5, 1.0, 2.0, 5.0):
    x = nu3 ** (1 - gam); a = beta3 * pi3 * x
    wS = beta3 / (1 - beta3)
    wG = (a + beta3 * (1 - pi3) / (1 - beta3)) / (1 - a)
    # direct sum: E[(d_{t+j}/d_t)^{1-gamma}] = sum_{n=0}^{j-1} (1-pi) pi^n x^n + (pi x)^j  (growth stops after n more periods)
    J = 3000; j = np.arange(1, J + 1)
    Eg = np.cumsum((1 - pi3) * (pi3 * x) ** np.arange(J))            # entry j-1 holds sum_{n=0}^{j-1}
    wG_sum = np.sum(beta3 ** j * (Eg + (pi3 * x) ** j))
    RfG = 1 / (beta3 * (pi3 * nu3 ** (-gam) + 1 - pi3)); RfS = 1 / beta3
    print(f"     {gam:4.1f}    {wG:10.4f}      {wS:10.4f}      {100*(wS/wG-1):+8.3f}%      {(1+wS)/wG:.4f}       {RfG:.4f}      {RfS:.4f}     {wG_sum:.4f}")

# =====================================================================================================
# 14.4  Term structure with log-AR(1) consumption: exact check of the closed form
# =====================================================================================================
b4, g4, phi4, sig4, logCs = .96, 2.0, .9, .03, .01
s2_4 = np.log(1 + sig4 ** 2); cbar4 = (logCs - s2_4 / 2) / (1 - phi4)      # eps lognormal, mean 1, variance sig4^2


def logQ4(j, c):   # log price of a j-period zero-coupon bond when log consumption is c
    return (j * np.log(b4) - g4 * (1 - phi4 ** j) * (cbar4 - c)
            + g4 ** 2 * s2_4 * (1 - phi4 ** (2 * j)) / (2 * (1 - phi4 ** 2)))


# exact check: Q_{j+1}(c) = E[ beta exp(-gamma (c'-c)) Q_j(c') ],  c' = log C* + phi c + e,  e ~ N(-s^2/2, s^2)
err4 = 0.
for c in (cbar4 - .1, cbar4, cbar4 + .05):
    for j in range(0, 12):
        cp = lambda e: logCs + phi4 * c - s2_4 / 2 + np.sqrt(s2_4) * e
        rhs = EN(lambda e: np.exp(np.log(b4) - g4 * (cp(e) - c) + (logQ4(j, cp(e)) if j else 0.)))
        err4 = max(err4, abs(np.log(rhs) - logQ4(j + 1, c)))
print(f"14.4 quadrature check of the bond-pricing recursion for j=1..12: max |log error| = {err4:.1e}")
print("14.4 slopes gamma(1-phi^j)/j for j=1,2,5,10,40:", np.round([g4 * (1 - phi4 ** j) / j for j in (1, 2, 5, 10, 40)], 4))

# =====================================================================================================
# 14.5  Risk-sensitive recursion: value function, likelihood ratio, and the plans of Figures 14.7.1-14.7.2
# =====================================================================================================
bet, th, mu5, sc5 = .95, .7, .004, .02
k1 = 1 / (1 - bet); k0 = bet / (1 - bet) ** 2 * (mu5 - sc5 ** 2 / (2 * th * (1 - bet)))
for cc in (-1.0, 0.0, 2.5):
    rhs = cc - bet * th * np.log(EN(lambda e: np.exp(-(k0 + k1 * (cc + mu5 + sc5 * e)) / th)))
    rhs3 = cc + bet * (k0 + k1 * (cc + mu5)) - bet / (2 * th) * (k1 * sc5) ** 2
    assert abs(rhs - (k0 + k1 * cc)) < 1e-9 and abs(rhs3 - rhs) < 1e-9
w5 = -sc5 / (th * (1 - bet))
gnum = lambda e: np.exp(-(k0 + k1 * (0.3 + mu5 + sc5 * e)) / th)
ee = np.array([-1.5, 0.0, 2.0])
print("14.5 quadrature check of k0,k1 and of (3): passed;  g(eps) direct vs exp(w eps - w^2/2):",
      np.round(gnum(ee) / EN(gnum), 8), np.round(np.exp(w5 * ee - w5 ** 2 / 2), 8))


def T_exact(vals, probs, th):   # -theta log E exp(-U/theta)
    vals, probs = np.asarray(vals, float), np.asarray(probs, float)
    return -th * np.log(np.sum(probs * np.exp(-vals / th)))


def T_mv(vals, probs, th):      # E U - var(U)/(2 theta)
    vals, probs = np.asarray(vals, float), np.asarray(probs, float)
    m = np.sum(probs * vals); return m - np.sum(probs * (vals - m) ** 2) / (2 * th)


def plan_value(T, b, th, plan):
    h = [.5, .5]
    if plan == "A":   # t=1 reveals whether c2 = 2 or 0
        U1 = [1 + b * T([2, 2], h, th), 1 + b * T([0, 0], h, th)]
    elif plan == "B":  # nothing learned at t=1
        U1 = [1 + b * T([2, 0], h, th)] * 2
    elif plan == "C":  # iid
        U1 = [2 + b * T([2, 1], h, th), 1 + b * T([2, 1], h, th)]
    else:              # D: persistent
        U1 = [2 + b * T([2, 2], h, th), 1 + b * T([1, 1], h, th)]
    return 1 + b * T(U1, h, th)


for th_ in (1.0, 5.0, 1e9):
    vals = {p: (plan_value(T_exact, bet, th_, p), plan_value(T_mv, bet, th_, p)) for p in "ABCD"}
    print(f"14.5 plans (beta={bet}, theta={th_:g}): " + "; ".join(f"{p}: exact {v[0]:.5f}, mean-var {v[1]:.5f}" for p, v in vals.items()))
    print(f"     U_A-U_B mean-var {vals['A'][1]-vals['B'][1]:.6f} (formula b^2(1-b)/(2th) = {bet**2*(1-bet)/(2*th_):.6f});"
          f" U_C-U_D mean-var {vals['C'][1]-vals['D'][1]:.6f} (formula b^2(1+b)/(8th) = {bet**2*(1+bet)/(8*th_):.6f})")

# =====================================================================================================
# 14.8  Lucas tree with a two-state chain, log utility; bond/equity split
# =====================================================================================================
b8, yL, yH, piL, piH, eta = .95, 1.0, 1.25, .8, .9, .5
P8 = np.array([[piL, 1 - piL], [1 - piH, piH]]); y8 = np.array([yL, yH])
M = b8 * P8 * (y8[:, None] / y8[None, :])        # M[s,s'] = P(s,s') m(s,s'): one-period Arrow prices
Rf8 = 1 / M.sum(1)
p8 = np.linalg.solve(np.eye(2) - M, M @ y8)      # p = M (y + p)
B8 = np.linalg.solve(np.eye(2) - M, M @ (eta * np.ones(2)))
E8 = np.linalg.solve(np.eye(2) - M, M @ (y8 - eta))
ER8 = (P8 * (y8[None, :] / (b8 * y8[:, None]))).sum(1)   # expected gross return on the tree
# closed forms: p = beta y/(1-beta);  B(s) = eta y_s [ (I - beta P)^{-1} beta P (1/y) ]_s
Bcf = eta * y8 * np.linalg.solve(np.eye(2) - b8 * P8, b8 * P8 @ (1 / y8))
print(f"14.8 beta={b8}, y=({yL},{yH}), piL={piL}, piH={piH}, eta={eta}")
print("     Rf(L,H) =", np.round(Rf8, 5), " p =", np.round(p8, 5), " beta y/(1-beta) =", np.round(b8 * y8 / (1 - b8), 5))
print("     bonds B =", np.round(B8, 5), " closed form", np.round(Bcf, 5), "; equity E =", np.round(E8, 5), "; B+E-p =", B8 + E8 - p8)
print("     E_s[R_tree] =", np.round(ER8, 5), " premium E R - Rf =", np.round(ER8 - Rf8, 5))

# =====================================================================================================
# 14.9  Long-run risk with CRRA: checks of E_t(m m), yields and the value function
# =====================================================================================================
rng = np.random.default_rng(1409)
b9, g9, mu9, rho9, sc9, sz9 = .995, 2.0, .005, .95, .005, .0005
z0 = .002
def logEmm(n, z):   # log E_t[m_{t+1}...m_{t+n}]
    k = np.arange(n); s = sc9 + sz9 * (1 - rho9 ** k) / (1 - rho9)
    return n * np.log(b9) - g9 * (n * mu9 + z * (1 - rho9 ** n) / (1 - rho9)) + g9 ** 2 / 2 * np.sum(s ** 2)
N = 2000000; e1, e2, e3 = rng.standard_normal((3, N))
e1, e2, e3 = np.r_[e1, -e1], np.r_[e2, -e2], np.r_[e3, -e3]          # antithetic pairs
z1 = rho9 * z0 + sz9 * e1; z2 = rho9 * z1 + sz9 * e2
dc1 = mu9 + z0 + sc9 * e1; dc2 = mu9 + z1 + sc9 * e2; dc3 = mu9 + z2 + sc9 * e3
mm2 = b9 ** 2 * np.exp(-g9 * (dc1 + dc2)); mm3 = mm2 * b9 * np.exp(-g9 * dc3)
print(f"14.9 MC check: log E(m1 m2) {np.log(mm2.mean()):.6f} vs formula {logEmm(2, z0):.6f};"
      f" log E(m1 m2 m3) {np.log(mm3.mean()):.6f} vs {logEmm(3, z0):.6f}")
yld = lambda n, z: -logEmm(n, z) / n
print(f"     parameters beta={b9}, gamma={g9}, mu={mu9}, rho={rho9}, sigma_c={sc9}, sigma_z={sz9}")
print("     yields (quarterly %, z=0): n=1,4,20,100,400:", np.round([100 * yld(n, 0.) for n in (1, 4, 20, 100, 400)], 4),
      " loading on z:", np.round([g9 * (1 - rho9 ** n) / (n * (1 - rho9)) for n in (1, 4, 20, 100, 400)], 3),
      f" limit of constant: {100*(-np.log(b9) + g9*mu9 - g9**2/2*(sc9 + sz9/(1-rho9))**2):.4f}")
# value function U_t = C^{1-g} f(z), f(z) = (1/(1-g)) sum_t beta^t exp(a_t + b_t z): verify f solves its recursion by quadrature
Tn = 6000; tt = np.arange(Tn)
sk = sc9 + sz9 * (1 - rho9 ** tt) / (1 - rho9)
a_t = tt * (1 - g9) * mu9 + (1 - g9) ** 2 / 2 * np.concatenate(([0.], np.cumsum(sk ** 2)))[:-1]
b_t = (1 - g9) * (1 - rho9 ** tt) / (1 - rho9)
f9 = lambda z: np.sum(b9 ** tt * np.exp(a_t + b_t * np.atleast_1d(z)[:, None]), axis=1) / (1 - g9)
kap9 = b9 * np.exp((1 - g9) * mu9 + (1 - g9) ** 2 / 2 * (sc9 + sz9 / (1 - rho9)) ** 2)
for zz in (-.003, 0., .004):
    rhs = 1 / (1 - g9) + b9 * EN(lambda e: np.exp((1 - g9) * (mu9 + zz + sc9 * e)) * f9(rho9 * zz + sz9 * e))
    assert abs(rhs / f9(zz)[0] - 1) < 1e-8
print(f"     value function recursion verified by quadrature; kappa = {kap9:.6f} (<1 needed); f(0) = {f9(0.)[0]:.4f}")

# =====================================================================================================
# 14.10  Long-run risk with the risk-sensitive recursion
# =====================================================================================================
b10, mu10, rho10, sc10, sz10 = .995, .005, .95, .005, .0005
th10 = 1 / ((10 - 1) * (1 - b10))                       # theta implied by gamma = 10
K1 = 1 / (1 - b10); K2 = b10 / ((1 - b10) * (1 - b10 * rho10)); sgt = sc10 + b10 * sz10 / (1 - b10 * rho10)
K0 = b10 / (1 - b10) ** 2 * (mu10 - sgt ** 2 / (2 * th10 * (1 - b10)))
for cc, zz in ((0., 0.), (1.3, .004), (-2., -.006)):
    U1 = lambda e: K0 + K1 * (cc + mu10 + zz + sc10 * e) + K2 * (rho10 * zz + sz10 * e)
    rhs = cc - b10 * th10 * np.log(EN(lambda e: np.exp(-U1(e) / th10)))
    assert abs(rhs - (K0 + K1 * cc + K2 * zz)) < 1e-8
w10 = -sgt / (th10 * (1 - b10)); gam10 = 1 + 1 / (th10 * (1 - b10))
zz = .003
Em_q = EN(lambda e: b10 * np.exp(-(mu10 + zz + sc10 * e)) * np.exp(w10 * e - w10 ** 2 / 2))
Em_f = np.exp(np.log(b10) - mu10 - zz + sc10 ** 2 / 2 - sc10 * w10)
print(f"14.10 guess verified by quadrature; theta={th10:.3f}, k1={K1:.1f}, k2={K2:.3f}; sigma_tilde={sgt:.6f};"
      f" w={w10:.4f} = (1-gamma)*sigma_tilde with gamma={gam10:.1f}: {(1-gam10)*sgt:.4f};"
      f" price of risk sigma_c-w = {sc10-w10:.4f} (without long-run risk gamma*sigma_c = {gam10*sc10:.4f}); E m: {Em_q:.8f} vs {Em_f:.8f}")

# =====================================================================================================
# 14.11  Stochastic volatility: yields in the two volatility states
# =====================================================================================================
rng = np.random.default_rng(1411)
b11, g11, mu11, sL, sH, p0_, p1_ = .995, 10.0, .005, .004, .008, .95, .90
P11 = np.array([[p0_, 1 - p0_], [1 - p1_, p1_]]); D11 = np.diag(np.exp(.5 * g11 ** 2 * np.array([sL, sH]) ** 2))
h = D11 @ np.ones(2); Y = [(-np.log(b11) + g11 * mu11) - np.log(h)]
for n in range(2, 401):
    h = D11 @ P11 @ h; Y.append(-np.log(b11) + g11 * mu11 - np.log(h) / n)
Y = np.array(Y); Yinf = -np.log(b11) + g11 * mu11 - np.log(np.max(np.linalg.eigvals(D11 @ P11).real))
print(f"14.11 beta={b11}, gamma={g11}, mu={mu11}, sigma_L={sL}, sigma_H={sH}, pi0={p0_}, pi1={p1_}")
nn = [0, 1, 3, 19, 99, 399]
print("      yields (% per quarter) n=1,2,4,20,100,400,inf  low vol:", np.round(100 * np.r_[Y[nn, 0], Yinf], 4))
print("                                                    high vol:", np.round(100 * np.r_[Y[nn, 1], Yinf], 4))
print("      low-minus-high spread (basis points) n=1,2,4,20,100,400:", np.round(1e4 * (Y[nn, 0] - Y[nn, 1]), 2),
      "; slope y_20 - y_1 (bp): low", round(1e4 * (Y[19, 0] - Y[0, 0]), 2), " high", round(1e4 * (Y[19, 1] - Y[0, 1]), 2))
for s0 in (0, 1):
    N = 2000000; sig0 = (sL, sH)[s0]; s1 = np.where(rng.random(N) < P11[s0, 1], 1, 0); sig1 = np.where(s1 == 1, sH, sL)
    x_ = sig0 * rng.standard_normal(N) + sig1 * rng.standard_normal(N)
    mm = b11 ** 2 * np.exp(-2 * g11 * mu11) * np.cosh(g11 * x_)          # antithetic average of exp(-g x) and exp(+g x)
    print(f"      MC check two-period yield, state {s0}: {-np.log(mm.mean())/2:.6f} vs {Y[1, s0]:.6f}")

# =====================================================================================================
# 14.12-14.13  Learning about mu (and z): Kalman filter with a common shock, risk-free rate
# =====================================================================================================
rng = np.random.default_rng(1413)
b13, muhat0, zhat0, rho13, sc13, sz0 = .995, .005, 0.0, .99, .005, .00005     # the exercise's values
smu, sz13 = .0025, .00005                                                        # our choices (not given in the book)
Acal = np.diag([1.0, rho13]); Ccal = np.array([0.0, sz13]); Gcal = np.array([1.0, 1.0])


def kalman_common(y, xhat0, Sig0, A, C, G, D):
    """x_{t+1} = A x_t + C e_{t+1}, y_{t+1} = G x_t + D e_{t+1} (same scalar shock).  Returns xhat_t, Sigma_t, Omega_t
    for t = 0..T (xhat_t, Sigma_t condition on y_1..y_t; Omega_t = var(y_{t+1} | y^t))."""
    T = len(y); xh = np.zeros((T + 1, 2)); Ss = np.zeros((T + 1, 2, 2)); Om = np.zeros(T + 1)
    xh[0], Ss[0] = xhat0, Sig0
    for t in range(T):
        S = Ss[t]; Om[t] = G @ S @ G + D * D
        K = (A @ S @ G + C * D) / Om[t]
        xh[t + 1] = A @ xh[t] + K * (y[t] - G @ xh[t])
        Ss[t + 1] = A @ S @ A.T + np.outer(C, C) - np.outer(K, K) * Om[t]
    Om[T] = G @ Ss[T] @ G + D * D
    return xh, Ss, Om


T13 = 10000; mu_true, zt = .005, 0.0
eps = rng.standard_normal(T13)
dc = np.zeros(T13); zpath = np.zeros(T13 + 1)
for t in range(T13):
    dc[t] = mu_true + zpath[t] + sc13 * eps[t]; zpath[t + 1] = rho13 * zpath[t] + sz13 * eps[t]
Sig0 = np.diag([smu ** 2, sz0 ** 2])
xh, Ss, Om = kalman_common(dc, np.array([muhat0, zhat0]), Sig0, Acal, Ccal, Gcal, sc13)
Rf13 = np.exp(-np.log(b13) + xh @ Gcal - Om / 2)                      # gross risk-free rate at t = 0..T
Rfi13 = np.exp(-np.log(b13) + mu_true + zpath - sc13 ** 2 / 2)       # full-information rate
# batch (GLS) cross-check of the filter for the first Tb observations
Tb = 30
H = np.zeros((Tb, Tb + 2)); m0 = np.r_[muhat0, zhat0, np.zeros(Tb)]; S0 = np.diag(np.r_[smu ** 2, sz0 ** 2, np.ones(Tb)])
for s in range(1, Tb + 1):
    H[s - 1, 0] = 1; H[s - 1, 1] = rho13 ** (s - 1)
    for kk in range(1, s): H[s - 1, 1 + kk] = sz13 * rho13 ** (s - 1 - kk)
    H[s - 1, 1 + s] = sc13
L = np.zeros((2, Tb + 2)); L[0, 0] = 1; L[1, 1] = rho13 ** Tb
for kk in range(1, Tb + 1): L[1, 1 + kk] = sz13 * rho13 ** (Tb - kk)
HS = H @ S0; Kb = L @ HS.T @ np.linalg.inv(HS @ H.T)
xb = L @ m0 + Kb @ (dc[:Tb] - H @ m0); Sb = L @ S0 @ L.T - Kb @ HS @ L.T
print(f"14.13 batch GLS vs Kalman at t={Tb}: |xhat diff| = {np.max(np.abs(xb - xh[Tb])):.2e},"
      f" |Sigma diff|/|Sigma| = {np.max(np.abs(Sb - Ss[Tb])) / np.max(np.abs(Ss[Tb])):.2e}")
# 14.12: mu unknown, no z (data dc12 = mu + sc eps); the same filter with sz = sz0 = 0 must reproduce the conjugate normal formulas
dc12 = mu_true + sc13 * eps
xh12, Ss12, Om12 = kalman_common(dc12, np.array([muhat0, 0.]), np.diag([smu ** 2, 0.]), Acal, np.zeros(2), Gcal, sc13)
tt13 = np.arange(T13 + 1)
post_var = 1 / (1 / smu ** 2 + tt13 / sc13 ** 2)
post_mean = post_var * (muhat0 / smu ** 2 + np.r_[0, np.cumsum(dc12)] / sc13 ** 2)
print(f"14.12 conjugate-normal check: max|var diff| {np.max(np.abs(Ss12[:, 0, 0] - post_var)):.1e},"
      f" max|mean diff| {np.max(np.abs(xh12[:, 0] - post_mean)):.1e}")
Rf12 = np.exp(-np.log(b13) + xh12[:, 0] - Om12 / 2)
Rfi12 = np.exp(-np.log(b13) + mu_true - sc13 ** 2 / 2)
for t in (0, 1, 10, 100, 1000, 10000):
    print(f"14.13 t={t:5d}: Rf {Rf13[t]:.6f} (full info {Rfi13[t]:.6f});  post sd mu {np.sqrt(Ss[t,0,0]):.2e}, z {np.sqrt(Ss[t,1,1]):.2e};"
          f" Omega-sc^2 {Om[t]-sc13**2:.2e};  14.12: Rf {Rf12[t]:.6f}, post sd mu {np.sqrt(Ss12[t,0,0]):.2e}")
seg = lambda a, lo, hi: a[lo:hi]
print("14.13 std of Rf over t in [0,100), [100,1000), [1000,10000]:", [f"{np.std(seg(Rf13, lo, hi)):.2e}" for lo, hi in ((0, 100), (100, 1000), (1000, 10001))],
      " full-info:", [f"{np.std(seg(Rfi13, lo, hi)):.2e}" for lo, hi in ((0, 100), (100, 1000), (1000, 10001))])
print(f"14.13 max|Rf - Rf_fullinfo| for t >= 5000: {np.max(np.abs(Rf13[5000:] - Rfi13[5000:])):.2e};"
      f" 14.12: max|Rf - Rf_fullinfo| for t>=5000: {np.max(np.abs(Rf12[5000:] - Rfi12)):.2e}")
print(f"14.13 stationary sd of z = {sz13/np.sqrt(1-rho13**2):.2e}; long-run sd of dc = sc + sz/(1-rho) = {sc13+sz13/(1-rho13):.4f};"
      f" sd of Rf_fullinfo = {np.std(Rfi13):.2e}")
# expected drift of log Rf under the consumer's beliefs in 14.12: (sigma_t^2 - sigma_{t+1}^2)/2
print(f"14.12 precautionary drift of log Rf between t=0 and t=1: {(post_var[0]-post_var[1])/2:.2e}; total from t=0 to 10000: {(post_var[0]-post_var[-1])/2:.2e}")

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
axs[0].plot(tt13, Rfi13, color="0.65", lw=.6, label="full information")
axs[0].plot(tt13, Rf13, "k-", lw=.7, label="learning (14.13)")
axs[0].set_xlabel("$t$ (quarters)"); axs[0].set_title("gross risk-free rate, $t=0,\\ldots,10000$", fontsize=9)
axs[0].legend(fontsize=7, frameon=False, loc="lower right")
n0 = 401
axs[1].plot(tt13[:n0], Rfi13[:n0], color="0.65", lw=.8, label="full information (14.13)")
axs[1].plot(tt13[:n0], Rf13[:n0], "k-", lw=.9, label="learning (14.13)")
axs[1].plot(tt13[:n0], Rf12[:n0], "k--", lw=.8, label=r"learning $\mu$ only (14.12)")
axs[1].axhline(Rfi12, color="k", ls=":", lw=.8, label="full information (14.12)")
axs[1].set_xlabel("$t$ (quarters)"); axs[1].set_title("first 400 quarters", fontsize=9); axs[1].legend(fontsize=6.5, frameon=False, loc="lower right")
axs[2].loglog(tt13[1:], np.sqrt(Ss[1:, 0, 0]), "k-", lw=1, label=r"sd of $\mu$ (14.13)")
axs[2].loglog(tt13[1:], np.sqrt(Ss[1:, 1, 1]), "k-.", lw=1, label=r"sd of $z_t$ (14.13)")
axs[2].loglog(tt13[1:], np.sqrt(Ss12[1:, 0, 0]), "k--", lw=1, label=r"sd of $\mu$ (14.12)")
axs[2].set_xlabel("$t$ (quarters)"); axs[2].set_title("posterior standard deviations", fontsize=9); axs[2].legend(fontsize=7, frameon=False)
for a_ in axs[:2]: a_.ticklabel_format(axis="y", useOffset=False)
plt.tight_layout(); plt.savefig(FIG + "ch14_13.pdf"); plt.close()

# =====================================================================================================
# 14.14  Affine term structure: recursion vs risk-neutral closed form vs Monte Carlo under P
# =====================================================================================================
mu14 = np.array([.001, 0.]); phi14 = np.array([[.95, .02], [0., .8]]); C14 = np.array([[.01, 0.], [.004, .008]])
d0, d1 = .01, np.array([1., .5]); L0 = np.array([.2, -.1]); Lz = np.array([[5., 0.], [0., 3.]])
nmax = 12; Abar = [-d0]; Bbar = [-d1]
for n in range(1, nmax):
    Abar.append(Abar[-1] + Bbar[-1] @ (mu14 - C14 @ L0) + .5 * Bbar[-1] @ C14 @ C14.T @ Bbar[-1] - d0)
    Bbar.append(Bbar[-1] @ (phi14 - C14 @ Lz) - d1)
muQ, phiQ = mu14 - C14 @ L0, phi14 - C14 @ Lz


def logp_Q(n, z):   # log E^Q exp(-sum_{i<n} r_{t+i}) computed from the Gaussian distribution of the sum
    mean = 0.; Ez = z.copy(); var = 0.
    for i in range(n):
        mean += d0 + d1 @ Ez; Ez = muQ + phiQ @ Ez
    for kk in range(1, n):                                  # shock at t+kk affects z_{t+kk},...,z_{t+n-1}
        Mj = sum(np.linalg.matrix_power(phiQ, j) for j in range(n - kk))
        v = d1 @ Mj @ C14; var += v @ v
    return -mean + var / 2


zt14 = np.array([.02, -.03])
err = max(abs(Abar[n - 1] + Bbar[n - 1] @ zt14 - logp_Q(n, zt14)) for n in range(1, nmax + 1))
# exact 2-d Gauss-Hermite check of p_t(n+1) = E_t[m_{t+1} p_{t+1}(n)] under the physical measure
E1, E2 = np.meshgrid(gh_x, gh_x, indexing="ij"); W2 = np.outer(gh_w, gh_w)
EPS = np.stack([E1.ravel(), E2.ravel()], 1); WW = W2.ravel()
errq = 0.
for zz in (zt14, np.array([-.05, .04]), np.zeros(2)):
    lam = L0 + Lz @ zz; r = d0 + d1 @ zz
    logm = -r - .5 * lam @ lam - EPS @ lam
    znext = mu14 + phi14 @ zz + EPS @ C14.T
    for n in range(1, nmax):
        rhs = np.sum(WW * np.exp(logm + Abar[n - 1] + znext @ Bbar[n - 1]))
        errq = max(errq, abs(np.log(rhs) - (Abar[n] + Bbar[n] @ zz)))
print(f"14.14 max |recursion - risk-neutral closed form| over n=1..{nmax}: {err:.2e};"
      f" max |log p(n+1) - log E[m p(n)]| by 2-d quadrature: {errq:.2e}")

# =====================================================================================================
# 14.19  Relative entropy of the twisted density
# =====================================================================================================
rng = np.random.default_rng(1419)
lam19 = np.array([.3, -.2, .5]); N = 2000000; e = rng.standard_normal((N, 3))
l = np.exp(-lam19 @ lam19 / 2 - e @ lam19)
tw = -lam19 + rng.standard_normal((N, 3))                    # draws from the twisted density N(-lambda, I)
print(f"14.19 MC: E l = {l.mean():.4f};  E[l log l] = {np.mean(l*np.log(l)):.5f};  E_twisted[log l] = "
      f"{np.mean(-lam19@lam19/2 - tw@lam19):.5f};  lambda'lambda/2 = {lam19@lam19/2:.5f};"
      f"  E_twisted eps = {np.round(np.mean(tw,0),3)} vs E[l eps] = {np.round(np.mean(l[:,None]*e,0),3)}")

# =====================================================================================================
# 14.20  Likelihood-ratio process: lognormal cdf, moments, and convergence to zero in distribution
# =====================================================================================================
rng = np.random.default_rng(1420)
def lognormal_cdf(x, m, s):
    """P(X <= x) for log X ~ N(m, s^2), written from scratch with the error function (via ndtr)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x); pos = x > 0
    out[pos] = ndtr((np.log(x[pos]) - m) / s)
    return out


xs = np.array([.01, .5, 1., 2., 10.])
for t in (1, 5):   # cross-checks against scipy and Monte Carlo
    sc_ = lognorm.cdf(xs, s=np.sqrt(t), scale=np.exp(-t / 2))
    sim = np.exp(-np.sum(rng.standard_normal((200000, t)), 1) - t / 2)
    assert np.max(np.abs(lognormal_cdf(xs, -t / 2, np.sqrt(t)) - sc_)) < 1e-12
    print(f"14.20 t={t}: cdf at {xs}: ours {np.round(lognormal_cdf(xs, -t/2, np.sqrt(t)), 5)}, MC {np.round([(sim<=x).mean() for x in xs], 5)}; sample mean {sim.mean():.4f}")
print("14.20   t   log10 median   F(0.01)     F(0.5)      F(1)        log10 P(xi>1)   log10 sd   log10 skew   log10 mode")
for t in (1, 5, 100, 1000, 10000):
    s = np.sqrt(t); m = -t / 2
    Fv = [ndtr((np.log(x) - m) / s) for x in (.01, .5, 1.)]
    lp = log_ndtr(-(0 - m) / s) / LN10                           # P(xi > 1) = Phi(-sqrt(t)/2)
    lsd = (t + np.log(-np.expm1(-t))) / 2 / LN10                  # log10 sqrt(e^t - 1)
    lsk = (t + np.log1p(2 * np.exp(-t)) + (t + np.log(-np.expm1(-t))) / 2) / LN10   # log10[(e^t+2) sqrt(e^t-1)]
    print(f"      {t:5d}   {m/LN10:9.2f}     {Fv[0]:.6f}   {Fv[1]:.6f}   {Fv[2]:.8f}   {lp:10.2f}    {lsd:9.2f}   {lsk:9.2f}   {-1.5*t/LN10:9.2f}")
print("14.20 E[xi_t 1{xi_t > 1}] = Phi(sqrt(t)/2) for t = 1,5,100:", np.round([ndtr(np.sqrt(t) / 2) for t in (1, 5, 100)], 7))
sm = []
for t in (1, 5, 20, 50, 100):
    ssum = np.zeros(1000000)
    for _ in range(t): ssum += rng.standard_normal(1000000)
    sm.append(np.exp(-ssum - t / 2).mean())
print("14.20 sample means of 10^6 simulated xi_t, t = 1,5,20,50,100:", [f"{v:.3g}" for v in sm])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
xx = np.linspace(1e-6, 3, 1200)
for t, ls in zip((1, 5, 100, 1000, 10000), ("-", "--", "-.", ":", (0, (1, 3)))):
    axs[0].plot(xx, lognormal_cdf(xx, -t / 2, np.sqrt(t)), color="k", ls=ls, lw=1.1, label=f"$t={t}$")
axs[0].set_xlabel(r"$x$"); axs[0].set_title(r"$\Pr(\xi_t\leq x)$", fontsize=9); axs[0].legend(fontsize=7, frameon=False, loc="lower right")
axs[0].set_ylim(0, 1.03)
lx = np.linspace(-30, 1.5, 1000)
for t, ls in zip((1, 5, 20, 50), ("-", "--", "-.", ":")):
    axs[1].plot(lx, ndtr((lx * LN10 + t / 2) / np.sqrt(t)), color="k", ls=ls, lw=1.1, label=f"$t={t}$")
axs[1].set_xlabel(r"$\log_{10}x$"); axs[1].set_title(r"$\Pr(\xi_t\leq x)$ against $\log_{10}x$", fontsize=9); axs[1].legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(FIG + "ch14_20.pdf"); plt.close()

# =====================================================================================================
# 14.21-14.23  Calibrations using the Mehra-Prescott moments of exercise 14.1
# =====================================================================================================
sg = np.sqrt(s2g)
pr_need = np.sqrt(np.log(1 + mpr14 ** 2))                 # sigma_c - w that delivers the minimal market price of risk
w21 = sg - pr_need
rho21 = -np.log(vstar14) - mg + s2g / 2 - sg * w21
print(f"14.21 annual MP data: min market price of risk {mpr14:.4f} at E(m)={vstar14:.4f}: need sigma_c - w = {pr_need:.4f},"
      f" i.e. w = {w21:.4f} (mean distortion sigma_c*w = {sg*w21:.5f} vs mu = {mg:.5f}); rho = {rho21:.4f} (beta = {np.exp(-rho21):.4f});"
      f" Tallarini gamma = 1 - w/sigma_c = {1 - w21/sg:.2f}; entropy w^2/2 = {w21**2/2:.4f}")
# detection error probability for N(0,1) vs N(w,1) with T = 90 annual observations: Phi(-sqrt(T)|w|/2); check by simulation
Tobs = 90; det = ndtr(-np.sqrt(Tobs) * abs(w21) / 2)
rng = np.random.default_rng(1421); ea = rng.standard_normal((200000, Tobs)); eb = w21 + rng.standard_normal((200000, Tobs))
llr = lambda e: np.sum(w21 * e - w21 ** 2 / 2, 1)          # log likelihood ratio of model B (mean w) to model A (mean 0)
det_mc = .5 * (np.mean(llr(ea) > 0) + np.mean(llr(eb) < 0))
print(f"14.21 detection error probability with T={Tobs}: {det:.4f} (simulated {det_mc:.4f})")
# 14.22: costs of fluctuations with the chapter's quarterly estimates (Table 14.6.1)
b22, mu22, sc22 = .995, .004952, .005050
w22 = sc22 * (1 - 50)
risk = b22 * sc22 ** 2 / (2 * (1 - b22)); amb_S = -b22 * sc22 * w22 / (1 - b22)
print(f"14.22 beta={b22}, mu={mu22}, sigma_c={sc22}, w = (1-50) sigma_c = {w22:.4f}: cost of risk {100*risk:.3f}%;"
      f" payment to become type P {100*amb_S:.2f}% (log points); with entropy penalty {50*amb_S:.2f}%; risk + ambiguity (multiplier) {100*(risk+amb_S/2):.2f}%"
      f" = Tallarini {100*50*risk:.2f}%")
# 14.23: market price of risk sqrt(exp(gamma^2 sigma^2)-1)
for gam in (2, 10, 50):
    print(f"14.23 gamma={gam}: market price of risk quarterly (sigma_x={sc22}) {np.sqrt(np.expm1((gam*sc22)**2)):.4f};"
          f" annual (sigma_x={sg:.4f}) {np.sqrt(np.expm1((gam*sg)**2)):.4f}")
