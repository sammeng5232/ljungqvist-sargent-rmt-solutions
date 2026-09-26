"""Chapter 29 (Equilibrium Search, Matching, and Lotteries): computations for exercises 29.1, 29.2, 29.4-29.6 and 29.8.

29.1  island economy: existence condition (1) and the labor allocation (2) for f(n) = n^.5.
29.2  Gomes-Greenwood-Rebelo: reservation wages in booms and recessions, simulated unemployment (cf. the book's Figure 29.1).
29.4  Mortensen-Pissarides with skill-specific markets: aggregate unemployment under mean-preserving spreads of skills.
29.5  Marimon-Zilibotti match-specific productivity: unemployment under mean-preserving spreads of G(p).
29.6  Ljungqvist-Sargent turbulence: unemployment as a function of the skill loss h1 for several replacement ratios.
29.8  matching model: transition after a permanent productivity increase.
All models use the book's timing of section 29.3 (production, then separation at the end of the period); periods are months.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
import scipy.sparse as sp
from rmtlib import FIG

np.set_printoptions(precision=4, suppress=True, linewidth=130)

# ================================================================== 29.1 island economy
print("=== 29.1  f(n) = n^.5, theta_L = 1, theta_H = 2, beta = .95, xhat = 1")
f1 = lambda n: .5 * n ** -.5
thL, thH, b1, xhat = 1., 2., .95, 1.
for pi in (.9, .7, .6):
    cond = b1 * (2 * pi - 1) * thH > thL
    if cond:
        x2 = brentq(lambda x: (thL + b1 * (1 - pi) * thH) * f1(2 * xhat - x) - b1 * pi * thH * f1(x), xhat, 2 * xhat - 1e-12)
        # value functions from the equilibrium conditions: beta(1-beta)vu = beta pi thH f'(x2)
        vu = pi * thH * f1(x2) / (1 - b1)
        vH2 = vu * (1 - (1 - pi) * b1) / pi
        chk = thH * f1(x2) + b1 * (pi * vH2 + (1 - pi) * b1 * vu) - vH2
        print(f"  pi = {pi}: beta(2pi-1)thH = {b1 * (2 * pi - 1) * thH:.3f} > thL: x1 = {2 * xhat - x2:.6f}, x2 = {x2:.6f};"
              f" vu = {vu:.4f}, Bellman residual at (H,x2) {chk:.1e}")
    else:
        print(f"  pi = {pi}: beta(2pi-1)thH = {b1 * (2 * pi - 1) * thH:.3f} <= thL: no stationary equilibrium with labor movements")

# ================================================================== 29.2 GGR
print("\n=== 29.2  GGR: F uniform on [0,1], b = .5, z = .05, beta = .99, mu = .01")
bb, zz, be, mu = .5, .05, .99, .01
bt = be * (1 - mu)
wg = (np.arange(400) + .5) / 400
# no business cycle: reservation wage wbar - b = bt/(1-bt) int_wbar (w - wbar) dF
wbar = brentq(lambda x: x - bb - bt / (1 - bt) * (1 - x) ** 2 / 2, 0, 1)
print(f"  no cycle: reservation wage {wbar:.6f}; steady-state unemployment mu/(1 - (1-mu)F(wbar)) = {mu / (1 - (1 - mu) * wbar):.6f}")
# booms and recessions: V_s(w) = max{w + z 1[B] + bt E V(w), Q}, Q = b + bt E_s' E_w' V_s'(w')
V = np.zeros((2, len(wg)))                                         # row 0 = boom, row 1 = recession
for it in range(20000):
    EV = V.mean(0)                                                 # E_s' V_s'(w) (states i.i.d., prob .5)
    Q = bb + bt * EV.mean()
    Vn = np.vstack([np.maximum(wg + zz + bt * EV, Q), np.maximum(wg + bt * EV, Q)])
    if np.max(np.abs(Vn - V)) < 1e-12:
        break
    V = Vn
EV = V.mean(0); Q = bb + bt * EV.mean()
wB = brentq(lambda x: x + zz + bt * np.interp(x, wg, EV) - Q, 0, 1)
wR = brentq(lambda x: x + bt * np.interp(x, wg, EV) - Q, 0, 1)
print(f"  booms/recessions: w_B = {wB:.6f}, w_R = {wR:.6f} (difference {wR - wB:.6f} < z = {zz})")
rng = np.random.default_rng(29)
T = 200; states = rng.integers(0, 2, T)                           # 0 = boom, 1 = recession
U, G = [mu / (1 - (1 - mu) * wR)], [0.]
for t in range(1, T):
    if states[t] == 0:
        U.append(mu + (1 - mu) * wB * U[-1]); G.append((1 - mu) * (G[-1] + (wR - wB) * U[-2]))
    else:
        U.append(mu + (1 - mu) * (wR * U[-1] + G[-1])); G.append(0.)
U = np.array(U); G = np.array(G)
print(f"  simulated unemployment (t = 100..199): mean {U[100:].mean():.4f}, min {U[100:].min():.4f}, max {U[100:].max():.4f}")

# ================================================================== 29.4 skill markets
print("\n=== 29.4  MP with skill markets: monthly r = .004, s = .02, alpha = phi = .5, A = .35, c = .65 (vacancy cost c h)")
r, s, al, A, c = .004, .02, .5, .35, .65
phi = al
q = lambda th: A * th ** (-al)
def theta_mp(y, z, cc):
    """Solve y - z = (r + s + phi theta q)/((1-phi) q) cc; returns 0 if the market shuts down."""
    if y <= z:
        return 0.
    g = lambda th: (r + s + phi * th * q(th)) / ((1 - phi) * q(th)) * cc - (y - z)
    return brentq(g, 1e-12, 1e6)
urate = lambda th: 1. if th == 0 else s / (s + th * q(th))
print(f"  h = 1, b = .4: theta = {theta_mp(1, .4, c):.4f}, u = {urate(theta_mp(1, .4, c)):.4f}")
spreads = np.linspace(0, .6, 31); bs4 = (0., .2, .4, .6)
res4 = {}
for bb4 in bs4:
    out = []
    for D in spreads:
        hs = np.linspace(1 - D, 1 + D, 201) if D > 0 else np.array([1.])
        out.append(np.mean([urate(theta_mp(h, bb4, c * h)) for h in hs]))
    res4[bb4] = np.array(out)
    print(f"  b = {bb4}: aggregate u for spread D = 0, .2, .4, .6: {res4[bb4][[0, 10, 20, 30]]}")
print(f"  benefits proportional to skill, b_i = .4 h_i: u = {urate(theta_mp(1, .4, c)):.6f} in every market (theta independent of h)")

# ================================================================== 29.5 match-specific productivity
print("\n=== 29.5  Marimon-Zilibotti: log p ~ N(-sig^2/2, sig^2), h = 1, same matching parameters")
def mz(bb5, sig):
    mu5 = -sig ** 2 / 2
    def I(k):                                                     # E[(p - k)^+]
        if k <= 0:
            return 1 - k
        d1 = (mu5 + sig ** 2 - np.log(k)) / sig
        return np.exp(mu5 + sig ** 2 / 2) * norm.cdf(d1) - k * norm.cdf(d1 - sig)
    th = lambda ps: (A * (1 - phi) * I(ps) / (c * (r + s))) ** (1 / al)
    ps = brentq(lambda ps: ps - bb5 - phi * c * th(ps) / (1 - phi), -5, 20)
    t = th(ps); Gp = norm.cdf((np.log(ps) - mu5) / sig) if ps > 0 else 0.
    return ps, t, s / (s + t * q(t) * (1 - Gp)), Gp
sigs = np.linspace(.05, .8, 31); bs5 = (0., .4, .6, .8)
res5 = {}
for bb5 in bs5:
    res5[bb5] = np.array([mz(bb5, sg)[2] for sg in sigs])
    ps, t, u5, Gp = mz(bb5, .3)
    print(f"  b = {bb5}: u for sigma = .05, .3, .55, .8: {res5[bb5][[0, 10, 20, 30]]};  at sigma = .3: p* = {ps:.4f}, theta = {t:.4f}, reject prob {Gp:.4f}")

# ================================================================== 29.6 turbulence
print("\n=== 29.6  LS turbulence: monthly beta = .995, death .002, s = .02, pi_e = .99, pi_u = .2, w ~ U[0,1], h2 = 1")
def ls98(h1, gam, N=100, be6=.995, dth=.002, s6=.02, pie=.99, piu=.2, h2=1.):
    w = (np.arange(N) + .5) / N; g = np.full(N, 1 / N); hh = np.array([h1, h2])
    bt6 = be6 * (1 - dth)
    Bl = np.r_[0., gam * w * h1, gam * w * h2]                   # benefit levels: 0, then indexed by (j, k)
    bidx = lambda j, k: 1 + k * N + j
    W = np.zeros((N, 2)); Uv = np.zeros((len(Bl), 2))
    jj = np.arange(N)
    for it in range(100000):
        # unemployed: U[b,k] = b + bt E max(W[j,k], U[b,k])
        Un = np.empty_like(Uv)
        for k in range(2):
            Un[:, k] = Bl + bt6 * (g[None, :] * np.maximum(W[None, :, k], Uv[:, k][:, None])).sum(1)
        Wn = np.empty_like(W)
        for k in range(2):
            lay = piu * Uv[bidx(jj, k), k] + (1 - piu) * Uv[bidx(jj, k), 0]
            cont = pie * W[:, k] + (1 - pie) * W[:, 1]
            Wn[:, k] = np.maximum(w * hh[k] + bt6 * ((1 - s6) * cont + s6 * lay), Uv[0, k])
        err = max(np.max(np.abs(Un - Uv)), np.max(np.abs(Wn - W)))
        Uv, W = Un, Wn
        if err < 1e-9:
            break
    work = np.array([[w[j] * hh[k] + bt6 * ((1 - s6) * (pie * W[j, k] + (1 - pie) * W[j, 1])
                      + s6 * (piu * Uv[bidx(j, k), k] + (1 - piu) * Uv[bidx(j, k), 0])) >= Uv[0, k] for k in range(2)] for j in range(N)])
    # Markov chain on (employed j,k) and (unemployed b,k): index employed e(j,k) = k*N + j, unemployed 2N + b*2 + k
    nE, nU = 2 * N, 2 * len(Bl); rows, cols, vals = [], [], []
    def add(a, b_, v):
        rows.append(a); cols.append(b_); vals.append(v)
    uidx = lambda b_, k: nE + 2 * b_ + k
    newborn = uidx(0, 0)
    for k in range(2):
        for j in range(N):
            src = k * N + j
            add(src, newborn, dth)
            for k2, pk in ((k, piu), (0, 1 - piu)):                # layoff
                add(src, uidx(bidx(j, k), k2), (1 - dth) * s6 * pk)
            for k2, pk in ((k, pie), (1, 1 - pie)):                # job continues, maybe upgrade, maybe quit
                dst = k2 * N + j if work[j, k2] else uidx(0, k2)
                add(src, dst, (1 - dth) * (1 - s6) * pk)
    for bi in range(len(Bl)):
        for k in range(2):
            src = uidx(bi, k); add(src, newborn, dth)
            acc = W[:, k] >= Uv[bi, k]
            pa = g[acc].sum()
            for j in np.where(acc)[0]:
                add(src, k * N + j, (1 - dth) * g[j])
            add(src, src, (1 - dth) * (1 - pa))
    Tm = sp.csr_matrix((vals, (rows, cols)), shape=(nE + nU, nE + nU))
    lam = np.full(nE + nU, 1 / (nE + nU)); TT = Tm.T.tocsr()
    for it2 in range(200000):
        new = TT @ lam
        if np.max(np.abs(new - lam)) < 1e-13:
            break
        lam = new
    lam = new / new.sum()
    # reservation wage of a worker laid off from (w = .75, h2) who lost skills
    jm = int(.75 * N); rw = w[np.argmax(W[:, 0] >= Uv[bidx(jm, 1), 0])] if (W[:, 0] >= Uv[bidx(jm, 1), 0]).any() else np.inf
    return lam[nE:].sum(), rw
h1s = np.array([1., .9, .8, .7, .6, .5, .4]); gams = (0., .3, .5, .7)
res6 = {}
for gm in gams:
    out = [ls98(h1, gm) for h1 in h1s]
    res6[gm] = np.array([o[0] for o in out])
    print(f"  gamma = {gm}: u for h1 = {h1s}: {res6[gm]};  reservation wage after skill loss from (w=.75,h2): {[round(o[1], 3) for o in out]}")

# ================================================================== 29.8 transition after a productivity increase
print("\n=== 29.8  y: 1 -> 1.05, z = .4, matching as in 29.4")
th0, th1 = theta_mp(1., .4, c), theta_mp(1.05, .4, c)
f0, f1_ = th0 * q(th0), th1 * q(th1)
u0 = s / (s + f0); ut = [u0]
for t in range(40):
    ut.append(ut[-1] + s * (1 - ut[-1]) - f1_ * ut[-1])
ut = np.array(ut)
print(f"  theta {th0:.4f} -> {th1:.4f}; job-finding {f0:.4f} -> {f1_:.4f}; u: {u0:.4f} -> {s / (s + f1_):.4f}")
print(f"  JC_t = f' u_t: {np.round(f1_ * ut[:6], 5)} ...; JD_t = s(1-u_t): {np.round(s * (1 - ut[:6]), 5)} ...; JC_-1 = JD_-1 = {s * (1 - u0):.5f}")

# ================================================================== figures
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for bb4, ls in zip(bs4, ("-", "--", "-.", ":")):
    ax[0].plot(spreads, 100 * res4[bb4], "k" + ls, label=f"b = {bb4}")
ax[0].set_xlabel("spread of skills (h uniform on [1-D, 1+D])"); ax[0].set_ylabel("aggregate unemployment, %")
ax[0].set_title("29.4: skill-biased change", fontsize=9); ax[0].legend(fontsize=7)
for bb5, ls in zip(bs5, ("-", "--", "-.", ":")):
    ax[1].plot(sigs, 100 * res5[bb5], "k" + ls, label=f"b = {bb5}")
ax[1].set_xlabel(r"dispersion $\sigma$ of log match productivity"); ax[1].set_title("29.5: dispersion of match values", fontsize=9)
ax[1].legend(fontsize=7)
for gm, ls in zip(gams, ("-", "--", "-.", ":")):
    ax[2].plot(h1s, 100 * res6[gm], "k" + ls, marker=".", label=rf"$\gamma$ = {gm}")
ax[2].invert_xaxis(); ax[2].set_xlabel(r"human capital after a skill loss $h_1$ ($h_2=1$)")
ax[2].set_title("29.6: turbulence and benefits", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch29_04.pdf"); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 3.2))
tt = np.arange(50)
ax.plot(tt, U[150:], "k", lw=1)
for t in tt:
    if states[150 + t] == 1:
        ax.axvspan(t - .5, t + .5, color="0.88", lw=0)
ax.set_xlabel("time (shaded: recessions)"); ax.set_ylabel("unemployment rate")
ax.set_title("29.2: simulated unemployment with booms and recessions", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch29_02.pdf"); plt.close()
