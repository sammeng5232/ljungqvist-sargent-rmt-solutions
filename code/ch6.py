"""Chapter 6 (Search and Unemployment): numerical parts of the exercises."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=5, suppress=True, linewidth=140)

# ---------------------------------------------------------------- 6.19 value iteration vs Howard (F uniform on [0,1])
beta, c = .95, .5
S = lambda w: (1 - w) ** 2 / 2                     # int_w^1 (w'-w) dF
Emax = lambda w: w + S(w)                          # E max{w', w}
wstar = brentq(lambda w: w - c - beta / (1 - beta) * S(w), 0, 1)
T_vfi = lambda w: (1 - beta) * c + beta * Emax(w)
H_how = lambda w: ((1 - beta) * c + beta * (1 - w ** 2) / 2) / (1 - beta * w)
K = -beta * 1.0 / (2 * (1 - beta * wstar))         # f = 1 on [0,1]
print(f"6.19 w* = {wstar:.10f}; predicted Howard constant K = {K:.5f}; VFI local rate beta*F(w*) = {beta*wstar:.5f}")
w_v = w_h = 0.0
print("     n   VFI error        Howard error     (w_{n+1}-w*)/(w_n-w*)^2")
for n in range(1, 9):
    w_v_new, w_h_new = T_vfi(w_v), H_how(w_h)
    ratio = (w_h_new - wstar) / (w_h - wstar) ** 2 if abs(w_h - wstar) > 1e-7 else np.nan
    print(f"    {n:2d}  {w_v_new - wstar: .3e}     {w_h_new - wstar: .3e}     {ratio: .4f}")
    w_v, w_h = w_v_new, w_h_new

# ---------------------------------------------------------------- 6.20 mean-preserving spread with atoms at 0 and B
def resw(alpha, c, beta=.95, B=1.0):
    # S_alpha(w) = int_w^B (w'-w) dF_alpha: uniform density 1/B on (aB, B-aB) plus atom alpha at B
    def S_a(w):
        lo, hi = alpha * B, B - alpha * B
        s = alpha * (B - w)                                         # atom at B
        a, b = max(w, lo), hi
        if b > a:
            s += ((b - w) ** 2 - (a - w) ** 2) / (2 * B)
        return s
    wr = brentq(lambda w: w - c - beta / (1 - beta) * S_a(w), 0, B)
    Fw = alpha if wr <= alpha * B else (wr / B if wr < B - alpha * B else 1 - alpha)
    return wr, 1 - Fw
print("6.20  c     wbar(alpha=0)  accept(0)   wbar(alpha=.3)  accept(.3)")
for cc in (.05, .2, .5, .8):
    w0, p0 = resw(0.0, cc); w3, p3 = resw(.3, cc)
    print(f"     {cc:.2f}   {w0:.4f}        {p0:.4f}      {w3:.4f}          {p3:.4f}")

# ---------------------------------------------------------------- 6.16 seasonal firing: example
def seasons2(beta=.95, c=.5, pi=.2, n=201):
    wg = np.linspace(0, 1, n); fw = np.full(n, 1 / n)             # F uniform on [0,1] (discretised)
    We = wg / (1 - beta); Wo = wg / (1 - beta); Ue = np.zeros(n); Uo = np.zeros(n)
    for it in range(20000):
        Vuo = np.maximum(wg + beta * We, c + beta * fw @ Ue)       # unemployed at an odd date with offer w
        Vue = np.maximum(We, c + beta * fw @ Uo)                   # unemployed at an even date with offer w
        Won = (1 - pi) * (wg + beta * We) + pi * (fw @ Vuo)        # employed at w, start of an odd date (firing risk)
        Wen = wg + beta * Won                                     # employed at w at an even date (no risk)
        if max(np.max(np.abs(Won - Wo)), np.max(np.abs(Wen - We)), np.max(np.abs(Vuo - Uo)), np.max(np.abs(Vue - Ue))) < 1e-11: break
        Wo, We, Uo, Ue = Won, Wen, Vuo, Vue
    wbar_odd = wg[np.argmax(wg + beta * We >= c + beta * fw @ Ue)]
    wbar_even = wg[np.argmax(We >= c + beta * fw @ Uo)]
    return wbar_odd, wbar_even
for pi in (.05, .2, .5):
    print(f"6.16 pi={pi}: reservation wage at odd dates {seasons2(pi=pi)[0]:.4f}, at even dates {seasons2(pi=pi)[1]:.4f}")

# ---------------------------------------------------------------- 6.25 Markov wages
P = np.array([[.8, .2, 0, 0, 0], [.18, .8, .02, 0, 0], [.25, .25, 0, .25, .25], [0, 0, .02, .8, .18], [0, 0, 0, .2, .8]])
w = np.arange(1, 6.0); c = 1.0
for beta in (.95, .99):
    v = np.zeros(5)
    for it in range(100000):
        vn = np.maximum(w / (1 - beta), c + beta * P @ v)
        if np.max(np.abs(vn - v)) < 1e-12: break
        v = vn
    acc = w / (1 - beta) >= c + beta * P @ v
    print(f"6.25 beta={beta}: v = {v}; reject value c+beta*Pv = {c + beta * P @ v}; accept? {acc} ({it} its)")

# ---------------------------------------------------------------- 6.26 Neal (1999) model and its Markov chain
beta = .95; n = 50
th = np.linspace(0, 5, n); ep = np.linspace(0, 5, n); f = np.full(n, 1 / n); g = np.full(n, 1 / n)
v = np.zeros((n, n))
for it in range(5000):
    stay = th[:, None] + ep[None, :] + beta * v
    newjob = (th + (ep[None, :] + beta * v) @ g)[:, None] * np.ones((1, n))          # C(theta)
    Q = f @ (th[:, None] + ep[None, :] + beta * v) @ g
    vn = np.maximum(np.maximum(stay, newjob), Q)
    if np.max(np.abs(vn - v)) < 1e-10: break
    v = vn
stay = th[:, None] + ep[None, :] + beta * v
Cth = th + (ep[None, :] + beta * v) @ g
Q = f @ (th[:, None] + ep[None, :] + beta * v) @ g
choice = np.where(stay >= np.maximum(Cth[:, None], Q) - 1e-12, 3, np.where(Cth[:, None] >= Q, 2, 1))   # 1 new life, 2 new job, 3 stay
P12 = f @ (choice == 2) @ g; P13 = f @ (choice == 3) @ g
# from state 2 (theta >= theta_bar) the next state is 3 iff the new epsilon' is accepted
theta_bar = th[np.argmax(Cth >= Q)]
rows2 = np.where((choice == 2).any(axis=1))[0]
P23_by_theta = [(g @ (choice[i] == 3)) for i in rows2]
print(f"6.26 Neal: {it} its; theta_bar = {theta_bar:.4f}; Q = {Q:.4f}; P12 = {P12:.4f}, P13 = {P13:.4f}, P11 = {1-P12-P13:.4f}")
print("     P23 across thetas in the new-job region:", np.round(np.unique(np.round(P23_by_theta, 10)), 4),
      " reservation epsilon there:", np.unique([ep[np.argmax(choice[i] == 3)] for i in rows2 if (choice[i] == 3).any()]))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(choice.T, origin="lower", extent=[0, 5, 0, 5], cmap="Greys_r", vmin=0.5, vmax=3.5, aspect="auto")
ax.set_xlabel(r"career $\theta$"); ax.set_ylabel(r"job $\epsilon$"); ax.set_title("Neal's model: new life (black), new job (grey), stay (white)", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "ch6_26.pdf"); plt.close()

# ---------------------------------------------------------------- 6.27 Neal with unemployment
for cu in (0.0, 2.0):
    v = np.zeros((n, n))
    for it in range(20000):
        stay = (th[:, None] + ep[None, :]) + beta * v
        Cth = cu + beta * v @ g
        Q = cu + beta * f @ v @ g
        vn = np.maximum(np.maximum(stay, Cth[:, None]), Q)
        if np.max(np.abs(vn - v)) < 1e-10: break
        v = vn
    stay = (th[:, None] + ep[None, :]) + beta * v; Cth = cu + beta * v @ g; Q = cu + beta * f @ v @ g
    choice = np.where(stay >= np.maximum(Cth[:, None], Q) - 1e-12, 3, np.where(Cth[:, None] >= Q, 2, 1))
    keep = np.where(Cth >= Q)[0]
    eb = [ep[np.argmax(choice[i] == 3)] if (choice[i] == 3).any() else np.nan for i in keep]
    print(f"6.27 c={cu}: theta_bar={th[keep[0]] if len(keep) else np.nan:.3f}; reservation job eps_bar(theta) for retained careers (theta={th[keep[0]]:.2f}..5):",
          np.round(np.array(eb)[::7], 3))
    # analytic check: eps_bar(theta) solves eps-(c-theta) = beta/(1-beta) sum_{eps'>=eps}(eps'-eps) g  (grid version)
    def epsbar(theta):
        return brentq(lambda e: e - (cu - theta) - beta / (1 - beta) * np.sum(g * np.maximum(ep - e, 0)), -10, 10)
    print("     continuous-threshold formula at theta=", th[keep[::7]], ":", np.round([epsbar(t) for t in th[keep[::7]]], 3))

# ---------------------------------------------------------------- 6.18 Jovanovic (1979b): investment in job-specific capital and on-the-job search
beta, A, alpha, delta = .95, .8, .5, .1
mu = np.linspace(1, 20, 20); fmu = np.full(20, .05)                # offers mu' uniform on {1,...,20}
xg = np.linspace(.5, 40, 200)                                     # plotted only up to 30 to avoid edge effects
phis = np.linspace(0, .95, 58); ss = np.linspace(0, .95, 58)
PH, SS = np.meshgrid(phis, ss, indexing="ij"); feas = (PH + SS <= .95)
X = xg[:, None, None]
XP = X + A * (X * PH[None]) ** alpha - delta * X                  # x' = x + g(x phi) - delta x
flow = X * (1 - PH[None] - SS[None])
V = xg / (1 - beta)
for it in range(5000):
    Vxp = np.interp(XP, xg, V)
    Vmu = np.interp(mu, xg, V)
    withoffer = np.maximum(Vmu[None, None, None, :], Vxp[..., None]) @ fmu     # an offer mu' arrives: keep the better job
    val = flow + beta * (np.sqrt(SS)[None] * withoffer + (1 - np.sqrt(SS)[None]) * Vxp)
    val = np.where(feas[None], val, -np.inf).reshape(len(xg), -1)
    j = np.argmax(val, axis=1)
    Vn = val[np.arange(len(xg)), j]
    if np.max(np.abs(Vn - V)) < 1e-8: break
    V = Vn
pol_phi = PH.ravel()[j]; pol_s = SS.ravel()[j]
xss = xg[np.argmin(np.abs(xg + A * (xg * pol_phi) ** alpha - delta * xg - xg))]
print(f"6.18 VFI converged in {it} its; phi(x) at x=1,4,8,12: {np.interp([1,4,8,12], xg, pol_phi)}; s(x): {np.interp([1,4,8,12], xg, pol_s)}")
print("     accept mu' iff V(mu') >= V(x'), i.e. mu' >= x'  (V increasing: min diff", np.min(np.diff(V)), ")")
growth = xg + A * (xg * pol_phi) ** alpha - delta * xg - xg
sign_change = [(xg[k], xg[k + 1]) for k in range(len(xg) - 1) if np.sign(growth[k]) != np.sign(growth[k + 1])]
print("     x' - x changes sign between:", [(round(a, 2), round(b, 2)) for a, b in sign_change])
print("     phi, s at x=2,5,10,15,20,25:", np.interp([2,5,10,15,20,25], xg, pol_phi), np.interp([2,5,10,15,20,25], xg, pol_s))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
m16 = xg <= 30
ax[0].plot(xg[m16], pol_phi[m16], "k-"); ax[0].set_title(r"investment share $\phi(x)$", fontsize=9)
ax[1].plot(xg[m16], pol_s[m16], "k-"); ax[1].set_title(r"search share $s(x)$", fontsize=9)
for a_ in ax: a_.set_xlabel("job-specific capital $x$")
plt.tight_layout(); plt.savefig(FIG + "ch6_18.pdf"); plt.close()
