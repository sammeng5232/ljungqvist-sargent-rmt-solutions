"""Chapter 5 (Linear Quadratic Dynamic Programming): numerical parts of the exercises."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rmtlib import sc, FIG, doublej, olrp, lq_howard

np.set_printoptions(precision=6, suppress=True, linewidth=150)

# ---------------------------------------------------------------- 5.1 / 5.2: check the cross-product Riccati equation
rng = np.random.default_rng(5)
n, k = 3, 2
A = rng.normal(size=(n, n)) * .5; B = rng.normal(size=(n, k))
M = rng.normal(size=(n, n)); R = M @ M.T + np.eye(n); Q = np.eye(k) * 2; H = rng.normal(size=(k, n)) * .3
bt = .95
F, P, it = olrp(bt, A, B, Q, R, H)
resid = P - (R + bt * A.T @ P @ A - (bt * A.T @ P @ B + H.T) @ np.linalg.solve(Q + bt * B.T @ P @ B, bt * B.T @ P @ A + H))
# the same problem after removing the cross product: u* = u + Q^{-1}H x
Rt = R - H.T @ np.linalg.solve(Q, H); At = A - B @ np.linalg.solve(Q, H)
Ft, Pt, _ = olrp(bt, At, B, Q, Rt)
print("5.1 Riccati residual", np.max(np.abs(resid)), "iterations", it, "; transformed problem gives same P:", np.max(np.abs(P - Pt)),
      "and F = Ft + Q^-1 H:", np.max(np.abs(F - (Ft + np.linalg.solve(Q, H)))))

# ---------------------------------------------------------------- 5.3
bt, r, bliss, gam, r1, r2 = .95, 1 / .95 - 1, 30., 1., 1.2, -.3
A3 = np.array([[1, 0, 0, 0], [0, r1, r2, 0], [0, 1, 0, 0], [0, 0, 0, 1.]]); B3 = np.array([[1.], [0], [0], [0]])
h = np.array([r, 1, 0, -bliss])
R3 = np.outer(h, h); Q3 = np.array([[1 + gam]]); H3 = -h.reshape(1, 4)
F3, P3, it3 = olrp(bt, A3, B3, Q3, R3, H3)
F3h, _, _ = lq_howard(A3, B3, R3, Q3, bt, np.zeros((1, 4)), H=H3)
print("5.3 Howard cross-check max|F-F_howard| =", np.max(np.abs(F3 - F3h)))
cons = np.array([r, 1, 0, 0]) + F3.ravel()
print("5.3 F (i_t = -F x_t, x=[a,y,y_-1,1]):", F3.ravel(), " iterations", it3)
print("    investment rule i_t =", -F3.ravel(), "\n    consumption rule c_t =", cons, "  eig(A-BF):", np.sort(np.abs(np.linalg.eigvals(A3 - B3 @ F3))))

# ---------------------------------------------------------------- 5.4
def durable_problem(lam, pi, dl, th):
    A = np.array([[1, 0, 0, 0, 0], [0, r1, r2, 0, 0], [0, 1, 0, 0, 0], [th * r, th, 0, dl, 0], [0, 0, 0, 0, 1.]])
    B = np.array([[1.], [0], [0], [-th], [0]])
    ht = np.array([pi * r, pi, 0, lam, -bliss])
    return A, B, np.outer(ht, ht), np.array([[pi ** 2 + gam]]), -pi * ht.reshape(1, 5)
for tag, pars in (("b durables", (1, .05, .95, 1)), ("c habits", (-1, 1, .95, 1))):
    A4, B4, R4, Q4, H4 = durable_problem(*pars)
    F4, P4, it4 = olrp(bt, A4, B4, Q4, R4, H4)
    F4h, _, _ = lq_howard(A4, B4, R4, Q4, bt, np.zeros((1, 5)), H=H4)     # i = 0 is a stabilising start
    print(f"5.4{tag}: Howard cross-check max|F-F_howard| = {np.max(np.abs(F4 - F4h)):.2e}")
    cons4 = np.array([r, 1, 0, 0, 0]) + F4.ravel()
    ev = np.linalg.eigvals(A4 - B4 @ F4)
    print(f"5.4{tag}: i_t = {-F4.ravel()} ; c_t = {cons4}  (x=[a,y,y_-1,h,1]) its {it4}; |eig(A-BF)| {np.sort(np.abs(ev))}")

# ---------------------------------------------------------------- 5.5
bt, r1, r2, g = .95, 1.2, -.4, .5
A5 = np.array([[r1, r2, g], [1, 0, 0], [0, 0, 0.]]); G5 = np.array([[1., 0, 0]])
coef = G5 @ np.linalg.inv(np.eye(3) - bt * A5)
print("5.5 E_t sum beta^j y_t+j = ", coef.ravel(), "* [y_t, y_t-1, w_t]")
# direct check by simulation of forecasts
x = np.array([1.3, -.7, .4]); s = 0; xx = x.copy()
for j in range(3000):
    s += bt ** j * sc(G5 @ xx); xx = A5 @ xx
print("    check at x=[1.3,-.7,.4]:", s, sc(coef @ x))

# ---------------------------------------------------------------- 5.7 dynamic Laffer curves
def laffer(g1, g2, g, Mm1=100.):
    lam = np.sort(np.roots([g2, -(g1 + g2 - g), g1]).real)          # small, large root
    p0min = Mm1 / (g1 - g - g2 * lam[0])
    return lam, p0min
def path(p0, g1, g2, g, T=60, Mm1=100.):
    p = np.empty(T + 1); p[0] = p0; p[1] = ((g1 - g) * p0 - Mm1) / g2
    for t in range(1, T):
        p[t + 1] = ((g1 + g2 - g) * p[t] - g1 * p[t - 1]) / g2
    M = g1 * p[:-1] - g2 * p[1:]
    return p, M
for g in (.05, .075):
    lam, p0min = laffer(100, 50, g)
    print(f"5.7 g={g}: stationary gross inflation rates {lam}, lowest p0 = {p0min:.6f}")
lam, p0min = laffer(100, 50, .05)
p, M = path(p0min, 100, 50, .05)
print("    at p0min: inflation", (p[1:6] / p[:5]), " money growth", M[1:6] / M[:5])
for p0 in (p0min * (1 - 1e-6), p0min * 1.001, 3., 10.):
    p, M = path(p0, 100, 50, .05, T=80)
    pos = np.all(p > 0) and np.all(M > 0)
    print(f"    p0={p0:.6f}: all positive over 80 periods: {pos}; inflation at t=1,10,40,79: {p[1]/p[0]:.5f} {p[10]/p[9]:.5f} {p[40]/p[39]:.5f} {p[79]/p[78]:.5f}; money growth t=79: {M[79]/M[78]:.5f}")
# Laffer curve of stationary seigniorage
pis = np.linspace(1, 2, 400); seig = (100 - 50 * pis) * (1 - 1 / pis)
print("    Laffer curve peak at pi=", np.sqrt(2), " max seigniorage", (100 - 50 * np.sqrt(2)) * (1 - 1 / np.sqrt(2)))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for p0, sty in ((p0min, "k-"), (p0min * 1.0001, "k--"), (p0min * 1.01, "k-."), (5.0, "k:")):
    p, M = path(p0, 100, 50, .05, T=40)
    ax[0].plot(np.arange(1, 41), p[1:] / p[:-1], sty, label=f"$p_0={p0:.4f}$")
ax[0].set_xlabel("t"); ax[0].set_ylabel("gross inflation $p_t/p_{t-1}$"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(pis, seig, "k-"); ax[1].axhline(.05, color="grey", lw=.7); ax[1].axhline(.075, color="grey", lw=.7, ls="--")
ax[1].set_xlabel("stationary gross inflation $\\pi$"); ax[1].set_ylabel("seigniorage $(\\gamma_1-\\gamma_2\\pi)(1-1/\\pi)$")
ax[1].set_ylim(0, 9)
plt.tight_layout(); plt.savefig(FIG + "ch5_07.pdf"); plt.close()
# zoom on the relevant region for g = .05, .075
for g in (.05, .075):
    lam, _ = laffer(100, 50, g)
    print(f"    g={g}: check seigniorage at roots:", [(100 - 50 * l) * (1 - 1 / l) for l in lam])

# ---------------------------------------------------------------- 5.8 illustration: tax smoothing with beta = q
bt = q = .95; rho, muT, c1, c2, w1, w2 = .8, 1.0, .1, .2, 1.0, 2.0
# x = [b, gP, gT, 1]; u = T ; b' = (T + b - gP - gT)/q
A8 = np.array([[1 / q, -1 / q, -1 / q, 0], [0, 1, 0, 0], [0, 0, rho, (1 - rho) * muT], [0, 0, 0, 1.]])
B8 = np.array([[1 / q], [0], [0], [0]])
Fnaive, _, _ = olrp(bt, A8, B8, np.array([[.5 * w2]]), np.zeros((4, 4)), np.array([[0, 0, 0, .5 * w1]]))
print("5.8 Riccati iteration from P=0 (ignores the no-Ponzi condition): T = -F x with -F =", -Fnaive.ravel())
F0 = -np.array([[-(1 - q), 1, 1, 0]])          # T = g - (1-q) b keeps assets constant: a stabilising start
F8, P8, it8 = lq_howard(A8, B8, np.zeros((4, 4)), np.array([[.5 * w2]]), bt, F0, H=np.array([[0, 0, 0, .5 * w1]]))
closed = np.array([-(1 - q), 1, (1 - q) / (1 - q * rho), muT * (1 - (1 - q) / (1 - q * rho))])
print("    Howard from a stabilising rule: T = -F x with -F =", -F8.ravel(), " closed form:", closed, " its", it8)
F8b, _, _ = lq_howard(A8, B8, np.zeros((4, 4)), np.array([[.5 * 5.0]]), bt, F0, H=np.array([[0, 0, 0, .5 * 3.0]]))
print("    same with w1=3, w2=5:", -F8b.ravel(), " eig(A-BF):", np.sort(np.abs(np.linalg.eigvals(A8 - B8 @ F8))))
# beta != q: taxes drift, and w1/w2 enters the constant
Fq, _, _ = lq_howard(A8 * 1, B8, np.zeros((4, 4)), np.array([[.5 * w2]]), .96, F0, H=np.array([[0, 0, 0, .5 * w1]]))
Fq2, _, _ = lq_howard(A8 * 1, B8, np.zeros((4, 4)), np.array([[.5 * 5.0]]), .96, F0, H=np.array([[0, 0, 0, .5 * 3.0]]))
print("    beta=.96 > q=.95:", -Fq.ravel(), "; with w1=3,w2=5:", -Fq2.ravel())

# ---------------------------------------------------------------- 5.11 / 5.12
bt, d, A0, A1 = .95, 2., 100., 1.
R11 = np.array([[0, A1 / 2, -A0 / 2], [A1 / 2, 0, 0], [-A0 / 2, 0, 0]])
A11 = np.array([[1, 0, 0], [0, .8, 200], [0, 0, 1.]]); B11 = np.array([[1.], [0], [0]])
F11, P11, it11 = olrp(bt, A11, B11, np.array([[d / 2]]), R11)
print("5.11 x=[y,Y,1], u=y'-y: F =", F11.ravel(), " its", it11, "\n     P =\n", P11)
print("     decision rule y' =", np.array([1, 0, 0]) - F11.ravel(), " eig(A-BF)", np.linalg.eigvals(A11 - B11 @ F11))
R12 = np.array([[0, A1 / 2, -.5, -A0 / 2], [A1 / 2, 0, 0, 0], [-.5, 0, 0, 0], [-A0 / 2, 0, 0, 0]])
A12 = np.array([[1, 0, 0, 0], [0, .8, 2, 200], [0, 0, .9, 0], [0, 0, 0, 1.]]); B12 = np.array([[1.], [0], [0], [0]])
C12 = np.array([[0], [0], [.05], [0]])
F12, P12, it12 = olrp(bt, A12, B12, np.array([[d / 2]]), R12)
d12 = bt / (1 - bt) * np.trace(P12 @ C12 @ C12.T)
print("5.12 x=[y,Y,u,1]: F =", F12.ravel(), " its", it12, " d =", d12, "\n     P =\n", P12)
print("     decision rule y' =", np.array([1, 0, 0, 0]) - F12.ravel())

# ---------------------------------------------------------------- 5.13 / 5.14 permanent income
def pim(lags, rho, eps, b=1000., bt=.95, sig=.05):
    """x = [a, y_t, ..., y_{t-lags+1}, 1], u = c - b"""
    Rg = 1 / bt; nY = lags; n = 2 + nY
    A = np.zeros((n, n)); A[0, 0] = Rg; A[0, 1] = 1; A[0, -1] = -b
    for j, rj in rho.items(): A[1, 1 + j] = rj
    A[1, -1] = 1 - sum(rho.values())
    for i in range(2, 1 + nY): A[i, i - 1] = 1
    A[-1, -1] = 1
    Bm = np.zeros((n, 1)); Bm[0, 0] = -1
    C = np.zeros((n, 1)); C[1, 0] = sig
    Rm = np.zeros((n, n)); Rm[0, 0] = .5 * eps
    F, P, it = olrp(bt, A, Bm, np.array([[.5]]), Rm, tol=1e-13, maxit=2000000)
    # stabilising solution via Howard, started from c - b = (R-1)a + y - b (assets stay constant)
    F0 = np.zeros((1, n)); F0[0, 0] = -(Rg - 1); F0[0, 1] = -1; F0[0, -1] = b
    Fh, Ph, ith = lq_howard(A, Bm, Rm, np.array([[.5]]), bt, F0)
    dcon = bt / (1 - bt) * np.trace(P @ C @ C.T); dh = bt / (1 - bt) * np.trace(Ph @ C @ C.T)
    return A, Bm, F, P, dcon, it, Fh, Ph, dh, ith
for name, lags, rho in (("5.13", 2, {0: 1.2, 1: -.4}), ("5.14", 4, {0: .55, 3: .3})):
    for eps in (1e-6, 0.0):
        A_, B_, F_, P_, d_, it_, Fh, Ph, dh, ith = pim(lags, rho, eps)
        ev = np.sort(np.abs(np.linalg.eigvals(A_ - B_ @ F_))); evh = np.sort(np.abs(np.linalg.eigvals(A_ - B_ @ Fh)))
        print(f"{name} eps={eps}: Riccati from P=0 ({it_} its): c_t - b = {(-F_).ravel()}; d={d_:.6f}; |eig| {ev}")
        print(f"        Howard ({ith} its): c_t - b = {(-Fh).ravel()}; d={dh:.6f}; |eig| {evh}")
        print(f"        P (Howard) =\n{Ph}")
