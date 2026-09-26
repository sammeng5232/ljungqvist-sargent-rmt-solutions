"""Chapter 19 (Dynamic Stackelberg Problems): exercises 19.1-19.3 and the book's section 19.5.3 example.

Every Stackelberg plan is computed twice: recursively (Riccati equation (19.4.5) plus x0 = -P22^{-1} P21 z0)
and by brute force (a long finite-horizon sequence problem solved as one quadratic program).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from rmtlib import FIG, olrp, doublej

np.set_printoptions(precision=5, suppress=True, linewidth=120)
beta, eps = .95, 1e-5


def stackelberg(A, B, R, Q, nz):
    """Recursive solution: F, P, the x0 rule -P22^{-1}P21, the closed loop A - BF and its discounted spectral radius."""
    F, P, it = olrp(beta, A, B, Q, R)
    X0 = -np.linalg.solve(P[nz:, nz:], P[nz:, :nz])
    Acl = A - B @ F
    rho = np.max(np.abs(np.linalg.eigvals(np.sqrt(beta) * Acl)))
    return F, P, X0, Acl, rho, it


# ================================================================== 19.1: optimal (Stackelberg) money supply
def money_system(alpha):
    lam = alpha / (1 + alpha)
    A = np.array([[1., 0.], [-1 / alpha, (1 + alpha) / alpha]])      # y = [m, p]:  p' = (p - (1-lam) m)/lam
    B = np.array([[1.], [0.]])
    return lam, A, B, np.diag([eps, 1.]), np.array([[1.]])


def brute_money(alpha, m0=1.0, T=400):
    """Choose u_0..u_{T-1} (u_t = 0 afterwards) to minimise sum beta^t (p^2 + u^2 + eps m^2),
    with m_t = m0 + sum_{s<t} u_s and p_t = (1-lam) sum_j lam^j m_{t+j} (money constant from T on)."""
    lam = alpha / (1 + alpha)
    L = np.tril(np.ones((T + 1, T)), -1)                 # m = m0*1 + L u, t = 0..T
    Kf = np.zeros((T + 1, T + 1))                        # p_t = sum_s Kf[t,s] m_s, t = 0..T
    for t in range(T + 1):
        s = np.arange(t, T)
        Kf[t, s] = (1 - lam) * lam ** (s - t)
        Kf[t, T] = lam ** (T - t)                        # tail: m_s = m_T for s >= T
    w = beta ** np.arange(T + 1.); w[T] = beta ** T / (1 - beta)      # tail t >= T: p = m = m_T, u = 0
    W = np.diag(w)
    one = np.ones(T + 1) * m0
    M = Kf.T @ W @ Kf + eps * W
    H = L.T @ M @ L + np.diag(beta ** np.arange(T + 0.))
    u = np.linalg.solve(H, -L.T @ M @ one)
    m = one + L @ u
    return u, m, Kf @ m, one @ M @ one + u @ L.T @ M @ one


def mpe(alpha):
    """Markov perfect equilibrium of 19.2: fixed point g = G(g)."""
    lam = alpha / (1 + alpha)
    hf = lambda g: (1 - lam) / (1 - lam * (1 + g))
    Wf = lambda g: (hf(g) ** 2 + g ** 2 + eps) / (1 - beta * (1 + g) ** 2)
    G = lambda g: -(lam ** 2 * hf(g) ** 2 + lam * (1 - lam) * hf(g) + beta * Wf(g)) / (1 + lam ** 2 * hf(g) ** 2 + beta * Wf(g))
    grid = np.linspace(-.9999, 0, 4001)
    fx = np.array([G(g) - g for g in grid])
    roots = [brentq(lambda g: G(g) - g, grid[i], grid[i + 1], xtol=1e-15)
             for i in range(len(grid) - 1) if fx[i] * fx[i + 1] < 0]
    g = 0.; it = 0                                       # the algorithm of 19.2(e): g_{j+1} = G(g_j)
    while True:
        gn = G(g); it += 1
        if abs(gn - g) < 1e-14: break
        g = gn
    return lam, g, hf(g), Wf(g), roots, it, G


print("=== 19.1 / 19.2  (beta = .95, loss p^2 + u^2 + 1e-5 m^2); Stackelberg u_t = -F_m m_t - F_p p_t, p_0 = x0 m_0; MPE u_t = g m_t, p_t = h m_t")
print(" alpha  lambda     F_m       F_p       x0     loss(commit)   g_MPE     h_MPE   loss(MPE)  #roots  brute-force check")
for alpha in (.5, 1., 2., 5.):
    lam, A, B, R, Q = money_system(alpha)
    F, P, X0, Acl, rho, it = stackelberg(A, B, R, Q, 1)
    Wc = P[0, 0] - P[0, 1] ** 2 / P[1, 1]                # w(m0) = -Wc m0^2
    u_b, m_b, p_b, loss_b = brute_money(alpha)
    y = np.array([1., X0.item()]); us = []
    for t in range(60):
        us.append(-(F @ y).item()); y = Acl @ y
    lam, g, h, Wm, roots, itg, G = mpe(alpha)
    err = max(np.max(np.abs(np.array(us) - u_b[:60])), abs(loss_b - Wc))
    print(f" {alpha:4.1f}  {lam:.4f}  {F[0,0]:8.5f}  {F[0,1]:8.5f}  {X0.item():8.5f}   {Wc:8.5f}   {g:8.5f}  {h:8.5f}  {Wm:8.5f}   {len(roots)}    {err:.1e}")
    print(f"        (Riccati iterations {it}; eigenvalues of A-BF = {np.round(np.linalg.eigvals(Acl), 5)}; MPE iterations {itg})")

alpha = 1.
lam, A, B, R, Q = money_system(alpha)
F, P, X0, Acl, rho, it = stackelberg(A, B, R, Q, 1)
print(f"\n19.1 detail for alpha = 1: P =\n{P}\n F = {F}, p0 = {X0.item():.6f} m0")
T = 25; y = np.array([1., X0.item()]); path = []
for t in range(T):
    path.append((y[0], y[1], -(F @ y).item(), (P[1] @ y).item())); y = Acl @ y
path = np.array(path)
print("  t    m_t       p_t       u_t      mu_p,t")
for t in range(8):
    print(f" {t:2d}  {path[t,0]:8.5f}  {path[t,1]:8.5f}  {path[t,2]:8.5f}  {path[t,3]:9.6f}")
y = np.array([1., X0.item()]); ms = []
for t in range(600):
    ms.append(y[0]); y = Acl @ y
ms = np.array(ms)
pf = np.array([(1 - lam) * np.sum(lam ** np.arange(600 - t) * ms[t:]) for t in range(10)])
print("  forward-solution check max|p_t - (1-lam) sum lam^j m_(t+j)| =", np.max(np.abs(pf - path[:10, 1])))
# representation in (m, mu_p): mu_p0 = 0
Tm = np.array([[1., 0.], [-P[1, 0] / P[1, 1], 1 / P[1, 1]]])            # [m; p] = Tm [m; mu]
Mmu = np.linalg.solve(Tm, Acl @ Tm)
print("  law of motion of [m; mu_p]:\n", Mmu, "\n  u_t in terms of [m; mu_p]:", -(F @ Tm))
# time inconsistency: continuing versus re-optimising at t = 1
y1 = path[1, :2]; y1s = np.array([y1[0], X0.item() * y1[0]])
print(f"  at t=1: m1 = {y1[0]:.5f}, promised p1 = {y1[1]:.5f}, mu_p1 = {path[1,3]:.5f}; re-optimised p1 = {y1s[1]:.5f}")
print(f"  value of continuing the plan {-(y1 @ P @ y1):.6f} < value of a fresh Stackelberg plan {-(y1s @ P @ y1s):.6f}")

lam, g, h, Wm, roots, itg, G = mpe(alpha)
print(f"\n19.2 MPE for alpha = 1: g = {g:.6f} (iterations from g=0: {itg}; fixed points on (-1,0]: {np.round(roots, 6)}), h = {h:.6f}, W = {Wm:.6f}")
print(f"  G(g) at g = -.9, -.5, 0: {G(-.9):.5f}, {G(-.5):.5f}, {G(0.):.5f};  |G'(g*)| ~ {abs(G(g + 1e-6) - G(g - 1e-6)) / 2e-6:.5f}")
print(f"  commitment loss coefficient {P[0,0]-P[0,1]**2/P[1,1]:.6f} vs MPE {Wm:.6f}")
m0 = 1.
obj = lambda u0: (lam * h * (m0 + u0) + (1 - lam) * m0) ** 2 + u0 ** 2 + eps * m0 ** 2 + beta * Wm * (m0 + u0) ** 2
res = minimize_scalar(obj, bounds=(-2, 2), method="bounded", options={"xatol": 1e-12})
print(f"  numerical maximiser of (3) at m0 = 1: u0 = {res.x:.8f} (g = {g:.8f}); v(m0) = {-res.fun:.6f} vs w(m0) = {-Wm:.6f}")
mm = [1.]; pp = []
for t in range(T):
    pp.append(h * mm[-1]); mm.append(mm[-1] * (1 + g))

fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
tt = np.arange(T)
ax[0].plot(tt, path[:, 0], "k.-", label="commitment (Stackelberg)"); ax[0].plot(tt, mm[:T], "b.--", label="Markov perfect")
ax[0].set_title("money $m_t$", fontsize=9)
ax[1].plot(tt, path[:, 1], "k.-"); ax[1].plot(tt, pp, "b.--"); ax[1].set_title("price level $p_t$", fontsize=9)
ax[2].plot(tt, path[:, 3], "k.-"); ax[2].axhline(0, color="grey", lw=.6)
ax[2].set_title(r"multiplier $\mu_{p,t}=P_{21}m_t+P_{22}p_t$", fontsize=9)
for a in ax: a.set_xlabel("t")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch19_01.pdf"); plt.close()


# ================================================================== section 19.5.3 and exercise 19.3
def industry(A0, A1, rho, c, d, e, g, h, follower):
    """y = [1, v, Q, q, i]; the last row of (19.5.6) is the follower's Euler equation:
    fringe (price taker, costs d, h) or duopolist (internalises its price effect, costs e, g)."""
    cst, slope = (A0 - d, -A1 - h) if follower == "fringe" else (A0 - e, -(2 * A1 + g))
    Gm = np.eye(5); Gm[4] = [cst, 1, -A1, slope, c]
    Ah = np.array([[1, 0, 0, 0, 0], [0, rho, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, c / beta]], float)
    Bh = np.array([[0], [0], [1], [0], [0]], float)
    A = np.linalg.solve(Gm, Ah); B = np.linalg.solve(Gm, Bh)
    R = -np.array([[0, 0, (A0 - e) / 2, 0, 0], [0, 0, .5, 0, 0], [(A0 - e) / 2, .5, -A1 - .5 * g, -A1 / 2, 0],
                   [0, 0, -A1 / 2, 0, 0], [0, 0, 0, 0, 0]], float)
    return A, B, R, np.array([[c / 2]])


A0, A1, rho_v, c, d, e, gc, hc = 100., 1., .8, 1., 20., 20., .2, .2
A, B, R, Q = industry(A0, A1, rho_v, c, d, e, gc, hc, "fringe")
F, P, X0, Acl, rho, it = stackelberg(A, B, R, Q, 4)
print("\n=== Section 19.5.3 check (large firm + competitive fringe, the book's parameters)")
print("  u_t = -F y_t with -F =", np.round(-F.ravel(), 2), "  (book: [-83.98 -0.78 0.95 1.31 2.07])")
print("  x0 =", np.round(X0.ravel(), 2), "z0   (book: [31.08 0.29 -0.15 -0.56])")
print(f"  spectral radius of sqrt(beta)(A-BF) = {rho:.5f}; Riccati iterations {it}")

A, B, R, Q = industry(A0, A1, rho_v, c, d, e, gc, hc, "duopolist")
F, P, X0, Acl, rho, it = stackelberg(A, B, R, Q, 4)
print("\n=== 19.3 duopoly with parameters A0, A1, rho, c, e, g, beta = 100, 1, .8, 1, 20, .2, .95")
print("  leader: u_t = Q1_{t+1} - Q1_t = -F y_t, -F =", np.round(-F.ravel(), 5))
print("  follower's initial investment x0 = Q2_1 - Q2_0 =", np.round(X0.ravel(), 5), "z0")
print(f"  spectral radius sqrt(beta)(A-BF) = {rho:.5f}; eigenvalues of A-BF: {np.round(np.sort(np.abs(np.linalg.eigvals(Acl))), 5)}")
phi = 1 + 1 / beta + (2 * A1 + gc) / c
d1 = (phi - np.sqrt(phi ** 2 - 4 / beta)) / 2
print(f"  follower's Euler roots: delta1 = {d1:.6f}, 1/(beta delta1) = {1/(beta*d1):.6f}")

Q10 = Q20 = 25.                                          # static Cournot outputs (A0-e)/(3A1+g)
z0 = np.array([1., 0., Q10, Q20]); y0 = np.r_[z0, X0 @ z0]
T = 400
Y = np.zeros((T + 1, 5)); U = np.zeros(T); Y[0] = y0
for t in range(T):
    U[t] = -(F @ Y[t]).item(); Y[t + 1] = Acl @ Y[t]
Q1, Q2, v = Y[:, 2], Y[:, 3], Y[:, 1]
price = A0 - A1 * (Q1 + Q2) + v
pi1 = price[:T] * Q1[:T] - e * Q1[:T] - .5 * gc * Q1[:T] ** 2 - .5 * c * U ** 2
pi2 = price[:T] * Q2[:T] - e * Q2[:T] - .5 * gc * Q2[:T] ** 2 - .5 * c * (Q2[1:] - Q2[:T]) ** 2
disc = beta ** np.arange(T)
V1, V2 = disc @ pi1, disc @ pi2
R2 = np.zeros((5, 5))                                    # firm 2's profit = -y'R2 y
R2[0, 3] = R2[3, 0] = -(A0 - e) / 2; R2[1, 3] = R2[3, 1] = -.5; R2[2, 3] = R2[3, 2] = A1 / 2
R2[3, 3] = A1 + .5 * gc; R2[4, 4] = .5 * c
P2 = doublej(np.sqrt(beta) * Acl.T, R2)
print(f"  from Q1_0 = Q2_0 = 25, v0 = 0:  x0 = {y0[4]:.5f}")
print(f"    leader value  {V1:.4f}  (= -y0'P y0 = {-(y0 @ P @ y0):.4f})")
print(f"    follower value {V2:.4f}  (Lyapunov: {-(y0 @ P2 @ y0):.4f})")
print(f"    Stackelberg steady state: Q1 = {Q1[-1]:.4f}, Q2 = {Q2[-1]:.4f}, p = {price[-1]:.4f}")
print("    t, Q1, Q2, p, mu_x:", *[f"\n     {t:2d} {Q1[t]:8.4f} {Q2[t]:8.4f} {price[t]:8.4f} {(P[4] @ Y[t]):9.5f}" for t in range(6)])


# ---- brute-force check: follower's best response and the leader's sequence problem (finite horizon TT)
def follower_foc(TT, Q20):
    """Firm 2 maximises sum_{t<TT} beta^t [(A0-A1(Q1+Q2)+v)Q2 - eQ2 - .5gQ2^2 - .5c(Q2'-Q2)^2] over Q2_1..Q2_TT (v = 0).
    FOC: Hm Q2 = b0 + Bq Q1, Q1 = (Q1_0..Q1_TT)."""
    Hm = np.zeros((TT, TT)); b0 = np.zeros(TT); Bq = np.zeros((TT, TT + 1))
    for t in range(TT):
        wt = beta ** t
        if t >= 1:
            Hm[t - 1, t - 1] += wt * (2 * A1 + gc); b0[t - 1] += wt * (A0 - e); Bq[t - 1, t] -= wt * A1
        Hm[t, t] += wt * c
        if t >= 1:
            Hm[t - 1, t - 1] += wt * c; Hm[t - 1, t] -= wt * c; Hm[t, t - 1] -= wt * c
        else:
            b0[0] += wt * c * Q20
    return Hm, b0, Bq


TT = 300
Hm, b0, Bq = follower_foc(TT, Q20)
Q2br = np.linalg.solve(Hm, b0 + Bq @ Q1[:TT + 1])
print(f"  check 1: firm 2's best response to the leader's path reproduces Q2: max|diff|, t<=80: {np.max(np.abs(Q2br[:80] - Q2[1:81])):.2e}")
# leader: Q2_{1..TT} = a + M Q1_{0..TT}; choose Q1_{1..TT}
a = np.linalg.solve(Hm, b0); Mbr = np.linalg.solve(Hm, Bq)
S1 = np.vstack([np.zeros((1, TT)), np.eye(TT)]); a1 = np.r_[Q10, np.zeros(TT)]
a2 = np.r_[Q20, a + Mbr @ a1]; S2 = np.vstack([np.zeros((1, TT)), Mbr @ S1])
J = np.hstack([np.eye(TT), np.zeros((TT, 1))]); Dl = np.diff(np.eye(TT + 1), axis=0)
X1, x1, X2, x2, Yd, yd = J @ S1, J @ a1, J @ S2, J @ a2, Dl @ S1, Dl @ a1
D = np.diag(beta ** np.arange(TT + 0.)); w = beta ** np.arange(TT + 0.)
Hq = 2 * (A1 + .5 * gc) * X1.T @ D @ X1 + A1 * (X1.T @ D @ X2 + X2.T @ D @ X1) + c * Yd.T @ D @ Yd
rhs = (A0 - e) * X1.T @ w - 2 * (A1 + .5 * gc) * X1.T @ D @ x1 - A1 * (X1.T @ D @ x2 + X2.T @ D @ x1) - c * Yd.T @ D @ yd
q1 = np.linalg.solve(Hq, rhs)
print(f"  check 2: brute-force leader problem reproduces Q1: max|diff|, t<=80: {np.max(np.abs(q1[:80] - Q1[1:81])):.2e}")

# ---- monopoly owning both plants
Am = np.diag([1., rho_v, 1., 1.]); Bm = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
Rm = -np.array([[0, 0, (A0 - e) / 2, (A0 - e) / 2], [0, 0, .5, .5], [(A0 - e) / 2, .5, -A1 - .5 * gc, -A1],
                [(A0 - e) / 2, .5, -A1, -A1 - .5 * gc]])
Fm, Pm, itm = olrp(beta, Am, Bm, (c / 2) * np.eye(2), Rm)
VM = -(z0 @ Pm @ z0)
zz = z0.copy(); Qm = [zz[2:].copy()]
for t in range(T):
    zz = (Am - Bm @ Fm) @ zz; Qm.append(zz[2:].copy())
Qm = np.array(Qm)
print(f"  monopoly (both plants): value {VM:.4f}; steady state Q1 = Q2 = {Qm[-1,0]:.4f} (static formula (A0-e)/(4A1+g) = {(A0-e)/(4*A1+gc):.4f}), p = {A0 - A1*Qm[-1].sum():.4f}")
print(f"  firm 2's maximum willingness to pay V_M - V_F = {VM - V2:.4f}; firm 1's reservation price V_L = {V1:.4f}; gain from merger V_M - V_L - V_F = {VM - V1 - V2:.4f}")
IX = np.vstack([np.eye(4), X0])                          # y0 = IX z0
PF = IX.T @ P2 @ IX; PL = IX.T @ P @ IX
print("  quadratic forms (value = -z0' P z0): P_M - P_F =\n", Pm - PF)
for z in (np.array([1., 0., 0., 0.]), np.array([1., 0., 40., 10.]), np.array([1., 5., 25., 25.])):
    print(f"   z0 = {z}: V_L = {-(z @ PL @ z):.3f}, V_F = {-(z @ PF @ z):.3f}, V_M = {-(z @ Pm @ z):.3f}, WTP = {-(z @ (Pm - PF) @ z):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
tt = np.arange(30)
ax[0].plot(tt, Q1[:30], "k.-", label="Stackelberg"); ax[0].plot(tt, Qm[:30, 0], "b.--", label="monopoly (both plants)")
ax[0].set_title("output of firm 1, $Q_{1t}$", fontsize=9)
ax[1].plot(tt, Q2[:30], "k.-"); ax[1].plot(tt, Qm[:30, 1], "b.--"); ax[1].set_title("output of firm 2, $Q_{2t}$", fontsize=9)
ax[2].plot(tt, price[:30], "k.-"); ax[2].plot(tt, A0 - A1 * Qm[:30].sum(1), "b.--"); ax[2].set_title("price $p_t$", fontsize=9)
for a_ in ax: a_.set_xlabel("t")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch19_03.pdf"); plt.close()
