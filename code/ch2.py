"""Chapter 2 (Time Series): numerical parts of the exercises."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rmtlib import sc, FIG, doublej, stationary_mean, lq_howard, lq_riccati, kalman_steady

np.set_printoptions(precision=6, suppress=True, linewidth=140)
rng = np.random.default_rng(20260919)

# ---------------------------------------------------------------- 2.1
P = np.array([[.9, .1], [.3, .7]]); pi0 = np.array([.5, .5])
def lik(states):
    L = pi0[states[0]]
    for a, b in zip(states[:-1], states[1:]):
        L *= P[a, b]
    return L
print("2.1", [lik(s) for s in ([0, 1, 0, 1, 0], [0] * 5, [1] * 5)])

# ---------------------------------------------------------------- 2.2
h = np.array([[1., 1.], [5., 25.]]); J = np.array([[1.8, 5.8], [3.4, 15.4]])
print("2.2 P =\n", J @ np.linalg.inv(h))

# ---------------------------------------------------------------- 2.3
c = np.array([1., 5.]); beta = .95
for gam in (2.5, 4.0):
    u = c ** (1 - gam) / (1 - gam)
    for name, PP in (("proc1", np.eye(2)), ("proc2", np.full((2, 2), .5))):
        v = np.linalg.solve(np.eye(2) - beta * PP, u)
        print("2.3 gamma", gam, name, "v", v, "V", pi0 @ v)
L1 = .5 * 1 ** 9; L2 = .5 ** 10
print("2.3c likelihoods", L1, L2, " posterior M1", L1 / (L1 + L2), "M2", L2 / (L1 + L2))

# ---------------------------------------------------------------- 2.4
def ar4_system(rho, mu, cc):
    rho = np.asarray(rho, float); alpha = mu * (1 - rho.sum())
    A = np.zeros((5, 5)); A[0, :4] = rho; A[0, 4] = alpha
    A[1, 0] = A[2, 1] = A[3, 2] = 1; A[4, 4] = 1
    C = np.zeros((5, 1)); C[0, 0] = cc
    return A, C
cases = {"i": ([1.2, -.3, 0, 0], 10, 1), "ii": ([1.2, -.3, 0, 0], 10, 2), "iii": ([.9, 0, 0, 0], 5, 1),
         "iv": ([.2, 0, 0, .5], 5, 1), "v": ([.8, .3, 0, 0], 5, 1)}
G = np.array([[1., 0, 0, 0, 0]])
for k, (rho, mu, cc) in cases.items():
    A, C = ar4_system(rho, mu, cc)
    ev = np.linalg.eigvals(A[:4, :4]); stable = np.max(np.abs(ev)) < 1
    h5 = (G @ np.linalg.matrix_power(A, 5)).ravel()
    out = f"2.4 case {k}: max|eig| of AR block {np.max(np.abs(ev)):.4f} stable={stable}; E_t y_t+5: h={h5[:4]} gamma0={h5[4]:.4f}"
    if np.max(np.abs(.95 * ev)) < 1:
        ht = (G @ np.linalg.inv(np.eye(5) - .95 * A)).ravel()
        out += f"\n      discounted sum: htilde={ht[:4]} const={ht[4]:.4f}"
    else:
        out += "\n      discounted sum diverges (.95*max|eig| >= 1)"
    if stable:
        m = stationary_mean(A); Sig = doublej(A, C @ C.T)
        acov = [sc(G @ np.linalg.matrix_power(A, j) @ Sig @ G.T) for j in (0, 1, 5, 10)]
        out += f"\n      mean {m[0]:.4f}, var {acov[0]:.4f}, autocov k=1,5,10: {acov[1]:.4f} {acov[2]:.4f} {acov[3]:.4f}"
    print(out)

# ---------------------------------------------------------------- 2.5 / 2.6
def cz_system(rho, alpha_c, delta, gamma, phi, psi1, psi2):
    A = np.zeros((5, 5))
    A[0] = [rho[0], rho[1], delta[0], delta[1], alpha_c]
    A[1, 0] = 1
    A[2] = [gamma[0], gamma[1], phi[0], phi[1], 0]
    A[3, 2] = 1; A[4, 4] = 1
    C = np.zeros((5, 2)); C[0, 0] = psi1; C[2, 1] = psi2
    return A, C
H = np.array([[1., 0, 0, 0, -60.]])            # c_t - 60 = H x_t
for tag, psi1 in (("i", 1.), ("ii", 2.)):
    A, C = cz_system([.8, -.3], 1., [.2, 0], [0, 0], [.7, -.2], psi1, 1.)
    nu = doublej(np.sqrt(.95) * A.T, H.T @ H)   # nu = H'H + .95 A' nu A
    sig = .95 / (1 - .95) * np.trace(nu @ C @ C.T)
    m = stationary_mean(A); Sig = doublej(A, C @ C.T)
    V_at_mean = -.5 * (m @ nu @ m + sig)
    V_uncond = -.5 * (m @ nu @ m + np.trace(nu @ Sig) + sig)
    print(f"2.5 case {tag}: mean c={m[0]:.4f} z={m[2]:.4f}; sigma-term {sig:.4f}; V0 at x0=mean {V_at_mean:.3f}; E V0 (x0~stationary) {V_uncond:.3f}")
    print("     nu =\n", nu)
    if tag == "i":
        print("2.6a stationary mean", m, "\n     covariance (c_t,c_t-1,z_t,z_t-1) =\n", Sig[:4, :4])
        cov = lambda j: np.linalg.matrix_power(A, j) @ Sig    # E(x_t+j - mu)(x_t - mu)'
        # regressors X = [z_t, z_t-4], dependent c_t+2
        Sxx = np.array([[Sig[2, 2], cov(4)[2, 2]], [cov(4)[2, 2], Sig[2, 2]]])
        Sxy = np.array([cov(2)[0, 2], cov(6)[0, 2]])
        a12 = np.linalg.solve(Sxx, Sxy); a0 = m[0] - a12.sum() * m[2]
        print(f"2.6b alpha0={a0:.6f} alpha1={a12[0]:.6f} alpha2={a12[1]:.6f}")

# ---------------------------------------------------------------- 2.7
procs = {"a": ([1.], [1.]), "b": ([1.], [1., .5]), "c": ([1.], [1., .5, .4]), "d": ([1., -.999], [1., -.4]),
         "e": ([1., -.8], [1., .5, .4]), "f": ([1., .8], [1.]), "g": ([1.], [1., -.6])}
labels = {"a": r"$y_t=w_t$", "b": r"$y_t=(1+.5L)w_t$", "c": r"$y_t=(1+.5L+.4L^2)w_t$",
          "d": r"$(1-.999L)y_t=(1-.4L)w_t$", "e": r"$(1-.8L)y_t=(1+.5L+.4L^2)w_t$",
          "f": r"$(1+.8L)y_t=w_t$", "g": r"$y_t=(1-.6L)w_t$"}
def arma_irf(a, b, n):
    y = np.zeros(n); w = np.zeros(n); w[0] = 1
    for t in range(n):
        y[t] = sum(b[j] * w[t - j] for j in range(len(b)) if t - j >= 0) - sum(a[j] * y[t - j] for j in range(1, len(a)) if t - j >= 0)
    return y
def arma_sim(a, b, n, burn=200):
    w = rng.standard_normal(n + burn); y = np.zeros(n + burn)
    for t in range(n + burn):
        y[t] = sum(b[j] * w[t - j] for j in range(len(b)) if t - j >= 0) - sum(a[j] * y[t - j] for j in range(1, len(a)) if t - j >= 0)
    return y[burn:]
om = np.linspace(0, np.pi, 400)
fig, ax = plt.subplots(7, 3, figsize=(10, 15))
for r, (k, (a, b)) in enumerate(procs.items()):
    z = np.exp(-1j * om)
    S = np.abs(np.polyval(b[::-1], z)) ** 2 / np.abs(np.polyval(a[::-1], z)) ** 2
    ax[r, 0].plot(arma_irf(a, b, 30), "k.-", ms=3); ax[r, 0].set_ylabel(f"({k})", rotation=0, labelpad=12)
    ax[r, 1].semilogy(om, S, "k-")
    ax[r, 2].plot(arma_sim(a, b, 80), "k-", lw=.8)
    ax[r, 1].set_title(labels[k], fontsize=9)
    print(f"2.7 ({k}) spectrum at 0: {S[0]:.4g}, at pi: {S[-1]:.4g}")
ax[0, 0].set_title("impulse response", fontsize=9); ax[0, 2].set_title("simulation (T=80)", fontsize=9)
for a_ in ax[-1]: a_.set_xlabel("")
ax[-1, 1].set_xlabel(r"frequency $\omega$"); ax[-1, 0].set_xlabel("lag"); ax[-1, 2].set_xlabel("t")
plt.tight_layout(); plt.savefig(FIG + "ch2_07.pdf"); plt.close()

# ---------------------------------------------------------------- 2.8e
t = np.arange(0, 11); m = t + 2.0                          # m_{-1}=1, m_{-2}=0, rho=1
mlag = np.r_[1.0, m[:-1]]
p_fund = 2 * m - mlag
plt.figure(figsize=(6, 3.4))
for cc in (-.02, -.01, 0, .01, .02):
    plt.plot(t, p_fund + cc * 2.0 ** t, "k-" if cc == 0 else "k--", lw=1.6 if cc == 0 else .9)
plt.plot(t, m, "k:", lw=1.2, label="$m_t$")
plt.xlabel("t"); plt.ylabel("log price level")
plt.text(10.1, p_fund[-1], "$f_t=0$", fontsize=8, va="center")
plt.tight_layout(); plt.savefig(FIG + "ch2_08.pdf"); plt.close()
print("2.8e fundamental p_t for t=0..5:", p_fund[:6])

# ---------------------------------------------------------------- 2.13
b = .95
A = np.array([[1, 0, 0, 0], [30 / b, 1 / b, -1 / b, 0], [.5, 0, 1.3, -.4], [0, 0, 1, 0]])
B = np.array([[0], [1 / b], [0], [0]]); C = np.array([[0], [0], [.05], [0]])
R = np.diag([0, 1e-6, 0, 0]); Q = np.array([[1.0]])
print("2.13a eig(A):", np.sort(np.linalg.eigvals(A).real), " roots of 1-1.3z+.4z^2:", np.sort(np.roots([.4, -1.3, 1]).real))
# initial rule c_t - 30 = -(1-beta) b_t: it keeps debt from exploding, so sqrt(beta)(A-BF0) is stable
F0 = np.zeros((1, 4)); F0[0, 1] = 1 - b
Ao = np.sqrt(b) * (A - B @ F0)
print("2.13b initial rule spectral radius of sqrt(beta)(A-BF0):", np.max(np.abs(np.linalg.eigvals(Ao))))
F, Pm, its = lq_howard(A, B, R, Q, b, F0, tol=1e-10)
print("2.13b Howard iterations:", its, " F* =", F.ravel())
F2, P2, its2 = lq_riccati(A, B, R, Q, b, tol=1e-12)
print("2.13b check with Riccati iteration (", its2, "iterations): max|F-F2| =", np.max(np.abs(F - F2)))
Acl = A - B @ F
print("2.13c eig(A-BF*):", np.sort(np.abs(np.linalg.eigvals(Acl))))
n = 40; x = C.copy(); irf_c = []; irf_b = []
for j in range(n):
    irf_c.append(sc(-F @ x)); irf_b.append(sc(x[1])); x = Acl @ x
# theory (2.12.18): z_t = [1, y_t, y_t-1], A22, C2, Uy
A22 = np.array([[1, 0, 0], [.5, 1.3, -.4], [0, 1, 0]]); C2 = np.array([[0], [.05], [0]]); Uy = np.array([[0, 1, 0]])
M = Uy @ np.linalg.inv(np.eye(3) - b * A22)
dc = sc((1 - b) * M @ C2)
zt = C2.copy(); th_c = []; th_b = []
for j in range(n):
    th_c.append(dc); th_b.append(sc(M @ zt) - dc / (1 - b)); zt = A22 @ zt
print("2.13c theoretical consumption response (1-beta)d(beta) =", dc, "; LQ response at j=0,10,39:", irf_c[0], irf_c[10], irf_c[39])
print("      debt response LQ j=0,1,5,39:", irf_b[0], irf_b[1], irf_b[5], irf_b[39], "; theory:", th_b[0], th_b[1], th_b[5], th_b[39])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(irf_c, "k-", label="LQ (Howard)"); ax[0].plot(th_c, "k--", label="(2.12.18)")
ax[0].set_title("consumption $c_{t+j}$"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(irf_b, "k-"); ax[1].plot(th_b, "k--"); ax[1].set_title("debt $b_{t+j}$")
for a_ in ax: a_.set_xlabel("j")
plt.tight_layout(); plt.savefig(FIG + "ch2_13.pdf"); plt.close()

# ---------------------------------------------------------------- 2.15 Slutsky
om2 = np.linspace(1e-4, np.pi - 1e-4, 2000)
fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
for a_, (mm, nn) in zip(ax, [(10, 10), (10, 40), (40, 10), (120, 30)]):
    logS = nn * np.log(2 + 2 * np.cos(om2)) + mm * np.log(2 - 2 * np.cos(om2))
    S = np.exp(logS - logS.max())
    wstar = 2 * np.arctan(np.sqrt(mm / nn))
    a_.plot(np.r_[-om2[::-1], om2], np.r_[S[::-1], S], "k-"); a_.axvline(wstar, color="grey", lw=.6); a_.axvline(-wstar, color="grey", lw=.6)
    a_.set_title(f"m={mm}, n={nn}", fontsize=9); a_.set_xlim(-np.pi, np.pi); a_.set_xticks([-3, 0, 3])
    print(f"2.15 m={mm} n={nn}: peak at omega*={wstar:.4f} (period {2*np.pi/wstar:.3f}); max S = 4^(m+n) * {np.exp(logS.max()-(mm+nn)*np.log(4)):.4g} ; half-power width {np.ptp(om2[S>.5]):.4f}")
ax[0].set_ylabel("$S_y(\\omega)/\\max S_y$")
plt.tight_layout(); plt.savefig(FIG + "ch2_15.pdf"); plt.close()

# ---------------------------------------------------------------- 2.17
lam, dlt = .05, .25
pu, pe = dlt / (lam + dlt), lam / (lam + dlt)
g = {"11": pu * (1 - lam), "21": pu * lam, "12": pe * dlt, "22": pe * (1 - dlt)}
print("2.17 pi_u", pu, "pi_e", pe, "g", g, "avar lam", lam * (1 - lam) / pu, "avar delta", dlt * (1 - dlt) / pe)

# ---------------------------------------------------------------- 2.19 IQ
T = 51
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for d in range(10):
    th = 100 + 10 * rng.standard_normal(); y = th + 10 * rng.standard_normal(T)
    IQ = np.empty(T); IQ[0] = 100.0
    for t_ in range(T - 1):
        K = 1.0 / (t_ + 2)
        IQ[t_ + 1] = IQ[t_] + K * (y[t_] - IQ[t_])
    ax[0].plot(IQ, lw=.9); ax[0].axhline(th, lw=.3, color="grey")
ax[0].set_title("$IQ_t$ for 10 draws of $\\theta$ (grey: true $\\theta$)", fontsize=9); ax[0].set_xlabel("t")
ax[1].plot(np.arange(T), 100 / (np.arange(T) + 1), "k-"); ax[1].set_title("$E(IQ_t-\\theta)^2=100/(t+1)$", fontsize=9); ax[1].set_xlabel("t")
plt.tight_layout(); plt.savefig(FIG + "ch2_19.pdf"); plt.close()

# ---------------------------------------------------------------- 2.20
A1 = np.array([[1.]]); C1 = np.array([[1.]]); G2 = np.array([[1.], [1.]]); R2 = np.eye(2)
S0 = np.array([[1.36602540378444]])
K = A1 @ S0 @ G2.T @ np.linalg.inv(G2 @ S0 @ G2.T + R2)
S1 = C1 @ C1.T + K @ R2 @ K.T + (A1 - K @ G2) @ S0 @ (A1 - K @ G2).T
print("2.20 K =", K, " Sigma_1 =", S1, " (1+sqrt3)/2 =", (1 + 3 ** .5) / 2, " A-KG =", A1 - K @ G2, " 2-sqrt3 =", 2 - 3 ** .5)
print("     innovation covariance =\n", G2 @ S0 @ G2.T + R2)

# ---------------------------------------------------------------- 2.24
Kz, Sz, _ = kalman_steady(np.array([[.9]]), np.array([[1.]]), np.array([[1.]]), np.array([[1.]]), Sigma0=np.array([[1 / .19]]))
print("2.24 steady-state Sigma", Sz, " K", Kz, " A-KG", .9 - Kz, " Omega", Sz + 1, " formula", (.81 + np.sqrt(.81 ** 2 + 4)) / 2)
S = 1 / .19
for t_ in range(6):
    print(f"     t={t_}: Sigma_t={S:.5f} Omega_t={S+1:.5f}")
    S = .81 * S + 1 - .81 * S ** 2 / (S + 1)

# ---------------------------------------------------------------- 2.25
for s1, s2 in ((1., 1.), (2., 1.)):
    Qv, Rv = s1 ** 2, s2 ** 2
    Sg = (Qv + np.sqrt(Qv ** 2 + 4 * Qv * Rv)) / 2; K = Sg / (Sg + Rv)
    print(f"2.25 sigma1={s1} sigma2={s2}: Sigma={Sg:.6f} K={K:.6f} 1-beta(1-K)={1-.95*(1-K):.6f} var(a)={Sg+Rv:.6f} "
          f"var dc ex1={s1**2+(.05*s2)**2:.6f} ex2={(1-.95*(1-K))**2*(Sg+Rv):.6f}")

# ---------------------------------------------------------------- 2.26
Ge, Go = np.array([[.9, .1]]), np.array([[.01, .99]]); Rv = 50.
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
Sig = 100 * np.eye(2); tr = []
for t_ in range(T):
    tr.append(Sig.copy()); Gt = Ge if t_ % 2 == 0 else Go
    Sig = Sig - Sig @ Gt.T @ np.linalg.inv(Gt @ Sig @ Gt.T + Rv) @ Gt @ Sig
tr = np.array(tr)
for d in range(10):
    th = 100 + 10 * rng.standard_normal(2); IQ = np.array([100., 100.]); path = [IQ.copy()]; S_ = 100 * np.eye(2)
    for t_ in range(T - 1):
        Gt = Ge if t_ % 2 == 0 else Go
        y = sc(Gt @ th) + np.sqrt(Rv) * rng.standard_normal()
        Kt = S_ @ Gt.T / sc(Gt @ S_ @ Gt.T + Rv)
        IQ = IQ + (Kt * (y - sc(Gt @ IQ))).ravel(); S_ = S_ - Kt @ Gt @ S_
        path.append(IQ.copy())
    path = np.array(path)
    ax[0].plot(path[:, 0], lw=.8); ax[1].plot(path[:, 1], lw=.8)
ax[0].set_title("math $IQ_{1t}$", fontsize=9); ax[1].set_title("verbal $IQ_{2t}$", fontsize=9)
ax[2].semilogy(tr[:, 0, 0], "k-", label="$\\Sigma_{11}$"); ax[2].semilogy(tr[:, 1, 1], "k--", label="$\\Sigma_{22}$")
ax[2].semilogy(np.abs(tr[:, 0, 1]), "k:", label="$|\\Sigma_{12}|$"); ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("$E(IQ_t-\\theta)(IQ_t-\\theta)'$", fontsize=9)
for a_ in ax: a_.set_xlabel("t")
plt.tight_layout(); plt.savefig(FIG + "ch2_26.pdf"); plt.close()
print("2.26 Sigma_50 =\n", tr[-1])

# ---------------------------------------------------------------- 2.28
A28 = np.array([[0., 0.], [1., 0.]]); C28 = np.array([[1.], [0.]]); G28 = np.array([[1., -2.]])
K28, S28, it28 = kalman_steady(A28, C28, G28, np.array([[0.]]), Sigma0=np.eye(2))
Om = G28 @ S28 @ G28.T
print("2.28 steady-state K =", K28.ravel(), " var(a) =", Om, " iterations", it28)
# innovations rep: y_t = G xhat_t + a_t, xhat_{t+1} = A xhat_t + K a_t  =>  MA coefficients G A^{j-1} K
print("     MA coefficients of innovation rep:", [1.0] + [sc(G28 @ np.linalg.matrix_power(A28, j - 1) @ K28) for j in range(1, 4)])

# ---------------------------------------------------------------- 2.29e dual Howard for the Kalman filter (example: exercise 2.24)
def kalman_howard(A, C, G, Rm, K0, tol=1e-13):
    K = K0
    for it in range(1, 200):
        Acl = A - K @ G
        S = doublej(Acl, C @ C.T + K @ Rm @ K.T)
        Kn = A @ S @ G.T @ np.linalg.inv(G @ S @ G.T + Rm)
        if np.max(np.abs(Kn - K)) < tol:
            return Kn, S, it
        K = Kn
Kh, Sh, ith = kalman_howard(np.array([[.9]]), np.array([[1.]]), np.array([[1.]]), np.array([[1.]]), np.array([[0.]]))
print("2.29e dual Howard: K", Kh, "Sigma", Sh, "iterations", ith)

# ---------------------------------------------------------------- 2.30 numerical illustration of the MPE algorithm
A30 = np.array([[1.0, .1], [0., .9]]); B30 = np.array([[0.], [1.]]); R30 = np.eye(2); Q30 = np.array([[1.]]); bt30 = .95
for dl in (1.0, .7):
    F = np.zeros((1, 2))
    for it in range(1, 500):
        Pv = doublej(np.sqrt(bt30) * (A30 - B30 @ F).T, R30 + F.T @ Q30 @ F)
        Gn = dl * bt30 * np.linalg.solve(Q30 + dl * bt30 * B30.T @ Pv @ B30, B30.T @ Pv @ A30)
        if np.max(np.abs(Gn - F)) < 1e-12: break
        F = Gn
    Fs, Ps, _ = lq_riccati(A30, B30, R30, Q30, bt30)
    Gs = dl * bt30 * np.linalg.solve(Q30 + dl * bt30 * B30.T @ Ps @ B30, B30.T @ Ps @ A30)
    print(f"2.30 delta={dl}: MPE F={F.ravel()} ({it} its); dictator u0=-G*x with G*={Gs.ravel()}, later F*={Fs.ravel()}")
