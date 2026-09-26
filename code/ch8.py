"""Chapter 8 (Equilibrium with Complete Markets): numerical parts of the exercises, with cross-checks."""
import itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=6, suppress=True, linewidth=140)


def hist_sum(Q0, i0, T, f):
    """Brute force: sum over all histories (s_1..s_T) of prod Q0[s_{t-1}, s_t] * f(history) (cross-check of matrix formulas)."""
    n = Q0.shape[0]; tot = 0.0
    for h in itertools.product(range(n), repeat=T):
        w, s = 1.0, i0
        for k in h:
            w *= Q0[s, k]; s = k
        tot += w * f(h)
    return tot


# ---------------------------------------------------------------- 8.4 Mehra-Prescott term structure
P = np.array([[.8, .2], [.15, .85]]); lam = np.array([.98, 1.03]); beta, gam = .96, 2.0
w, V = np.linalg.eig(P.T); pis = np.real(V[:, np.argmin(abs(w - 1))]); pis /= pis.sum()
print("8.4 stationary distribution", pis, " mean gross growth", pis @ lam, " E[lambda] under pi0=(.5,.5):", .5 * lam.sum())
pi0 = np.array([.5, .5]); print("    E[lambda_t] under pi0 for t=0..4:", [round((pi0 @ np.linalg.matrix_power(P, t)) @ lam, 6) for t in range(5)])
Q = beta * P * lam[None, :] ** (-gam)                              # Q_ij = beta P_ij lambda_j^{-gamma}
Qj = [np.linalg.matrix_power(Q, j) for j in range(3)]
print("8.4 Q =\n", Q)
for j in range(3):
    print(f"    j={j}: riskless {Qj[j][0].sum():.6f}   contingent on lambda_j=.98: {Qj[j][0, 0]:.6f}   on lambda_j=1.03: {Qj[j][0, 1]:.6f}")
chk = hist_sum(Q, 0, 2, lambda h: 1.0)
print("    brute-force check of the j=2 riskless price:", round(chk, 10))

# ---------------------------------------------------------------- 8.11 equivalent martingale (forward) measure, using the 8.4 economy
b1, b2 = Qj[1][0].sum(), Qj[2][0].sum()
pt1 = Qj[1][0] / b1                                                  # pi~_1 over s_1
pt2 = np.array([[Q[0, i] * Q[i, j] for j in range(2)] for i in range(2)]) / b2   # pi~_2 over (s_1, s_2)
print("8.11 b_1, b_2 =", b1, b2, "; pi~_1 =", pt1, " sums", pt1.sum(), "; pi~_2 marginal on s_1 =", pt2.sum(1), " sums", pt2.sum())

# ---------------------------------------------------------------- 8.6 Mehra-Prescott with P=[.8 .2;.1 .9]
P6 = np.array([[.8, .2], [.1, .9]]); lam6 = np.array([.97, 1.03]); beta6, gam6 = .95, 2.0
def ad_price(hist):
    pr, s, d = 1.0, 0, 1.0
    for k in hist:
        pr *= P6[s, k]; d *= lam6[k]; s = k
    return beta6 ** len(hist) * pr * d ** (-gam6), pr, d
for hist in [(0, 0, 1, 0, 1), (1, 1, 1, 1, 0)]:
    q, pr, d = ad_price(hist)
    print(f"8.6 history {lam6[list(hist)]}: prob {pr:.6f}, d5 {d:.6f}, price q = {q:.8f}")
Q6 = beta6 * P6 * lam6[None, :] ** (-gam6)
A6 = beta6 * P6 * lam6[None, :] ** (1 - gam6)
print("8.6 spectral radius of A (must be <1):", max(abs(np.linalg.eigvals(A6))))
vbar = np.linalg.solve(np.eye(2) - A6, np.ones(2))                  # cum-dividend price/dividend ratio by state
print("8.6(d) price of the whole endowment stream (cum dividend), d0=1, lambda0=.97:", vbar[0], "; by state:", vbar)
Q65 = np.linalg.matrix_power(Q6, 5)
print("8.6(e) price of date-5 claim contingent on lambda_5=.97:", Q65[0, 0], " brute force:",
      hist_sum(Q6, 0, 5, lambda h: 1.0 if h[-1] == 0 else 0.0))
print("8.6(g) natural debt limit / d_t by state (= cum-dividend P/D ratio):", vbar, "; ex-dividend:", vbar - 1)
print("8.6(h) Q =\n", Q6)
print("8.6(i) one-period riskless bond price by state:", Q6.sum(1), " gross rates", 1 / Q6.sum(1),
      "; two-period (pays at t+2):", (Q6 @ Q6).sum(1))

# ---------------------------------------------------------------- 8.7 two consumers, log utility, Y(s) = 1 + s
P7 = np.array([[.8, .2], [.3, .7]]); beta7 = .95; Y7 = np.array([1.0, 2.0]); y1 = np.array([0.0, 1.0]); y2 = np.ones(2)
Q7 = beta7 * P7 * Y7[:, None] / Y7[None, :]
print("8.7(h) Q =\n", Q7)
print("8.7(i) one-period riskless price by state:", Q7.sum(1), " gross rate", 1 / Q7.sum(1))
print("8.7(j) two-period riskless price by state:", (Q7 @ Q7).sum(1))
print("8.7(k) five-period Arrow prices Q^5 =\n", np.linalg.matrix_power(Q7, 5))
M7 = np.linalg.inv(np.eye(2) - Q7)
A1, A2 = M7 @ y1, M7 @ y2
print("8.7(g) natural debt limits  A1(s) =", A1, " A2(s) =", A2, " sum", A1 + A2, " vs Y/(1-beta)", Y7 / (1 - beta7))
theta = (1 - beta7) * A1 / Y7
print("8.7 consumption share of consumer 1 (=Pareto weight theta) for s0 = 0, 1:", theta)
for s0 in (0, 1):   # cross-check: Negishi iteration on theta by bisection on consumer 1's budget
    f = lambda th: th * Y7[s0] / (1 - beta7) - A1[s0]           # value of c1 = theta*Y minus value of y1
    print(f"    s0={s0}: theta by bisection {brentq(f, 1e-9, 1 - 1e-9):.10f}")
for s0 in (0, 1):
    th = theta[s0]; Ups1 = M7 @ (th * Y7 - y1)
    print(f"    s0={s0}: wealth (Arrow holdings) of consumer 1 by current state: {Ups1}")

# ---------------------------------------------------------------- 8.8 optimal taxation with a small open economy
P8 = np.array([[.8, .2], [.5, .5]]); g8 = np.array([.2, 1.0])
for b8 in (.9, .95):
    v = np.linalg.solve(np.eye(2) - b8 * P8, g8)
    R = (1 - b8) * v[0]
    B = R / (1 - b8) - v                                                 # debt owed at the start of a period = PV(R) - PV(g)
    print(f"8.8 beta={b8}: Rbar={R:.6f} (formula {(.2 + .1 * b8) / (1 - .3 * b8):.6f}); B(peace)={B[0]:.6f}, B(war)={B[1]:.6f} "
          f"(formula -.8/(1-.3b)={-.8 / (1 - .3 * b8):.6f})")
    for s in (0, 1):   # one-period budget check: g + B(s) = R + sum_s' beta P B(s')
        print(f"      budget residual in state {s}: {g8[s] + B[s] - R - b8 * P8[s] @ B:.2e}")

# ---------------------------------------------------------------- 8.9 / 8.10 linear-utility and log-utility consumers (truncated-horizon checks)
def corners(mu, alpha, beta, T=2000):
    t = np.arange(T); even = (t % 2 == 0)
    y1 = np.full(T, mu); y2 = np.where(even, 0.0, alpha); Y = y1 + y2
    thr = mu * (1 + 1 / beta)
    if alpha <= thr + 1e-12:                         # interior prices q_t = beta^t, c2 constant
        q = beta ** t; c2 = np.full(T, alpha * beta / (1 + beta))
    else:                                            # type 1 at a corner in even periods
        k = alpha * beta / (1 + beta)
        c2 = np.where(even, mu, k); q = np.where(even, beta ** t, beta ** t * mu / k)
    c1 = Y - c2
    return dict(q=q, c1=c1, c2=c2, W1=q @ y1, W2=q @ y2, bud2=q @ (c2 - y2), bud1=q @ (c1 - y1),
                R=q[:-1] / q[1:], foc1=[np.max((beta ** t / q)[even]) * q[1] / beta, np.max((beta ** t / q)[~even]) * q[1] / beta])
    # foc1 = max of beta^t/(mu_1 q_t) over even and over odd dates, with mu_1 = beta/q_1 from an odd date (must be <= 1, = 1 where c1 > 0)
for alpha in (1 + 1 / .95, 3.0):
    r = corners(1.0, alpha, .95)
    print(f"8.10 alpha={alpha:.4f}: c1(even,odd)={r['c1'][:2]}, c2(even,odd)={r['c2'][:2]}, q_0..3={r['q'][:4]}, "
          f"R(even->odd, odd->even)={r['R'][:2]}, W1={r['W1']:.6f}, W2={r['W2']:.6f}, budget residuals {r['bud1']:.1e},{r['bud2']:.1e}, "
          f"max beta^t/(mu1 q_t) even, odd = {np.round(r['foc1'], 6)}")

# ---------------------------------------------------------------- 8.14 Weill: identical CRRA, endowments owned by type 0 (recessions) and type 1 (booms)
y0b, y1b, b14 = 1.0, 2.0, .95
Yb = np.array([y0b, y1b])
def share1(P, gam, s0=1):
    Q = b14 * P * (Yb[None, :] / Yb[:, None]) ** (-gam)
    M = np.linalg.inv(np.eye(2) - Q)
    W1 = M[s0, 1] * y1b; W0 = M[s0, 0] * y0b
    return W1 / (W0 + W1), Q
P14 = np.array([[.6, .4], [.3, .7]])
for gam in (.5, 2.0, 5.0):
    a1, Q14 = share1(P14, gam)
    f = (1 - b14 * P14[0, 0]) / ((1 - b14 * P14[0, 0]) + b14 * P14[1, 0] * (y0b / y1b) ** (1 - gam))
    print(f"8.14(a) gamma={gam}: alpha_1 = {a1:.6f} (closed form {f:.6f}); one-period bond prices by state {Q14.sum(1)}")
Palt = np.array([[0.0, 1.0], [1.0, 0.0]])
gstar = 1 + np.log(1 / b14) / np.log(y1b / y0b)
for gam in (.5, 1.0, gstar, 2.0, 5.0):
    a1, _ = share1(Palt, gam)
    print(f"8.14(d) gamma={gam:.4f}: alpha_1 = {a1:.6f} (closed form {1 / (1 + b14 * (y0b / y1b) ** (1 - gam)):.6f})")
print(f"8.14(d) threshold gamma* = 1 + log(1/beta)/log(y1/y0) = {gstar:.6f}")
def default_gain(gam):   # at t=1 (recession), type 0: value of autarky minus value of honouring the contract (gamma < 1)
    a0 = 1 - 1 / (1 + b14 * (y0b / y1b) ** (1 - gam))
    u = lambda c: c ** (1 - gam) / (1 - gam)
    honour = (u(a0 * y0b) + b14 * u(a0 * y1b)) / (1 - b14 ** 2)
    autarky = u(y0b) / (1 - b14 ** 2)
    return autarky - honour
gd = brentq(default_gain, 1e-6, 1 - 1e-6)
print(f"8.14(e) default threshold for gamma in (0,1): {gd:.6f}; gain at gamma=.1,.5,.9: {[round(default_gain(g), 5) for g in (.1, .5, .9)]}")

# ---------------------------------------------------------------- 8.15 diverse beliefs I: solve consumer 1's budget for r = mu2/mu1
for b15 in (.5, .9, .95):
    def bud(r, b=b15):
        c10 = r / (1 + r); q1 = (2 + r) / (3 * (1 + r)); q2 = (1 + 2 * r) / (3 * (1 + r))
        c1h1, c1h2 = r / (2 + r), 2 * r / (1 + 2 * r)
        return c10 - .5 + b / (1 - b) * (q1 * (c1h1 - 1) + q2 * c1h2)
    print(f"8.15 beta={b15}: r = mu2/mu1 = {brentq(bud, 1e-6, 1e6):.10f}")

# ---------------------------------------------------------------- 8.17 diverse beliefs III (part II): Negishi weight by bisection
for b17 in (.9, .95):
    def bud17(lmb, b=b17):
        q0, q2 = .5 * lmb + .4 * (1 - lmb), .5 * lmb + .6 * (1 - lmb)          # beta^{-t} q_t on the s=0 and s=2 branches
        c0, c2 = .5 * lmb / q0, .5 * lmb / q2                                   # consumer 1's consumption on each branch
        return lmb - .5 + b / (1 - b) * (q0 * c0 + q2 * (c2 - 1))
    lm = brentq(bud17, 1e-9, 1 - 1e-9)
    print(f"8.17 beta={b17}: lambda = {lm:.10f} (formula (5+b)/(10+b) = {(5 + b17) / (10 + b17):.10f}); "
          f"c1 on s=0 branch {(5 + b17) / (9 + b17):.6f}, on s=2 branch {(5 + b17) / (11 + b17):.6f}; "
          f"Arrow prices at t=0: Q(0)={b17 * (9 + b17) / (2 * (10 + b17)):.6f}, Q(2)={b17 * (11 + b17) / (2 * (10 + b17)):.6f}")

# ---------------------------------------------------------------- 8.20 risk-free rate with log utility
P20 = np.array([[1.0, 0.0], [.5, .5]]); lam20 = np.array([1.0, 1.02]); b20 = .95
pb = b20 * P20 @ (1 / lam20)
print("8.20 bond price by state:", pb, " gross rate R:", 1 / pb)

# ---------------------------------------------------------------- 8.22 heterogeneous beliefs: simulation of the market-selection result
rng = np.random.default_rng(8)
a1, a2, lam22, b22, T22, npath = .4, .6, .5, .95, 150, 20          # consumer 2 knows the truth alpha~ = .6 = Prob(s=0)
ys = np.array([1.0, .5])
KL = a2 * np.log(a2 / a1) + (1 - a2) * np.log((1 - a2) / (1 - a1))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for p in range(npath):
    s = (rng.random(T22 + 1) > a2).astype(int)                       # s_t = 0 with probability .6
    loglr = np.cumsum(np.where(s == 0, np.log(a2 / a1), np.log((1 - a2) / (1 - a1))))   # log L2_t = log pi2_t/pi1_t
    w1 = lam22 / (lam22 + (1 - lam22) * np.exp(loglr))            # consumer 1's consumption share c1/y
    implied = w1 * a1 + (1 - w1) * a2                                # Q~(0|s^t) y(0)/(beta y(s_t)): price-implied prob of s'=0
    ax[0].semilogy(np.arange(T22 + 1), w1, lw=.6, color="0.4")
    ax[1].plot(np.arange(T22 + 1), implied, lw=.6, color="0.4")
    if p == 0:
        print(f"8.22 path 0: consumer-1 share at t=0,50,100,150: {w1[[0, 50, 100, 150]]}; implied prob: {implied[[0, 50, 100, 150]]}")
ax[0].semilogy(np.arange(T22 + 1), np.exp(-KL * np.arange(T22 + 1)), "k--", lw=1.2, label=r"$e^{-t\,\mathrm{ent}}$")
ax[0].set_title("consumer 1's consumption share $c^1_t/y_t$", fontsize=9); ax[0].legend(fontsize=8)
ax[1].axhline(a2, color="k", ls="--", lw=1.2); ax[1].set_ylim(.38, .62)
ax[1].set_title(r"$\tilde Q_t(0|s^t)\,y(0)/[\beta y(s_t)]$ (price-implied Prob$(s'=0)$)", fontsize=9)
for a_ in ax: a_.set_xlabel("t")
plt.tight_layout(); plt.savefig(FIG + "ch8_22.pdf"); plt.close()
print(f"8.22 relative entropy ent(pi, pi^1) = {KL:.6f}")

# ---------------------------------------------------------------- 8.24 recovery in a level (stationary) economy
b24, g24, P24, c24 = .96, 2.0, np.array([[.8, .2], [.15, .85]]), np.array([1.0, 1.2])
Q24 = b24 * P24 * (c24[None, :] / c24[:, None]) ** (-g24)
Q24sq = Q24 @ Q24
r21 = (Q24sq[0, 0] - Q24[0, 0] ** 2) / Q24[0, 1]; r22 = Q24sq[0, 1] / Q24[0, 1] - Q24[0, 0]
print("8.24(b) row 2 recovered from row 1 of Q and Q^2:", [r21, r22], " true:", Q24[1])
x = np.roots([Q24[1, 0], Q24[1, 1] - Q24[0, 0], -Q24[0, 1]]); x = x[x > 0][0]      # x = (c2/c1)^{-gamma}
bh = Q24[0, 0] + Q24[0, 1] / x
print(f"8.24(d) recovered beta={bh:.10f}, P11={Q24[0, 0] / bh:.10f}, P22={Q24[1, 1] / bh:.10f}, x={x:.10f} (true {(c24[1] / c24[0]) ** (-g24):.10f})")
ev, evec = np.linalg.eig(Q24); k = np.argmax(ev.real); z = np.abs(evec[:, k].real)
print(f"        Perron-Frobenius: spectral radius {ev[k].real:.10f}, eigenvector ratio z2/z1 = {z[1] / z[0]:.10f} (= (c2/c1)^gamma = {(c24[1] / c24[0]) ** g24:.10f})")

# ---------------------------------------------------------------- 8.25 recovery in a growth economy
b25, g25, P25, l25 = .96, 2.0, np.array([[.8, .2], [.15, .85]]), np.array([.98, 1.03])
Q25 = b25 * P25 * l25[None, :] ** (-g25)
zinv = np.linalg.solve(Q25, np.ones(2)); Phat = Q25 * zinv[None, :]
print("8.25 P recovered from Q alone:", Phat.ravel(), "; z_j = beta*lambda_j^-gamma:", 1 / zinv, b25 * l25 ** (-g25))
for bb in (.90, .96, .99):     # observationally equivalent (beta, lambda) given gamma
    lalt = (bb * zinv) ** (1 / g25)
    print(f"      beta={bb}: lambda=({lalt[0]:.6f},{lalt[1]:.6f}) reproduces Q exactly: {np.allclose(bb * P25 * lalt[None, :] ** (-g25), Q25)}")
ev, evec = np.linalg.eig(Q25); k = np.argmax(ev.real); z = np.abs(evec[:, k].real)
Pross = Q25 * z[None, :] / (ev[k].real * z[:, None])
print(f"      naive Ross recovery: beta_hat={ev[k].real:.6f}, P_hat rows {Pross[0]}, {Pross[1]} (true beta .96, P {P25.ravel()})")

# ---------------------------------------------------------------- 8.26 risk-free bonds only: observational equivalence
def bonds(b, P11, P22, r, H=4, s=0):
    P = np.array([[P11, 1 - P11], [1 - P22, P22]]); d = np.array([1.0, r])      # d_j = marginal utility c(j)^-gamma relative to state 1
    Q = b * P * d[None, :] / d[:, None]
    return np.array([np.linalg.matrix_power(Q, h)[s].sum() for h in range(1, H + 1)])
pA = bonds(.95, .8, .7, .7); pB = bonds(.95, .7, .8, .8)
print("8.26 bond prices p1..p4, parameter set A:", pA, "\n     parameter set B:", pB, " max diff", np.max(abs(pA - pB)))
e = np.linalg.solve(np.array([[pA[0], -1.0], [pA[1], -pA[0]]]), pA[1:3])   # Prony: p_{h+2} = e1 p_{h+1} - e2 p_h, p_0 = 1
roots = np.roots([1, -e[0], e[1]])
print("     Prony roots (beta, beta*rho):", roots, " predicted p4:", e[0] * pA[2] - e[1] * pA[1], " actual:", pA[3])

# ---------------------------------------------------------------- 8.27 linear and log consumers in a Markov economy
def corners_markov(lamP, delP, y1v, y2v, b=.95, s0=0):
    P = np.array([[lamP, 1 - lamP], [1 - delP, delP]]); Y = y1v + y2v; H = np.linalg.inv(np.eye(2) - b * P)
    c2bar = (1 - b) * (H @ y2v)[s0]
    if c2bar <= Y.min():                                            # Auxiliary Assumption 1: interior, risk-neutral pricing
        c2 = np.full(2, c2bar); regime = "interior"
    else:                                                           # c1 = 0 in the low state (state 0 here, Y[0] < Y[1])
        lo, hi = np.argmin(Y), np.argmax(Y)
        def f(k):
            c2 = np.empty(2); c2[lo] = Y[lo]; c2[hi] = k
            return (H @ (y2v / c2))[s0] - 1 / (1 - b)
        k = brentq(f, Y[lo], Y[hi]); c2 = np.empty(2); c2[lo] = Y[lo]; c2[hi] = k; regime = "corner in low state"
    c1 = Y - c2
    Q = b * P * c2[:, None] / c2[None, :]                            # type 2's Euler equation (log utility)
    M = np.linalg.inv(np.eye(2) - Q)
    return dict(regime=regime, c1=c1, c2=c2, Q=Q, A1=M @ y1v, A2=M @ y2v, a1=M @ (c1 - y1v), a2=M @ (c2 - y2v),
                foc1=c2 / c2.max())   # type 1: beta^t pi_t/(mu_1 q_t) = c2(s)/max c2 must be <= 1, with = 1 where c1 > 0
for aa in (.5, 1.5, 4.0):
    r = corners_markov(.3, .4, np.array([1.0, 1.0]), np.array([0.0, aa]))
    print(f"8.27 y2=(0,{aa}): {r['regime']}; c1={r['c1']}, c2={r['c2']}; Q=\n{r['Q']}\n     type-1 corner check c2/max(c2) = {r['foc1']}\n"
          f"     natural debt limits A1={r['A1']}, A2={r['A2']}; Arrow holdings a1={r['a1']}, a2={r['a2']}")

H27 = np.linalg.inv(np.eye(2) - .95 * np.array([[.3, .7], [.6, .4]]))
print(f"8.27 regime switch (s0 = low state): interior iff a <= a* = 1/((1-beta) H_01) = {1 / (.05 * H27[0, 1]):.6f}")

# ---------------------------------------------------------------- 8.28 inverse problem from one row of Q(1) and Q(2)
lm, dl, b28, g28, yb1, yb2 = .7, .6, .95, 3.0, 1.0, 1.1
P28 = np.array([[lm, 1 - lm], [1 - dl, dl]]); yv = np.array([yb1, yb2])
Q1 = b28 * P28 * (yv[None, :] / yv[:, None]) ** (-g28); Q2 = Q1 @ Q1
a_, b_, c_, d_ = Q1[0, 0], Q1[0, 1], Q2[0, 0], Q2[0, 1]
zr = np.roots([b_, -(d_ - 2 * a_ * b_), -(c_ - a_ ** 2) * b_]); zr = zr[zr > 0][0]      # z = beta(1-lambda)
bet = a_ + zr; lam_h = a_ / bet; del_h = (d_ / b_ - a_) / bet
print(f"8.28 observed row: Q1(1,.)={Q1[0]}, Q2(1,.)={Q2[0]}; recovered beta={bet:.10f}, lambda={lam_h:.10f}, delta={del_h:.10f}")
print(f"     x=(y2/y1)^-gamma: recovered {b_ / zr:.10f}, true {(yb2 / yb1) ** (-g28):.10f}")

# END
