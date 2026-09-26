"""Chapter 22 (Equilibrium without Commitment): numerical parts of exercises 22.3-22.5."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=5, suppress=True, linewidth=130)
np.seterr(invalid="ignore")                                # inf - inf on infeasible grid points is handled explicitly

# ============================================================== 22.3  quadratic utility, endowments 0/2
print("=== 22.3  u(c) = 4c - c^2/2, beta = .8, endowments 0 or 2 (perfectly negatively correlated)")
u3 = lambda c: 4 * c - c ** 2 / 2
up3 = lambda c: 4 - c
b3 = .8
vaut3 = .5 * (u3(2.) + u3(0.)) / (1 - b3)


def best_tau(beta):
    """largest transfer tau <= 1 from the high to the low agent that satisfies the high agent's constraint (memoryless)."""
    va = .5 * (u3(2.) + u3(0.)) / (1 - beta)
    pc = lambda t: u3(2 - t) + beta * .5 * (u3(2 - t) + u3(t)) / (1 - beta) - (u3(2.) + beta * va)
    if pc(1.) >= 0:
        return 1.
    if pc(1e-9) < 0:
        return 0.
    return brentq(pc, 1e-9, 1.)


tau = best_tau(b3)
vbar = .5 * (u3(2 - tau) + u3(tau)) / (1 - b3)
print(f"  (a) v_aut = {vaut3:.4f}; best memoryless transfer tau = {tau:.6f} (closed form 6 beta - 4 = {6*b3-4:.4f}); c_high = {2-tau:.4f}, c_low = {tau:.4f}")
print(f"      value of the contract v = {vbar:.4f}; full insurance would give {u3(1.)/(1-b3):.4f}")
for beta in (.6, 2 / 3, .7, .75, .8, .83, 5 / 6, .9, .99):
    print(f"      beta = {beta:.4f}: tau* = {best_tau(beta):.4f}  (formula min(1, max(0, 6 beta - 4)) = {min(1, max(0, 6*beta-4)):.4f})")

# (b) ex ante constrained Pareto frontier by value iteration, each step solved through its concave dual
ch, cl = 2 - tau, tau                                      # consumption of the high / low agent in the memoryless contract
vmax = (u3(2.) + u3(cl) + b3 * vbar) / (2 - b3)
print(f"  (b) kink slopes at v = {vbar:.2f}: right {-up3(cl)/up3(ch):.4f}, left {-up3(ch)/up3(cl):.4f};  v_max = {vmax:.4f}, P(v_max) = v_aut = {vaut3}")
V = np.linspace(vaut3, u3(2.) / (1 - b3), 1201)             # candidate promised values to agent 1
P = np.where(V <= u3(2.) / (1 - b3), u3(2 - np.clip(4 - np.sqrt(np.maximum(16 - 2 * (1 - b3) * V, 0)), 0, 2)) / (1 - b3), -np.inf)  # first-best frontier (upper bound)
mus = np.r_[np.linspace(0, 3, 3001), np.linspace(3.01, 60, 400)]
Y1 = np.array([2., 0.])                                    # endowment of agent 1 in the two equally likely states
for it in range(400):
    fin = np.isfinite(P)
    W, PW = V[fin], P[fin]
    D = np.zeros(len(mus))
    for y1 in Y1:
        y2 = 2 - y1
        # PC1: u(c) + b w >= u(y1) + b vaut ;  PC2: u(2-c) + b P(w) >= u(y2) + b vaut
        need1 = u3(y1) + b3 * vaut3 - b3 * W                # u(c) >= need1  -> c >= cmin(w)
        need2 = u3(y2) + b3 * vaut3 - b3 * PW               # u(2-c) >= need2 -> c <= cmax(w)
        cmin = np.where(need1 <= u3(0.), 0., 4 - np.sqrt(np.maximum(16 - 2 * need1, 0)))
        cmin = np.where(need1 > u3(2.), np.inf, cmin)
        cmax = np.where(need2 <= u3(0.), 2., 2 - (4 - np.sqrt(np.maximum(16 - 2 * need2, 0))))
        cmax = np.where(need2 > u3(2.), -np.inf, cmax)
        ok = cmin <= cmax + 1e-12
        cint = np.clip((4 * mus - 2) / (1 + mus), 0, 2)      # u'(2-c) = mu u'(c)
        cc = np.clip(cint[:, None], cmin[None, :], cmax[None, :])
        val = u3(2 - cc) + b3 * PW[None, :] + mus[:, None] * (u3(cc) + b3 * W[None, :])
        val = np.where(ok[None, :], val, -np.inf)
        D += .5 * val.max(axis=1)
    Pn = np.min(D[:, None] - mus[:, None] * V[None, :], axis=0)
    # values of v that cannot be delivered: the dual is unbounded below (use a large mu as the test)
    feasible = np.isfinite(Pn) & (Pn > vaut3 - 1e-9)
    Pn = np.where(feasible, Pn, -np.inf)
    d = np.max(np.abs(np.where(np.isfinite(Pn) & np.isfinite(P), Pn - P, 0))) + (0 if np.array_equal(np.isfinite(Pn), np.isfinite(P)) else 1)
    P = Pn
    if d < 1e-9:
        break
fin = np.isfinite(P)
S = V[fin] + P[fin]
imax = np.argmax(S)
print(f"      value iteration on the ex ante frontier: {it+1} steps; domain [{V[fin][0]:.3f}, {V[fin][-1]:.3f}] (theory [15, {vmax:.3f}])")
print(f"      max_v v + P(v) = {S[imax]:.5f} at v = {V[fin][imax]:.4f}  (memoryless contract: 2 v = {2*vbar:.5f});  P(v_bar) = {np.interp(vbar, V[fin], P[fin]):.5f}")
h = V[1] - V[0]
k = np.argmin(np.abs(V - vbar))
print(f"      one-sided slopes of P at v_bar: left {(P[k]-P[k-8])/(8*h):.4f}, right {(P[k+8]-P[k])/(8*h):.4f}")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(V[fin], P[fin], "k", label="constrained frontier $P(v)$")
ax.plot(V, 2 * vbar - V, color="grey", lw=.7, ls=":", label=r"$v+P=2\bar v$")
ax.plot([vbar], [vbar], "ko", ms=4); ax.set_xlim(14.8, 19.3); ax.set_ylim(14.8, 19.3)
ax.set_xlabel("value of agent 1"); ax.set_ylabel("value of agent 2"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch22_03.pdf"); plt.close()

# ============================================================== 22.4  Kehoe-Levine without risk
print("\n=== 22.4  u(c) = (c+b)^(1-gamma)/(1-gamma), beta = .8, b = 5, gamma = 2, eps = .5")
bb, gg, ee = 5., 2., .5
u4 = lambda c: (c + bb) ** (1 - gg) / (1 - gg)
up4 = lambda c: (c + bb) ** (-gg)


def kl(beta):
    hi, lo = 1 + ee, 1 - ee
    vh = (u4(hi) + beta * u4(lo)) / (1 - beta ** 2)
    vl = (u4(lo) + beta * u4(hi)) / (1 - beta ** 2)
    c1 = (1 - beta) * (hi + beta * lo) / (1 - beta ** 2)    # constant CE consumption of the type rich at t = 0
    f = lambda c: u4(c) + beta * u4(2 - c) - (u4(hi) + beta * u4(lo))
    cch = 1. if f(1.) >= 0 else brentq(f, 1., hi - 1e-12)
    R_enf = up4(cch) / (beta * up4(2 - cch))
    return vh, vl, c1, cch, R_enf


b4 = .8
vh, vl, c1, cch, Renf = kl(b4)
print(f"  (a) autarky: v_aut,h = {vh:.6f}, v_aut,l = {vl:.6f}")
print(f"  (b) CE: q_t = beta^t, R = 1/beta = {1/b4:.4f}; c_1 = {c1:.6f}, c_2 = {2-c1:.6f}")
v1ce, v2ce = u4(c1) / (1 - b4), u4(2 - c1) / (1 - b4)
print(f"  (c) CE values v_1 = {v1ce:.6f}, v_2 = {v2ce:.6f}")
print(f"  (d) odd dates: type 2 (high endowment) compares CE {v2ce:.6f} with autarky {vh:.6f} -> prefers autarky: {vh > v2ce};"
      f"  even dates: type 1 {v1ce:.6f} vs {vh:.6f} -> stays: {v1ce > vh}")
vhe = (u4(cch) + b4 * u4(2 - cch)) / (1 - b4 ** 2); vle = (u4(2 - cch) + b4 * u4(cch)) / (1 - b4 ** 2)
print(f"  (f) c-check = {cch:.6f}: v_h = {vhe:.6f} (= v_aut,h {vh:.6f}), v_l = {vle:.6f} (> v_aut,l {vl:.6f}: {vle > vl})")
print(f"      full smoothing c = 1 would give the high agent {(1 + b4) * u4(1.) / (1 - b4 ** 2):.6f} < {vh:.6f}")
print(f"  (g) gross interest rates: complete markets {1/b4:.6f}; with enforcement {Renf:.6f}")
bstar = (u4(1 + ee) - u4(1.)) / (u4(1.) - u4(1 - ee))
print(f"  (h) beta* = (u(1.5) - u(1))/(u(1) - u(.5)) = {bstar:.6f}")
betas = np.linspace(.8, .99, 20)
rows = [kl(x) for x in betas]
for x, r_ in zip(betas[::3], rows[::3]):
    print(f"      beta = {x:.3f}: c-check = {r_[3]:.5f}, R_enf = {r_[4]:.5f}, 1/beta = {1/x:.5f}")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
T = np.arange(8)
even = T % 2 == 0
ax[0].plot(T, np.where(even, 1 + ee, 1 - ee), "k.-", label="autarky")
ax[0].plot(T, np.full(8, c1), "b.--", label="complete markets")
ax[0].plot(T, np.where(even, cch, 2 - cch), "r.-.", label="with enforcement")
ax[0].set_title("consumption of type 1", fontsize=9); ax[0].legend(fontsize=7); ax[0].set_xlabel("t")
ax[1].plot(T, np.where(even, vh, vl), "k.-", label="autarky")
ax[1].plot(T, np.full(8, v1ce), "b.--", label="complete markets")
ax[1].plot(T, np.where(even, vhe, vle), "r.-.", label="with enforcement")
ax[1].plot(T, np.where(even, vl, vh), "k.:", lw=.8); ax[1].plot(T, np.full(8, v2ce), "b:", lw=.8); ax[1].plot(T, np.where(even, vle, vhe), "r:", lw=.8)
ax[1].set_title("continuation values (type 1 thick, type 2 thin)", fontsize=9); ax[1].set_xlabel("t")
ax[2].plot(betas, [r_[4] for r_ in rows], "r.-", label="with enforcement"); ax[2].plot(betas, 1 / betas, "b--", label=r"complete markets $1/\beta$")
ax[2].axvline(bstar, color="grey", lw=.6); ax[2].set_xlabel(r"$\beta$"); ax[2].set_title("gross interest rate", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch22_04.pdf"); plt.close()

# ============================================================== 22.5  the kink: book's example (beta, gamma, y) = (.85, 1.1, .6)
print("\n=== 22.5  u(c) = c^(1-gamma)/(1-gamma), (beta, gamma, y) = (.85, 1.1, .6)")
b5, g5, y5 = .85, 1.1, .6
u5 = lambda c: c ** (1 - g5) / (1 - g5)
up5 = lambda c: c ** (-g5)
vaut5 = .5 * (u5(y5) + u5(1 - y5)) / (1 - b5)
fi = u5(.5) - ((1 - b5 / 2) * u5(y5) + b5 / 2 * u5(1 - y5))
print(f"  (e) full insurance sustainable iff u(.5) >= (1-beta/2) u(y) + (beta/2) u(1-y): difference {fi:.6f} -> {'yes' if fi >= 0 else 'no'}")
pc5 = lambda c: u5(c) + b5 * .5 * (u5(c) + u5(1 - c)) / (1 - b5) - (u5(y5) + b5 * vaut5)
c5 = brentq(pc5, .5 + 1e-12, y5 - 1e-9)
v5 = .5 * (u5(c5) + u5(1 - c5)) / (1 - b5)
ratio = up5(1 - c5) / up5(c5)
R5 = 1 / (.5 * b5 * (1 + ratio))
print(f"  (g) c = {c5:.6f} (book: .536), v = {v5:.6f}, v_aut = {vaut5:.6f}")
print(f"  (i) Arrow prices q(y|y) = {.5*b5:.6f}, q(1-y|y) = {.5*b5*ratio:.6f};  R = {R5:.6f}  vs complete markets 1/beta = {1/b5:.6f}")
cR = brentq(lambda c: 1 / (.5 * b5 * (1 + up5(1 - c) / up5(c))) - 1.0146, .5, .7)
print(f"      the chapter's footnote 14 reports R = 1.0146; with this pricing formula that rate would require c = {cR:.4f}, not {c5:.4f}")
cs = np.linspace(.5, .6, 201)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
ax[0].plot(cs, u5(1 - cs) + b5 * .5 * (u5(cs) + u5(1 - cs)) / (1 - b5), "k"); ax[0].set_xlabel("c")
ax[0].set_title(r"low-endowment agent: $u(1-c)+\beta v(c)$", fontsize=9)
ax[1].plot(cs, [pc5(x) for x in cs], "k"); ax[1].axhline(0, color="grey", lw=.6); ax[1].axvline(c5, color="grey", lw=.6, ls="--")
ax[1].set_xlabel("c"); ax[1].set_title(r"$u(c)+\beta v(c)-[u(y)+\beta v_{aut}]$", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch22_05.pdf"); plt.close()
