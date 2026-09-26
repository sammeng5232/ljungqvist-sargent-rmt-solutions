"""Chapter 16 (Optimal Taxation with Commitment): numerical parts of exercises 16.5, 16.11, 16.12 and 16.13.

16.11 and 16.12 leave the utility function and beta open; we use (our choices)
  16.11: u = c + H(l), H(l) = kappa*log(l), kappa = .2, beta = .95, b0 = .2 (consumption tax only);
  16.12: u = kappa*log(c) + l, kappa = .8, beta = .95, b0 = .2 (labor tax only), with the 7-state chain needed for draws at t = 2 AND 3;
  16.13: u = log(c) + kappa*l as in section 16.13.4, kappa = 2.5, beta = .95, T = 10.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=6, suppress=True, linewidth=130)
beta = .95


def largest_root(f, lo, hi, n=4000):
    """Largest root of f on (lo, hi): the good side of the Laffer curve (high consumption, low tax)."""
    xs = np.linspace(hi, lo, n); fs = np.array([f(x) for x in xs])
    k = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0][0]
    return brentq(f, xs[k + 1], xs[k])


def smallest_root(f, lo, hi, n=4000):
    """Smallest root of f on (lo, hi): the lowest tax rate that raises the required revenue."""
    return -largest_root(lambda x: f(-x), -hi, -lo, n)


def debt_by_state(surp, P):
    """Debt owed on entering each state = expected PV of primary surpluses, Arrow prices beta*P: (I - beta P)^{-1} surp."""
    return np.linalg.solve(np.eye(len(surp)) - beta * P, surp)


# ================================================================== 16.5
print("=== 16.5  u = c - .5(1-x)^2, tau(1-tau) = (1-beta) PV(g)")
for lab, pv in (("(c) g = 0,.2,0,.2,...", .2 * beta / (1 - beta ** 2)), ("(d) g = .2,0,.2,0,...", .2 / (1 - beta ** 2))):
    rhs = (1 - beta) * pv
    tau = (1 - np.sqrt(1 - 4 * rhs)) / 2
    print(f"  {lab}: (1-beta)PV(g) = {rhs:.6f}, tau = {tau:.6f} (other root {1 - tau:.6f} is on the wrong side of the Laffer curve)")

# ================================================================== 16.11  consumption tax, u = c + kappa log l
print("\n=== 16.11  u = c + .2 log(l), consumption tax, beta = .95")
kap = .2
gL, gH = .1, .2
# states 1..5: (0,L),(1,L),(2,L),(2,H),(>=3,L); index 0..4
P11 = np.zeros((5, 5)); P11[0, 1] = 1; P11[1, 2] = P11[1, 3] = .5; P11[2, 4] = P11[3, 4] = 1; P11[4, 4] = 1
g11 = np.array([gL, gL, gL, gH, gL])
nlab = lambda tau: 1 - kap * (1 + tau)                 # H'(1-n) = 1/(1+tau)
print("  P =\n", P11)
# (e) balanced budget, b0 = 0: tau*c = g with c = n(tau) - g
tau_bb = {}
for g in (gL, gH):
    tau_bb[g] = brentq(lambda t: t * (nlab(t) - g) - g, 0, 1.0)
    print(f"  (e) g = {g}: tau = {tau_bb[g]:.6f}, n = {nlab(tau_bb[g]):.6f}, c = {nlab(tau_bb[g]) - g:.6f}")
taus_e = np.array([tau_bb[g] for g in g11])
Q_e = beta * P11 * (1 + taus_e)[:, None] / (1 + taus_e)[None, :]
print("  (e) Arrow prices Q(s'|s) =\n", Q_e, "\n      risk-free gross rates by state:", 1 / Q_e.sum(1))
b0 = .2
# (f) roll over b: tau_s c_s + sum_s' Q(s'|s) b = g_s + b, Q = beta P (1+tau_s)/(1+tau_s'); solve backward
def rollover11(b):
    tf = np.zeros(5)
    tf[4] = smallest_root(lambda t: t * (nlab(t) - gL) + beta * b - gL - b, -.5, 3)
    for s in (3, 2, 1, 0):
        succ_inv = P11[s] @ (1 / (1 + tf))
        tf[s] = smallest_root(lambda t: t * (nlab(t) - g11[s]) + beta * (1 + t) * succ_inv * b - g11[s] - b, -.5, 3)
    return tf, beta * P11 * (1 + tf)[:, None] / (1 + tf)[None, :]
for b in (0., .1, .2):
    tf_b, Qf_b = rollover11(b)
    print(f"  (f) b0 = {b}: taxes by state {tf_b}; gross return on government debt by state {1 / Qf_b.sum(1)}")
tf, Qf = rollover11(b0)
# (g) constant tax: Q = beta P; budget sum beta^t E[tau c - g] = b0
PVg = gL + beta * gL + beta ** 2 * .15 + beta ** 3 * gL / (1 - beta)
def pv_surplus(t):
    n = nlab(t)
    PVc = n / (1 - beta) - PVg
    return t * PVc - PVg
tbar = brentq(lambda t: pv_surplus(t) - b0, 0, 1.5)
n_g = nlab(tbar)
surp = tbar * (n_g - g11) - g11                          # primary surplus by state
B = debt_by_state(surp, P11)
print(f"  (g) constant tax tau = {tbar:.6f}; n = {n_g:.6f}; primary surplus by state {surp};  debt owed on entering each state B = {B} (B[0] = b0: {B[0]:.6f})")
print(f"  (h) history 1,2,3,5,...: government pays {B[2]:.6f} at t=2, then {B[4]:.6f} per period rolled over;"
      f" history 1,2,4,5,...: pays {B[3]:.6f} at t=2, then {B[4]:.6f}")
# (i) Ramsey: l constant for t >= 1; l0 from the t = 0 condition; Phi from the implementability condition
H1, H2 = (lambda l: kap / l), (lambda l: -kap / l ** 2)
def l_bar(Phi):
    return brentq(lambda l: -1 + H1(l) + Phi * (-1 - H2(l) * (1 - l) + H1(l)), 1e-6, 1 - 1e-9)
def ramsey11(b, g_first, PVg_rest):
    """Ramsey plan started with debt b; g_first = g at the initial date, PVg_rest = sum_{j>=1} beta^j E g."""
    def l_zero(Phi):
        return brentq(lambda l: -1 + H1(l) + Phi * (-1 - H2(l) * (1 - l) + H1(l)) - Phi * H2(l) * b, 1e-6, 1 - 1e-9)
    def impl(Phi):
        lb, l0 = l_bar(Phi), l_zero(Phi)
        return (1 - l0) * (1 - H1(l0)) - g_first + beta / (1 - beta) * (1 - lb) * (1 - H1(lb)) - PVg_rest - H1(l0) * b
    Phi = brentq(impl, 1e-8, 5)
    return Phi, l_bar(Phi), l_zero(Phi)
Phi, lb, l0 = ramsey11(b0, gL, PVg - gL)
print(f"  (i) Ramsey: Phi = {Phi:.6f}; l_t = {lb:.6f} for t >= 1 (tax {lb/kap - 1:.6f}), l_0 = {l0:.6f} (tax {l0/kap - 1:.6f}); constant-tax policy of (g): {tbar:.6f}")
tR = lb / kap - 1; t0R = l0 / kap - 1
BR = debt_by_state(tR * (1 - lb - g11) - g11, P11)      # debt entering each state t >= 1 under the Ramsey plan
chk = t0R * (1 - l0 - gL) - gL + beta * (1 + t0R) / (1 + tR) * BR[1]
print(f"      Ramsey debt owed on entering states 2..5: {BR[1:]};  time-0 budget check: {chk:.6f} = b0")
U = lambda l, g: 1 - l - g + kap * np.log(l)
def welfare(tax_by_state):                               # expected discounted utility of a state-dependent tax policy
    l = kap * (1 + tax_by_state)
    return np.linalg.solve(np.eye(5) - beta * P11, U(l, g11))[0]
tram = np.array([t0R] + [tR] * 4)
print(f"  (j) welfare: Ramsey {welfare(tram):.6f}, constant tax {welfare(np.full(5, tbar)):.6f}, roll-over policy (f) {welfare(tf):.6f}")
# (k) a Ramsey planner re-optimizing at t = 1 in state (1,L), inheriting the debt BR[1]
Phi1, lb1, l01 = ramsey11(BR[1], gL, beta * .15 + beta ** 2 * gL / (1 - beta))
print(f"  (k) time-1 Ramsey plan with b1 = {BR[1]:.6f}: Phi = {Phi1:.6f}, tax at t = 1 {l01/kap - 1:.6f} (continuation prescribes {tR:.6f}),"
      f" later taxes {lb1/kap - 1:.6f}")

# ================================================================== 16.12  labor tax, u = kappa log(c) + l
k12 = .8
print(f"\n=== 16.12  u = {k12} log(c) + l, labor tax, beta = .95; 7 states (0,L),(1,L),(2,L),(2,H),(3,L),(3,H),(>=4,L)")
P12 = np.zeros((7, 7)); P12[0, 1] = 1; P12[1, 2] = P12[1, 3] = .5; P12[2, 4] = P12[2, 5] = .5; P12[3, 4] = P12[3, 5] = .5
P12[4, 6] = P12[5, 6] = 1; P12[6, 6] = 1
g12 = np.array([gL, gL, gL, gH, gL, gH, gL])
print("  P =\n", P12)
# household: 1 = u'(c)(1 - tau)  ->  c = kappa (1 - tau);  Arrow prices Q(s'|s) = beta P c(s)/c(s')
# (e) balanced budget: tau (c + g) = g  ->  tau = g/kappa, c = kappa - g, n = kappa
for g in (gL, gH):
    print(f"  (e) g = {g}: tau = g/kappa = {g / k12:.6f}, c = {k12 - g:.6f}, n = kappa = {k12}")
c_e = k12 - g12
Q12 = beta * P12 * c_e[:, None] / c_e[None, :]
print("  (e) risk-free gross rates by state:", np.round(1 / Q12.sum(1), 6))
# (f) roll over b: tau n + sum_s' Q(s'|s) b = g + b, tau = 1 - c/kappa, n = c + g
def rollover12(b):
    resid = lambda c, s, succ: (1 - c / k12) * (c + g12[s]) + beta * c * succ * b - g12[s] - b   # succ = sum P/c'
    cf = np.full(7, np.nan)
    cf[6] = largest_root(lambda c: resid(c, 6, 1 / c), 1e-4, k12 - 1e-6)
    for s in (5, 4, 3, 2, 1, 0):
        nxt = P12[s] > 0                                  # only successors with positive probability
        succ = P12[s][nxt] @ (1 / cf[nxt])
        cf[s] = largest_root(lambda c: resid(c, s, succ), 1e-4, k12 - 1e-6)
    return cf, beta * P12 * cf[:, None] / cf[None, :]
for b in (0., .1, .2):
    cf_b, Qf_b = rollover12(b)
    print(f"  (f) b0 = {b}: taxes by state {np.round(1 - cf_b / k12, 6)}; gross return on debt by state {np.round(1 / Qf_b.sum(1), 6)}")
cf, Qf12 = rollover12(b0)
# (g) constant tax: c = kappa(1 - tau) constant, Q = beta P; budget sum beta^t E[tau (c + g) - g] = b0
PVg12 = gL + beta * gL + beta ** 2 * .15 + beta ** 3 * .15 + beta ** 4 * gL / (1 - beta)
tb12 = brentq(lambda t: t * (k12 * (1 - t) / (1 - beta) + PVg12) - PVg12 - b0, 0, .5)
cg = k12 * (1 - tb12)
surp12 = tb12 * (cg + g12) - g12
B12 = debt_by_state(surp12, P12)
print(f"  (g) constant tax tau = {tb12:.6f}, c = {cg:.6f}; primary surplus by state {np.round(surp12, 6)}")
print(f"      debt owed on entering each state: {np.round(B12, 6)} (B[0] = b0: {B12[0]:.6f})")
# (i) Ramsey: c = kappa/(1+Phi), tau = Phi/(1+Phi) for t >= 1; (1+Phi) c0^2 - kappa c0 - Phi kappa b = 0
def ramsey12(b, g_first, PVg_rest):
    c0f = lambda Phi: (k12 + np.sqrt(k12 ** 2 + 4 * (1 + Phi) * Phi * k12 * b)) / (2 * (1 + Phi))
    def impl(Phi):
        c, c0 = k12 / (1 + Phi), c0f(Phi)
        return (k12 - c0 - g_first) + beta * (k12 - c) / (1 - beta) - PVg_rest - k12 * b / c0
    Phi = brentq(impl, 1e-8, 5)
    return Phi, k12 / (1 + Phi), c0f(Phi)
Phi12, c12, c012 = ramsey12(b0, gL, PVg12 - gL)
t12 = Phi12 / (1 + Phi12); t012 = 1 - c012 / k12
print(f"  (i) Ramsey: Phi = {Phi12:.6f}; c = {c12:.6f}, tau = {t12:.6f} for t >= 1; c0 = {c012:.6f}, tau0 = {t012:.6f};"
      f" constant-tax policy of (g): {tb12:.6f}")
BR12 = debt_by_state(t12 * (c12 + g12) - g12, P12)
chk = t012 * (c012 + gL) - gL + beta * c012 / c12 * BR12[1]
print(f"      Ramsey debt owed on entering states 2..7: {np.round(BR12[1:], 6)}; time-0 budget check {chk:.6f} = b0;"
      f" gross interest 0->1: {c12 / (beta * c012):.6f} (1/beta = {1/beta:.6f})")
def welfare12(tax):                                      # tax by state -> expected discounted utility
    c = k12 * (1 - tax); u = k12 * np.log(c) + 1 - (c + g12)
    return np.linalg.solve(np.eye(7) - beta * P12, u)[0]
W12 = {"Ramsey": welfare12(np.r_[t012, np.full(6, t12)]), "constant": welfare12(np.full(7, tb12)), "rollover": welfare12(1 - cf / k12)}
print(f"  (j) welfare: Ramsey {W12['Ramsey']:.6f}, constant tax {W12['constant']:.6f}, roll-over policy (f) {W12['rollover']:.6f}")
Phi1, c1b, c01 = ramsey12(BR12[1], gL, beta * .15 + beta ** 2 * .15 + beta ** 3 * gL / (1 - beta))
print(f"  (k) time-1 Ramsey plan with b1 = {BR12[1]:.6f}: Phi = {Phi1:.6f}, tax at t = 1 {1 - c01/k12:.6f} (continuation prescribes {t12:.6f}),"
      f" later taxes {Phi1/(1+Phi1):.6f}")

# ================================================================== 16.13  u = log c + kappa*l, examples 1-3 with b0 = 0 and b0 > 0
print("\n=== 16.13  u = log(c) + 2.5 l, beta = .95")
k13 = 2.5
def ramsey13(Eg_path, g0, b0_):
    """Eg_path(t) = expected g_t for t >= 1 (a function); returns Phi, c (t >= 1), c0, taxes."""
    PVg1 = sum(beta ** t * Eg_path(t) for t in range(1, 2000))
    def c0f(Phi):
        a = (1 + Phi) * k13
        return (1 + np.sqrt(1 + 4 * a * Phi * b0_)) / (2 * a)     # (1+Phi) kappa c0^2 - c0 - Phi b0 = 0
    def f(Phi):
        c = 1 / ((1 + Phi) * k13); c0 = c0f(Phi)
        lhs = (1 - k13 * (c0 + g0)) + (beta / (1 - beta)) * (1 - k13 * c) - k13 * PVg1
        return lhs - b0_ / c0
    Phi = brentq(f, 0, 50) if f(0) < 0 else 0.0
    c = 1 / ((1 + Phi) * k13); c0 = c0f(Phi)
    return Phi, c, c0, 1 - k13 * c, 1 - k13 * c0
T = 10
examples = {"1: g_t = .1": (lambda t: .1, .1), "2: g_T = .3 only": (lambda t: .3 if t == T else 0., 0.),
            "3: g_T = .3 w.p. .5": (lambda t: .15 if t == T else 0., 0.)}
paths = {}
for name, (Eg13, g0) in examples.items():
    for b0_ in (0., .1):
        Phi, c, c0, tau, tau0 = ramsey13(Eg13, g0, b0_)
        R0 = c / (beta * c0)
        print(f"  example {name}, b0 = {b0_}: Phi = {Phi:.5f}; c = {c:.5f}, tau = {tau:.5f} (t >= 1); c0 = {c0:.5f}, tau0 = {tau0:.5f};"
              f" gross interest 0->1 = {R0:.5f} (1/beta = {1/beta:.5f})")
        # debt owed at the start of t >= 1 (prices beta from t = 1 on): b_t = tau c/(1-beta) - (1-tau) sum_j beta^(j-t) E_t g_j
        tt = np.arange(1, 2 * T + 1)
        pvg = np.array([sum(beta ** (j - t) * Eg13(j) for j in range(t, 3000)) for t in tt])
        bt = tau * c / (1 - beta) - (1 - tau) * pvg
        chk = tau0 * (c0 + g0) - g0 + beta * c0 / c * bt[0]
        bH = bL = None; extra = ""
        if name.startswith("3"):
            bH = tau * c / (1 - beta) - (1 - tau) * .3          # g_T = .3 realized, nothing afterwards
            bL = tau * c / (1 - beta)
            extra = f"; at T: b_T(g=.3) = {bH:.5f}, b_T(g=0) = {bL:.5f}"
        print(f"      surplus t>=1, t != T: {tau * c:.5f}; debt b_1..b_(T+1) = {np.round(bt[:T + 1], 4)}; time-0 budget check {chk:.6f} = b0{extra}")
        paths[(name[0], b0_)] = (tt, bt, bH, bL)

# ================================================================== figure: 16.12 taxes by state; 16.13 debt paths
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
labs = ["(0,L)", "(1,L)", "(2,L)", "(2,H)", "(3,L)", "(3,H)", "(4+,L)"]
xs = np.arange(7); w = .2
pols = [(r"balanced (e), $b_0=0$", g12 / k12), ("roll-over (f)", 1 - cf / k12), ("constant (g)", np.full(7, tb12)),
        ("Ramsey (i)", np.r_[t012, np.full(6, t12)])]
for i, (lab, tx) in enumerate(pols):
    ax[0].bar(xs + (i - 1.5) * w, tx, w, label=lab, color=str(.15 + .22 * i), edgecolor="k", lw=.4)
ax[0].set_xticks(xs); ax[0].set_xticklabels(labs, fontsize=8); ax[0].set_ylabel(r"labor tax $\tau^n$")
ax[0].legend(fontsize=7); ax[0].set_title(r"16.12: tax by state, $u=.8\ln c+\ell$, $b_0=.2$", fontsize=9)
for (ex, b), ls in ((("2", 0.), "-"), (("2", .1), "--"), (("3", 0.), "-"), (("3", .1), "--")):
    tt, bt, bH, bL = paths[(ex, b)]
    col = "k" if ex == "2" else "0.55"
    if ex == "2":
        ax[1].plot(np.r_[0, tt], np.r_[b, bt], ls, color=col, label=f"example 2, $b_0={b}$")
    else:
        pre = tt <= T - 1
        ax[1].plot(np.r_[0, tt[pre]], np.r_[b, bt[pre]], ls, color=col, label=f"example 3, $b_0={b}$")
        for bT in (bH, bL):                                   # the two date-T states, then b_t = tau c/(1-beta) for t > T
            ax[1].plot([T - 1, T, T + 1], [bt[T - 2], bT, bL], ":", color=col, lw=.9, marker=".", ms=4)
        ax[1].plot([T + 1, 2 * T], [bL, bL], ls, color=col)
ax[1].axhline(0, color="grey", lw=.5); ax[1].axvline(T, color="grey", lw=.5, ls=":")
ax[1].set_xlabel("t"); ax[1].set_ylabel("government debt $b_t$"); ax[1].legend(fontsize=7); ax[1].set_xticks(range(0, 2 * T + 1, 2))
ax[1].set_title(r"16.13: Ramsey debt, $u=\ln c+2.5\ell$, $g_T$ at $T=10$", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch16_01.pdf"); plt.close()
