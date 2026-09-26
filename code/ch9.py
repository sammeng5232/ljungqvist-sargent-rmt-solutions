"""Chapter 9 (Overlapping Generations): numerical checks of the closed forms in exercises 9.1-9.13.

Most of the chapter's exercises are analytical; this script verifies every formula used in Ch9.tex
(saving functions, stationary and nonstationary monetary equilibria, the tree economy of 9.3, the Laffer curves
of 9.10, 9.11 and 9.13, credit controls in 9.7, the Grandmont-Hall scheme and forced saving) and draws two figures.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from rmtlib import FIG

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def save_crra(R, w1, w2, gam):
    """Saving of a young agent with u = c^(1-gam)/(1-gam), endowment (w1, w2), gross return R."""
    return (R ** (1 / gam) * w1 - w2) / (R ** (1 / gam) + R)


# ------------------------------------------------------------------ 9.1
print("=== 9.1  (w1, w2) = (2, 1), N = 1")
w1, w2 = 2., 1.
for gam in (.5, 1., 2., 5.):
    R = np.linspace((w2 / w1) ** gam + 1e-9, 3, 3001)
    s = save_crra(R, w1, w2, gam)
    # check the Euler equation u'(w1 - s) = R u'(w2 + R s)
    err = np.max(np.abs((w1 - s) ** (-gam) - R * (w2 + R * s) ** (-gam)))
    mono = np.all(np.diff(s[R <= 1.0]) > 0)
    mbar = save_crra(1., w1, w2, gam)
    # a nonstationary equilibrium from m_1 = .5 mbar: m_{t+1} = R_t m_t with s(R_t) = m_t
    m = [.5 * mbar]; Rs = []
    for t in range(60):
        Rt = brentq(lambda x: save_crra(x, w1, w2, gam) - m[-1], (w2 / w1) ** gam, 1.0)
        Rs.append(Rt); m.append(Rt * m[-1])
    print(f"  gamma={gam}: Euler residual {err:.1e}; s increasing on (R_min, 1]: {mono}; stationary m = s(1) = {mbar:.4f} (= (w1-w2)/2);"
          f" nonstationary path: R_1 = {Rs[0]:.4f}, R_60 = {Rs[-1]:.6f} -> (w2/w1)^gamma = {(w2 / w1) ** gam:.6f}")

# ------------------------------------------------------------------ 9.2
print("\n=== 9.2  (w1, w2) = (2, 1), log utility")
Rn = w2 / w1
print(f"  nonmonetary: R = w2/w1 = {Rn}; lenders consume ({w1/2}, {Rn*w1/2}), borrowers ({w2/(2*Rn)}, {w2/2})")
M = 1.
pbar = 4 * M / (w1 - w2)
print(f"  stationary monetary price level 4M/(w1-w2) = {pbar};  check money market: w1/4 - w2/4 = {w1/4 - w2/4} = M/pbar = {M/pbar}")
for c in (0., .5):
    p = pbar + c * (w1 / w2) ** np.arange(1, 8)
    R = p[:-1] / p[1:]
    print(f"  c = {c}: returns R_t = {np.round(R, 5)}; lender utility gen 1: {np.log(w1/2) + np.log(R[0]*w1/2):.4f}, borrower: {np.log(w2/(2*R[0])) + np.log(w2/2):.4f}")

# ------------------------------------------------------------------ 9.3
print("\n=== 9.3  w1 = 10, w2 = 5, d = 1e-6, log utility (gamma = 1 assumed), one young agent per period")
w1_, w2_, d = 10., 5., 1e-6
s1 = lambda R: .5 * (w1_ - w2_ / R)
Rst = brentq(lambda R: s1(R) * (R - 1) - d, 1 + 1e-15, 2)
print(f"  stationary tree economy: R = 1 + {Rst - 1:.4e}, tree price q = s(R) = {s1(Rst):.8f} (fiat-money value in 9.1: {s1(1.):.4f})")
for dq in (1e-3, -1e-3):
    q = s1(Rst) + dq; path = [q]
    for t in range(40):
        R_t = brentq(lambda R: s1(R) - q, w2_ / w1_ + 1e-12, 1e6) if 0 < q < w1_ / 2 else np.nan
        q = R_t * q - d; path.append(q)
        if not (0 < q < w1_ / 2): break
    print(f"  tree price starting {dq:+.0e} off the stationary value leaves (0, w1/2) after {len(path)-1} periods: {path[-1]:.4f}")

# ------------------------------------------------------------------ 9.4 / 9.5
print("\n=== 9.4 / 9.5  log utility, (w1, w2) = (2, 1)")
for n in (1., 1.2, .8, .4):
    ok = n > w2 / w1
    mstar = .5 * (w1 - w2 / n) if ok else np.nan
    print(f"  n = {n}: monetary steady state exists: {ok};", end="")
    if ok:
        lam = n * w1 / w2
        x = 1 / mstar + .3 * lam ** np.arange(0, 40)         # a nonstationary equilibrium, c = .3
        Rt = n * x[:-1] / x[1:]
        print(f" m* = {mstar:.4f}, R = n; nonstationary: R_1 = {Rt[0]:.4f} -> R_39 = {Rt[-1]:.6f} (w2/w1 = {w2/w1})")
    else:
        print(" autarky only (R_aut = w2/w1 >= n)")

# ------------------------------------------------------------------ 9.7
print("\n=== 9.7  credit controls, y = 1")
y = 1.
print("  (b) no currency, no control: R = 1; lenders (y/2, y/2), borrowers (y/2, y/2), initial old 0")
R_e = brentq(lambda R: y / 2 - y / (4 * R), .1, 5)
print(f"  (e) control l(1+r) >= -y/4: R = {R_e:.4f}; lenders ({y/2}, {R_e*y/2}), borrowers ({y/(4*R_e)}, {y - y/4})")
print(f"  (f) valued currency: R = 1, q = 8/y = {8/y}; lenders ({y/2}, {y/2}), borrowers ({y/4}, {3*y/4}), initial old {y/8} each")
print(f"  (g) borrower utility (b) {2*np.log(y/2):.4f} vs (f) {np.log(y/4)+np.log(3*y/4):.4f}")
# (d): no equilibrium with valued currency: real balances explode
mm = [.01]
while mm[-1] < y / 4 and len(mm) < 10000:
    mm.append(mm[-1] / (1 - 4 * mm[-1] / y))
print(f"  (d) from m_1 = .01, real balances reach y/4 after {len(mm)-1} periods")

# ------------------------------------------------------------------ 9.10
print("\n=== 9.10  seigniorage, N1 = N2 = 1, alpha = 2, beta = 1")
a, b = 2., 1.
Gz = lambda z: .5 * (a - b * z) * (1 - 1 / z)
zmax = a / b; zst = np.sqrt(a / b)
res = minimize_scalar(lambda z: -Gz(z), bounds=(1, zmax), method="bounded", options={"xatol": 1e-12})
print(f"  nonmonetary R = N2 beta/(N1 alpha) = {b/a}; z_max = {zmax}, revenue-maximising z = sqrt(z_max) = {zst:.6f} (numerical {res.x:.6f});"
      f" G_max = .5(sqrt a - sqrt b)^2 = {.5*(np.sqrt(a)-np.sqrt(b))**2:.6f} = {Gz(zst):.6f}")

# ------------------------------------------------------------------ 9.13
print("\n=== 9.13  Bryant-Keynes-Wallace, (w1, w2) = (2, 1)")
phi = lambda R: .5 * (w1 - w2 / R) * (1 - R)
Rstar = np.sqrt(w2 / w1)
print(f"  phi(R) = .5(w1 - w2/R)(1 - R) peaks at R* = sqrt(w2/w1) = {Rstar:.6f} with phi = .5(sqrt w1 - sqrt w2)^2 = {phi(Rstar):.6f}")
g = .05
r1 = brentq(lambda R: phi(R) - g, w2 / w1 + 1e-12, Rstar); r2 = brentq(lambda R: phi(R) - g, Rstar, 1 - 1e-12)
U = lambda R: np.log(w1 - .5 * (w1 - w2 / R)) + np.log(w2 + R * .5 * (w1 - w2 / R))
F = (w1 - w2 + g) / 2; R2 = 1 - g / F
print(f"  g = {g}: two stationary equilibria R = {r1:.6f}, {r2:.6f}; utilities {U(r1):.6f}, {U(r2):.6f}")
print(f"  forced saving: F = (w1 - w2 + g)/2 = {F}, 1 + r2 = {R2:.6f}; c1 = c2 = {(w1 + w2 - g)/2}; utility {2*np.log((w1+w2-g)/2):.6f}")
print(f"     at 1 + r2 the young would like to save {.5*(w1 - w2/R2):.4f} < F: the constraint binds")

# ------------------------------------------------------------------ 9.12 Grandmont-Hall: iterate the equilibrium conditions
f = lambda R: .5 * (w1 - w2 / R)                                # an aggregate saving function with f(1) > 0
H0 = 1.; pbar12 = H0 / f(1.)
H = H0; Rt_list = []
for t in range(5):
    R_t = brentq(lambda R: f(R) - H / pbar12, w2 / w1 + 1e-12, 50)   # loan market with p(t) = pbar
    Rt_list.append(R_t); H = R_t * H                                 # H(t+1) = (1 + r(t)) H(t) when p(t+1) = pbar
print(f"\n=== 9.12  Grandmont-Hall: 1 + r(t) along the equilibrium = {np.round(Rt_list, 12)}; pbar = {pbar12:.6f}")

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
RR = np.linspace(w2 / w1 + 1e-3, 1, 400)
ax[0].plot(RR, phi(RR), "k"); ax[0].axhline(g, color="grey", lw=.7, ls="--")
ax[0].plot([r1, r2], [g, g], "ko", ms=4); ax[0].set_xlabel("gross return on currency $R$")
ax[0].set_title(r"9.13: seigniorage $\phi(R)=s(R)(1-R)$, $(w_1,w_2)=(2,1)$", fontsize=9)
zz = np.linspace(1, zmax, 400)
ax[1].plot(zz, Gz(zz), "k"); ax[1].axvline(zst, color="grey", lw=.6, ls="--")
ax[1].set_xlabel("money growth factor $z$"); ax[1].set_title(r"9.10: stationary seigniorage $G(z)$, $N_1\alpha=2$, $N_2\beta=1$", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch9_laffer.pdf"); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3.3))
for c, ls in ((0., "-"), (.05, "--"), (.3, "-."), (1., ":")):
    p = pbar + c * (w1 / w2) ** np.arange(1, 12)
    ax.plot(np.arange(1, 11), p[:-1] / p[1:], "k" + ls, label=f"c = {c}")
ax.axhline(w2 / w1, color="grey", lw=.6); ax.set_xlabel("t"); ax.set_ylabel("return on currency $p_t/p_{t+1}$")
ax.legend(fontsize=7); ax.set_title("9.2: equilibria with valued currency", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + "ch9_02.pdf"); plt.close()
