"""Chapter 27 (Fiscal-Monetary Theories of Inflation): numerical checks for exercises 27.1-27.8.

27.1  the adjustment factor alpha_t for nominal vs real interest components of the deficit.
27.2  Brock's example: stationary equilibrium, the difference equation for real balances and its instability.
27.3  Ireland's CIA model with alpha = 0, labor taxes and seigniorage: the Ramsey allocation and two implementations.
27.4  shopping-time money demand, the maximal deficit, and the price path of an anticipated open market purchase.
27.5  unpleasant monetarist arithmetic: sign of dp0/dB against the condition e > 1/i.
27.8  growth and seigniorage at a constant price level.
"""
import numpy as np
from scipy.optimize import brentq

np.set_printoptions(precision=6, suppress=True, linewidth=130)

# ------------------------------------------------------------------ 27.1
print("=== 27.1")
R, by = 1.02, .5
for infl in (0., 1.):
    pp = 1 + infl                                                   # p_t / p_{t-1}
    alpha = (R - 1) / (R - 1 / pp)
    nominal = by * (1 - 1 / (R * pp)); real = by * (1 - 1 / R)
    print(f"  inflation {infl:.0%}: alpha = {alpha:.6f}; nominal interest component {nominal:.6f} of GDP, real {real:.6f}, ratio {real / nominal:.6f}")

# ------------------------------------------------------------------ 27.2 Brock
print("\n=== 27.2  Brock: u = ln c + gamma ln(m/p), example beta = .95, gamma = .5, y = 1, g = .2, tau = .1")
beta, gam, y, g, tau, M0 = .95, .5, 1., .2, .1, 1.
c = y - g; x = g - tau
k = x / (gam * c)
Rm = (1 - k) / (1 - k * beta)
h = gam * c / (1 - beta * Rm)
p0 = M0 / (h - x)
print(f"  deficit x = g - tau = {x}; max financeable gamma(y-g) = {gam * c}; Rm = {Rm:.6f}, h = {h:.6f} (fixed point {(gam * c - beta * x) / (1 - beta):.6f}), p0 = {p0:.6f}")
for dh in (1e-3, -1e-3):
    ht = [h + dh]
    for t in range(200):
        nxt = x + (ht[-1] - gam * c) / beta
        ht.append(nxt)
        if nxt <= x or nxt > 1e6:
            break
    print(f"  start {dh:+.0e} from the fixed point: after {len(ht) - 1} periods h = {ht[-1]:.4g}" + ("  (R_m <= 0: not an equilibrium)" if ht[-1] <= x else ""))

# ------------------------------------------------------------------ 27.3 Ireland, alpha = 0
print("\n=== 27.3  CIA economy, u = c^gamma/gamma - n, example gamma = .5, beta = .95, g = .1")
gam3, b3, g3 = .5, .95, .1
f = lambda cc: cc + g3 - b3 * cc ** gam3                          # stationary allocations: c + g = beta c^gamma
cs = np.linspace(1e-9, 1, 200001); fv = f(cs)
roots = [brentq(f, cs[i], cs[i + 1]) for i in np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]]
cL, cH = roots
n = cH + g3
slope = lambda cc: cc ** (1 - gam3) / (gam3 * b3)                  # derivative of the forward map at a fixed point
print(f"  steady states c_low = {cL:.6f} (map slope {slope(cL):.4f}), c_high = {cH:.6f} (slope {slope(cH):.4f}); n = {n:.6f}")
print(f"  effective tax on labor income 1 - c/n = g/n = {g3 / n:.6f}")
tauF = 1 - cH ** (1 - gam3)
print(f"  Friedman rule implementation: tau = {tauF:.6f}, gross inflation pi = beta = {b3}, 1 + i = {cH ** (gam3 - 1) * (1 - tauF):.6f}")
pi0 = b3 * cH ** (gam3 - 1)
print(f"  zero labor tax implementation: pi = {pi0:.6f}; 1 + i = {cH ** (gam3 - 1):.6f}; effective tax 1 - 1/pi = {1 - 1 / pi0:.6f}")
for tau_ in (tauF, .1, 0., -.1):
    pi_ = b3 * (1 - tau_) * cH ** (gam3 - 1)
    print(f"   tau = {tau_:+.4f}: pi = {pi_:.6f}, effective after-tax real wage (1-tau)/pi = {(1 - tau_) / pi_:.6f} (= c/n = {cH / n:.6f})")

# ------------------------------------------------------------------ 27.4
print("\n=== 27.4  m/p = kappa c (1+i)/i, kappa = (1-alpha)/(alpha - eta(1-alpha))")
al, eta = .6, .3
kappa = (1 - al) / (al - eta * (1 - al))
print(f"  example alpha = {al}, eta = {eta}: kappa = {kappa:.6f}; max deficit -> kappa (y - g)")
# (e) anticipated open market purchase: q_t = p_t/p, q_T = 1 + mu, q_t = q_{t+1}/((1-beta) q_{t+1} + beta)
mu, T, b4 = .1, 10, .95
q = [1 + mu]
for t in range(T):
    q.append(q[-1] / ((1 - b4) * q[-1] + b4))
q = q[::-1]                                                        # q_0 ... q_T
print(f"  mu = {mu}, T = {T}, beta = {b4}: price path p_t/p = {np.round(q, 5)}")
print(f"  jump at announcement {q[0] - 1:.5f}; inflation T-1 -> T: {q[-1] / q[-2]:.5f} = 1 + (1-beta) mu = {1 + (1 - b4) * mu:.5f} < 1 + mu")

# ------------------------------------------------------------------ 27.5
print("\n=== 27.5  constant-elasticity money demand f(Rm) = A Rm^e; open market sale dB > 0")
b5 = .96; R5 = 1 / b5
def dp0_sign(e, Rm, A=1.):
    fR = A * Rm ** e; fp = e * fR / Rm
    Sp = fp * (1 - Rm) - fR                                        # slope of seigniorage
    dMp = (1 / R5) * (1 + fp * (R5 - 1) / Sp)                      # d(M0/p0)/dB
    return -np.sign(dMp), Sp
for Rm in (.9, .7):
    i = R5 / Rm - 1
    for e in (.5, 1 / i * .9, 1 / i * 1.1, Rm / (1 - Rm) * .99):
        sgn, Sp = dp0_sign(e, Rm)
        print(f"  Rm = {Rm}, i = {i:.4f} (1/i = {1/i:.4f}, good-side bound Rm/(1-Rm) = {Rm/(1-Rm):.4f}): e = {e:.4f}: S'(Rm) = {Sp:+.4f},"
              f" p0 {'rises' if sgn > 0 else 'falls'}")

# ------------------------------------------------------------------ 27.8
print("\n=== 27.8  seigniorage at a constant price level with growth: (M_{t+1}-M_t)/p_t = h_t (1 - 1/gamma)")
for gr in (1.02, 1.05, 1.10):
    print(f"  gamma = {gr}: seigniorage / real balances = {1 - 1 / gr:.4f}")
