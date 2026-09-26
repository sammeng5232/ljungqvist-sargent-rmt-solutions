"""Chapter 28 (Credit and Currency): numerical checks for the Townsend-turnpike exercises, log utility unless stated.

28.1  Arrow-Debreu with period-3 endowments.
28.6  initial distribution of currency: the equilibrium that is stationary from t = 1.
28.7  storage versus currency.
28.9  inside money: the four regimes in F; 28.10 the date-0 transition; 28.11 the real-bills experiment.
"""
import numpy as np
from scipy.optimize import brentq

beta = .9
c0 = 1 / (1 + beta)                                   # root of beta = u'(c0)/u'(1-c0) with log utility
print(f"log utility, beta = {beta}: c0 = 1/(1+beta) = {c0:.6f}, 1 - c0 = {1 - c0:.6f}")

# ------------------------------------------------------------------ 28.1
co = (1 + beta) / (1 + beta + beta ** 2); ce = beta ** 2 / (1 + beta + beta ** 2)
print(f"\n=== 28.1  c^o = (1+b)/(1+b+b^2) = {co:.6f}, c^e = b^2/(1+b+b^2) = {ce:.6f}")
PVo = (1 + beta) / (1 - beta ** 3); PVe = beta ** 2 / (1 - beta ** 3)
print(f"  budget check: c^o/(1-b) = {co / (1 - beta):.6f} = PV(y^o) = {PVo:.6f};  c^e/(1-b) = {ce / (1 - beta):.6f} = PV(y^e) = {PVe:.6f}")

# ------------------------------------------------------------------ 28.6
print("\n=== 28.6  H = 1 (money per odd-even pair)")
H = 1.
pbar = H * (1 + beta) / beta
for al in (0., .25, .5, 1.):
    p0 = H * (1 + beta * (1 - al)) / beta
    co0 = 1 / (1 + beta * (1 - al))
    # checks: odd Euler 1/(co0 p0) = beta (1+beta)/(beta pbar); odd budget p0 co0 + H = p0 + al H
    e1 = 1 / (co0 * p0) - (1 + beta) / pbar; e2 = p0 * co0 + H - p0 - al * H
    ok_even = 1 / ((1 - al) * H) >= beta ** 2 / H if al < 1 else True
    print(f"  alpha = {al}: p0 = {p0:.6f} (pbar = {pbar:.6f}), c^o_0 = {co0:.6f}, c^e_0 = {1 - co0:.6f}; residuals {e1:.1e}, {e2:.1e}; even Euler ok {ok_even}")

# ------------------------------------------------------------------ 28.7
print("\n=== 28.7  endowments (1-eps, eps), storage return delta")
eps, delta = .05, .8
k = (beta * delta * (1 - eps) - eps) / (delta * (1 + beta))
cH_a = 1 - eps - k; cL_a = eps + delta * k
cH_m, cL_m = 1 / (1 + beta), beta / (1 + beta)
U = lambda a, b: (np.log(a) + beta * np.log(b)) / (1 - beta ** 2)
print(f"  autarky: store k = {k:.6f}; c_H = {cH_a:.6f}, c_L = {cL_a:.6f} (c_L/c_H = {cL_a / cH_a:.4f} = beta delta)")
print(f"  money:   c_H = {cH_m:.6f}, c_L = {cL_m:.6f}; real balances per pair 1 - eps - c_H = {1 - eps - cH_m:.6f}")
print(f"  lifetime utility from a high date: autarky {U(cH_a, cL_a):.6f} < money {U(cH_m, cL_m):.6f};"
      f" from a low date (even agent at t = 0): autarky {np.log(eps) + beta * U(cH_a, cL_a):.6f} < money {np.log(cL_m) + beta * U(cH_m, cL_m):.6f}")

# ------------------------------------------------------------------ 28.9 regimes, log utility
print("\n=== 28.9  inside money: thresholds (1-c0)/2 = %.6f, 1/(2(1+b)) = %.6f, b/(1+b) = %.6f" % ((1 - c0) / 2, 1 / (2 * (1 + beta)), beta / (1 + beta)))
M = 1.
def regime(F):
    if F < (1 - c0) / 2:
        return f"money valued, R = 1, p = M/(1 - c0 - 2F) = {M / (1 - c0 - 2 * F):.6f}, allocation ({c0:.6f}, {1 - c0:.6f})"
    if F < 1 / (2 * (1 + beta)):
        R = brentq(lambda R: beta * R ** 2 - F * (1 + R) * (1 + beta * R), 1 - 1e-12, 1 / beta + 1e-12)
        cL = F * (1 + 1 / R)
        return f"loans only, borrower at the limit, R = {R:.6f}, (c_H, c_L) = ({1 - cL:.6f}, {cL:.6f}); Euler u'(cH)/(bR u'(cL)) = {cL / (1 - cL) / (beta * R):.6f}"
    cod = 1 / (1 + beta) - (1 - beta) * F; ced = beta / (1 + beta) + (1 - beta) * F
    X = beta * (1 / (1 + beta) - F)                                  # odd's lending at even dates
    return (f"R = 1/beta, constant consumption c^o = {cod:.6f}, c^e = {ced:.6f}; even's borrowing at even dates {X:.6f} <= limit beta F = {beta * F:.6f}:"
            f" {X <= beta * F + 1e-12}")
for F in (0., .1, (1 - c0) / 2, .25, 1 / (2 * (1 + beta)), beta / (1 + beta)):
    print(f"  F = {F:.6f}: {regime(F)}")

# ------------------------------------------------------------------ 28.10 transition (log utility): x = p0/p = 1 - F
print("\n=== 28.10  date-0 transition, log utility")
for F in (.05, .1, .2):
    x = brentq(lambda x: x * (1 - (1 - c0 - F) / x) - c0, 1e-6, 1)  # u'(1 - (1-c0-F)/x) = x u'(c0)
    p = M / (1 - c0 - 2 * F)
    print(f"  F = {F}: x = p0/p = {x:.6f} (1 - F = {1 - F:.6f}); p0 = {x * p:.6f}, p = {p:.6f}; R_0 = {x:.6f};"
          f" c^o_0 = {c0 / x:.6f}, c^e_0 = {1 - c0 / x:.6f}")
# printed initial debts of 28.9 (even owes F): x = (1 - F)/(1 + F)
for F in (.05, .1):
    x = brentq(lambda x: x * (1 + F - (1 - c0 - F) / x) - c0, 1e-6, 1)
    print(f"  28.9 with the printed initial debts, F = {F}: p0/p = {x:.6f} = (1-F)/(1+F) = {(1 - F) / (1 + F):.6f}")

# ------------------------------------------------------------------ 28.11
print("\n=== 28.11  real bills: M-bar/p = 1 - c0 - 2F + Delta, M-bar = M + Delta p  =>  p = M/(1 - c0 - 2F) for every Delta")
F = .1
for D in (0., .03, .08):
    p = M / (1 - c0 - 2 * F); Mbar = M + D * p
    print(f"  Delta = {D}: p = {p:.6f}, M-bar = {Mbar:.6f}, M-bar/p = {Mbar / p:.6f} = 1 - c0 - 2F + Delta = {1 - c0 - 2 * F + D:.6f}")
