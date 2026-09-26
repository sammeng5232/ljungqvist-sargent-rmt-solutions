"""Chapter 17 (Self-Insurance): numerical checks for exercises 17.2 and 17.5."""
import numpy as np

# 17.2: y0 = 2; saving x solves r(r+1)x^2 + (r+2-pi)x - 2(1-pi) = 0
for r, pi in ((.05, .5), (.05, .2), (.10, .5)):
    a, b, c = r * (r + 1), r + 2 - pi, -2 * (1 - pi)
    x = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    lhs = 1 / (2 - x); rhs = pi / (2 + r * x) + (1 - pi) / (1 + r * x)
    print(f"17.2 r={r}, pi={pi}: saving x = {x:.6f}, c0 = {2-x:.6f}, later c = y1 + {r*x:.6f}; FOC check {lhs:.8f} = {rhs:.8f}")
    # y0 = 1: marginal value of saving at s=0 is below the marginal utility of consumption -> corner
    print(f"      y0=1: u'(1) = 1 vs E[u'(y1)] = {pi/2 + (1-pi):.4f}  -> no saving, c_t = y_t")

# 17.5: check E[beta R (c_{t+1}/c_t)^(-gamma)] = 1 when c = y and mu = .5 gamma sigma^2 (Gauss-Hermite quadrature)
xg, wg = np.polynomial.hermite_e.hermegauss(60)
for gam, sig in ((2., .1), (5., .2)):
    mu = .5 * gam * sig ** 2
    Em = np.sum(wg * np.exp(-gam * (sig * xg + mu))) / np.sqrt(2 * np.pi)
    Ey = np.sum(wg * np.exp(sig * xg + mu)) / np.sqrt(2 * np.pi)
    print(f"17.5 gamma={gam}, sigma={sig}: E[(y'/y)^(-gamma)] = {Em:.12f};  expected income growth E[y'/y] = {Ey:.6f}")
