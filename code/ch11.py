"""Chapter 11 (Fiscal Policies in a Growth Model): numerical parts of the exercises.

Perfect-foresight equilibrium paths are computed by Newton's method on the stacked Euler equations: the unknowns
are k_1..k_{S-1}, k_0 is given and the terminal condition puts k_S on the linearized saddle path of the terminal
steady state, k_S - kbar = lambda_2 (k_{S-1} - kbar).  Cross-checks: the book's shooting algorithm (section 11.6.3),
the linear approximation (11.10.8), closed forms, and a longer horizon.
Baseline parameters are those of section 11.9: f(k) = k^.33, delta = .2, beta = .95, gamma = 2, g = .2.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from rmtlib import FIG

np.set_printoptions(precision=5, suppress=True, linewidth=140)
AL, DE, BE, GA, Z = .33, .2, .95, 2.0, 1.0
RHO = 1 / BE - 1
SH = 300                                             # truncation horizon S
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "lines.linewidth": 1.2})

f = lambda k: Z * k ** AL
fp = lambda k: AL * Z * k ** (AL - 1)
fpp = lambda k: AL * (AL - 1) * Z * k ** (AL - 2)


def ext(v, n):
    """Policy sequence of length n (t = 0..n-1); a scalar is constant, a short list is extended by its last value."""
    v = np.atleast_1d(np.asarray(v, float))
    return np.full(n, v[0]) if v.size == 1 else np.r_[v[:n], np.full(max(0, n - v.size), v[-1])]


def step(t0, before, after, n=SH + 1):
    return np.where(np.arange(n) < t0, before, after).astype(float)


# ------------------------------------------------------------------ Newton solver with a banded Jacobian
def newton(F, x0, band, tol=1e-12, maxit=100):
    """Damped Newton; the Jacobian is banded (row i depends on x[i-l..i+u]) and built by finite differences."""
    x = np.array(x0, float); l, u = band; n = x.size; w = l + u + 1
    with np.errstate(all="ignore"):
        fx = F(x)
    if not np.all(np.isfinite(fx)):
        raise RuntimeError("infeasible initial guess")
    for it in range(maxit):
        err = np.max(np.abs(fx))
        if err < tol:
            return x, it
        h = 1e-7 * np.maximum(1, np.abs(x)); ab = np.zeros((w, n))
        for r in range(w):
            cols = np.arange(r, n, w); xp = x.copy(); xp[cols] += h[cols]
            with np.errstate(all="ignore"):
                d = F(xp) - fx
            for j in cols:
                i0, i1 = max(0, j - u), min(n, j + l + 1)
                ab[u + np.arange(i0, i1) - j, j] = d[i0:i1] / h[j]
        dx = solve_banded((l, u), ab, -fx); lam = 1.0
        while True:
            xn = x + lam * dx
            with np.errstate(all="ignore"):
                fn = F(xn)
            if np.all(np.isfinite(fn)) and np.max(np.abs(fn)) <= (1 - 1e-4 * lam) * err:
                break
            lam *= .5
            if lam < 1e-14:
                raise RuntimeError(f"line search failed (residual {err:.2e})")
        x, fx = xn, fn
    raise RuntimeError("Newton did not converge")


# ------------------------------------------------------------------ inelastic-labor model (sections 11.6-11.10)
def kss(tk=0.0, psi=1.0):
    """Steady state: (1-tk)(f'(k) - delta/psi) + 1/psi = 1/(beta psi), i.e. f'(k) = (delta + rho/(1-tk))/psi."""
    return (AL * Z * psi / (DE + RHO / (1 - tk))) ** (1 / (1 - AL))


def lam2(kb, cb, tk=0.0, psi=1.0, gam=GA):
    """Stable root of the linearized Euler equation x_{t+2} - (1 + psi P + psi A) x_{t+1} + psi P x_t = 0."""
    P = fp(kb) + (1 - DE) / psi                      # pre-tax gross return
    A = -cb * BE * psi * (1 - tk) * fpp(kb) / gam
    s = 1 + psi * P + psi * A
    return (s - np.sqrt(s * s - 4 * psi * P)) / 2


def path(k0, g=.2, tk=0., tc=0., psi=1., phi=0., gam=GA, S=SH, x0=None):
    """Equilibrium from k0 for policy sequences g_t, tau_kt, tau_ct, psi_t (investment technology, exercise 11.3)
    and phi_t (capital levy per unit of k_t, exercise 11.4); scalars are constant, the value at S is terminal.
    x0 is an optional initial guess for k_1..k_{S-1} (continuation)."""
    g, tk, tc, psi, phi = (ext(v, S + 1) for v in (g, tk, tc, psi, phi))
    kT = kss(tk[-1], psi[-1]); cT = f(kT) - DE * kT / psi[-1] - g[-1]; l2 = lam2(kT, cT, tk[-1], psi[-1], gam)
    if x0 is None:                                   # geometric approach to kT, capped so that c_t >= .1 (max c_t)
        x0 = np.empty(S - 1); kk = k0
        for t in range(S - 1):
            kk = min(kT + l2 * (kk - kT), .9 * (psi[t] * (f(kk) - g[t]) + (1 - DE) * kk)); x0[t] = kk
    K = lambda x: np.concatenate(([k0], x, [kT + l2 * (x[-1] - kT)]))
    cons = lambda k: f(k[:-1]) + ((1 - DE) * k[:-1] - k[1:]) / psi[:-1] - g[:-1]
    retf = lambda k: (1 - tk[1:]) * (fp(k[1:]) - DE / psi[1:]) + 1 / psi[1:] - phi[1:]   # ret_{t+1}, t = 0..S-1

    def resid(x):
        k = K(x); c = cons(k); a = -gam * np.log(c) - np.log(1 + tc[:-1])
        return a[:-1] - np.log(psi[:-2]) - np.log(BE) - a[1:] - np.log(retf(k)[:-1])
    x, it = newton(resid, x0, band=(1, 1))
    k = K(x); c = cons(k); ret = retf(k)
    Rb = psi[:-1] * (1 + tc[:-1]) / (1 + tc[1:]) * ret            # Rbar_{t+1}: u'(c_t) = beta u'(c_{t+1}) Rbar_{t+1}
    q = BE ** np.arange(S) * c ** -gam / (1 + tc[:-1]); q = q / q[0]
    return dict(k=k, c=c, Rb=Rb, q=q, eta=fp(k), kT=kT, cT=cT, l2=l2, it=it, g=g, tk=tk, tc=tc, psi=psi, gam=gam,
                res=np.max(np.abs(resid(x))), x=x)


def shoot(k0, g=.2, tk=0., tc=0., gam=GA, S=70):
    """The book's shooting algorithm (section 11.6.3): bisect on c0 until k_S hits the terminal steady state."""
    g, tk, tc = (ext(v, S + 1) for v in (g, tk, tc)); kT = kss(tk[-1])

    def run(c0):
        k, c = np.empty(S + 1), np.empty(S); k[0], c[0] = k0, c0
        for t in range(S):
            k[t + 1] = f(k[t]) + (1 - DE) * k[t] - g[t] - c[t]
            if k[t + 1] <= 0:
                return k, c, -1.0                                       # c0 too high: capital collapses
            if t + 1 < S:
                R = (1 + tc[t]) / (1 + tc[t + 1]) * ((1 - tk[t + 1]) * (fp(k[t + 1]) - DE) + 1)
                c[t + 1] = c[t] * (BE * R) ** (1 / gam)
        return k, c, np.sign(k[S] - kT)
    lo, hi = 0.0, f(k0) + (1 - DE) * k0 - g[0]
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if run(mid)[2] > 0 else (lo, mid)
    return (lo + hi) / 2, run((lo + hi) / 2)


def util(c, gam=GA):
    return np.log(c) if gam == 1 else c ** (1 - gam) / (1 - gam)


def welfare(s):
    S = s["c"].size
    return np.sum(BE ** np.arange(S) * util(s["c"], s["gam"])) + BE ** S * util(s["cT"], s["gam"]) / (1 - BE)


def ce(W, cref, gam=GA):
    """Permanent proportional change in the constant consumption cref that yields lifetime utility W."""
    return np.exp((1 - BE) * W - np.log(cref)) - 1 if gam == 1 else ((1 - BE) * (1 - gam) * W) ** (1 / (1 - gam)) / cref - 1


def pv(s, x, xT):
    """sum_t q_t x_t, with the steady-state tail beyond S."""
    return np.sum(s["q"] * x) + s["q"][-1] * BE / (1 - BE) * xT


kb = kss(); fb = f(kb); cb = fb - DE * kb - .2                                   # baseline steady state (g = .2)
print(f"baseline: kbar = {kb:.6f}, f(kbar) = {fb:.6f}, cbar(g=.2) = {cb:.6f}, rho = {RHO:.6f}, "
      f"lambda_2(gamma=2) = {lam2(kb, cb):.6f}, lambda_2(gamma=.2) = {lam2(kb, cb, gam=.2):.6f}")
s0 = path(kb)                                                               # no policy change: must stay put
print(f"check: steady state to steady state, max|k_t - kbar| = {np.max(np.abs(s0['k'] - kb)):.1e}")

# (11.10.16) as printed versus the root of the linearized Euler equation
a1, a2, a3, a4, a5 = .975, .0329, .0642, .00063, .0011
print("(11.10.16): gamma, exact lambda_2, printed formula, corrected formula gamma/(a1 gamma + a2 + (a3 gamma + a4 gamma^2 + a5)^.5)")
for gm in (.2, 1, 2, 5):
    print(f"   {gm:4.1f}  {lam2(kb, cb, gam=gm):.5f}  {gm / (a1 / gm + a2 + a3 * np.sqrt(1 / gm + a4 / gm ** 2 + a5)):.5f}"
          f"  {gm / (a1 * gm + a2 + np.sqrt(a3 * gm + a4 * gm ** 2 + a5)):.5f}")
R0 = 1 + RHO
print(f"   exact constants: (1+R)/2R = {(1 + R0) / (2 * R0):.5f}, a/2R = {-BE * cb * fpp(kb) / (2 * R0):.5f}, "
      f"2(1+R)a/4R^2 = {2 * (1 + R0) * (-BE * cb * fpp(kb)) / (4 * R0 ** 2):.5f}, (R-1)^2/4R^2 = {(R0 - 1) ** 2 / (4 * R0 ** 2):.6f}, "
      f"a^2/4R^2 = {(BE * cb * fpp(kb)) ** 2 / (4 * R0 ** 2):.5f}")


def panel(ax, t, y, title, **kw):
    ax.plot(t, y, **kw); ax.set_title(title)


# ================================================================== 11.1 tax reform I (consumption tax)
tcd = .2 / cb
print(f"\n11.1 (d) constant consumption tax tau_c = g/cbar = {tcd:.6f}; allocation unchanged")
sd = path(kb, tc=tcd)
print(f"     check: path with constant tau_c from kbar: max|k_t - kbar| = {np.max(np.abs(sd['k'] - kb)):.1e}")


def gap11(tau):
    s = path(kb, tc=step(10, 0, tau))
    return pv(s, s["tc"][:-1] * s["c"], tau * s["cT"]) - pv(s, .2 * np.ones(SH), .2)
tce = brentq(gap11, .2, .9, xtol=1e-13)
se = path(kb, tc=step(10, 0, tce))
We, Wd = welfare(se), util(cb) / (1 - BE)
# government debt B_t (value of debt issued at t): B_t = R_{t-1,t} B_{t-1} + g_t - T_t, B_{-1} = 0
Rq = se["q"][:-1] / se["q"][1:]; Tt = se["tc"][:-1] * se["c"]; B = np.zeros(SH); B[0] = .2 - Tt[0]
for t in range(1, SH):
    B[t] = Rq[t - 1] * B[t - 1] + .2 - Tt[t]
print(f"11.1 (e) tau_c from t=10 on = {tce:.6f} (vs {tcd:.6f});  k_10 = {se['k'][10]:.5f}, c_0 = {se['c'][0]:.5f}, "
      f"c_9 = {se['c'][9]:.5f}, c_10 = {se['c'][10]:.5f}, min k = {se['k'].min():.5f}, Rbar_10 = {se['Rb'][9]:.5f}")
print(f"     debt: B_9 = {B[9]:.5f}, B_40 = {B[40]:.5f}, B_S = {B[-1]:.5f}; steady-state check (tau c - g)/rho = {(tce * cb - .2) / RHO:.5f}")
print(f"     welfare: consumption-equivalent loss of (e) relative to (d) = {-100 * ce(We, cb):.4f}% of consumption per period")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.3)); tt = np.arange(41)
panel(ax[0], tt, se["k"][:41], "$k_t$", color="k"); panel(ax[1], tt, se["c"][:41], "$c_t$", color="k")
panel(ax[2], tt, np.r_[R0, se["Rb"][:40]], r"$\bar R_t$", color="k"); panel(ax[3], tt, B[:41], "government debt $B_t$", color="k")
for a_, v in zip(ax, (kb, cb, R0, 0)):
    a_.axhline(v, ls="--", color="grey", lw=.8); a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_01.pdf"); plt.close()

# ================================================================== 11.2 tax reform II (capital tax versus consumption tax)
rev = lambda tau: tau * RHO * kss(tau) / (1 - tau)               # steady-state capital-tax revenue
taus = np.linspace(0, .999, 20001); rv = rev(taus); imax = np.argmax(rv)
print(f"\n11.2 steady-state capital-tax revenue peaks at tau_k = {taus[imax]:.4f} with revenue {rv[imax]:.5f} (< g = .2)")
g2 = .05
tk2 = brentq(lambda t: rev(t) - g2, 1e-9, taus[imax])
tk2b = brentq(lambda t: rev(t) - g2, taus[imax], .9999)
k2o = kss(tk2); c2o = f(k2o) - DE * k2o - g2; c2n = fb - DE * kb - g2
print(f"     g = {g2}: tau_k = {tk2:.6f} (other root on the wrong side of the Laffer curve: {tk2b:.4f}); kbar_0 = {k2o:.5f}, "
      f"cbar_0 = {c2o:.5f}, f'(k)-delta = {fp(k2o) - DE:.5f} = rho/(1-tau) = {RHO / (1 - tk2):.5f}")
s2 = path(k2o, g=g2)
tc2 = pv(s2, g2 * np.ones(SH), g2) / pv(s2, s2["c"], s2["cT"])
print(f"     reform: kbar = {kb:.5f}, cbar = {c2n:.5f}; constant tau_c = {tc2:.6f};  c_0 = {s2['c'][0]:.5f}, "
      f"Rbar_1 = {s2['Rb'][0]:.5f}; periods with c_t < cbar_0: {np.sum(s2['c'] < c2o)}")
print(f"     consumption-equivalent gain: steady-state comparison {100 * (c2n / c2o - 1):.4f}%, "
      f"with the transition {100 * ce(welfare(s2), c2o):.4f}%")

# ================================================================== 11.3 anticipated productivity (investment-technology) shift
k31, k32 = kss(), kss(psi=2)
c31, c32 = f(k31) - DE * k31, f(k32) - DE * k32 / 2
print(f"\n11.3 steady states: psi=1: k = {k31:.5f}, c = {c31:.5f}, x = {DE * k31:.5f};  psi=2: k = {k32:.5f}, "
      f"c = {c32:.5f}, x = {DE * k32 / 2:.5f}")
psi3 = step(4, 1, 2)
s3 = path(k31, g=0, psi=psi3)
x3 = (s3["k"][1:] - (1 - DE) * s3["k"][:-1]) / psi3[:-1]
print("     t:      ", np.arange(9)); print("     k_t:    ", s3["k"][:9]); print("     c_t:    ", s3["c"][:9])
print("     x_t:    ", x3[:9]); print("     R_{t,t+1}", s3["Rb"][:9]); print(f"     min x_t = {x3.min():.5f} (investment stays positive)")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.3)); tt = np.arange(31)
panel(ax[0], tt, s3["k"][:31], "$k_t$", color="k"); panel(ax[1], tt, s3["c"][:31], "$c_t$", color="k")
panel(ax[2], tt, x3[:31], "$x_t$", color="k"); panel(ax[3], tt[1:], s3["Rb"][:30], r"$R_{t-1,t}$", color="k")
for a_, v in zip(ax, (k31, c31, DE * k31, R0)):
    a_.axhline(v, ls="--", color="grey", lw=.8); a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_03.pdf"); plt.close()

# ================================================================== 11.4 capital levy
k40 = kb; g4 = .02
levy_path = lambda phi, g, x0=None: path(k40, g=g, phi=np.where(np.arange(SH + 1) == 10, phi, 0.0), x0=x0)
print(f"\n11.4 (b) k0 = {k40:.5f};  (c) levy at T=0: phi = g/((1-beta) k0) = {g4 / ((1 - BE) * k40):.5f} (g = {g4}); "
      f"c = {fb - DE * kb - g4:.5f}; phi > 1 iff g > (1-beta) k0 = {(1 - BE) * k40:.5f}")
# Laffer curve of the foreseen levy: for each phi the g it finances (a fixed point in g), by continuation in phi
lf, xs, xw = [], [], None
for phi in np.linspace(.02, 4, 200):
    gg = lf[-1][1] if lf else .001
    for _ in range(100):
        s = levy_path(phi, gg, xw); xw = s["x"]
        gn = s["q"][10] * phi * s["k"][10] / pv(s, np.ones(SH), 1.0)
        if abs(gn - gg) < 1e-13:
            break
        gg = gn
    lf.append((phi, gg, s["k"][10], s["Rb"][9])); xs.append(xw)
lf = np.array(lf); im = np.argmax(lf[:, 1])
print(f"     Laffer curve of a levy at T=10: max financeable g = {lf[im, 1]:.6f} at phi = {lf[im, 0]:.3f} (k_10 = {lf[im, 2]:.4f}, "
      f"R_(9,10) = {lf[im, 3]:.4f}); at phi = 4: g = {lf[-1, 1]:.5f}.  A levy at T=0 finances any g < f(k0) - delta k0 = {fb - DE * kb:.4f}")
j4 = np.argmax(lf[:, 1] > g4)
phi4 = brentq(lambda p: (lambda s: s["q"][10] * p * s["k"][10] - pv(s, g4 * np.ones(SH), g4))(levy_path(p, g4, xs[j4])),
              lf[j4 - 1, 0], lf[j4, 0], xtol=1e-14)
s4 = levy_path(phi4, g4, xs[j4]); W4_0 = util(fb - DE * kb - g4) / (1 - BE)
print(f"     (d) g = {g4}: phi = {phi4:.6f} (vs {g4 / ((1 - BE) * k40):.6f} at T=0);  c_0 = {s4['c'][0]:.5f} (T=0 levy: {fb - DE * kb - g4:.5f}),"
      f" c_9 = {s4['c'][9]:.5f}, c_10 = {s4['c'][10]:.5f}, k_10 = {s4['k'][10]:.5f}, R_(9,10) net of levy = {s4['Rb'][9]:.5f}, "
      f"R_(10,11) = {s4['Rb'][10]:.5f}")
print("     k_t, t=0..14:", s4["k"][:15]); print("     c_t, t=0..14:", s4["c"][:15])
print(f"     welfare loss of the T=10 levy relative to the T=0 levy: {-100 * ce(welfare(s4), fb - DE * kb - g4):.4f}% of consumption")
# (e) sequential trading: government debt B_t (value at t of debt issued at t) and household bond holdings b_{t+1} = B_t
L = np.where(np.arange(SH) == 10, phi4 * s4["k"][:SH], 0.0); Rq4 = s4["q"][:-1] / s4["q"][1:]; B4 = np.zeros(SH); B4[0] = g4 - L[0]
for t in range(1, SH):
    B4[t] = Rq4[t - 1] * B4[t - 1] + g4 - L[t]
print(f"     (e) government debt B_t: B_0 = {B4[0]:.5f}, B_9 = {B4[9]:.5f}, B_10 = {B4[10]:.5f}, B_S = {B4[-1]:.5f}; "
      f"-g/rho = {-g4 / RHO:.5f}; levy = {L[10]:.5f}")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.3)); tt = np.arange(31)
panel(ax[0], tt, s4["k"][:31], "$k_t$", color="k"); panel(ax[1], tt, s4["c"][:31], "$c_t$", color="k")
panel(ax[2], tt[1:], s4["Rb"][:30], r"$R_{t-1,t}$ (after levy)", color="k"); panel(ax[3], tt, B4[:31], "government debt $B_t$", color="k")
ax[1].axhline(fb - DE * kb - g4, ls=":", color="k", lw=.8)
for a_, v in zip(ax, (kb, fb - DE * kb, R0, 0)):
    a_.axhline(v, ls="--", color="grey", lw=.8); a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_04.pdf"); plt.close()

# ================================================================== 11.5 log utility, permanent jump in g
c5 = fb - DE * kb; kgold = (AL * Z / DE) ** (1 / (1 - AL)); cgold = f(kgold) - DE * kgold
s5 = path(kb, g=c5 / 2, gam=1)
print(f"\n11.5 k = {kb:.5f}, c = {c5:.5f}; golden rule k~ = {kgold:.5f}, c~ = {cgold:.5f};  g = c/2 = {c5 / 2:.5f}: "
      f"max|k_t - k| = {np.max(np.abs(s5['k'] - kb)):.1e}, c_t = {s5['c'][0]:.5f}, Rbar = {s5['Rb'][0]:.6f}")

# ================================================================== 11.6 trade and growth (k0 < kbar)
kbar0 = .5 * kb; wb = fb - fp(kb) * kb
s6 = path(kbar0, g=0)
co6 = fb - DE * kb - RHO * (kb - kbar0)
def mono(s, sg):
    """Monotonicity of k, c (direction sg) and Rbar (opposite) while the path is visibly away from the steady state."""
    m = np.argmax(np.abs(s["k"] - s["kT"]) < 1e-9)
    return (bool(np.all(sg * np.diff(s["k"][:m]) > 0)), bool(np.all(sg * np.diff(s["c"][:m - 1]) > 0)),
            bool(np.all(-sg * np.diff(s["Rb"][:m - 2]) > 0)), int(m))
print(f"\n11.6 closed economy from k0 = .5 kbar: c_0 = {s6['c'][0]:.5f}, c_inf = {s6['cT']:.5f}, Rbar_1 = {s6['Rb'][0]:.5f}; "
      f"k up, c up, Rbar down monotonically: {mono(s6, 1)}")
print(f"     open economy: B_-1 = {kb - kbar0:.5f}, c = f(kbar) - delta kbar - rho(kbar - k0) = w + rho k0 = {co6:.5f} (w = {wb:.5f})")
cce6 = co6 * (1 + ce(welfare(s6), co6))                       # constant consumption equivalent to the closed path
print(f"     closed-economy certainty-equivalent constant consumption = {cce6:.5f}; gain of opening = {100 * (co6 / cce6 - 1):.4f}% of consumption")
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3)); tt = np.arange(31)
ax[0].plot(tt, s6["c"][:31], "k-", label="closed"); ax[0].plot(tt, np.full(31, co6), "k--", label="open"); ax[0].set_title("$c_t$")
ax[1].plot(tt, s6["k"][:31], "k-", label="closed"); ax[1].plot(tt, np.r_[kbar0, np.full(30, kb)], "k--", label="open"); ax[1].set_title("$k_t$")
ax[0].legend(frameon=False)
for a_ in ax:
    a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_06.pdf"); plt.close()

# ================================================================== 11.7 capital tax, unforeseen and foreseen, two gammas
tk7 = .2; k7 = kss(tk7); c7 = f(k7) - DE * k7 - .2
print(f"\n11.7 (a) kbar = {kb:.5f}, cbar = {cb:.5f}, Rbar = {R0:.5f};  new steady state (tau_k=.2): k = {k7:.5f}, c = {c7:.5f}, "
      f"f'(k) = {fp(k7):.5f}")
fig, ax = plt.subplots(2, 3, figsize=(9, 4.4)); tt = np.arange(41)
for gm, ls in ((2.0, "-"), (.2, "-.")):
    su = path(kb, tk=tk7, gam=gm); sf = path(kb, tk=step(10, 0, tk7), gam=gm)
    l2 = su["l2"]; Rp = fp(k7) + 1 - DE
    print(f"     gamma={gm}: lambda_2 = {l2:.4f}, 1/lambda_1 = lambda_2/(f'+1-delta) = {l2 / Rp:.4f};  unforeseen: c_0 = {su['c'][0]:.5f}"
          f" (jump {su['c'][0] - cb:+.5f}), Rbar_1 = {su['Rb'][0]:.5f}, k_10 = {su['k'][10]:.5f};  "
          f"foreseen: c_0 = {sf['c'][0]:.5f}, c_9 = {sf['c'][9]:.5f}, k_10 = {sf['k'][10]:.5f}, Rbar_9 = {sf['Rb'][8]:.5f}, Rbar_10 = {sf['Rb'][9]:.5f}")
    ratio = (sf["k"][21:26] - k7) / (sf["k"][20:25] - k7)
    print(f"        foreseen, convergence ratios (k_t+1 - k)/(k_t - k), t=20..24: {ratio}; unforeseen: k down, c down, Rbar up "
          f"monotonically: {mono(su, -1)}; c_0 - cbar' vs (P - lambda_2)(kbar - kbar') = {su['c'][0] - c7:.5f} vs {(Rp - l2) * (kb - k7):.5f}")
    for row, s in enumerate((su, sf)):
        ax[row, 0].plot(tt, s["k"][:41], "k" + ls, label=rf"$\gamma={gm}$"); ax[row, 1].plot(tt, s["c"][:41], "k" + ls)
        ax[row, 2].plot(tt, np.r_[R0, s["Rb"][:40]], "k" + ls)
for row, lab in enumerate(("unforeseen at $t=0$", "foreseen, at $t=10$")):
    for j, (nm, v) in enumerate((("$k_t$", kb), ("$c_t$", cb), (r"$\bar R_t$", R0))):
        ax[row, j].axhline(v, ls="--", color="grey", lw=.8); ax[row, j].set_title(nm + ", " + lab)
ax[0, 0].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG + "ch11_07.pdf"); plt.close()

# ================================================================== 11.8 trade and growth II (k0 > kbar)
s8 = path(2 * kb, g=0)
print(f"\n11.8 closed economy from k0 = 2 kbar: c_0 = {s8['c'][0]:.5f}, c_1 = {s8['c'][1]:.5f}, cbar = {s8['cT']:.5f}, "
      f"Rbar_1 = {s8['Rb'][0]:.5f}, k_1 = {s8['k'][1]:.5f}; k down, c down, Rbar up monotonically: {mono(s8, -1)}")
print(f"     w = f(kbar) - kbar f'(kbar) = {wb:.5f};  k0/kbar, CE of closed and of 'sell all' relative to the optimal open plan (g)")
print(f"     cutoff for 'closed beats sell-all' via constant k: f(k0)/k0 >= rho+delta  <=>  k0/kbar <= alpha^(-1/(1-alpha)) = {AL ** (-1 / (1 - AL)):.4f}")
for m in (1.5, 2, 5, 10, 20, 40):
    k0 = m * kb; sc_ = path(k0, g=0); cg = wb + RHO * k0; csell = RHO * k0
    Wc = welfare(sc_)
    print(f"     {m:5.1f}   closed: {100 * ce(Wc, cg):8.3f}%   sell-all: {100 * (csell / cg - 1):8.3f}%   "
          f"(closed {'>' if Wc > util(csell) / (1 - BE) else '<'} sell-all)")
ms = np.linspace(5, 60, 111); dW = [welfare(path(m * kb, g=0)) - util(RHO * m * kb) / (1 - BE) for m in ms]
mstar = brentq(lambda m: welfare(path(m * kb, g=0)) - util(RHO * m * kb) / (1 - BE), ms[np.argmax(np.array(dW) < 0) - 1],
               ms[np.argmax(np.array(dW) < 0)])
print(f"     sell-all overtakes the closed economy at k0 = {mstar:.3f} kbar")

# ================================================================== 11.10 term structure after an unforeseen capital tax
s10 = path(kb, tk=tk7)
r_short = np.log(s10["q"][:-1] / s10["q"][1:])                  # r_{t-1,t}, t = 1..S-1
r_long = np.cumsum(r_short) / np.arange(1, SH)                  # r_{0,t}
r_euler = -np.log(BE) + GA * np.log(s10["c"][1:] / s10["c"][:-1])
print(f"\n11.10 (a) flat: r = -log beta = {-np.log(BE):.6f};  (d) tau_k = .2: r_(t-1,t), t=1..10:", r_short[:10])
print("      r_(0,t), t=1..10:", r_long[:10], f"\n      checks: max|r - log Rbar| = {np.max(np.abs(r_short - np.log(s10['Rb'][:-1]))):.1e}, "
      f"max|r - Euler formula| = {np.max(np.abs(r_short - r_euler)):.1e}; r_(0,40) = {r_long[39]:.5f}")
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3)); tt = np.arange(1, 11)
ax[0].plot(tt, r_short[:10], "ko-", ms=3, label=r"$\tau_k=.2$"); ax[0].plot(tt, np.full(10, -np.log(BE)), "k--", label="(a)")
ax[1].plot(tt, r_long[:10], "ko-", ms=3); ax[1].plot(tt, np.full(10, -np.log(BE)), "k--")
ax[0].set_title("$r_{t-1,t}$"); ax[1].set_title("$r_{0,t}$"); ax[0].legend(frameon=False)
for a_ in ax:
    a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_10.pdf"); plt.close()

# ================================================================== 11.11 rising, concave consumption
s11a = path(.5 * kb)                                            # story 1: k0 below the steady state
k11 = kss(.2); s11b = path(k11, tk=0)                           # story 2: unforeseen cut of tau_k from .2 to 0 at t=0
g11 = np.diff(np.log(s11a["c"]))
print(f"\n11.11 story 1 (k0 = .5 kbar): c_0 = {s11a['c'][0]:.5f}, growth rates t=0..4: {g11[:5]}, Rbar_1..5: {s11a['Rb'][:5]}")
print(f"      story 2 (tau_k .2 -> 0 unforeseen): c before = {f(k11) - DE * k11 - .2:.5f}, c_0 = {s11b['c'][0]:.5f}, c_30 = {s11b['c'][30]:.5f},"
      f" Rbar_1 = {s11b['Rb'][0]:.5f}; second differences of c negative: {np.all(np.diff(s11a['c'], 2)[:60] < 0)}, {np.all(np.diff(s11b['c'], 2)[:60] < 0)}")
fig, ax = plt.subplots(1, 3, figsize=(9, 2.3)); tt = np.arange(41)
for s, ls, lab in ((s11a, "-", r"$k_0=.5\bar k$"), (s11b, "--", r"$\tau_k$: $.2\to0$")):
    ax[0].plot(tt, s["c"][:41], "k" + ls, label=lab); ax[1].plot(tt, s["k"][:41], "k" + ls); ax[2].plot(tt[1:], s["Rb"][:40], "k" + ls)
for a_, nm in zip(ax, ("$c_t$", "$k_t$", r"$\bar R_t$")):
    a_.set_title(nm); a_.set_xlabel("$t$")
ax[2].axhline(R0, ls=":", color="grey", lw=.8); ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG + "ch11_11.pdf"); plt.close()

# ================================================================== 11.12 flat, then a drop at t = 12 and a gradual decline
T12, gA, tkA = 12, .23, .3
sA = path(kb, g=gA, tk=tkA)                 # story A: at t=12 an unforeseen permanent rise of g and tau_k (time counted from 12)
sB = path(kb, g=step(10, .2, .4))           # story B: at t=12 unforeseen news that g will rise permanently to .4 at t=22
print(f"\n11.12 story A (unforeseen at t=12: g .2->{gA}, tau_k 0->{tkA}): c_12 = {sA['c'][0]:.5f} (before {cb:.5f}; drop {sA['c'][0] - cb:+.5f}),"
      f" new cbar = {sA['cT']:.5f} (further decline {sA['cT'] - sA['c'][0]:+.5f}); Rbar_12 = {1 + (1 - tkA) * (fp(kb) - DE):.5f}, "
      f"Rbar_13 = {sA['Rb'][0]:.5f}; new kbar = {sA['kT']:.5f}; convex decline: {np.all(np.diff(sA['c'], 2)[:80] > 0)}")
print(f"      story B (news at t=12, g rises to .4 at t=22): c_12 = {sB['c'][0]:.5f}, c_22 = {sB['c'][10]:.5f}, max k = {sB['k'].max():.5f}; "
      f"second differences of c before 22 negative: {np.all(np.diff(sB['c'], 2)[:8] < 0)}, after 22 positive: {np.all(np.diff(sB['c'], 2)[11:60] > 0)}")
fig, ax = plt.subplots(1, 3, figsize=(9, 2.3)); tt = np.arange(51)
ax[0].plot(tt, np.r_[np.full(T12, cb), sA["c"][:51 - T12]], "k-", label=r"A: $g$ and $\tau_k$ rise at 12")
ax[0].plot(tt, np.r_[np.full(T12, cb), sB["c"][:51 - T12]], "k--", label="B: news at 12, $g$ rises at 22")
ax[1].plot(tt, np.r_[np.full(T12, kb), sA["k"][:51 - T12]], "k-"); ax[1].plot(tt, np.r_[np.full(T12, kb), sB["k"][:51 - T12]], "k--")
ax[2].plot(tt[1:], np.r_[np.full(T12 - 1, R0), 1 + (1 - tkA) * (fp(kb) - DE), sA["Rb"][:50 - T12]], "k-")
ax[2].plot(tt[1:], np.r_[np.full(T12, R0), sB["Rb"][:50 - T12]], "k--")
for a_, nm in zip(ax, ("$c_t$", "$k_t$", r"$\bar R_t$")):
    a_.set_title(nm); a_.set_xlabel("$t$")
ax[0].legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(FIG + "ch11_12.pdf"); plt.close()

# ================================================================== 11.13 linear utility; saving rates
kmin = brentq(lambda k: f(k) + (1 - DE) * k - kb, 1e-9, kb)
print(f"\n11.13 kbar = {kb:.5f}, cbar = {fb - DE * kb:.5f}; s(kbar) = delta kbar/f(kbar) = {DE * kb / fb:.5f} = alpha delta/(rho+delta) = "
      f"{AL * DE / (RHO + DE):.5f};  s = 1 for k <= {kmin:.5f}; s < 0 for k > kbar/(1-delta) = {kb / (1 - DE):.5f}")
kg = np.linspace(.1, 2.6, 60) * kb
s_lin = np.minimum(1, (kb - (1 - DE) * kg) / f(kg))
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(kg / kb, s_lin, "k-", lw=1.8, label=r"linear utility ($\gamma=0$)")
for gm, ls in ((.02, ":"), (.5, "-."), (2, "--")):
    sr = np.array([(path(k0, g=0, gam=gm)["k"][1] - (1 - DE) * k0) / f(k0) for k0 in kg])
    ax.plot(kg / kb, sr, "k" + ls, label=rf"$\gamma={gm}$")
    print(f"      gamma={gm}: s(.5 kbar) = {np.interp(.5, kg / kb, sr):.4f}, s(kbar) = {np.interp(1, kg / kb, sr):.4f}, s(2 kbar) = {np.interp(2, kg / kb, sr):.4f}")
ax.axhline(0, color="grey", lw=.6); ax.set_xlabel(r"$k/\bar k$"); ax.set_ylabel("saving rate $s$"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch11_13.pdf"); plt.close()

# ================================================================== 11.14 transition from k0 = .5 kbar: g = 0, lump-sum g, capital-tax-financed g
k14 = .5 * kb; phi14 = .1; g14 = phi14 * f(k14)
sc14 = path(k14, g=0); sd14 = path(k14, g=g14)
print(f"\n11.14 (b) kbar = {kb:.5f}, x/f = {AL * DE / (RHO + DE):.5f};  (c) c_0 = {sc14['c'][0]:.5f}, k_1 = {sc14['k'][1]:.5f}, "
      f"Rbar_1 = {sc14['Rb'][0]:.5f}, lambda_2 = {sc14['l2']:.4f}")
print(f"      (d) g = phi f(k0) = {g14:.5f}: c_0 = {sd14['c'][0]:.5f}, k_1 = {sd14['k'][1]:.5f}, Rbar_1 = {sd14['Rb'][0]:.5f}, "
      f"lambda_2 = {sd14['l2']:.4f}; k_10: {sc14['k'][10]:.5f} vs {sd14['k'][10]:.5f}")


def gap14(tau):
    s = path(k14, g=g14, tk=tau)
    taxes = tau * (fp(s["k"][:-1]) - DE) * s["k"][:-1]
    return pv(s, taxes, tau * (fp(s["kT"]) - DE) * s["kT"]) - pv(s, g14 * np.ones(SH), g14), s
kr = (AL ** 2 * Z / DE) ** (1 / (1 - AL))
print(f"      bound on capital-tax revenue per period: max_k [alpha f(k) - delta k] = {AL * f(kr) - DE * kr:.5f} (at k = {kr:.4f}); "
      f"g with phi = .2 would be {.2 * f(k14):.5f}")
taug = np.linspace(.05, .95, 19); gaps = [gap14(t)[0] for t in taug]
i1 = np.argmax(np.array(gaps) > 0)
tk14 = brentq(lambda t: gap14(t)[0], taug[i1 - 1], taug[i1], xtol=1e-13); se14 = gap14(tk14)[1]
print(f"      (e) PV budget surplus on a tau grid: {np.round(gaps, 4)}")
print(f"      (e) tau_k = {tk14:.6f}; steady state k = {se14['kT']:.5f}, c = {se14['cT']:.5f}; c_0 = {se14['c'][0]:.5f}, "
      f"Rbar_1 = {se14['Rb'][0]:.5f}; steady-state revenue = {tk14 * (fp(se14['kT']) - DE) * se14['kT']:.5f} vs g = {g14:.5f}")
fig, ax = plt.subplots(1, 3, figsize=(9, 2.3)); tt = np.arange(31)
for s, ls, lab in ((sc14, "-", "(c) $g=0$"), (sd14, "--", "(d) $g$, lump sum"), (se14, "-.", r"(e) $g$, $\bar\tau_k$")):
    ax[0].plot(tt, s["k"][:31], "k" + ls, label=lab); ax[1].plot(tt, s["c"][:31], "k" + ls); ax[2].plot(tt[1:], s["Rb"][:30], "k" + ls)
for a_, nm in zip(ax, ("$k_t$", "$c_t$", r"$\bar R_t$")):
    a_.set_title(nm); a_.set_xlabel("$t$")
ax[2].axhline(R0, ls=":", color="grey", lw=.8); ax[0].legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(FIG + "ch11_14.pdf"); plt.close()

# ================================================================== 11.15 upward-sloping yield curve
print(f"\n11.15 r_(0,t) after the unforeseen tau_k (11.10(d)) at t = 1, 5, 10, 20, 40: {r_long[[0, 4, 9, 19, 39]]}; "
      f"concave: {np.all(np.diff(r_long[:40], 2) < 0)}")
s15 = path(1.5 * kb)
r15 = np.cumsum(np.log(s15["q"][:-1] / s15["q"][1:])) / np.arange(1, SH)
print(f"      k0 = 1.5 kbar, constant policy: r_(0,t) at t = 1, 5, 10, 20, 40: {r15[[0, 4, 9, 19, 39]]}")
fig, ax = plt.subplots(figsize=(4, 2.4)); tt = np.arange(1, 41)
ax.plot(tt, r_long[:40], "k-", label=r"unforeseen $\tau_k=.2$ at $t=0$"); ax.plot(tt, r15[:40], "k--", label=r"$k_0=1.5\bar k$")
ax.axhline(-np.log(BE), ls=":", color="grey", lw=.8); ax.set_xlabel("maturity $t$"); ax.set_title("$r_{0,t}$"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "ch11_15.pdf"); plt.close()

# ================================================================== 11.16 low versus high gamma
fig, ax = plt.subplots(1, 3, figsize=(9, 2.3)); tt = np.arange(31)
for gm, ls in ((2.0, "-"), (.2, "--")):
    s = path(.5 * kb, gam=gm)
    print(f"\n11.16 gamma={gm}: lambda_2 = {s['l2']:.4f}; periods until |k_t - kbar| < 1% of kbar: "
          f"{np.argmax(np.abs(s['k'] - kb) < .01 * kb)}; c_0 = {s['c'][0]:.5f}, c_5 = {s['c'][5]:.5f}, cbar = {cb:.5f}")
    ax[0].plot(tt, s["k"][:31], "k" + ls, label=rf"$\gamma={gm}$"); ax[1].plot(tt, fp(s["k"][:31]), "k" + ls); ax[2].plot(tt, s["c"][:31], "k" + ls)
for a_, nm, v in zip(ax, ("$k_t$", r"$f'(k_t)$", "$c_t$"), (kb, RHO + DE, cb)):
    a_.set_title(nm); a_.set_xlabel("$t$"); a_.axhline(v, ls=":", color="grey", lw=.8)
ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG + "ch11_16.pdf"); plt.close()

# ================================================================== 11.17 anticipated versus unanticipated jump in g (figure 11.6)
g17 = step(10, .2, .4); s17 = path(kb, g=g17)
c0sh, (ksh, csh, _) = shoot(kb, g=g17, S=70)
print(f"\n11.17 anticipated: c_0 = {s17['c'][0]:.6f} (shooting: {c0sh:.6f}, |diff| = {abs(c0sh - s17['c'][0]):.1e}; "
      f"max|k diff| t<=40 = {np.max(np.abs(ksh[:41] - s17['k'][:41])):.1e}), k_10 = {s17['k'][10]:.5f}, c_10 = {s17['c'][10]:.5f}, "
      f"min Rbar = {s17['Rb'].min():.5f} at t = {np.argmin(s17['Rb']) + 1}; cbar_new = {s17['cT']:.5f}")
print(f"      Newton iterations {s17['it']}, residual {s17['res']:.1e}; horizon check S=300 vs 600: "
      f"{np.max(np.abs(path(kb, g=g17, S=600)['k'][:301] - s17['k'])):.1e}")
# linear approximation (11.10.8) around the terminal steady state: x_{t+1} = l2 x_t + (1/(a l1)) sum_j l1^-j (e z_{t+j} + h z_{t+j+1})
gT = .4; cT17 = fb - DE * kb - gT


def H(k0_, k1_, k2_, g0_, g1_):
    c0_ = f(k0_) + (1 - DE) * k0_ - g0_ - k1_; c1_ = f(k1_) + (1 - DE) * k1_ - g1_ - k2_
    return -GA * np.log(c0_) - np.log(BE) + GA * np.log(c1_) - np.log(fp(k1_) + 1 - DE)
eps = 1e-6; base = (kb, kb, kb, gT, gT)
dH = [(H(*[b + eps * (i == j) for j, b in enumerate(base)]) - H(*[b - eps * (i == j) for j, b in enumerate(base)])) / (2 * eps) for i in range(5)]
d_, b_, a_, e_, h_ = dH
lr = np.roots([a_, b_, d_]); l1, l2l = max(lr), min(lr)
z = g17 - gT; xl = np.zeros(61); xl[0] = kb - kb
v = e_ * z[:-1] + h_ * z[1:]
for t in range(60):
    xl[t + 1] = l2l * xl[t] + np.sum(l1 ** -np.arange(SH - t) * v[t:]) / (a_ * l1)
print(f"      linear approximation: lambda_2 = {l2l:.6f} (formula {lam2(kb, cT17):.6f}), lambda_1 = {l1:.6f}, beta lambda_1 lambda_2 = "
      f"{BE * l1 * l2l:.6f}; max|k_lin - k| (t<=60) = {np.max(np.abs(kb + xl - s17['k'][:61])):.4f}, k_10 lin = {kb + xl[10]:.5f}")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.3)); tt = np.arange(41)
ax[0].plot(tt, np.where(tt < 10, cb, cb - .2), "k-"); ax[0].set_title("(a) $c_t$, unanticipated")
ax[1].plot(tt, s17["c"][:41], "k-"); ax[1].axhline(cb - .2, ls="--", color="grey", lw=.8); ax[1].set_title("(b) $c_t$, anticipated")
ax[2].plot(tt, s17["k"][:41], "k-", label="anticipated"); ax[2].plot(tt, np.full(41, kb), "k--", label="unanticipated"); ax[2].set_title("$k_t$")
ax[3].plot(tt[1:], s17["Rb"][:40], "k-"); ax[3].plot(tt[1:], np.full(40, R0), "k--"); ax[3].set_title(r"$\bar R_t$")
ax[2].legend(frameon=False, fontsize=6)
for a_ in ax:
    a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_17.pdf"); plt.close()

# ================================================================== 11.18 habits and durability
def path_habit(k0, cm1, ah, gam=GA, S=SH):
    kT = kss(); cT = f(kT) - DE * kT
    up = lambda x: x ** -gam

    def parts(x):
        k = np.r_[k0, x, kT, kT]                                  # k_0..k_{S+1}
        c = f(k[:-1]) + (1 - DE) * k[:-1] - k[1:]                 # c_0..c_S
        cc = np.r_[cm1, c, cT]                                    # c_{-1}..c_{S+1}
        xx = cc[1:] - ah * cc[:-1]                                # x_0..x_{S+1}
        mu = up(xx[:-1]) - ah * BE * up(xx[1:])                   # mu_0..mu_S
        return k, c, mu

    def resid(x):
        k, c, mu = parts(x)
        return np.log(mu[:S - 1]) - np.log(BE) - np.log(mu[1:S]) - np.log(fp(k[1:S]) + 1 - DE)
    x, it = newton(resid, kT + (k0 - kT) * .85 ** np.arange(1, S), band=(2, 2))
    k, c, mu = parts(x)
    return k, c, mu, it


def habit_roots(ah, gam=GA, n=40, j=20):
    kT = kss(); cT = f(kT) - DE * kT

    def resid_full(k):                     # Euler residuals E_t, t = 1..n-4, for a path k_0..k_{n-1} (x_t, mu_t start at t = 1)
        c = f(k[:-1]) + (1 - DE) * k[:-1] - k[1:]; x = c[1:] - ah * c[:-1]
        mu = x[:-1] ** -gam - ah * BE * x[1:] ** -gam
        return np.log(mu[:-1]) - np.log(BE) - np.log(mu[1:]) - np.log(fp(k[2:-2]) + 1 - DE)
    kk = np.full(n, kT); r0 = resid_full(kk); coef = []
    for d in range(-1, 4):                 # E_j depends on k_{j-1}, ..., k_{j+3}
        kp = kk.copy(); kp[j + d] += 1e-6; coef.append((resid_full(kp) - r0)[j - 1] / 1e-6)
    return np.roots(coef[::-1])
print("\n11.18 steady state (independent of the habit parameter): k = %.5f, f'(k) = %.5f, c = %.5f" % (kb, fp(kb), fb - DE * kb))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3)); tt = np.arange(31)
s18 = path(.5 * kb, g=0)
for ah, ls in ((.5, "-"), (0, ":"), (-.5, "--")):
    k18, c18, mu18, it18 = path_habit(.5 * kb, fb - DE * kb, ah)
    if ah != 0:
        rts = habit_roots(ah); st = np.sort(np.abs(rts))
        print(f"      alpha={ah:+.1f}: Newton its {it18}; c_0 = {c18[0]:.5f}, k_1 = {k18[1]:.5f}, k_10 = {k18[10]:.5f}, min mu = {mu18.min():.4f}; "
              f"roots {np.round(rts, 4)}; stable {np.round(st[:2], 4)}; products with partners {np.round(st[:2] * st[:1:-1], 4)} (1/beta = {1 / BE:.4f})")
        emp = (k18[31:36] - kb) / (k18[30:35] - kb)
        print(f"         empirical ratios (k_t+1 - k)/(k_t - k), t = 30..34: {emp}")
    else:
        print(f"      alpha=0 reproduces the standard model: max|k diff| = {np.max(np.abs(k18[:SH] - s18['k'][:SH])):.1e}; "
              f"c_0 = {c18[0]:.5f}, k_1 = {k18[1]:.5f}, k_10 = {k18[10]:.5f}, lambda_2 = {s18['l2']:.4f}")
    ax[0].plot(tt, c18[:31], "k" + ls, label=rf"$\alpha={ah}$"); ax[1].plot(tt, k18[:31], "k" + ls)
ax[0].set_title("$c_t$"); ax[1].set_title("$k_t$"); ax[0].legend(frameon=False, fontsize=7)
for a_ in ax:
    a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_18.pdf"); plt.close()

# ================================================================== 11.19 Big K, little k (foreseen jump in g)
K, C = s17["k"], s17["c"]; eta = fp(K); w = f(K) - K * fp(K); q = s17["q"]; tauh = g17[:SH]
R_ = eta[1:SH] + 1 - DE                                           # R_{t,t+1}, t = 0..S-2 (price ratios q_t/q_{t+1})
print(f"\n11.19 price check: max|q_t/q_t+1 - (eta_t+1 + 1 - delta)| = {np.max(np.abs(q[:-1] / q[1:] - R_)):.1e}")


def household(c0, n=SH - 1):
    c = np.empty(n + 1); k = np.empty(n + 1); c[0] = c0; k[0] = kb
    for t in range(n):
        k[t + 1] = (eta[t] + 1 - DE) * k[t] + w[t] - tauh[t] - c[t]
        c[t + 1] = c[t] * (BE * R_[t]) ** (1 / GA)
    return c, k
lo, hi = 0.0, 2.0
for _ in range(200):
    mid = (lo + hi) / 2; lo, hi = (mid, hi) if household(mid)[1][-1] > K[SH - 1] else (lo, mid)
ch, kh = household((lo + hi) / 2)
W0 = (eta[0] + 1 - DE) * kb + pv(s17, w[:SH] - tauh, (f(kb) - kb * fp(kb)) - .4)   # (eta_0 + 1 - delta) k_0 + PV(w - tau_h)
ratio = (q / BE ** np.arange(SH)) ** (-1 / GA)                                          # c_t/c_0 implied by prices alone
c0pv = W0 / pv(s17, ratio, ratio[-1])
print(f"      household: shooting c_0 = {ch[0]:.8f}, budget formula c_0 = W_0/sum q_t (c_t/c_0) = {c0pv:.8f}, Big K C_0 = {C[0]:.8f}; "
      f"max|c_t - C_t| (t<=100) = {np.max(np.abs(ch[:101] - C[:101])):.1e}, max|k_t - K_t| (t<=100) = {np.max(np.abs(kh[:101] - K[:101])):.1e}")
print(f"      PV of lump-sum taxes = PV of g = {pv(s17, tauh, .4):.5f} (old policy: {pv(s0, .2 * np.ones(SH), .2):.5f}); "
      f"household wealth W_0 = {W0:.5f} (old: {(fp(kb) + 1 - DE) * kb + pv(s0, (f(kb) - kb * fp(kb) - .2) * np.ones(SH), f(kb) - kb * fp(kb) - .2):.5f})")
print(f"      at the new prices the old plan c_t = {cb:.5f} would cost {cb * pv(s17, np.ones(SH), 1.0):.5f} > W_0; "
      f"R_(t,t+1) - 1 at t = 0, 5, 9, 10, 20: {R_[[0, 5, 9, 10, 20]] - 1} (rho = {RHO:.5f})")

# ================================================================== 11.20 elastic labor, foreseen jump in tau_n (figure 11.12.3)
BB = 3.0


def ss_lab(tn=0.0, g=.2):
    kt = kss(); w_ = (1 - AL) * kt ** AL; c = (1 - tn) * w_ / BB; n = (c + g) / (f(kt) - DE * kt)
    return kt, w_, c, n, n * kt


def path_lab(k0, tn, g=.2, S=SH):
    tn, g = ext(tn, S + 1), ext(g, S + 1)
    kt, w_, cT, nT, kT = ss_lab(tn[-1], g[-1])

    def parts(x):
        n = x[0::2]; k = np.r_[k0, x[1::2], kT]                   # n_0..n_{S-1}; k_0..k_S
        y = k[:-1] ** AL * n ** (1 - AL); c = y + (1 - DE) * k[:-1] - g[:-1] - k[1:]
        wage = (1 - AL) * (k[:-1] / n) ** AL; Rk = AL * (k[1:-1] / n[1:]) ** (AL - 1) + 1 - DE
        return n, k, c, wage, Rk

    def resid(x):
        n, k, c, wage, Rk = parts(x); r = np.empty(x.size)
        r[0::2] = np.log(BB * c) - np.log((1 - tn[:-1]) * wage)                          # B c_t = (1-tau_nt) w_t
        r[1::2] = -np.log(c[:-1]) - np.log(BE) + np.log(c[1:]) - np.log(Rk)             # 1/c_t = beta Rbar_{t+1}/c_{t+1}
        return r
    _, _, c0_, n0_, k0_ = ss_lab(tn[0], g[0])
    x0 = np.empty(2 * S - 1); x0[0::2] = np.where(np.arange(S) < 10, n0_, nT); x0[1::2] = np.linspace(k0, kT, S - 1)
    x, it = newton(resid, x0, band=(2, 2))
    n, k, c, wage, Rk = parts(x)
    return dict(n=n, k=k, c=c, w=wage, Rb=Rk, it=it)
kt0, w0_, c0_, n0_, k0_ = ss_lab(0.0); _, _, c1_, n1_, k1_ = ss_lab(.2)
print(f"\n11.20 steady states: k/n = {kt0:.5f}, w = {w0_:.5f};  tau_n=0: c = {c0_:.5f}, n = {n0_:.5f}, k = {k0_:.5f};  "
      f"tau_n=.2: c = {c1_:.5f}, n = {n1_:.5f}, k = {k1_:.5f}")
s20 = path_lab(k0_, step(10, 0, .2))
kap = s20["k"][:-1] / s20["n"]
print(f"      Newton its {s20['it']}; c_t/c - 1 (t=0..9): {s20['c'][:10] / c0_ - 1}")
print(f"      k/n (t=0..11): {kap[:12]}; max deviations before t=10: c {100 * np.max(np.abs(s20['c'][:10] / c0_ - 1)):.3f}%, "
      f"k/n {100 * np.max(np.abs(kap[:10] / kt0 - 1)):.3f}%; root of k' = [f(k~)/k~ + 1 - delta] k - g - c: {f(kt0) / kt0 + 1 - DE:.4f}; "
      f"pre-tax wage jump at 10: {100 * (s20['w'][10] / w0_ - 1):.2f}%")
print(f"      n_0 = {s20['n'][0]:.5f}, n_9 = {s20['n'][9]:.5f}, n_10 = {s20['n'][10]:.5f}; k_10 = {s20['k'][10]:.5f}; w_10 = {s20['w'][10]:.5f}; "
      f"c_10 = {s20['c'][10]:.5f}; Rbar_10 = {s20['Rb'][9]:.5f}; after-tax wage at 10 = {.8 * s20['w'][10]:.5f} ({100 * (.8 * s20['w'][10] / w0_ - 1):.2f}%)")
su20 = path_lab(k0_, .2)            # candidate with no response before t=10: from t=10 on it is the unforeseen-change path
print(f"      'no response before 10' candidate: c_10/c_9 = {su20['c'][0] / c0_:.5f} but beta*Rbar_10 = "
      f"{BE * (AL * (k0_ / su20['n'][0]) ** (AL - 1) + 1 - DE):.5f}, so the Euler equation between 9 and 10 fails")
print(f"      checks: max|B c_t - (1-tau_n) w_t| = {np.max(np.abs(BB * s20['c'] - (1 - step(10, 0, .2)[:SH]) * s20['w'])):.1e}; "
      f"horizon S=400: max|k diff| = {np.max(np.abs(path_lab(k0_, step(10, 0, .2), S=400)['k'][:SH] - s20['k'][:SH])):.1e}")
fig, ax = plt.subplots(2, 3, figsize=(9, 4.2)); tt = np.arange(41)
for a_, y, nm, v in zip(ax.ravel(), (s20["k"][:41], s20["c"][:41], s20["n"][:41], np.r_[R0, s20["Rb"][:40]], s20["w"][:41], step(10, 0, .2)[:41]),
                        ("$k_t$", "$c_t$", "$n_t$", r"$\bar R_t$", "$w_t$", r"$\tau_{nt}$"), (k0_, c0_, n0_, R0, w0_, 0)):
    a_.plot(tt, y, "k-"); a_.axhline(v, ls="--", color="grey", lw=.8); a_.set_title(nm); a_.set_xlabel("$t$")
plt.tight_layout(); plt.savefig(FIG + "ch11_20.pdf"); plt.close()
