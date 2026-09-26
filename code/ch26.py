"""Chapter 26 (Two Topics in International Trade): exercise 26.1, the Bond-Park model with
u_L(t) = -.5(t-.5)^2, u_W(t) = -.5t^2 (so u_S(t) = u_W - u_L = .125 - .5t), gamma_S = .4.

The constrained Pareto frontier is computed in closed form (regions I-III of section 26.3.9) and checked by
value-function iteration on the Bellman equation (26.3.26)-(26.3.27), started from the unconstrained frontier.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rmtlib import FIG

gS = .4
uL = lambda t: -.5 * (t - .5) ** 2
uW = lambda t: -.5 * t ** 2
uS = lambda t: uW(t) - uL(t)                       # = .125 - .5 t
tN = min(.5, 1 - gS)                               # argmax u_L = .5 < 1 - gamma_S = .6
print(f"u_S(t) = {uS(0.):.4f} {uS(1.)-uS(0.):+.4f} t;  Nash tariff tN = {tN};  u_L(tN) = {uL(tN)}, u_L(0) = {uL(0.)}, u_S(0) = {uS(0.)}, u_S(tN) = {uS(tN)}")
bc = np.sqrt((uL(tN) - uL(0.)) / (uS(0.) - uS(tN)))
print(f"26.1(a) beta_c = sqrt({uL(tN)-uL(0.):.4f}/{uS(0.)-uS(tN):.4f}) = {bc:.6f}")


def objects(b):
    vNL, vNS, W = uL(tN) / (1 - b), uS(tN) / (1 - b), uW(0.) / (1 - b)
    vs = (vNL - uL(0.)) / b                           # (26.3.21)
    vss = (uL(0.) + b * (uS(0.) - uS(tN))) / (1 - b)  # (26.3.22)
    vmax = uL(tN) + b * (W - vNS)                    # end of region II (t = tN, P = vNS)
    return vNL, vNS, W, vs, vss, vmax


def frontier(v, b):
    """Closed-form constrained Pareto frontier P(v) on [vNL, vmax]."""
    vNL, vNS, W, vs, vss, vmax = objects(b)
    v = np.asarray(v, float)
    P = W - v
    II = v > vss
    t = .5 - np.sqrt(np.maximum(2 * (vmax - v[II]), 0.))   # u_L(t) = v - b(W - vNS), t < tN
    P[II] = uS(t) + b * vNS
    return P


for b in (.75, .8, .9, .95):
    vNL, vNS, W, vs, vss, vmax = objects(b)
    print(f"  beta = {b}: e_Smin = {(uL(tN)-uL(0.))/b:.5f}, e_Smax = {b*(uS(0.)-uS(tN)):.5f};  vN_L = {vNL:.5f}, vN_S = {vNS:.5f}, W = {W:.5f};"
          f"  v*_L = {vs:.5f}, v**_L = {vss:.5f}, vmax_L = {vmax:.5f}, P(v**_L) - vN_S = {W - vss - vNS:.5f}")

# ---------------------------------------------------------------- (b) region II table and (c) region III, beta = .8
b = .8
vNL, vNS, W, vs, vss, vmax = objects(b)
print(f"\n26.1(b) beta = {b}: region II, v_L in ({vss}, {vmax}]")
print("   v_L      t_L     theta    P(v_L)   e_S+b*y   [y=v**: e_S, e'_S]   [y=v*: e_S, e'_S]")
for v in np.linspace(vss, vmax, 6):
    t = .5 - np.sqrt(2 * (vmax - v)); th = .5 / (.5 - t) if t < .5 else np.inf
    tot = b * (W - vNS)
    ys = [vss, vs]
    es = [(tot - b * y, (1 - b) * y - uL(0.)) for y in ys]
    print(f"  {v:.4f}  {t:.5f}  {th:8.4f}  {frontier([v], b)[0]:.5f}  {tot:.4f}    [{es[0][0]:.4f}, {es[0][1]:.4f}]      [{es[1][0]:.4f}, {es[1][1]:.4f}]")
    # check the promise and both participation constraints
    for y, (e, e1) in zip(ys, es):
        assert abs(uL(t) + e + b * y - v) < 1e-12 and e >= -1e-12 and e1 <= e + 1e-12    # promise kept; e'_S <= e_S
        assert uL(t) + b * y >= vNL - 1e-12                        # PC_L (the transfer is kept either way)
        assert -e + b * (W - y) >= b * vNS - 1e-12                 # PC_S

print(f"\n26.1(c) beta = {b}: region III, v_L in [{vNL}, {vs:.5f}): baseline y = v*_L, e_S = v_L - vN_L, e'_S = (1-b) v*_L - u_L(0)")
for v in (0., .05, .1, .15):
    y = vs; e = v - uL(0.) - b * y; e1 = (1 - b) * y - uL(0.)
    ok = (abs(uL(0.) + e + b * y - v) < 1e-12, uL(0.) + b * y >= vNL - 1e-12, -e + b * (W - y) >= b * vNS - 1e-12,
          -e1 + b * (W - y) >= b * vNS - 1e-12, abs(uL(0.) + e1 + b * y - y) < 1e-12)
    ymax = min((v - uL(0.)) / b, vss)
    print(f"  v_L = {v:.3f}: y = {y:.5f}, e_S = {e:.5f}, e'_S = {e1:.5f}, P(v_L) = {W - v:.5f}; constraints ok: {all(ok)};"
          f" admissible y in [{vs:.5f}, {ymax:.5f}]")

# ---------------------------------------------------------------- value-function iteration check (beta = .8)
dv = 1 / 640
V = np.arange(0, W - vNS + dv / 2, dv)              # candidate promised values for L, [vN_L, W - vN_S]
Tg = np.linspace(0, 1 - gS, 601)                    # tariffs in [0, 1 - gamma_S]
P = W - V                                           # start from the unconstrained frontier
UL, US = uL(Tg)[:, None], uS(Tg)[:, None]
for it in range(400):
    Py = P[None, :]
    base = UL + b * V[None, :]                       # u_L(t) + b y
    okL = base >= vNL - 1e-12                        # PC_L
    Pn = np.full_like(P, -np.inf)
    for i, v in enumerate(V):
        e = np.maximum(0., v - base)                 # smallest transfer that keeps the promise
        ok = okL & np.isfinite(Py) & (e <= b * (Py - vNS) + 1e-12)
        val = np.where(ok, US - e + b * Py, -np.inf)
        Pn[i] = val.max()
    fin = np.isfinite(Pn) & np.isfinite(P)
    diff = np.max(np.abs(Pn[fin] - P[fin])) if fin.any() else np.inf
    same_dom = np.array_equal(np.isfinite(Pn), np.isfinite(P))
    P = Pn
    if diff < 1e-12 and same_dom:
        break
dom = V[np.isfinite(P)]
err = np.max(np.abs(P[np.isfinite(P)] - frontier(dom, b)))
print(f"\nVFI check (beta = {b}, grid dv = 1/640, 601 tariffs): {it+1} iterations; domain [{dom.min():.5f}, {dom.max():.5f}] (closed form [{vNL}, {vmax}]);"
      f" max |P_VFI - P_closed| = {err:.2e}")
reg = (dom <= vss + 1e-12)
print(f"   on regions III and I the error is {np.max(np.abs(P[np.isfinite(P)][reg] - frontier(dom[reg], b))):.2e}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, b_ in zip(ax, (.8, .9)):
    vNL, vNS, W, vs, vss, vmax = objects(b_)
    vv = np.linspace(vNL, vmax, 800)
    a.plot(vv, W - vv, color="grey", lw=.8, ls=":", label="unconstrained frontier")
    a.plot(vv, frontier(vv, b_), "k", lw=1.6, label="constrained frontier $P(v_L)$")
    for x, lab in ((vNL, r"$v_L^N$"), (vs, r"$v_L^*$"), (vss, r"$v_L^{**}$")):
        a.axvline(x, color="grey", lw=.5); a.text(x, 1.01, lab, fontsize=8, ha="center", va="bottom", transform=a.get_xaxis_transform())
    a.axhline(vNS, color="grey", lw=.5, ls="--"); a.text(vNL + .3 * (vmax - vNL), vNS, r"$v_S^N$", fontsize=8, va="bottom")
    if b_ == .8:
        a.plot(dom[::4], P[np.isfinite(P)][::4], "r.", ms=2, label="value-function iteration")
    a.set_title(rf"$\beta={b_}$", fontsize=9, pad=14); a.set_xlabel(r"$v_L$"); a.set_ylabel(r"$v_S$")
    a.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(FIG + "ch26_01.pdf"); plt.close()
