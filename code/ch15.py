"""Chapter 15 (Economic Growth): numerical illustration for exercise 15.3 (vintage capital echoes)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root
from rmtlib import FIG

beta, alpha, z = .95, .3, 1.0
up = lambda c: 1.0 / c                                   # log utility
f = lambda K: z * K ** alpha
fp = lambda K: z * alpha * K ** (alpha - 1)
Kss = (alpha * beta * (1 + beta)) ** (1 / (1 - alpha)); kss = Kss / 2
print(f"15.3 steady state: K* = {Kss:.6f}, k* = {kss:.6f}, c* = {f(Kss) - kss:.6f}; check z f'(K*) beta(1+beta) = {fp(Kss)*beta*(1+beta):.6f}")


def euler(kprev2, kprev, k, knext, knext2):
    """Euler equation for k_{t+1}=k, given k_{t-1}=kprev2, k_t=kprev, k_{t+2}=knext, k_{t+3}=knext2."""
    c0 = f(kprev + kprev2) - k
    c1 = f(k + kprev) - knext
    c2 = f(knext + k) - knext2
    return -up(c0) + beta * up(c1) * fp(k + kprev) + beta ** 2 * up(c2) * fp(knext + k)


# linearise: coefficients on (k_{t-1},...,k_{t+3})
h = 1e-6; a = []
for j in range(5):
    x = np.full(5, kss); x[j] += h; xm = np.full(5, kss); xm[j] -= h
    a.append((euler(*x) - euler(*xm)) / (2 * h))
roots = np.roots(a[::-1])                                  # a_4 lam^4 + ... + a_0 = 0
print("     characteristic roots:", np.round(np.sort_complex(roots), 6), " products of reciprocal pairs:",
      np.round(np.sort(np.abs(roots))[0] * np.sort(np.abs(roots))[3], 6), np.round(np.sort(np.abs(roots))[1] * np.sort(np.abs(roots))[2], 6), " 1/beta =", round(1 / beta, 6))

# optimal path from unequal vintages: k_{-1} = .5 k*, k_0 = 1.5 k* (same total capital)
T = 80; km1, k0 = .5 * kss, 1.5 * kss
def resid(kk):
    k = np.r_[km1, k0, kk, kss, kss]                       # k_{-1}, k_0, k_1..k_T, k_{T+1}, k_{T+2}
    return np.array([euler(k[t - 1], k[t], k[t + 1], k[t + 2], k[t + 3]) for t in range(1, T + 1)])
sol = root(resid, np.full(T, kss), method="hybr", tol=1e-13)
kpath = np.r_[km1, k0, sol.x]
print("     solver success:", sol.success, " max residual", np.max(np.abs(resid(sol.x))))
print("     k_t / k* for t=-1..8:", np.round(kpath[:10] / kss, 4))
K = kpath[1:] + kpath[:-1]
c = f(K[:-1]) - kpath[2:]
print("     K_t / K* for t=0..8:", np.round(K[:9] / Kss, 5), "\n     c_t / c* for t=0..8:", np.round(c[:9] / (f(Kss) - kss), 5))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(np.arange(-1, 25), kpath[:26] / kss, "k.-"); ax[0].axhline(1, color="grey", lw=.6)
ax[0].set_title(r"new capital $k_t/k^*$", fontsize=9); ax[0].set_xlabel("t")
ax[1].plot(np.arange(0, 25), c[:25] / (f(Kss) - kss), "k.-"); ax[1].axhline(1, color="grey", lw=.6)
ax[1].set_title(r"consumption $c_t/c^*$", fontsize=9); ax[1].set_xlabel("t")
plt.tight_layout(); plt.savefig(FIG + "ch15_03.pdf"); plt.close()
