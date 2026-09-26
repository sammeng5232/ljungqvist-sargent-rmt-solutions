"""Chapter 13 (Asset Pricing Theory): Harrison-Kreps tables (13.2) and numerical checks for 13.1, 13.3, 13.5."""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)
dvd = np.array([0., 1.])                        # dividend in states 1, 2


def hk_table(Pa, Pb, beta=.75, tol=1e-13):
    """Rows of Table 13.A.1: p_a, p_b (homogeneous), p (optimistic marginal investor), phat_a, phat_b, pcheck (pessimistic)."""
    I = np.eye(2)
    pa = beta * np.linalg.solve(I - beta * Pa, Pa @ dvd)
    pb = beta * np.linalg.solve(I - beta * Pb, Pb @ dvd)
    def fix(op):
        p = np.zeros(2)
        for it in range(10000):
            pn = beta * op(Pa @ (dvd + p), Pb @ (dvd + p))
            if np.max(np.abs(pn - p)) < tol:
                return pn, it
            p = pn
    p, it1 = fix(np.maximum)
    pc, it2 = fix(np.minimum)
    phat_a = beta * Pa @ (dvd + p)              # what type a would pay (valuing resale at p)
    phat_b = beta * Pb @ (dvd + p)
    return pa, pb, p, phat_a, phat_b, pc, it1, it2


for tag, Pa, Pb in (("book (appendix A)", np.array([[1/2, 1/2], [2/3, 1/3]]), np.array([[2/3, 1/3], [1/4, 3/4]])),
                    ("13.2(a)", np.array([[1/2, 1/2], [3/4, 1/4]]), np.array([[3/4, 1/4], [1/2, 1/2]])),
                    ("13.2(b)", np.array([[1/2, 1/2], [1/2, 1/2]]), np.array([[3/4, 1/4], [2/3, 1/3]]))):
    pa, pb, p, ha, hb, pc, i1, i2 = hk_table(Pa, Pb)
    print(f"{tag}:\n  p_a {pa}\n  p_b {pb}\n  p (optimists) {p}  [{i1} its]\n  phat_a {ha}\n  phat_b {hb}\n  pcheck (pessimists) {pc}  [{i2} its]")
    # who is marginal in each state under the optimistic price
    print("  marginal (optimistic) type by state:", ["a" if a >= b else "b" for a, b in zip(Pa @ (dvd + p), Pb @ (dvd + p))],
          " pessimistic:", ["a" if a <= b else "b" for a, b in zip(Pa @ (dvd + pc), Pb @ (dvd + pc))])

# ---------------------------------------------------------------- 13.1 / 13.3 illustration: 2-state growth chain
beta, gam = .95, 2.
lam = np.array([.97, 1.05]); P = np.array([[.6, .4], [.2, .8]])
Q = beta * P * lam[None, :] ** (-gam)                      # one-step Arrow prices Q_ij
Qt = Q * lam[None, :]                                      # beta P_ij lam_j^(1-gamma)
v = np.linalg.solve(np.eye(2) - Qt, Qt @ np.ones(2))       # price-dividend ratio of the Lucas tree
print("13.1 Arrow prices Q =\n", Q, "\n     two-step Arrow prices Q^2 =\n", Q @ Q, "\n     risk-free rates R_f = 1/rowsum(Q):", 1 / Q.sum(1))
print("     tree price/dividend v =", v, "; spectral radius of Qtilde =", max(abs(np.linalg.eigvals(Qt))))
vv = np.zeros(2)
for it in range(5000):
    vv = Qt @ (1 + vv)
print("     check by iterating v = Qtilde(1+v):", vv)
# 13.3: consol with coupon zeta, and call/put options with strike pS on the consol
zeta = 1.0
pcons = np.linalg.solve(np.eye(2) - Q, Q @ (zeta * np.ones(2)))
pS = pcons.mean()
w = np.zeros(2); J = np.zeros(2)
for it in range(20000):
    wn = np.maximum(pcons - pS, Q @ w); Jn = np.maximum(pS - pcons, Q @ J)
    if max(np.max(np.abs(wn - w)), np.max(np.abs(Jn - J))) < 1e-14: break
    w, J = wn, Jn
print(f"13.3 consol prices p(lambda) = {pcons}; strike pS = {pS:.4f}; call w = {w} (exercise where w = p - pS: {np.isclose(w, pcons - pS)}); put J = {J} (exercise: {np.isclose(J, pS - pcons)}); its {it}")
print("     contraction modulus = max row sum of Q =", Q.sum(1).max())

# ---------------------------------------------------------------- 13.5 check of the Arrow-security holdings formula
x = np.array([1.0, 1.3, .8]); P3 = np.array([[.5, .3, .2], [.2, .6, .2], [.3, .3, .4]]); b5, g5 = .95, 2.
Q3 = b5 * P3 * (x[None, :] / x[:, None]) ** (-g5)
Pt = P3 * (x[None, :] / x[:, None]) ** (1 - g5)
Psi = np.linalg.solve(np.eye(3) - b5 * Pt, np.ones(3))
print("13.5 Psi =", Psi, " c0 = 1/Psi(s0):", 1 / Psi)
# budget identity c + sum_j Q_ij a_j = W with a_j = c (x_j/x_i) Psi_j and W = c Psi_i (take c = 1)
lhs = 1 + (Q3 * (x[None, :] / x[:, None]) * Psi[None, :]).sum(1)
print("     budget check  1 + sum_j Q_ij (x_j/x_i) Psi_j  vs  Psi_i:", lhs, Psi)
