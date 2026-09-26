"""Chapter 7 (Recursive Competitive Equilibrium: I): numerical parts of exercises 7.1-7.5."""
import numpy as np
from rmtlib import olrp

np.set_printoptions(precision=6, suppress=True, linewidth=140)
A0, A1, beta, d = 100., .05, .95, 10.


def firm_rule(H0, H1):
    """Competitive firm, x = [y, Y, 1], u = y' - y.  Returns (h0, h1, h2) with y' = h0 + h1 y + h2 Y."""
    A = np.array([[1, 0, 0], [0, H1, H0], [0, 0, 1.]]); B = np.array([[1.], [0], [0]])
    R = np.array([[0, A1 / 2, -A0 / 2], [A1 / 2, 0, 0], [-A0 / 2, 0, 0]])     # minus the return A0 y - A1 y Y
    F, P, it = olrp(beta, A, B, np.array([[d / 2]]), R)
    return -F[0, 2], 1 - F[0, 0], -F[0, 1], P, it


# 7.1
h0, h1, h2, P, it = firm_rule(95.5, .95)
print(f"7.1 (H0,H1)=(95.5,.95): y' = {h0:.6f} + {h1:.6f} y + {h2:.6f} Y   ({it} its)\n    P =\n{P}")
for n in (1, 2, 10):
    print(f"    n={n}: actual law Y' = {n*h0:.6f} + {h1 + n*h2:.6f} Y")

# 7.2
for tag, (H0, H1) in (("i", (94.0888, .9211)), ("ii", (93.22, .9433)), ("iii", (95.08187459215024, .95245906270392))):
    h0, h1, h2, _, _ = firm_rule(H0, H1)
    print(f"7.2 ({tag}) perceived ({H0}, {H1}) -> actual ({h0:.6f}, {h1 + h2:.6f})")
# iterate the map M with damping from (95.5, .95)
H = np.array([95.5, .95])
for k in range(1, 2001):
    h0, h1, h2, _, _ = firm_rule(*H)
    Hn = np.array([h0, h1 + h2])
    if np.max(np.abs(Hn - H)) < 1e-12:
        break
    H = .5 * H + .5 * Hn
print(f"    damped iteration on M converges to H = {H} after {k} steps")

# 7.3 planner
Ap = np.eye(2); Bp = np.array([[1.], [0]]); Rp = np.array([[A1 / 2, -A0 / 2], [-A0 / 2, 0]])
Fp, Pp, itp = olrp(beta, Ap, Bp, np.array([[d / 2]]), Rp)
s1, s0 = 1 - Fp[0, 0], -Fp[0, 1]
print(f"7.3 planner: Y' = {s0:.10f} + {s1:.12f} Y ; steady state {s0/(1-s1):.4f} ; V(Y) = -[{Pp[0,0]:.6f} Y^2 + {2*Pp[0,1]:.4f} Y + {Pp[1,1]:.2f}]")

# 7.4 monopoly
Rm = np.array([[A1, -A0 / 2], [-A0 / 2, 0]])
Fm, Pm, itm = olrp(beta, Ap, Bp, np.array([[d / 2]]), Rm)
m1, m0 = 1 - Fm[0, 0], -Fm[0, 1]
print(f"7.4 monopoly: Y' = {m0:.6f} + {m1:.6f} Y ; steady state {m0/(1-m1):.4f}")

# 7.5 duopoly: x = [y1, y2, 1], u_i = y_i' - y_i; iterate best responses to a symmetric linear MPE
Ad = np.eye(3); B1 = np.array([[1.], [0], [0]]); B2 = np.array([[0.], [1], [0]])
R1 = np.array([[A1, A1 / 2, -A0 / 2], [A1 / 2, 0, 0], [-A0 / 2, 0, 0]])       # minus  (A0 - A1(y1+y2)) y1
R2 = np.array([[0, A1 / 2, 0], [A1 / 2, A1, -A0 / 2], [0, -A0 / 2, 0]])
Q = np.array([[d / 2]])
F1 = np.zeros((1, 3)); F2 = np.zeros((1, 3))
for k in range(1, 5001):
    F1n, P1, _ = olrp(beta, Ad - B2 @ F2, B1, Q, R1)
    F2n, P2, _ = olrp(beta, Ad - B1 @ F1n, B2, Q, R2)
    if max(np.max(np.abs(F1n - F1)), np.max(np.abs(F2n - F2))) < 1e-11:
        F1, F2 = F1n, F2n
        break
    F1, F2 = F1n, F2n
Acl = Ad - B1 @ F1 - B2 @ F2
print(f"7.5 duopoly MPE after {k} rounds:\n    y1' = {Acl[0,2]:.6f} + {Acl[0,0]:.6f} y1 + {Acl[0,1]:.6f} y2\n    y2' = {Acl[1,2]:.6f} + {Acl[1,0]:.6f} y1 + {Acl[1,1]:.6f} y2")
x = np.array([0., 0., 1.])
for t in range(4000):
    x = Acl @ x
print(f"    steady state y1 = y2 = {x[0]:.4f}, industry output {x[0]+x[1]:.4f}; eig(A-B1F1-B2F2) = {np.linalg.eigvals(Acl[:2,:2])}")
# check: each firm's rule is a best response (Riccati residual of firm 1's problem given firm 2)
F1c, _, _ = olrp(beta, Ad - B2 @ F2, B1, Q, R1)
print("    best-response check max|F1 - BR(F2)| =", np.max(np.abs(F1c - F1)))
# static Cournot benchmark for comparison
print(f"    static Cournot industry output {2*A0/(3*A1):.4f}, competitive {A0/A1:.4f}, monopoly {A0/(2*A1):.4f}")
