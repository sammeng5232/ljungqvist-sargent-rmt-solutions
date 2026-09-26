"""Small numerical toolkit shared by the chapter scripts (Python counterparts of the book's Matlab programs)."""
import numpy as np

FIG = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "figures", "")


def sc(x):
    """Scalar value of a 1x1 (or 0-d) array."""
    return np.asarray(x).item()


def doublej(A, CC, tol=1e-15, maxit=200):
    """Solve X = A X A' + CC by the doubling algorithm (the book's doublej.m).
    Works when A has a unit eigenvalue that CC does not excite (a constant in the state)."""
    A = np.asarray(A, float); X = np.asarray(CC, float).copy(); Ak = A.copy()
    for _ in range(maxit):
        Xn = X + Ak @ X @ Ak.T
        Ak = Ak @ Ak
        if np.max(np.abs(Xn - X)) < tol * max(1.0, np.max(np.abs(Xn))):
            return Xn
        X = Xn
    return X


def stationary_mean(A):
    """Eigenvector of A for the unit eigenvalue, scaled so that the constant (last state) equals one."""
    w, V = np.linalg.eig(A)
    k = np.argmin(np.abs(w - 1.0))
    v = np.real(V[:, k])
    return v / v[-1]


def lq_howard(A, B, R, Q, beta, F0, tol=1e-12, maxit=1000, H=None):
    """Howard policy improvement (2.4.30), (5.2.10)-(5.2.11) for min E sum beta^t (x'Rx + u'Qu + 2u'Hx),
    x' = Ax + Bu + Cw, started from a rule F0 with sqrt(beta)(A-BF0) stable.  Because every rule it visits keeps
    sqrt(beta)(A-BF) stable, it delivers the stabilising solution (the one that respects no-Ponzi conditions).
    Returns F, P, number of improvement steps."""
    F = np.atleast_2d(F0).astype(float)
    H = np.zeros((B.shape[1], A.shape[0])) if H is None else np.atleast_2d(H).astype(float)
    for it in range(1, maxit + 1):
        Ao = np.sqrt(beta) * (A - B @ F)
        Y = R + F.T @ Q @ F - F.T @ H - H.T @ F     # one-period loss matrix under u = -Fx
        P = doublej(Ao.T, Y)                        # P = Y + beta (A-BF)' P (A-BF)
        Fn = np.linalg.solve(Q + beta * B.T @ P @ B, beta * B.T @ P @ A + H)
        if np.max(np.abs(Fn - F)) < tol * max(1.0, np.max(np.abs(Fn))):
            return Fn, P, it
        F = Fn
    raise RuntimeError("Howard iterations did not converge")


def lq_riccati(A, B, R, Q, beta, tol=1e-12, maxit=100000):
    """Iterate on the Riccati equation (2.4.31) from P = 0."""
    P = np.zeros_like(R, dtype=float)
    for it in range(maxit):
        F = beta * np.linalg.solve(Q + beta * B.T @ P @ B, B.T @ P @ A)
        Pn = R + F.T @ Q @ F + beta * (A - B @ F).T @ P @ (A - B @ F)
        if np.max(np.abs(Pn - P)) < tol:
            return F, Pn, it
        P = Pn
    raise RuntimeError("Riccati iterations did not converge")


def olrp(beta, A, B, Q, R, H=None, tol=1e-12, maxit=200000, P0=None):
    """Discounted optimal linear regulator with cross products (the book's olrp.m):
    minimise sum beta^t (x'Rx + u'Qu + 2u'Hx) s.t. x' = Ax + Bu, by iterating the Riccati equation of exercise 5.1
        P = R + beta A'PA - (beta A'PB + H')(Q + beta B'PB)^{-1}(beta B'PA + H)
    from P0 (default 0).  Returns F (u = -Fx), P and the number of iterations."""
    A = np.atleast_2d(A).astype(float); B = np.atleast_2d(B).astype(float)
    Q = np.atleast_2d(Q).astype(float); R = np.atleast_2d(R).astype(float)
    H = np.zeros((B.shape[1], A.shape[0])) if H is None else np.atleast_2d(H).astype(float)
    P = np.zeros_like(R) if P0 is None else np.array(P0, float)
    for it in range(1, maxit + 1):
        F = np.linalg.solve(Q + beta * B.T @ P @ B, beta * B.T @ P @ A + H)
        Pn = R + beta * A.T @ P @ A - (beta * A.T @ P @ B + H.T) @ F
        Pn = (Pn + Pn.T) / 2
        if np.max(np.abs(Pn - P)) < tol * max(1.0, np.max(np.abs(Pn))):
            F = np.linalg.solve(Q + beta * B.T @ Pn @ B, beta * B.T @ Pn @ A + H)
            return F, Pn, it
        P = Pn
    raise RuntimeError("olrp: Riccati iterations did not converge")


def kalman_steady(A, C, G, R, Sigma0=None, tol=1e-14, maxit=100000):
    """Iterate the Kalman filter (2.7.14) to its time-invariant limit; returns K, Sigma, iterations."""
    n = A.shape[0]
    S = np.eye(n) if Sigma0 is None else np.atleast_2d(Sigma0).astype(float)
    for it in range(maxit):
        K = A @ S @ G.T @ np.linalg.inv(G @ S @ G.T + R)
        Sn = C @ C.T + K @ R @ K.T + (A - K @ G) @ S @ (A - K @ G).T
        if np.max(np.abs(Sn - S)) < tol:
            return K, Sn, it
        S = Sn
    raise RuntimeError("Kalman iterations did not converge")
