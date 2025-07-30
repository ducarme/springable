import numpy as np
from scipy.special import factorial
from typing import overload


def valid_t_among(t: np.ndarray):
    t = np.real(t[np.isreal(t)])
    return np.sort(t[np.logical_and(t >= 0.0, t <= 1.0)])

def evaluate_poly(t: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """ Computes and returns the value of the Bezier polynomial at t using de Castlejau's algorithm """
    beta = [c for c in coefs]  # copies array
    n = len(beta)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]


def elevate_order(coefs: np.ndarray) -> np.ndarray:
    n = coefs.shape[0] - 1
    new_coefs = np.empty(n + 2)
    w = np.append(coefs, 0.0)
    for i in range(n + 2):
        wi_1 = 0.0 if i == 0 else w[i-1]
        new_coefs[i] = ((n + 1 - i) * w[i] + i * wi_1) / (n + 1)
    return new_coefs


def lower_order(coefs: np.ndarray) -> np.ndarray:
    return coefs[:-1]


def get_monomial_coefs(coefs: np.ndarray):
    """ returns the coefficient for each power.
    Element at index -1 is the independent term, element at index -2 is the linear term, etc"""
    deg = coefs.shape[0] - 1
    monomial_coefs = np.empty(deg + 1)
    for j in range(deg + 1):
        monomial_coefs[j] = factorial(deg) / factorial(deg - j) \
                            * np.sum(
            [(-1.) ** (i + j) * coefs[i] / factorial(i) / factorial(j - i) for i in range(j + 1)])
    return monomial_coefs[::-1]


def get_roots(coefs: np.ndarray):
    return valid_t_among(np.roots(get_monomial_coefs(coefs)))


def evaluate_derivative_poly(t: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """ Computes and returns the value of the derivative of the Bezier polynomial at t using de Castlejau's algorithm"""
    coefs = np.diff(coefs) * (coefs.shape[0] - 1)
    return evaluate_poly(t, coefs)


def get_monomial_coefs_of_derivative(coefs: np.ndarray):
    coefs = np.diff(coefs) * (coefs.shape[0] - 1)
    return get_monomial_coefs(coefs)


def get_extrema(coefs: np.ndarray):
    return valid_t_among(np.roots(get_monomial_coefs_of_derivative(coefs)))


def is_monotonic(coefs: np.ndarray):
    return get_extrema(coefs).shape[0] == 0


def evaluate_inverse_poly(x: np.ndarray, coefs: np.ndarray):
    """ polynomial is assumed to be monotonic """
    mc = get_monomial_coefs(coefs)
    if isinstance(x, np.ndarray):
        t = np.empty_like(x)
        for i in range(x.shape[0]):
            mc_ = mc.copy()
            mc_[-1] -= x[i]
            roots = valid_t_among(np.roots(mc_))
            if roots.size > 0:
                t[i] = roots[0]
            else:
                poly_1 = evaluate_poly(np.array(1.0), coefs)
                slope_1 = evaluate_derivative_poly(np.array(1.0), coefs)
                intercept = poly_1 - slope_1 * 1.0
                t[i] = (x[i] - intercept) / slope_1
        return t
    else:
        mc = get_monomial_coefs(coefs)
        mc[-1] -= x
        return valid_t_among(np.roots(mc))[0]


def evaluate_second_derivative_poly(t: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """ Computes and returns the value of the second derivative of the Bezier polynomial at t using de Castlejau's
    algorithm"""
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return evaluate_poly(t, coefs)


def evaluate_third_derivative_poly(t: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    coefs = np.diff(coefs, 3) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2) * (coefs.shape[0] - 3)
    return evaluate_poly(t, coefs)


def get_monomial_coefs_of_second_derivative(coefs: np.ndarray):
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return get_monomial_coefs(coefs)


def get_inflexions(coefs: np.ndarray):
    return valid_t_among(np.roots(get_monomial_coefs_of_second_derivative(coefs)))


def create_antiderivative_of_parametric_bezier(y_coefs: np.ndarray, x_coefs: np.ndarray):
    """
        Computes the area under a parametric curve described by two Bezier poly
    """
    y = np.polynomial.Polynomial(get_monomial_coefs(y_coefs)[::-1])
    dx = np.polynomial.Polynomial(get_monomial_coefs_of_derivative(x_coefs)[::-1])
    ydx = y * dx
    return ydx.integ(1, k=[0.0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    def beta_plus(s, delta_inf=0.5, kappa=1.0):
        return 0.5 * (np.sqrt((s + delta_inf) ** 2 + kappa ** 2) + s + delta_inf)


    def beta_minus(s, delta_inf=0.5, kappa=1.0):
        return -beta_plus(-s, delta_inf, kappa)


    def gamma_plus(s, delta_s, delta_inf, kappa):
        return beta_plus(s - delta_s, delta_inf, kappa) + delta_s


    def gamma_minus(s, delta_s, delta_inf, kappa):
        return beta_minus(s - delta_s, delta_inf, kappa) + delta_s
    
    def smax(x, y, epsilon):
        return 0.5*(x+y) + 0.5*np.sqrt((x-y)**2 + epsilon**2)
    
    def smin(x, y, epsilon):
        return 0.5*(x+y) - 0.5*np.sqrt((x-y)**2 + epsilon**2)


    _u_i = np.array([0.0,
                     3.6469744295621784,
                     1.7703266451373811,
                     2.16166551961459,
                     0.8453438509185234,
                     1.1210598761183754,
                     1.3078352480279523,
                     3.6025041029170413])
    _f_i = np.array([0.0,
                     8.726919339164237,
                     -1.1127308066083579,
                     -0.9608843537414966,
                     11.439909297052154,
                     -2.448979591836734,
                     -2.9956268221574343,
                     6.692176870748298])

    _u_i = np.array([0.0, 1.0, -0.65, 1.0])
    _f_i = np.array([0.0, 3.0, -2.0, 2.0])
    _u_i = np.array([0.0, 1.0, 1.0, 2.0])
    _f_i = np.array([0.0, 3.0, -2.0, 2.0])
    _t = np.linspace(0, 1, 300)
    u = evaluate_poly(_t, _u_i)
    f = evaluate_poly(_t, _f_i)
    du = evaluate_derivative_poly(_t, _u_i)
    d2u = evaluate_second_derivative_poly(_t, _u_i)
    df = evaluate_derivative_poly(_t, _f_i)
    u_roots = get_roots(_u_i)
    u_extrema = get_extrema(_u_i)
    u_inflexions = get_inflexions(_u_i)

    delta_inf = 0.0
    kappa = 0.1
    delta_s = (np.max(df[du > 0] / du[du > 0]) + np.min(df[du < 0] / du[du < 0])) / 2.
    k = ((du >= 0) * gamma_plus(df / du, delta_s, delta_inf, kappa)
         + (du < 0) * gamma_minus(df / du, delta_s, delta_inf, kappa))

    # dfdu = df / du
    # kmax = np.max(dfdu[du>0])
    # kmin = np.min(dfdu[du<0] if (du<0).size != 0 else np.inf)
    # deltak = kmax / 20.0
    # kstar = min(kmin-deltak, kmax+deltak)
    # khat = smax(dfdu+deltak, kstar, deltak) * (du > 0)  + smin(dfdu-deltak, kstar, deltak) * (du <= 0)
    # k = kstar * np.ones_like(du) if kmin-kmax > 2 * deltak else khat

    dfdu = df / du
    kmax = np.max(dfdu[du>0])
    kmin = np.min(dfdu[du<0]) if np.any(du<0) else np.inf
    deltak = kmax / 20
    kstar = min(kmax/2 + kmin/2, kmax+2*deltak)
    khat = smax(dfdu+deltak, kstar, 10*deltak) * (du > 0)  + smin(dfdu-deltak, kstar, 10*deltak) * (du <= 0)
    k = khat


    # fig, ax = plt.subplots()
    # uuu = np.array([0.0, 1.0, 1.01, 5.0])
    # x = np.linspace(0.1, 4.9)
    # ax.plot(x, evaluate_inverse_poly(x, uuu))
    # ax.plot(_x, evaluate_poly(_x, evaluate_poly(_x, uuu)))
    # plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    # axs[0].plot(u, f, 'o', markersize=2)
    # axs[0].plot(evaluate_poly(u_roots, _u_i), evaluate_poly(u_roots, _f_i), '*')
    # axs[0].plot(evaluate_poly(u_extrema, _u_i), evaluate_poly(u_extrema, _f_i), 'o')
    # axs[0].plot(evaluate_poly(u_inflexions, _u_i), evaluate_poly(u_inflexions, _f_i), 's')
    # axs[1].plot(_x, u, 'o', markersize=2)
    # axs[1].plot(u_roots, evaluate_poly(u_roots, _u_i), '*')
    # axs[1].plot(u_extrema, evaluate_poly(u_extrema, _u_i), 'o')
    # axs[1].plot(u_inflexions, evaluate_poly(u_inflexions, _u_i), 's')
    # axs[2].plot(_x, du, 'o', markersize=2)
    # axs[2].plot(u_roots, evaluate_derivative_poly(u_roots, _u_i), '*')
    # axs[2].plot(u_extrema, evaluate_derivative_poly(u_extrema, _u_i), 'o')
    # axs[2].plot(u_inflexions, evaluate_derivative_poly(u_inflexions, _u_i), 's')
    # axs[3].plot(_x, d2u, 'o', markersize=2)
    # axs[3].plot(u_roots, evaluate_second_derivative_poly(u_roots, _u_i), '*')
    # axs[3].plot(u_extrema, evaluate_second_derivative_poly(u_extrema, _u_i), 'o')
    # axs[3].plot(u_inflexions, evaluate_second_derivative_poly(u_inflexions, _u_i), 's')
    axs[0].plot(_t, df / du, 'o', markersize=2)
    axs[0].plot(_t, k, 'ro', markersize=2)
    axs[0].plot(u_roots, evaluate_derivative_poly(u_roots, _f_i) / evaluate_derivative_poly(u_roots, _u_i), '*')
    axs[0].plot(u_extrema, evaluate_derivative_poly(u_extrema, _f_i) / evaluate_derivative_poly(u_extrema, _u_i), 'o')
    axs[0].plot(u_inflexions,
                evaluate_derivative_poly(u_inflexions, _f_i) / evaluate_derivative_poly(u_inflexions, _u_i), 's')
    axs[0].set_ylim((-7.5, 7.5))
    # axs[0].plot(_t, np.ones_like(_t) * kmax+2*deltak, label='kmax + delta')
    # axs[0].plot(_t, np.ones_like(_t) * kmax/2+kmin/2, label='kavg')
    axs[0].legend()
    for ax in axs:
        ax.grid()
    plt.show()
