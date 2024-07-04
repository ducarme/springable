import numpy as np
from scipy.special import factorial


def valid_t_among(t: np.ndarray):
    t = np.real(t[np.isreal(t)])
    return np.sort(t[np.logical_and(t >= 0.0, t <= 1.0)])


def evaluate_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
    """ Computes and returns the value of the Bezier polynomial at t using de Castlejau's algorithm """
    beta = [c for c in coefs]  # copies array
    n = len(beta)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]


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


def evaluate_derivative_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
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


def evaluate_inverse_poly(x: float | np.ndarray, coefs: np.ndarray):
    """ polynomial is assumed to be monotonic """
    if isinstance(x, np.ndarray):
        t = np.empty_like(x)
        for i in range(x.shape[0]):
            t[i] = evaluate_inverse_poly(x[i], coefs)
        return t
    else:
        mc = get_monomial_coefs(coefs)
        mc[-1] -= x
        return valid_t_among(np.roots(mc))[0]


def evaluate_second_derivative_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
    """ Computes and returns the value of the second derivative of the Bezier polynomial at t using de Castlejau's
    algorithm"""
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return evaluate_poly(t, coefs)


def evaluate_third_derivative_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
    coefs = np.diff(coefs, 3) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2) * (coefs.shape[0] - 3)
    return evaluate_poly(t, coefs)


def get_monomial_coefs_of_second_derivative(coefs: np.ndarray):
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return get_monomial_coefs(coefs)


def get_inflexions(coefs: np.ndarray):
    return valid_t_among(np.roots(get_monomial_coefs_of_second_derivative(coefs)))


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

    # _u_i = np.array([0.0, 1.0, -0.65, 1.0])
    # _f_i = np.array([0.0, 3.0, -2.0, 2.0])
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
    kappa = 2.0
    delta_s = (np.max(df[du > 0] / du[du > 0]) + np.min(df[du < 0] / du[du < 0])) / 2.
    k = ((du >= 0) * gamma_plus(df / du, delta_s, delta_inf, kappa)
         + (du < 0) * gamma_minus(df / du, delta_s, delta_inf, kappa))

    # fig, ax = plt.subplots()
    # uuu = np.array([0.0, 1.0, 1.01, 5.0])
    # x = np.linspace(0.1, 4.9)
    # ax.plot(x, evaluate_inverse_poly(x, uuu))
    # ax.plot(_t, evaluate_poly(_t, evaluate_poly(_t, uuu)))
    # plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(9, 20))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    # axs[0].plot(u, f, 'o', markersize=2)
    # axs[0].plot(evaluate_poly(u_roots, _u_i), evaluate_poly(u_roots, _f_i), '*')
    # axs[0].plot(evaluate_poly(u_extrema, _u_i), evaluate_poly(u_extrema, _f_i), 'o')
    # axs[0].plot(evaluate_poly(u_inflexions, _u_i), evaluate_poly(u_inflexions, _f_i), 's')
    # axs[1].plot(_t, u, 'o', markersize=2)
    # axs[1].plot(u_roots, evaluate_poly(u_roots, _u_i), '*')
    # axs[1].plot(u_extrema, evaluate_poly(u_extrema, _u_i), 'o')
    # axs[1].plot(u_inflexions, evaluate_poly(u_inflexions, _u_i), 's')
    # axs[2].plot(_t, du, 'o', markersize=2)
    # axs[2].plot(u_roots, evaluate_derivative_poly(u_roots, _u_i), '*')
    # axs[2].plot(u_extrema, evaluate_derivative_poly(u_extrema, _u_i), 'o')
    # axs[2].plot(u_inflexions, evaluate_derivative_poly(u_inflexions, _u_i), 's')
    # axs[3].plot(_t, d2u, 'o', markersize=2)
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
    for ax in axs:
        ax.grid()
    plt.show()
