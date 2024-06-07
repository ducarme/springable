import numpy as np
from scipy.special import factorial


def valid_t_among(t: np.ndarray):
    t = np.real(t[np.isreal(t)])
    return t[np.logical_and(t >= 0.0, t <= 1.0)]


def evaluate_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
    """ Computes and returns the value of the Bezier polynomial at t using de Castlejau's algorithm """
    beta = [c for c in coefs]  # copies array
    n = len(beta)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]


def get_monomial_coefs(coefs: np.ndarray):
    deg = coefs.shape[0] - 1
    monomial_coefs = np.empty(deg + 1)
    for j in range(deg + 1):
        monomial_coefs[j] = factorial(deg) / factorial(deg - j) \
                            * np.sum([(-1.) ** (i + j) * coefs[i] / factorial(i) / factorial(j - i) for i in range(j + 1)])
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


def evaluate_second_derivative_poly(t: float | np.ndarray, coefs: np.ndarray) -> float | np.ndarray:
    """ Computes and returns the value of the second derivative of the Bezier polynomial at t using de Castlejau's
    algorithm"""
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return evaluate_poly(t, coefs)


def get_monomial_coefs_of_second_derivative(coefs: np.ndarray):
    coefs = np.diff(coefs, 2) * (coefs.shape[0] - 1) * (coefs.shape[0] - 2)
    return get_monomial_coefs(coefs)


def get_inflexions(coefs: np.ndarray):
    return valid_t_among(np.roots(get_monomial_coefs_of_second_derivative(coefs)))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    u_i = np.array([0.0, 1.0, -0.65, 1.0])
    f_i = np.array([0.0, 3.0, -2.0, 2.0])
    t = np.linspace(0, 1, 100)
    u = evaluate_poly(t, u_i)
    f = evaluate_poly(t, f_i)
    du = evaluate_derivative_poly(t, u_i)
    d2u = evaluate_second_derivative_poly(t, u_i)
    df = evaluate_poly(t, f)
    u_roots = get_roots(u_i)
    u_extrema = get_extrema(u_i)
    u_inflexions = get_inflexions(u_i)

    fig, axs = plt.subplots(4, 1, figsize=(9, 20))
    axs[0].plot(u, f)
    axs[0].plot(evaluate_poly(u_roots, u_i), evaluate_poly(u_roots, f_i), '*')
    axs[0].plot(evaluate_poly(u_extrema, u_i), evaluate_poly(u_extrema, f_i), 'o')
    axs[0].plot(evaluate_poly(u_inflexions, u_i), evaluate_poly(u_inflexions, f_i), 's')
    axs[1].plot(t, u)
    axs[1].plot(u_roots, evaluate_poly(u_roots, u_i), '*')
    axs[1].plot(u_extrema, evaluate_poly(u_extrema, u_i), 'o')
    axs[1].plot(u_inflexions, evaluate_poly(u_inflexions, u_i), 's')
    axs[2].plot(t, du)
    axs[2].plot(u_roots, evaluate_derivative_poly(u_roots, u_i), '*')
    axs[2].plot(u_extrema, evaluate_derivative_poly(u_extrema, u_i), 'o')
    axs[2].plot(u_inflexions, evaluate_derivative_poly(u_inflexions, u_i), 's')
    axs[3].plot(t, d2u)
    axs[3].plot(u_roots, evaluate_second_derivative_poly(u_roots, u_i), '*')
    axs[3].plot(u_extrema, evaluate_second_derivative_poly(u_extrema, u_i), 'o')
    axs[3].plot(u_inflexions, evaluate_second_derivative_poly(u_inflexions, u_i), 's')
    for ax in axs:
        ax.grid()
    plt.show()


