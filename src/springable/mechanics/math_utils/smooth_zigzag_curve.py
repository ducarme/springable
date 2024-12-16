import numpy as np


def compute_zigzag_control_points(a, x, extra=1.0):
    cp_x = [0.0]
    cp_y = [0.0]
    for i in range(len(x)):
        last_x = cp_x[-1]
        last_y = cp_y[-1]
        cp_x.append(x[i])
        cp_y.append(last_y + a[i] * (x[i] - last_x))
    cp_x.append(cp_x[-1] + extra)
    cp_y.append(cp_y[-1] + a[-1] * extra)
    return np.array(cp_x), np.array(cp_y)


def compute_zizag_slopes_and_transitions_from_control_points(cp_x, cp_y) \
        -> tuple[list[float], list[float]]:
    a = (np.diff(cp_y) / np.diff(cp_x)).tolist()
    x = np.array(cp_x[1:-1]).tolist()
    return a, x


def _compute_intercepts(a, x):
    b = [0.0]
    for i in range(len(x)):
        last_b = b[-1]
        b.append(last_b + x[i] * (a[i] - a[i + 1]))
    return np.array(b)


def _create_interval_conditions(x, delta):
    x_new = np.empty(2 * len(x))
    x_new[0::2] = np.array(x) - delta
    x_new[1::2] = np.array(x) + delta
    conditions = [lambda xx: xx < x_new[0]]
    for i in range(1, 2 * len(x)):
        conditions.append(lambda xx, ix=i: (xx >= x_new[ix - 1]) * (xx < x_new[ix]))
    conditions.append(lambda xx: xx >= x_new[-1])
    return conditions


def _create_smoothing_function(a0, a1, x0, b0, delta):
    coefficients = np.array([(a1 - a0) / (4 * delta),
                             (a0 * delta + a0 * x0 + a1 * delta - a1 * x0) / (2 * delta),
                             ((a1 - a0) * (x0 - delta) ** 2) / (4 * delta) + b0,
                             ])
    return np.polynomial.Polynomial(coefficients[::-1])


def _create_derivative_smoothing_function(a0, a1, x0, b0, delta) -> np.polynomial.Polynomial:
    return _create_smoothing_function(a0, a1, x0, b0, delta).deriv()


def _create_second_derivative_smoothing_function(a0, a1, x0, b0, delta):
    return _create_smoothing_function(a0, a1, x0, b0, delta).deriv(m=2)


def _create_all_smoothing_functions(a, x, delta, b):
    funs = []
    for i in range(len(x)):
        fun = _create_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        funs.append(fun)
    return funs


def _create_all_derivative_smoothing_functions(a, x, delta, b) -> list[np.polynomial.Polynomial]:
    dfuns = []
    for i in range(len(x)):
        dfun = _create_derivative_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        dfuns.append(dfun)
    return dfuns


def _create_all_second_derivative_smoothing_functions(a, x, delta, b):
    dfuns = []
    for i in range(len(x)):
        dfun = _create_second_derivative_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        dfuns.append(dfun)
    return dfuns


def create_smooth_zigzag_function(a, x, delta):
    b = _compute_intercepts(a, x)
    smoothing_functions = _create_all_smoothing_functions(a, x, delta, b)
    conditions = _create_interval_conditions(x, delta)

    spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            spline_functions.append(lambda xx, ix=index: a[ix] * xx + b[ix])
        else:
            spline_functions.append(smoothing_functions[index])

    return lambda u: np.sign(u) * np.piecewise(np.abs(u), [condition(np.abs(u)) for condition in conditions],
                                               spline_functions)


def create_smooth_zigzag_derivative_function(a, x, delta):
    b = _compute_intercepts(a, x)
    derivative_smoothing_functions = _create_all_derivative_smoothing_functions(a, x, delta, b)
    conditions = _create_interval_conditions(x, delta)

    der_spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            der_spline_functions.append(a[index])
        else:
            der_spline_functions.append(derivative_smoothing_functions[index])

    return lambda u: np.piecewise(np.abs(u), [condition(np.abs(u)) for condition in conditions], der_spline_functions)


def create_smooth_zigzag_second_derivative_function(a, x, delta):
    b = _compute_intercepts(a, x)
    second_derivative_smoothing_functions = _create_all_second_derivative_smoothing_functions(a, x, delta, b)
    conditions = _create_interval_conditions(x, delta)

    second_der_spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            second_der_spline_functions.append(0.0)
        else:
            second_der_spline_functions.append(second_derivative_smoothing_functions[index])

    return lambda u: np.sign(u) * np.piecewise(np.abs(u), [condition(np.abs(u)) for condition in conditions],
                                               second_der_spline_functions)


def get_extrema(a, x, delta):
    conditions = _create_interval_conditions(x, delta)
    intercepts = _compute_intercepts(a, x)
    dfs = _create_all_derivative_smoothing_functions(a, x, delta, intercepts)
    extrema = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            pass
        else:
            roots = dfs[index].roots()
            real_roots = np.real(roots[np.isreal(roots)])
            if real_roots.size > 0 and conditions[i](real_roots[0]):
                extrema.append(real_roots[0])
    return np.array(extrema)


def get_extrema_from_control_points(cp_x, cp_y, delta):
    a, x = compute_zizag_slopes_and_transitions_from_control_points(cp_x, cp_y)
    return get_extrema(a, x, delta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    a = [2, 1, 1, -1, 5]
    x = [1, 2, 3, 6]
    delta = .1

    szz = create_smooth_zigzag_function(a, x, delta)

    u = np.linspace(x[0] - 1., x[-1] + 1., 300)
    u_extrema = get_extrema(a, x, delta)
    fig, ax = plt.subplots()
    ax.plot(u, szz(u))
    ax.plot(u_extrema, szz(u_extrema), 'o')
    plt.show()
