import numpy as np


def compute_zigzag_control_points(a, x):
    cp_x = [0.0]
    cp_y = [0.0]
    for i in range(len(x)):
        last_x = cp_x[-1]
        last_y = cp_y[-1]
        cp_x.append(x[i])
        cp_y.append(last_y + a[i] * (x[i] - last_x))
    extra = 1.0 * cp_x[-1]
    cp_x.append(cp_x[-1] + extra)
    cp_y.append(cp_y[-1] + a[-1] * extra)
    return np.array(cp_x), np.array(cp_y)


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


def _create_derivative_smoothing_function(a0, a1, x0, b0, delta):
    return _create_smoothing_function(a0, a1, x0, b0, delta).deriv()


def _create_all_smoothing_functions(a, x, delta, b):
    funs = []
    for i in range(len(x)):
        fun = _create_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        funs.append(fun)
    return funs


def _create_all_derivative_smoothing_functions(a, x, delta, b):
    dfuns = []
    for i in range(len(x)):
        dfun = _create_derivative_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
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
