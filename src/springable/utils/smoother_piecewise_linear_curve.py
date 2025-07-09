import numpy as np


def compute_piecewise_control_points(a, x, extra=1.0):
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


def compute_piecewise_slopes_and_transitions_from_control_points(cp_x, cp_y) \
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


def _create_smoothing_function0(a0, a1, x0, delta):
    coefficients = np.array([(a0 - a1)/(4*delta**3) / 4,
                             3*x0*(a1 - a0)/(4*delta**3) / 3,
                             3*(a0-a1) * (x0**2 - delta**2) / (4*delta**3) / 2,
                             a0/2 + a1/2 + 3*x0/(4*delta)*(a0-a1) + x0**3/(4*delta**3) * (a1 -a0),
                             0
                             ])
    return np.polynomial.Polynomial(coefficients[::-1])


def _create_derivative_smoothing_function(a0, a1, x0, delta) -> np.polynomial.Polynomial:
    return _create_smoothing_function0(a0, a1, x0, delta).deriv()


def _create_second_derivative_smoothing_function(a0, a1, x0, delta) -> np.polynomial.Polynomial:
    return _create_smoothing_function0(a0, a1, x0, delta).deriv(m=2)


def _create_all_smoothing_functions(a, x, delta):
    cp_x, cp_y = compute_piecewise_control_points(a, x)
    funs = []
    for i in range(len(x)):
        fun = _create_smoothing_function0(a[i], a[i + 1], x[i], delta)
        fxm = fun(x[i] - delta)
        fun.coef[0] = -fxm + cp_y[i+1] - a[i] * delta
        funs.append(fun)
    return funs


def _create_all_derivative_smoothing_functions(a, x, delta) -> list[np.polynomial.Polynomial]:
    dfuns = []
    for i in range(len(x)):
        dfun = _create_derivative_smoothing_function(a[i], a[i + 1], x[i], delta)
        dfuns.append(dfun)
    return dfuns


def _create_all_second_derivative_smoothing_functions(a, x, delta):
    dfuns = []
    for i in range(len(x)):
        dfun = _create_second_derivative_smoothing_function(a[i], a[i + 1], x[i], delta)
        dfuns.append(dfun)
    return dfuns


def create_smooth_piecewise_function(a, x, delta):
    b = _compute_intercepts(a, x)
    smoothing_functions = _create_all_smoothing_functions(a, x, delta)
    conditions = _create_interval_conditions(x, delta)

    spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            spline_functions.append(lambda xx, ix=index: a[ix] * xx + b[ix])
        else:
            spline_functions.append(smoothing_functions[index])

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], spline_functions)


def create_smooth_piecewise_derivative_function(a, x, delta):
    derivative_smoothing_functions = _create_all_derivative_smoothing_functions(a, x, delta)
    conditions = _create_interval_conditions(x, delta)

    der_spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            der_spline_functions.append(a[index])
        else:
            der_spline_functions.append(derivative_smoothing_functions[index])

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], der_spline_functions)


def create_smooth_piecewise_second_derivative_function(a, x, delta):
    second_derivative_smoothing_functions = _create_all_second_derivative_smoothing_functions(a, x, delta)
    conditions = _create_interval_conditions(x, delta)

    second_der_spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            second_der_spline_functions.append(0.0)
        else:
            second_der_spline_functions.append(second_derivative_smoothing_functions[index])

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], second_der_spline_functions)


def get_extrema(a, x, delta):
    conditions = _create_interval_conditions(x, delta)
    dfs = _create_all_derivative_smoothing_functions(a, x, delta)
    extrema = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            pass
        else:
            roots = dfs[index].roots()
            real_roots = np.real(roots[np.isreal(roots)])
            if real_roots.size > 0:
                for rr in real_roots:
                    if conditions[i](rr):  # should only happens once a loop
                        extrema.append(rr)
    return np.array(extrema)


def get_extrema_from_control_points(cp_x, cp_y, delta):
    a, x = compute_piecewise_slopes_and_transitions_from_control_points(cp_x, cp_y)
    return get_extrema(a, x, delta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    _a = [2, 1, 1, -1, 5]
    _x = [1, 2, 3, 6]
    _delta = .45

    cp_x, cp_y = compute_piecewise_control_points(_a, _x)
    spw = create_smooth_piecewise_function(_a, _x, _delta)
    dspw = create_smooth_piecewise_derivative_function(_a, _x, _delta)

    u = np.linspace(_x[0] - 1., _x[-1] + 1., 300)
    u_extrema = get_extrema(_a, _x, _delta)

    fig, (ax, ax1) = plt.subplots(2, 1)
    ax.plot(u, spw(u), lw=2)
    ax.plot(u_extrema, spw(u_extrema), 'ro')
    ax.plot(cp_x, cp_y, '-', lw=5, zorder=0.2)

    ax1.plot(u, dspw(u), lw=2)
    ax1.plot(u_extrema, dspw(u_extrema), 'ro')
    plt.show()
