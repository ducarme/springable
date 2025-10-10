import numpy as np
from scipy.integrate import solve_ivp


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

def _create_inverse_interval_conditions(a, x, delta):
    b = _compute_intercepts(a, x)
    x_new = np.empty(2 * len(x))
    x_new[0::2] = np.array(x) - delta
    x_new[1::2] = np.array(x) + delta
    
    conditions = [lambda yy: yy < a[0] * x_new[0] + b[0]]
    for i in range(1, 2 * len(x)):
        if i % 2 != 0:
            conditions.append(lambda yy, ix=i: (yy >= (a[ix//2] * x_new[ix - 1] + b[ix//2])) * (yy < a[ix//2+1] * x_new[ix] + b[ix//2+1]))
        else:
            conditions.append(lambda yy, ix=i: (yy >= (a[ix//2] * x_new[ix - 1] + b[ix//2])) * (yy < a[ix//2] * x_new[ix] + b[ix//2]))
    conditions.append(lambda yy: yy >= a[-1] * x_new[-1] + b[-1])
    return conditions


def _create_smoothing_function(a0, a1, x0, b0, delta) -> np.polynomial.Polynomial:
    coefficients = np.array([(a1 - a0) / (4 * delta),
                             (a0 * delta + a0 * x0 + a1 * delta - a1 * x0) / (2 * delta),
                             ((a1 - a0) * (x0 - delta) ** 2) / (4 * delta) + b0,
                             ])
    return np.polynomial.Polynomial(coefficients[::-1])

def _create_inverse_smoothing_function(a0, a1, x0, b0, delta):
    """ assuming that the smoothing function is monotonically INCREASING on its interval"""
    _a = (a1 - a0) / (4 * delta)
    _b = (a0 * delta + a0 * x0 + a1 * delta - a1 * x0) / (2 * delta)
    _c = ((a1 - a0) * (x0 - delta) ** 2) / (4 * delta) + b0
    return lambda y: 2 * (_c - y) / (-_b - np.sqrt(_b**2 - 4*_a * (_c - y)))
    # equivalent to (-_b + np.sqrt(_b**2 - 4 * _a * (_c - y)))/(2*_a),
    # but numerically stable when _a -> 0
    # (knowing that _b > 0 when _a -> 0 (a0 ~ a1),
    # since the smoothing function is monotonically increasing, meaning that a0 > 0, a1 > 0)

def _create_antiderivative_smoothing_function0(a0, a1, x0, b0, delta):
    return _create_smoothing_function(a0, a1, x0, b0, delta).integ(1, k=[0])


def _create_derivative_smoothing_function(a0, a1, x0, b0, delta) -> np.polynomial.Polynomial:
    return _create_smoothing_function(a0, a1, x0, b0, delta).deriv()


def _create_second_derivative_smoothing_function(a0, a1, x0, b0, delta):
    return _create_smoothing_function(a0, a1, x0, b0, delta).deriv(m=2)



def _create_all_smoothing_functions(a, x, delta, b) -> list[np.polynomial.Polynomial]:
    funs = []
    for i in range(len(x)):
        fun = _create_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        funs.append(fun)
    return funs

def _create_all_inverse_smoothing_functions(a, x, delta, b):
    funs = []
    for i in range(len(x)):
        fun = _create_inverse_smoothing_function(a[i], a[i + 1], x[i], b[i], delta)
        funs.append(fun)
    return funs

def _create_all_antiderivative_smoothing_functions0(a, x, delta, b):
    int_funs = []
    for i in range(len(x)):
        int_fun = _create_antiderivative_smoothing_function0(a[i], a[i + 1], x[i], b[i], delta)
        int_funs.append(int_fun)
    return int_funs


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


def create_smooth_piecewise_function(a, x, delta):
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

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], spline_functions)

def create_inverse_smooth_piecewise_function(a, x, delta):
    b = _compute_intercepts(a, x)
    inverse_smoothing_functions = _create_all_inverse_smoothing_functions(a, x, delta, b)
    conditions = _create_inverse_interval_conditions(a, x, delta)

    spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            spline_functions.append(lambda yy, ix=index: (yy - b[ix]) / a[ix])
        else:
            spline_functions.append(inverse_smoothing_functions[index])

    return lambda y: np.piecewise(y, [condition(y) for condition in conditions], spline_functions)

def create_smooth_piecewise_antiderivative_function(a, x, delta):
    b = _compute_intercepts(a, x)
    anti_der_smoothing_functions0 = _create_all_antiderivative_smoothing_functions0(a, x, delta, b)
    conditions = _create_interval_conditions(x, delta)

    running_integral = 0.0
    spline_functions = []
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:
            integral_f = lambda xx, ix=index: 0.5*a[ix] * xx**2 + b[ix] * xx
            if index > 0:
                constant = running_integral - integral_f(x[index-1] + delta)
            else:
                constant = 0.0
            spline_fun = lambda xx, int_f=integral_f, ix=index, cst=constant: int_f(xx, ix) + cst
            if index < len(x):
                running_integral = spline_fun(x[index] - delta)
        else:
            integral_f = lambda xx, ix=index: anti_der_smoothing_functions0[ix](xx)
            constant = running_integral - integral_f(x[index] - delta)
            spline_fun = lambda xx, int_f=integral_f, ix=index, cst=constant: int_f(xx, ix) + cst
            running_integral = spline_fun(x[index] + delta)
        spline_functions.append(spline_fun)

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], spline_functions)


def create_smooth_piecewise_derivative_function(a, x, delta):
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

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], der_spline_functions)


def create_smooth_piecewise_second_derivative_function(a, x, delta):
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

    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], second_der_spline_functions)


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
    a, x = compute_piecewise_slopes_and_transitions_from_control_points(cp_x, cp_y)
    return get_extrema(a, x, delta)

def create_antiderivative_of_parametric_piecewise(ah, av, x, delta):
    """
        Computes the area under a parametric curve described by
        two smooth piecewises sharing the same transitions and smoothing parameter (yet different slopes)
    """

    conditions = _create_interval_conditions(x, delta)

    bh = _compute_intercepts(av, x)
    bv = _compute_intercepts(av, x)
    dxs = _create_all_derivative_smoothing_functions(ah, x, delta, bh)
    ys = _create_all_smoothing_functions(av, x, delta, bv)

    int_splines = []
    running_integral = 0.0
    for i in range(len(conditions)):
        index = i // 2
        if i % 2 == 0:  # linear interval
            dx = np.polynomial.Polynomial([ah[index]])
            y = np.polynomial.Polynomial([bv[index], av[index]])
            ydx = y * dx
            int_ydx = ydx.integ(1, k=[0.0])
            if index > 0:
                cst = running_integral - int_ydx(x[index-1]+delta)
            else:
                cst = 0.0
            int_spline = ydx.integ(1, k=[cst])
            if index < len(x):
                running_integral = int_spline(x[index] - delta)
        else:  # quadractic interval
            dx = dxs[index]
            y = ys[index]
            ydx = y * dx
            int_ydx = ydx.integ(1, k=[0.0])
            cst = running_integral - int_ydx(x[index] - delta)
            int_spline = ydx.integ(1, k=[cst])
            running_integral = int_spline(x[index] + delta)
        int_splines.append(int_spline)
    
    return lambda u: np.piecewise(u, [condition(u) for condition in conditions], int_splines)





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 'ZIGZAG2(mode=-1; u_i=[1; 0.1; 0.321; 0.800]; f_i=[1; 0.04; 0.042; 1]; epsilon=5.000E-01)'

    _a = [2, -1, -2, -1, 5]
    _a2 = [1, 1, -1, 1, 1]
    _x = [1, 2, 3, 4]
    _delta = .1

    a_mono = [5.0, 1.0, 3.0, 5, 0.01]
    x_mono = [0.5, 2.0, 2.4, 10]
    delta_mono = 0.2
    spw_mono = create_smooth_piecewise_function(a_mono, x_mono, delta_mono)
    inv_spw_mono = create_inverse_smooth_piecewise_function(a_mono, x_mono, delta_mono)
    u_mono = np.linspace(0, x_mono[-1] + 1., 10000)
    f_mono = np.linspace(0, spw_mono(x_mono[-1] + 1.), 10000)
    fig, ax = plt.subplots()
    ax.plot(u_mono, spw_mono(u_mono))
    ax.plot(f_mono, inv_spw_mono(f_mono))
    ax.set_aspect('equal')
    ax.plot([0, u_mono[-1]], [0, u_mono[-1]])
    plt.show()


    spw = create_smooth_piecewise_function(_a, _x, _delta)
    spw2 = create_smooth_piecewise_function(_a2, _x, _delta)
    dspw2 = create_smooth_piecewise_derivative_function(_a2, _x, _delta)
    int_spw = create_smooth_piecewise_antiderivative_function(_a, _x, _delta)

    u = np.linspace(0, _x[-1] + 1., 10000)
    u_extrema = get_extrema(_a, _x, _delta)
    fig, ax = plt.subplots()
    ax.plot(u, spw(u))
    ax.plot(u_extrema, spw(u_extrema), 'o')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(u, int_spw(u))
    ax.plot(u, solve_ivp(lambda x, _: spw(x),
                         t_span=[u[0], u[-1]], y0=[0.0], t_eval=u, rtol=1e-8, atol=1e-8).y[0, :])
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(u[1:], np.diff(int_spw(u))/np.diff(u))
    ax.plot(u, spw(u))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(spw2(u), spw(u))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(u, create_antiderivative_of_parametric_piecewise(_a2, _a, _x, _delta)(u))
    ax.plot(u, solve_ivp(lambda x, _: spw(x) * dspw2(x),
                         t_span=[u[0], u[-1]], y0=[0.0], t_eval=u, rtol=1e-8, atol=1e-8).y[0, :])
    plt.show()

