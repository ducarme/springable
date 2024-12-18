from .math_utils.bezier_curve import *
from .math_utils import smooth_piecewise_linear_curve as spw
from scipy.interpolate import interp1d, griddata
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt


class MechanicalBehavior:
    _nb_dofs: int = None

    def __init__(self, natural_measure, /, **parameters):
        self._natural_measure = natural_measure
        self._parameters = parameters

    def elastic_energy(self, *variables: float) -> float:
        raise NotImplementedError("This method is abstract.")

    def gradient_energy(self, *variables: float):
        raise NotImplementedError("This method is abstract.")

    def hessian_energy(self, *variables: float):
        raise NotImplementedError("This method is abstract.")

    @classmethod
    def get_nb_dofs(cls) -> int:
        return cls._nb_dofs

    def get_parameters(self) -> dict[str, ...]:
        return self._parameters

    def get_natural_measure(self):
        return self._natural_measure

    def update(self, natural_measure=None, /, **parameters):
        if natural_measure is not None:
            self._natural_measure = natural_measure
        self._parameters.update(parameters)
        # TO BE EXTENDED IN SUBCLASSES IF NECESSARY

    def copy(self):
        parameters = self.get_parameters()
        copied_parameters = {k: (v.copy() if isinstance(v, list) else v)
                             for k, v in parameters.items()}
        return type(self)(self._natural_measure, **copied_parameters)


class UnivariateBehavior(MechanicalBehavior):
    _nb_dofs = 1

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        raise NotImplementedError("The method called is abstract.")

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        raise NotImplementedError("The method is abstract")

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        raise NotImplementedError("The method is abstract")


class BivariateBehavior(MechanicalBehavior):
    _nb_dofs = 2

    def elastic_energy(self, alpha: float, t: float) -> float:
        raise NotImplementedError("The method is abstract.")

    def gradient_energy(self, alpha: float, t: float) -> tuple[float]:
        raise NotImplementedError("The method is abstract")

    def hessian_energy(self, alpha: float, t: float) -> tuple[float]:
        raise NotImplementedError("The method is abstract")

    def get_hysteron_info(self) -> dict[str]:
        raise NotImplementedError("This method is abstract")


class ControllableByPoints:
    """ Interface to implement for curves that can be defined by control points """

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update_from_control_points(self, cp_x: np.ndarray, cp_y: np.ndarray):
        raise NotImplementedError


class LinearBehavior(UnivariateBehavior):

    def __init__(self, natural_measure: float, k: float):
        super().__init__(natural_measure, k=k)

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return 0.5 * self._parameters['k'] * (alpha - self._natural_measure) ** 2

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._parameters['k'] * (alpha - self._natural_measure),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._parameters['k'],

    def get_spring_constant(self) -> float:
        return self._parameters['k']


class LogarithmBehavior(UnivariateBehavior):

    def __init__(self, natural_measure: float, k: float):
        super().__init__(natural_measure, k=k)

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        energy = self._parameters['k'] * self._natural_measure * alpha * (np.log(alpha / self._natural_measure) - 1)
        return energy

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        force = self._parameters['k'] * self._natural_measure * np.log(alpha / self._natural_measure)
        return force,

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        stiffness = self._parameters['k'] * self._natural_measure / alpha
        return stiffness,


class BezierBehavior(UnivariateBehavior, ControllableByPoints):
    def __init__(self, natural_measure, u_i: list[float], f_i: list[float],
                 sampling: int = 100):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i)
        self._sampling = sampling
        self._check()
        self._make()

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        cp_x = np.array([0.0] + self._parameters['u_i'])
        cp_y = np.array([0.0] + self._parameters['f_i'])
        return cp_x, cp_y

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def _check(self):
        u_coefs = np.array([0.0] + self._parameters['u_i'])
        if not is_monotonic(u_coefs):
            raise InvalidBehaviorParameters("The Bezier behavior does not describe a function, try Bezier2 instead.")

    def _make(self):
        u_coefs = np.array([0.0] + self._parameters['u_i'])
        f_coefs = np.array([0.0] + self._parameters['f_i'])
        t = np.linspace(0, 1, self._sampling)
        u = evaluate_poly(t, u_coefs)

        def fdu(_t, _): return evaluate_poly(_t, f_coefs) * evaluate_derivative_poly(_t, u_coefs)

        energy = solve_ivp(fun=fdu, t_span=[0.0, 1.0], y0=[0.0], t_eval=t).y[0, :]
        self._energy = lambda uu: ((uu < u[-1]) * interp1d(u, energy, kind='linear',
                                                           bounds_error=False, fill_value=0.0)(np.abs(uu))
                                   + (np.abs(uu) >= u[-1]) * (energy[-1] + generalized_force[-1] * (np.abs(uu) - u[-1])
                                                              + 0.5 * generalized_stiffness[-1] * (
                                                                      np.abs(uu) - u[-1]) ** 2))

        generalized_force = evaluate_poly(t, f_coefs)
        self._first_der_energy = lambda uu: np.sign(uu) * interp1d(u, generalized_force, kind='linear',
                                                                   bounds_error=False, fill_value='extrapolate')(
            np.abs(uu))

        generalized_stiffness = evaluate_derivative_poly(t, f_coefs) / evaluate_derivative_poly(t, u_coefs)
        self._second_der_energy = lambda uu: interp1d(u, generalized_stiffness, kind='linear',
                                                      bounds_error=False, fill_value=generalized_stiffness[-1])(
            np.abs(uu))

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._first_der_energy(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._second_der_energy(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()

    @classmethod
    def compute_fitting_parameters(cls, force_displacement_curves: list[tuple], degree: int, show=False):
        umax = np.min([np.max(fd_curve[0]) for fd_curve in force_displacement_curves])
        fmax = np.min([np.max(fd_curve[1]) for fd_curve in force_displacement_curves])
        nb_samples = 500
        u_sampling = np.linspace(0.0, umax, nb_samples)

        def compute_mismatch(x: np.ndarray) -> float:
            natural_measure = 1.0
            u_i = x[::2].tolist()
            f_i = x[1::2].tolist()
            try:
                behavior = BezierBehavior(natural_measure, u_i=u_i, f_i=f_i)
                f_fit = behavior.gradient_energy(natural_measure + u_sampling)[0]
            except InvalidBehaviorParameters as e:
                mismatch = nb_samples * fmax ** 2
                print(f"Could not define a proper mismatch based on the parameters u_i={u_i} and f_i={f_i}. "
                      f"{e.get_message()}. Mismatch is defaulted to {mismatch:.3E}.")
            else:
                mismatch = 0.0
                for fd_curve in force_displacement_curves:
                    u_data = fd_curve[0]
                    f_data = fd_curve[1]
                    f_exp = interp1d(u_data, f_data, fill_value='extrapolate')(u_sampling)
                    mismatch += np.sum((f_exp - f_fit) ** 2)
            return mismatch

        u_guess = np.linspace(1 / degree, 1.0, degree) * umax
        f_guess = np.linspace(1 / degree, 1.0, degree) * fmax

        initial_values = np.array([u_guess[i // 2] if i % 2 == 0 else f_guess[i // 2] for i in range(2 * degree)])
        bounds = [(0.0, None) if i % 2 == 0 else (None, None) for i in range(2 * degree)]
        bounds[1] = (0.0, None)  # first control point should be above f=0 axis

        # control point abscissas should be monotonically increasing
        constraint_matrix = np.zeros((degree - 1, 2 * degree))
        for i in range(degree - 1):
            constraint_matrix[i, 2 * i] = -1.0
            constraint_matrix[i, 2 * i + 2] = 1.0
        lb = np.zeros(constraint_matrix.shape[0])
        ub = np.ones(constraint_matrix.shape[0]) * np.inf
        constraints = LinearConstraint(constraint_matrix, lb, ub)

        result = minimize(compute_mismatch,
                          x0=initial_values,
                          method='trust-constr',
                          bounds=bounds,
                          constraints=constraints,
                          options={'verbose': 3}
                          )
        optimal_parameters = result.x
        u_i = optimal_parameters[::2]
        f_i = optimal_parameters[1::2]
        u_i = u_i.tolist()
        f_i = f_i.tolist()
        if show:
            l0 = 1.0
            bb = BezierBehavior(l0, u_i, f_i)
            l = l0 + np.linspace(0.0, u_i[-1])
            f, = bb.gradient_energy(l)
            fig, ax = plt.subplots()
            ax.plot(l - l0, f, '-o')
            for fd_curve in force_displacement_curves:
                ax.plot(fd_curve[0], fd_curve[1], '--')
            plt.show()
        return u_i, f_i


class Bezier2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float]):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i)
        self._check()
        self._make()

    def _check(self):
        a_coefs = np.array([0.0] + self._parameters['u_i'])
        b_coefs = np.array([0.0] + self._parameters['f_i'])

        # checking validity of behavior
        da0 = evaluate_derivative_poly(0.0, a_coefs)
        db0 = evaluate_derivative_poly(0.0, b_coefs)
        if da0 == 0.0 or db0 == 0.0:
            raise InvalidBehaviorParameters('The initial slope of the behavior'
                                            'cannot be perfectly horizontal or vertical')
        a_extrema = get_extrema(a_coefs)
        b_extrema = get_extrema(b_coefs)
        if any(x in set(b_extrema) for x in a_extrema):
            raise InvalidBehaviorParameters('The behavior curve cannot have cusps.')
        a_extrema = [(a_extremum, 'a_max' if i % 2 == (0 if da0 > 0 else 1) else 'a_min')
                     for i, a_extremum in enumerate(a_extrema)]
        b_extrema = [(b_extremum, 'b_max' if i % 2 == (0 if db0 > 0 else 1) else 'b_min')
                     for i, b_extremum in enumerate(b_extrema)]
        extrema = []
        aa = 0
        bb = 0
        while aa < len(a_extrema) and bb < len(b_extrema):
            if a_extrema[aa][0] < b_extrema[bb][0]:
                ext = a_extrema[aa]
                aa += 1
            else:
                ext = b_extrema[bb]
                bb += 1
            extrema.append(ext)
        if aa < len(a_extrema):
            extrema += a_extrema[aa:]
        if bb < len(b_extrema):
            extrema += b_extrema[bb:]

        transitions = [extremum[1] for extremum in extrema]
        current_state = f'{"top" if db0 > 0 else "bottom"}-{"right" if da0 > 0 else "left"}'
        for transition in transitions:
            match current_state:
                case 'top-right':
                    if transition != 'b_max':
                        raise InvalidBehaviorParameters('The tangent cannot approach +inf, '
                                                        'when moving along the curve.')
                    current_state = 'bottom-right'
                case 'bottom-right':
                    if transition == 'a_max':
                        current_state = 'bottom-left'
                    elif transition == 'b_min':
                        current_state = 'top-right'
                    else:  # should be impossible
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case 'top-left':
                    if transition != 'b_max':
                        raise InvalidBehaviorParameters('The tangent cannot approach +inf, '
                                                        'when moving along the curve.')
                    current_state = 'bottom-left'
                case 'bottom-left':
                    if transition == 'a_min':
                        current_state = 'bottom-right'
                    elif transition == 'b_min':
                        current_state = 'top-left'
                    else:  # should be impossible
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case _:
                    print('error')

    def _make(self):
        a_coefs = np.array([0.0] + self._parameters['u_i'])
        b_coefs = np.array([0.0] + self._parameters['f_i'])
        a1 = evaluate_poly(1.0, a_coefs)
        da1 = evaluate_derivative_poly(1.0, a_coefs)
        b1 = evaluate_poly(1.0, b_coefs)
        db1 = evaluate_derivative_poly(1.0, b_coefs)
        self._a = lambda t: ((np.abs(t) <= 1) * np.sign(t) * evaluate_poly(np.abs(t), a_coefs)
                             + (np.abs(t) > 1) * np.sign(t) * (a1 + da1 * (np.abs(t) - 1)))
        self._b = lambda t: ((np.abs(t) <= 1) * np.sign(t) * evaluate_poly(np.abs(t), b_coefs)
                             + (np.abs(t) > 1) * np.sign(t) * (b1 + db1 * (np.abs(t) - 1)))
        self._da = lambda t: ((np.abs(t) <= 1) * evaluate_derivative_poly(np.abs(t), a_coefs)
                              + (np.abs(t) > 1) * da1)
        self._db = lambda t: ((np.abs(t) <= 1) * evaluate_derivative_poly(np.abs(t), b_coefs)
                              + (np.abs(t) > 1) * db1)
        self._d2a = lambda t: (np.abs(t) <= 1) * np.sign(t) * evaluate_second_derivative_poly(np.abs(t), a_coefs)
        self._d2b = lambda t: (np.abs(t) <= 1) * np.sign(t) * evaluate_second_derivative_poly(np.abs(t), b_coefs)
        self._d3a = lambda t: (np.abs(t) <= 1) * evaluate_third_derivative_poly(np.abs(t), a_coefs)
        self._d3b = lambda t: (np.abs(t) <= 1) * evaluate_third_derivative_poly(np.abs(t), b_coefs)
        self._dbda = lambda t: self._db(t) / self._da(t)
        self._d_dbda = lambda t: (self._d2b(t) * self._da(t) - self._db(t) * self._d2a(t)) / self._da(t) ** 2

        def d2_dbda(t):
            da = self._da(t)
            db = self._db(t)
            d2a = self._d2a(t)
            d2b = self._d2b(t)
            d3a = self._d3a(t)
            d3b = self._d3b(t)
            return (d3b * da ** 2 - db * d3a * da - 2 * d2b * da * d2a + 2 * db * d2a ** 2) / da ** 3

        self._d2_dbda = d2_dbda

        def int_adb(t):
            if isinstance(t, float):
                def adb(_t):
                    return self._db(_t) * self._a(_t)

                return quad(adb, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def adb(_t, _):
                    return self._db(_t) * self._a(_t)

                if t.ndim == 1:
                    return solve_ivp(fun=adb, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t).y[0, :]
                # if t.ndim == 2:
                #     int_adbdt = solve_ivp(fun=adb, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t[0, :]).y[0, :]
                #     return int_adbdt.reshape(1, -1).repeat(t.shape[0], axis=0)
                else:
                    raise TypeError('numpy array must be 1-dimensional')
            else:
                raise TypeError('must be float or numpy array')

        def int_bda(t):
            if isinstance(t, float):
                def bda(_t):
                    return self._b(_t) * self._da(_t)

                return quad(bda, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def bda(_t, _):
                    return self._b(_t) * self._da(_t)

                if t.ndim == 1:
                    return solve_ivp(fun=bda, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t).y[0, :]
                # if t.ndim == 2:
                #     int_bdadt = solve_ivp(fun=bda, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t[0, :]).y[0, :]
                #     return int_bdadt.reshape(1, -1).repeat(t.shape[0], axis=0)
                else:
                    raise TypeError('numpy array must be 1-dimensional')
            else:
                raise TypeError('must be float or 1-dimensional numpy array')

        self._int_adb = int_adb
        self._int_bda = int_bda
        self._hysteron_info = None

        all_t = np.linspace(0, 1, 50)
        da = self._da(all_t)
        db = self._db(all_t)
        db_da = db / da
        kmax = np.max(db_da[da > 0])
        kmin = np.min(db_da[da < 0]) if np.any(da < 0.0) else np.inf
        delta = 0.05 * kmax
        kstar = min(kmin - delta, kmax + delta)

        if kmin - kmax > 2 * delta:
            self._is_k_constant = True

            def k_fun(t):
                return np.ones_like(t) * kstar

            dk_fun = None  # should not be used
            d2k_fun = None  # should not be used

        else:
            self._is_k_constant = False

            def k_fun(t) -> np.ndarray:
                k_arr = np.zeros_like(t)
                da_pos = self._da(t) >= 0
                da_neg = self._da(t) < 0
                k_arr[da_pos] = np.maximum(self._dbda(t[da_pos]) + delta, kstar)
                k_arr[da_neg] = kstar
                return k_arr

            def dk_fun(t) -> np.ndarray:
                dk_arr = np.zeros_like(t)
                indices = np.logical_and(self._da(t) >= 0, self._dbda(t) + delta > kstar)
                dk_arr[indices] = 1.0 * self._d_dbda(t[indices])
                return dk_arr

            def d2k_fun(t) -> np.ndarray:
                d2k_arr = np.zeros_like(t)
                indices = np.logical_and(self._da(t) >= 0, self._dbda(t) + delta > kstar)
                d2k_arr[indices] = 1.0 * self._d2_dbda(t[indices])
                return d2k_arr

        self._k = k_fun
        self._dk = dk_fun
        self._d2k = d2k_fun

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        cp_x = np.array([0.0] + self._parameters['u_i'])
        cp_y = np.array([0.0] + self._parameters['f_i'])
        return cp_x, cp_y

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def elastic_energy(self, alpha: float, t: float) -> np.ndarray:
        y = alpha - self._natural_measure
        return 0.5 * self._k(t) * (y - self._a(t)) ** 2 + y * self._b(t) - self._int_adb(t)

    def gradient_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        dvdalpha = self._k(t) * (y - self._a(t)) + self._b(t)
        dvdt = (y - self._a(t)) * (self._db(t) - self._k(t) * self._da(t))
        if not self._is_k_constant:
            dvdt += 0.5 * (y - self._a(t)) ** 2 * self._dk(t)
        return dvdalpha, dvdt

    def hessian_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        d2vdalpha2 = self._k(t)
        d2vdalphadt = self._db(t) - self._k(t) * self._da(t)
        d2vdt2 = (y - self._a(t)) * (self._d2b(t) - self._k(t) * self._d2a(t)) + self._da(t) * (
                self._k(t) * self._da(t) - self._db(t))
        if not self._is_k_constant:
            d2vdalphadt += self._dk(t) * (y - self._a(t))
            d2vdt2 += (y - self._a(t)) * (0.5 * self._d2k(t) * (y - self._a(t)) - 2 * self._da(t) * self._dk(t))
        return d2vdalpha2, d2vdalphadt, d2vdt2

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()

    def get_hysteron_info(self) -> dict[str, float]:
        if self._hysteron_info is None:
            self._compute_hysteron_info()
        return self._hysteron_info

    def _compute_hysteron_info(self):
        a_coefs = np.array([0.0] + self._parameters['u_i'])
        extrema = np.sort(get_extrema(a_coefs))
        extrema = np.hstack((-extrema[::-1], extrema))
        nb_extrema = extrema.shape[0]
        if nb_extrema == 0:  # not a hysteron
            self._hysteron_info = {}
        else:
            critical_t = np.hstack(([-np.inf], extrema, [np.inf]))
            branch_intervals = [(critical_t[i], critical_t[i + 1]) for i in range(nb_extrema + 1)]
            is_branch_stable = [(i - len(branch_intervals) // 2) % 2 == 0 for i in range(len(branch_intervals))]
            branch_ids = []
            for i in range(len(branch_intervals)):
                if is_branch_stable[i]:
                    branch_id = str(abs(int((i - len(branch_intervals) // 2) / 2)))
                else:
                    branch_id = str(abs(int((i - len(branch_intervals) // 2) / 2))) + '-' + str(
                        abs(int((i - len(branch_intervals) // 2) / 2)) + 1)
                branch_ids.append(branch_id)
            self._hysteron_info = {'nb_stable_branches': nb_extrema,
                                   'branch_intervals': branch_intervals,
                                   'is_branch_stable': is_branch_stable,
                                   'branch_ids': branch_ids}


class PiecewiseBehavior(UnivariateBehavior, ControllableByPoints):
    def __init__(self, natural_measure, k, u, us):
        super().__init__(natural_measure, k=k, u=u, us=us)
        self._check()
        self._make()

    def _check(self):
        k, u, us = self._parameters['k'], self._parameters['u'], self._parameters['us']
        if u[0] - 0.0 < us or (len(u) > 1 and np.min(np.diff(u)) < 2 * us):
            raise InvalidBehaviorParameters(
                f'us ({us:.3E}) is too large for the piecewise intervals provided. '
                f'Should be less than {min(np.min(np.diff(u)) / 2, u[0] - 0.0):.3E}')
        if len(k) != len(u) + 1:
            raise InvalidBehaviorParameters(f'Expected {len(k) - 1} transitions, but only {len(k)} were provided.')

    def _make(self):
        k, u, us = self._parameters['k'], self._parameters['u'], self._parameters['us']
        self._u_i, self._f_i = spw.compute_piecewise_control_points(k, u, extra=4 * us)
        self._generalized_force_function = spw.create_smooth_piecewise_function(k, u, us)
        self._generalized_stiffness_function = spw.create_smooth_piecewise_derivative_function(k, u, us)

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self._u_i.copy(), self._f_i.copy()

    def update_from_control_points(self, cp_x, cp_y):
        if (np.diff(cp_x) < 0).any():
            raise InvalidBehaviorParameters(f'Each control point must be on the right of the previous one.')
        k, u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_x, cp_y)
        self.update(k=k, u=u)

    def elastic_energy(self, alpha: float) -> float:
        return quad(self._generalized_force_function, 0.0, alpha - self._natural_measure)[0]

    def gradient_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_force_function(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_stiffness_function(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()


class ZigzagBehavior(UnivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], epsilon: float,
                 sampling: int = 100):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, epsilon=epsilon)
        self._sampling = sampling
        self._check()
        self._make()

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        cp_x = np.array([0.0] + self._parameters['u_i'])
        cp_y = np.array([0.0] + self._parameters['f_i'])
        return cp_x, cp_y

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def _check(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')
        if len(u_i) != len(f_i):
            raise InvalidBehaviorParameters(f'u_i and f_i must contain the same number of elements')
        if (np.diff(u_i, prepend=0.0) <= 0.0).any():
            raise InvalidBehaviorParameters(
                f'u_i values should be positive and monotonically increasing. Try Zigzag2 instead')

    def _make(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)
        n = len(u_i) + 1
        cp_t = np.arange(n) / (n - 1)
        delta = epsilon / (2 * (n - 1))
        slopes_u, transitions_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        slopes_f, transitions_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        u = spw.create_smooth_piecewise_function(slopes_u, transitions_u, delta)
        f = spw.create_smooth_piecewise_function(slopes_f, transitions_f, delta)
        du = spw.create_smooth_piecewise_derivative_function(slopes_u, transitions_u, delta)
        df = spw.create_smooth_piecewise_derivative_function(slopes_f, transitions_f, delta)

        t = np.linspace(0, 1, self._sampling)
        u_s = u(t)
        f_s = f(t)
        k_s = df(t) / du(t)

        def fdu(_t, _): return f(_t) * du(_t)

        e_s = solve_ivp(fun=fdu, t_span=[0.0, 1.0], y0=[0.0], t_eval=t).y[0, :]
        e_inside = interp1d(u_s, e_s, kind='linear', bounds_error=False, fill_value=0.0)

        self._energy = lambda uu: ((uu < u_s[-1]) * e_inside(np.abs(uu))
                                   + (np.abs(uu) >= u_s[-1]) * (e_s[-1] + f_s[-1] * (np.abs(uu) - u_s[-1])
                                                                + 0.5 * k_s[-1] * (np.abs(uu) - u_s[-1]) ** 2))
        self._force = lambda uu: np.sign(uu) * interp1d(u_s, f_s, kind='linear', bounds_error=False,
                                                        fill_value='extrapolate')(np.abs(uu))
        self._stiffness = lambda uu: interp1d(u_s, k_s, kind='linear', bounds_error=False, fill_value=k_s[-1])(
            np.abs(uu))

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._force(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._stiffness(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()


class Zigzag2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], epsilon: float):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, epsilon=epsilon)
        self._check()
        self._make()

    def _check(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')
        if len(u_i) != len(f_i):
            raise InvalidBehaviorParameters(f'u_i and f_i must contain the same number of elements')

        n = len(u_i) + 1
        _cp_t = np.arange(n) / (n - 1)
        _delta = epsilon / (2 * (n - 1))
        _cp_u = np.array([0.0] + u_i)
        _cp_f = np.array([0.0] + f_i)

        # checking validity of behavior
        da0 = u_i[0]
        db0 = f_i[0]
        if da0 == 0.0 or db0 == 0.0:
            raise InvalidBehaviorParameters('The initial slope of the behavior'
                                            'cannot be perfectly horizontal or vertical')
        a_extrema = spw.get_extrema_from_control_points(_cp_t, _cp_u, _delta)
        b_extrema = spw.get_extrema_from_control_points(_cp_t, _cp_f, _delta)
        if any(x in set(b_extrema) for x in a_extrema):
            raise InvalidBehaviorParameters('The behavior curve cannot have cusps.')
        a_extrema = [(a_extremum, 'a_max' if i % 2 == (0 if da0 > 0 else 1) else 'a_min')
                     for i, a_extremum in enumerate(a_extrema)]
        b_extrema = [(b_extremum, 'b_max' if i % 2 == (0 if db0 > 0 else 1) else 'b_min')
                     for i, b_extremum in enumerate(b_extrema)]
        extrema = []
        aa = 0
        bb = 0
        while aa < len(a_extrema) and bb < len(b_extrema):
            if a_extrema[aa][0] < b_extrema[bb][0]:
                ext = a_extrema[aa]
                aa += 1
            else:
                ext = b_extrema[bb]
                bb += 1
            extrema.append(ext)
        if aa < len(a_extrema):
            extrema += a_extrema[aa:]
        if bb < len(b_extrema):
            extrema += b_extrema[bb:]

        transitions = [extremum[1] for extremum in extrema]
        current_state = f'{"top" if db0 > 0 else "bottom"}-{"right" if da0 > 0 else "left"}'
        for transition in transitions:
            match current_state:
                case 'top-right':
                    if transition != 'b_max':
                        raise InvalidBehaviorParameters('The tangent cannot approach +inf, '
                                                        'when moving along the curve.')
                    current_state = 'bottom-right'
                case 'bottom-right':
                    if transition == 'a_max':
                        current_state = 'bottom-left'
                    elif transition == 'b_min':
                        current_state = 'top-right'
                    else:  # should be impossible
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case 'top-left':
                    if transition != 'b_max':
                        raise InvalidBehaviorParameters('The tangent cannot approach +inf, '
                                                        'when moving along the curve.')
                    current_state = 'bottom-left'
                case 'bottom-left':
                    if transition == 'a_min':
                        current_state = 'bottom-right'
                    elif transition == 'b_min':
                        current_state = 'top-left'
                    else:  # should be impossible
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case _:
                    print('error')

    def _make(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        self._cp_u = np.array([0.0] + u_i)
        self._cp_f = np.array([0.0] + f_i)
        n = len(u_i) + 1
        self._cp_t = np.arange(n) / (n - 1)
        self._delta = epsilon / (2 * (n - 1))
        slopes_u, transitions_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(self._cp_t,
                                                                                                   self._cp_u)
        slopes_f, transitions_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(self._cp_t,
                                                                                                   self._cp_f)
        self._a = spw.create_smooth_piecewise_function(slopes_u, transitions_u, self._delta)
        self._b = spw.create_smooth_piecewise_function(slopes_f, transitions_f, self._delta)
        self._da = spw.create_smooth_piecewise_derivative_function(slopes_u, transitions_u, self._delta)
        self._db = spw.create_smooth_piecewise_derivative_function(slopes_f, transitions_f, self._delta)
        self._d2a = spw.create_smooth_piecewise_second_derivative_function(slopes_u, transitions_u, self._delta)
        self._d2b = spw.create_smooth_piecewise_second_derivative_function(slopes_f, transitions_f, self._delta)
        self._d3a = lambda t: np.zeros_like(t)
        self._d3b = lambda t: np.zeros_like(t)
        self._dbda = lambda t: self._db(t) / self._da(t)
        self._d_dbda = lambda t: (self._d2b(t) * self._da(t) - self._db(t) * self._d2a(t)) / self._da(t) ** 2

        def d2_dbda(t):
            da = self._da(t)
            db = self._db(t)
            d2a = self._d2a(t)
            d2b = self._d2b(t)
            d3a = self._d3a(t)
            d3b = self._d3b(t)
            return (d3b * da ** 2 - db * d3a * da - 2 * d2b * da * d2a + 2 * db * d2a ** 2) / da ** 3

        self._d2_dbda = d2_dbda

        def int_adb(t):
            if isinstance(t, float):
                def adb(_t):
                    return self._db(_t) * self._a(_t)

                return quad(adb, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def adb(_t, _):
                    return self._db(_t) * self._a(_t)

                if t.ndim == 1:
                    return solve_ivp(fun=adb, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t).y[0, :]
                # if t.ndim == 2:
                #     int_adbdt = solve_ivp(fun=adb, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t[0, :]).y[0, :]
                #     return int_adbdt.reshape(1, -1).repeat(t.shape[0], axis=0)
                else:
                    raise TypeError('numpy array must be 1-dimensional')
            else:
                raise TypeError('must be float or numpy array')

        def int_bda(t):
            if isinstance(t, float):
                def bda(_t):
                    return self._b(_t) * self._da(_t)

                return quad(bda, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def bda(_t, _):
                    return self._b(_t) * self._da(_t)

                if t.ndim == 1:
                    return solve_ivp(fun=bda, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t).y[0, :]
                # if t.ndim == 2:
                #     int_bdadt = solve_ivp(fun=bda, t_span=[np.min(t), np.max(t)], y0=[0.0], t_eval=t[0, :]).y[0, :]
                #     return int_bdadt.reshape(1, -1).repeat(t.shape[0], axis=0)
                else:
                    raise TypeError('numpy array must be 1-dimensional')
            else:
                raise TypeError('must be float or 1-dimensional numpy array')

        self._int_adb = int_adb
        self._int_bda = int_bda
        self._hysteron_info = None

        all_t = np.linspace(0, 1, 50)
        da = self._da(all_t)
        db = self._db(all_t)
        db_da = db / da
        kmax = np.max(db_da[da > 0])
        kmin = np.min(db_da[da < 0]) if np.any(da < 0.0) else np.inf
        delta = 0.05 * kmax
        kstar = min(kmin - delta, kmax + delta)

        if kmin - kmax > 2 * delta:
            self._is_k_constant = True

            def k_fun(t):
                return np.ones_like(t) * kstar

            dk_fun = None  # should not be used
            d2k_fun = None  # should not be used

        else:
            self._is_k_constant = False

            def k_fun(t) -> np.ndarray:
                k_arr = np.zeros_like(t)
                da_pos = self._da(t) >= 0
                da_neg = self._da(t) < 0
                k_arr[da_pos] = np.maximum(self._dbda(t[da_pos]) + delta, kstar)
                k_arr[da_neg] = kstar
                return k_arr

            def dk_fun(t) -> np.ndarray:
                dk_arr = np.zeros_like(t)
                indices = np.logical_and(self._da(t) >= 0, self._dbda(t) + delta > kstar)
                dk_arr[indices] = 1.0 * self._d_dbda(t[indices])
                return dk_arr

            def d2k_fun(t) -> np.ndarray:
                d2k_arr = np.zeros_like(t)
                indices = np.logical_and(self._da(t) >= 0, self._dbda(t) + delta > kstar)
                d2k_arr[indices] = 1.0 * self._d2_dbda(t[indices])
                return d2k_arr

        self._k = k_fun
        self._dk = dk_fun
        self._d2k = d2k_fun

        # fig_, ax = plt.subplots()
        # tt = np.linspace(-1, 0, 400)
        # k = self._k(tt)
        # da = self._da(tt)
        # db = self._db(tt)
        # a = self._a(tt)
        # b = self._b(tt)
        # ax.plot(tt, a, 'o')
        # # ax.plot(tt, db/da, 'o')
        # # ax.plot(tt, k, '-o')
        # plt.show()

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cp_u.copy(), self._cp_f.copy()

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def elastic_energy(self, alpha: float, t: float) -> np.ndarray:
        y = alpha - self._natural_measure
        return 0.5 * self._k(t) * (y - self._a(t)) ** 2 + y * self._b(t) - self._int_adb(t)

    def gradient_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        dvdalpha = self._k(t) * (y - self._a(t)) + self._b(t)
        dvdt = (y - self._a(t)) * (self._db(t) - self._k(t) * self._da(t))
        if not self._is_k_constant:
            dvdt += 0.5 * (y - self._a(t)) ** 2 * self._dk(t)
        return dvdalpha, dvdt

    def hessian_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        d2vdalpha2 = self._k(t)
        d2vdalphadt = self._db(t) - self._k(t) * self._da(t)
        d2vdt2 = (y - self._a(t)) * (self._d2b(t) - self._k(t) * self._d2a(t)) + self._da(t) * (
                self._k(t) * self._da(t) - self._db(t))
        if not self._is_k_constant:
            d2vdalphadt += self._dk(t) * (y - self._a(t))
            d2vdt2 += (y - self._a(t)) * (0.5 * self._d2k(t) * (y - self._a(t)) - 2 * self._da(t) * self._dk(t))
        return d2vdalpha2, d2vdalphadt, d2vdt2

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()

    def get_hysteron_info(self) -> dict[str, float]:
        if self._hysteron_info is None:
            self._compute_hysteron_info()
        return self._hysteron_info

    def _compute_hysteron_info(self):
        extrema = np.sort(spw.get_extrema_from_control_points(self._cp_t, self._cp_u, self._delta))
        extrema = np.hstack((-extrema[::-1], extrema))
        nb_extrema = extrema.shape[0]
        if nb_extrema == 0:  # not a hysteron
            self._hysteron_info = {}
        else:
            critical_t = np.hstack(([-np.inf], extrema, [np.inf]))
            branch_intervals = [(critical_t[i], critical_t[i + 1]) for i in range(nb_extrema + 1)]
            is_branch_stable = [(i - len(branch_intervals) // 2) % 2 == 0 for i in range(len(branch_intervals))]
            branch_ids = []
            for i in range(len(branch_intervals)):
                if is_branch_stable[i]:
                    branch_id = str(abs(int((i - len(branch_intervals) // 2) / 2)))
                else:
                    branch_id = str(abs(int((i - len(branch_intervals) // 2) / 2))) + '-' + str(
                        abs(int((i - len(branch_intervals) // 2) / 2)) + 1)
                branch_ids.append(branch_id)
            self._hysteron_info = {'nb_stable_branches': nb_extrema,
                                   'branch_intervals': branch_intervals,
                                   'is_branch_stable': is_branch_stable,
                                   'branch_ids': branch_ids}

    def plot_energy_landscape(self, ax=None):
        n = 500
        t = np.linspace(0, 2.0, n)
        y = np.linspace(np.min(self._a(t)), np.max(self._a(t)), n)
        f = np.linspace(np.min(self._b(t)), np.max(self._b(t)), n)
        t_grid, y_grid = np.meshgrid(t, y)
        v_grid = np.empty_like(t_grid)
        for i in range(t_grid.shape[0]):
            v_grid[i, :] = self.elastic_energy(y_grid[i, :], t_grid[i, :])
        if ax is None:
            make_new_fig = True
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            make_new_fig = False

        a = self._a(t)
        int_bda = self._int_bda(t)
        stable = np.logical_and(self._db(t) > 0, self._da(t) > 0)
        stabilizable = np.logical_and(self._db(t) < 0, self._da(t) > 0)
        unstable = self._da(t) < 0

        ax.plot_surface(t_grid, y_grid, v_grid, cmap='viridis')
        # ax.plot(t[stable], a[stable], int_bda[stable], 'bo', label='path')
        # ax.plot(t[stabilizable], a[stabilizable], int_bda[stabilizable], 'ko', label='path')
        # ax.plot(t[unstable], a[unstable], int_bda[unstable], 'ro', label='path')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$U$')
        if make_new_fig:
            plt.show()


class ContactBehavior(UnivariateBehavior):
    def __init__(self, natural_measure, f0, u0):
        super().__init__(natural_measure, f0=f0, u0=u0)
        self._p = 3.0

    def elastic_energy(self, alpha: float) -> float:
        dalpha = alpha - self._natural_measure
        f0 = self._parameters['f0']
        u0 = self._parameters['u0']
        p = self._p
        if dalpha >= u0:
            return 0.0
        else:
            return f0 * u0 / (p + 1) * ((u0 - dalpha) / u0) ** (p + 1)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float | np.ndarray]:
        if isinstance(alpha, np.ndarray):
            g = np.empty_like(alpha)
            for i, alpha_i in enumerate(alpha):
                g[i] = self.gradient_energy(alpha_i)[0]
            return g,

        dalpha = alpha - self._natural_measure
        f0 = self._parameters['f0']
        u0 = self._parameters['u0']
        p = self._p
        if dalpha >= u0:
            return 0.0,
        else:
            return -f0 * ((u0 - dalpha) / u0) ** p,

    def hessian_energy(self, alpha: float) -> tuple[float]:
        dalpha = alpha - self._natural_measure
        f0 = self._parameters['f0']
        u0 = self._parameters['u0']
        p = self._p
        if dalpha >= u0:
            return 0.0,
        else:
            return +f0 / u0 * p * ((u0 - dalpha) / u0) ** (p - 1),


class IdealGas(UnivariateBehavior):

    def __init__(self, v0: float, n: float, R: float, T0: float):
        """
            v0: the volume the amount of gas would take at ambient pressure
            n:  the amount of substance (number of gas particles, number of moles, ...)
            R:  is the proportionality factor between the gas temperature and the gas thermal energy per unit of substance
                (Boltzmann constant if the amount of substance is expressed as the number of particles,
                the molar gas constant if the amount of substance is expressed as the number of moles, ...)
            T0: the temperature of the gas at ambient pressure
        """
        super().__init__(v0, n=n, R=R, T0=T0)

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        raise NotImplementedError("This method is abstract")

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        raise NotImplementedError("This method is abstract")

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        raise NotImplementedError("This method is abstract")


class IsothermicGas(IdealGas):

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        v = alpha
        v0 = self._natural_measure
        if v < 0:
            return np.nan
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT * (v / v0 - 1.0 - np.log(v / v0))

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float | np.ndarray]:
        if isinstance(alpha, np.ndarray):
            g = np.empty_like(alpha)
            for i, alpha_i in enumerate(alpha):
                g[i] = self.gradient_energy(alpha_i)[0]
            return g,

        v = alpha
        v0 = self._natural_measure
        if v < 0:
            return np.nan,
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT * (v - v0) / (v * v0),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v = alpha
        if v < 0:
            return np.nan,
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT / v ** 2,


class IsentropicGas(IdealGas):

    def __init__(self, v0: float, n: float, R: float, T0: float, gamma: float):
        """
            v0: the volume the amount of gas would take at ambient pressure
            n:  the amount of substance (number of gas particles, number of moles, ...)
            R:  is the proportionality factor between the gas temperature and the gas thermal energy per unit of substance
                (Boltzmann constant if the amount of substance is expressed as the number of particles,
                the molar gas constant if the amount of substance is expressed as the number of moles, ...)
            T0: the temperature of the gas at ambient pressure
            gamma: the heat capacity ratio (a.k.a. adiabatic index), that is, cp/cv. Must be strictly greater than 1.
                   It is a non-dimensional property of the gas (for dry air, gamma = 1.4)
        """

        super().__init__(v0, n=n, R=R, T0=T0)
        self._parameters['gamma'] = gamma
        self._check()

    def _check(self):
        if self._parameters['gamma'] < 1.0:
            raise InvalidBehaviorParameters(f"The ratio of heat capacities gamma must be strictly greater than 1. "
                                            f"Current value = {self._parameters['gamma']:.3E}")

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        if v < 0:
            return np.nan
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * (v / v0 - 1) + nRT0 / (gamma - 1) * ((v0 / v) ** (gamma - 1) - 1)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float | np.ndarray]:
        if isinstance(alpha, np.ndarray):
            g = np.empty_like(alpha)
            for i, alpha_i in enumerate(alpha):
                g[i] = self.gradient_energy(alpha_i)[0]
            return g,

        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        if v < 0:
            return np.nan,
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * (1 / v0 - 1 / v * (v0 / v) ** (gamma - 1)),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        if v < 0:
            return np.nan,
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * gamma / v ** 2 * (v0 / v) ** (gamma - 1),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()


class InvalidBehaviorParameters(ValueError):
    """ raise this when one attempts to create a mechanical behavior with invalid parameters"""

    def __init__(self, message):
        super().__init__(message)
        self._msg = message

    def get_message(self):
        return self._msg
