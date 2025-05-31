from collections.abc import Callable

from ..utils import bezier_curve
from ..utils import smooth_piecewise_linear_curve as spw
from scipy.interpolate import interp1d, griddata, make_interp_spline, PPoly
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt

from ..utils.smooth_piecewise_linear_curve import create_smooth_piecewise_function


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

    def _check(self):
        pass
        # TO BE EXTENDED IN SUBCLASSES IF NECESSARY
        # should not return anything,
        # just raise InvalidBehaviorParameter exception in case the behavior is ill-defined

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

    def __init__(self, natural_measure, mode):
        super().__init__(natural_measure, mode=mode)

        # will only be updated when calling _make()
        self.a: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.b: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.da: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.db: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.d2a: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.d2b: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.d3a: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.d3b: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.k: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.dk: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None
        self.d2k: Callable[[float], float] | Callable[[np.ndarray], np.ndarray] | None = None

        # to update in the 'update' method
        self._hysteron_info = None

    def _check(self):
        super()._check()
        if self._parameters['mode'] not in (-1, 0, 1):
            raise InvalidBehaviorParameters('The mode should be -1, 0 or +1 (integer; no float number allowed).')
        # checking validity of behavior
        da0 = self._da_fun(0.0)
        db0 = self._db_fun(0.0)
        if da0 == 0.0 or db0 == 0.0:
            raise InvalidBehaviorParameters('The initial slope of the behavior'
                                            'cannot be perfectly horizontal or vertical')
        a_extrema = self.get_a_extrema()
        b_extrema = self.get_b_extrema()
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
        mode = self._parameters['mode']
        da0 = self._da_fun(0.0)
        db0 = self._db_fun(0.0)
        a1 = self._a_fun(1.0)
        da1 = self._da_fun(1.0)
        b1 = self._b_fun(1.0)
        db1 = self._db_fun(1.0)

        if mode == 0:
            self.a = lambda t: ((np.abs(t) <= 1) * np.sign(t) * self._a_fun(np.abs(t))
                                + (np.abs(t) > 1) * np.sign(t) * (a1 + da1 * (np.abs(t) - 1)))
            self.b = lambda t: ((np.abs(t) <= 1) * np.sign(t) * self._b_fun(np.abs(t))
                                + (np.abs(t) > 1) * np.sign(t) * (b1 + db1 * (np.abs(t) - 1)))
            self.da = lambda t: ((np.abs(t) <= 1) * self._da_fun(np.abs(t)) + (np.abs(t) > 1) * da1)
            self.db = lambda t: ((np.abs(t) <= 1) * self._db_fun(np.abs(t)) + (np.abs(t) > 1) * db1)
            self.d2a = lambda t: (np.abs(t) <= 1) * np.sign(t) * self._d2a_fun(np.abs(t))
            self.d2b = lambda t: (np.abs(t) <= 1) * np.sign(t) * self._d2b_fun(np.abs(t))
            self.d3a = lambda t: (np.abs(t) <= 1) * self._d3a_fun(np.abs(t))
            self.d3b = lambda t: (np.abs(t) <= 1) * self._d3b_fun(np.abs(t))

        elif mode == 1:
            self.a = lambda t: ((t < 0) * (da0 * t) + (t > 1) * (a1 + da1 * (t - 1))
                                + np.logical_and((t >= 0), (t <= 1)) * self._a_fun(t))
            self.b = lambda t: ((t < 0) * (db0 * t) + (t > 1) * (b1 + db1 * (t - 1))
                                + np.logical_and((t >= 0), (t <= 1)) * self._b_fun(t))
            self.da = lambda t: ((t < 0) * da0 + (t > 1) * da1
                                 + np.logical_and((t >= 0), (t <= 1)) * self._da_fun(t))
            self.db = lambda t: ((t < 0) * db0 + (t > 1) * db1
                                 + np.logical_and((t >= 0), (t <= 1)) * self._db_fun(t))
            self.d2a = lambda t: np.logical_and((t >= 0.), (t <= 1)) * self._d2a_fun(t)
            self.d2b = lambda t: np.logical_and((t >= 0.), (t <= 1)) * self._d2b_fun(t)
            self.d3a = lambda t: np.logical_and((t >= 0.), (t <= 1)) * self._d3a_fun(t)
            self.d3b = lambda t: np.logical_and((t >= 0.), (t <= 1)) * self._d3b_fun(t)

        elif mode == -1:
            self.a = lambda t: ((t > 0) * (da0 * t) + (t < -1) * (-a1 + da1 * (t + 1))
                                 + np.logical_and(t <= 0, t >= -1) * -self._a_fun(-t))
            self.b = lambda t: ((t > 0) * (db0 * t) + (t < -1) * (-b1 + db1 * (t + 1))
                                 + np.logical_and(t <= 0, t >= -1) * -self._b_fun(-t))
            self.da = lambda t: ((t > 0) * da0 + (t < -1) * da1
                                  + np.logical_and((t <= 0), (t >= -1)) * self._da_fun(-t))
            self.db = lambda t: ((t > 0) * db0 + (t < -1) * db1
                                  + np.logical_and((t <= 0), (t >= -1)) * self._db_fun(-t))
            self.d2a = lambda t: np.logical_and((t <= 0), (t >= -1)) * -self._d2a_fun(-t)
            self.d2b = lambda t: np.logical_and((t <= 0), (t >= -1)) * -self._d2b_fun(-t)
            self.d3a = lambda t: np.logical_and((t <= 0), (t >= -1)) * self._d3a_fun(-t)
            self.d3b = lambda t: np.logical_and((t <= 0), (t >= -1)) * self._d3b_fun(-t)
        else:
            raise InvalidBehaviorParameters('This error should never be triggered '
                                            '(error: mode not -1, 0 or 1, while making behavior)')

        self._dbda = lambda t: self.db(t) / self.da(t)
        self._d_dbda = lambda t: (self.d2b(t) * self.da(t) - self.db(t) * self.d2a(t)) / self.da(t) ** 2

        def d2_dbda(t):
            da = self.da(t)
            db = self.db(t)
            d2a = self.d2a(t)
            d2b = self.d2b(t)
            d3a = self.d3a(t)
            d3b = self.d3b(t)
            return (d3b * da ** 2 - db * d3a * da - 2 * d2b * da * d2a + 2 * db * d2a ** 2) / da ** 3

        self._d2_dbda = d2_dbda

        def int_adb(t):
            if isinstance(t, float):
                def adb(_t):
                    return self.db(_t) * self.a(_t)

                return quad(adb, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def adb(_t, _):
                    return self.db(_t) * self.a(_t)

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
                    return self.b(_t) * self.da(_t)

                return quad(bda, 0, t)[0]
            elif isinstance(t, np.ndarray):
                def bda(_t, _):
                    return self.b(_t) * self.da(_t)

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
        if mode == -1:
            all_t = np.linspace(-1, 0, round(1e3))
        else:
            all_t = np.linspace(0, 1, round(1e3))
        da = self.da(all_t)
        db = self.db(all_t)
        db_da = db / da
        kmax = np.max(db_da[da > 0]) if np.any(da > 0.0) else np.inf
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
                da_pos = self.da(t) >= 0
                da_neg = self.da(t) < 0
                k_arr[da_pos] = np.maximum(self._dbda(t[da_pos]) + delta, kstar)
                k_arr[da_neg] = kstar
                return k_arr

            def dk_fun(t) -> np.ndarray:
                dk_arr = np.zeros_like(t)
                indices = np.logical_and(self.da(t) >= 0, self._dbda(t) + delta > kstar)
                dk_arr[indices] = 1.0 * self._d_dbda(t[indices])
                return dk_arr

            def d2k_fun(t) -> np.ndarray:
                d2k_arr = np.zeros_like(t)
                indices = np.logical_and(self.da(t) >= 0, self._dbda(t) + delta > kstar)
                d2k_arr[indices] = 1.0 * self._d2_dbda(t[indices])
                return d2k_arr

        self.k = k_fun
        self.dk = dk_fun
        self.d2k = d2k_fun

    def _a_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _b_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _da_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _db_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _d2a_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _d2b_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _d3a_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def _d3b_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def get_a_extrema(self) -> np.ndarray:
        raise NotImplementedError("This method is abstract.")

    def get_b_extrema(self) -> np.ndarray:
        raise NotImplementedError("This method is abstract.")

    def elastic_energy(self, alpha: float, t: float) -> np.ndarray:
        y = alpha - self._natural_measure
        return 0.5 * self.k(t) * (y - self.a(t)) ** 2 + y * self.b(t) - self._int_adb(t)

    def gradient_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        dvdalpha = self.k(t) * (y - self.a(t)) + self.b(t)
        dvdt = (y - self.a(t)) * (self.db(t) - self.k(t) * self.da(t))
        if not self._is_k_constant:
            dvdt += 0.5 * (y - self.a(t)) ** 2 * self.dk(t)
        return dvdalpha, dvdt

    def hessian_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = alpha - self._natural_measure
        d2vdalpha2 = self.k(t)
        d2vdalphadt = self.db(t) - self.k(t) * self.da(t)
        d2vdt2 = (y - self.a(t)) * (self.d2b(t) - self.k(t) * self.d2a(t)) + self.da(t) * (
                self.k(t) * self.da(t) - self.db(t))
        if not self._is_k_constant:
            d2vdalphadt += self.dk(t) * (y - self.a(t))
            d2vdt2 += (y - self.a(t)) * (0.5 * self.d2k(t) * (y - self.a(t)) - 2 * self.da(t) * self.dk(t))
        return d2vdalpha2, d2vdalphadt, d2vdt2

    def get_hysteron_info(self) -> dict[str, float]:
        if self._hysteron_info is None:
            self._compute_hysteron_info()
        return self._hysteron_info

    def _compute_hysteron_info(self):
        extrema = self.get_a_extrema()
        if extrema.shape[0] == 0:  # not a hysteron
            self._hysteron_info = {}
        else:
            mode = self._parameters['mode']
            if mode == 0:
                extrema = np.hstack((-extrema[::-1], extrema))
                nb_extrema = extrema.shape[0]
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
                self._hysteron_info = {'nb_stable_branches': 2 * ((nb_extrema / 2) // 2 + 1) - 1,
                                       'branch_intervals': branch_intervals,
                                       'is_branch_stable': is_branch_stable,
                                       'branch_ids': branch_ids}
            if mode == 1:
                nb_extrema = extrema.shape[0]
                critical_t = np.hstack(([-np.inf], extrema, [np.inf]))
                branch_intervals = [(critical_t[i], critical_t[i + 1]) for i in range(nb_extrema + 1)]
                is_branch_stable = [i % 2 == 0 for i in range(len(branch_intervals))]
                branch_ids = []
                for i in range(len(branch_intervals)):
                    if is_branch_stable[i]:
                        branch_id = str(i // 2)
                    else:
                        branch_id = str(i // 2) + '-' + str(i // 2 + 1)
                    branch_ids.append(branch_id)
                self._hysteron_info = {'nb_stable_branches': nb_extrema + 1,
                                       'branch_intervals': branch_intervals,
                                       'is_branch_stable': is_branch_stable,
                                       'branch_ids': branch_ids}
            if mode == -1:
                nb_extrema = extrema.shape[0]
                critical_t = np.hstack(([-np.inf], extrema, [np.inf]))
                branch_intervals = [(critical_t[i], critical_t[i + 1]) for i in range(nb_extrema + 1)]
                is_branch_stable = [i % 2 == 0 for i in range(len(branch_intervals))][::-1]
                branch_ids = []
                for i in range(len(branch_intervals)):
                    if is_branch_stable[i]:
                        branch_id = str(i // 2)
                    else:
                        branch_id = str(i // 2) + '-' + str(i // 2 + 1)
                    branch_ids.append(branch_id)
                branch_ids = branch_ids[::-1]
                self._hysteron_info = {'nb_stable_branches': nb_extrema + 1,
                                       'branch_intervals': branch_intervals,
                                       'is_branch_stable': is_branch_stable,
                                       'branch_ids': branch_ids}

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._hysteron_info = None



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
    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], mode: int = 0,
                 sampling: int = 250):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, mode=mode)
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
        if not bezier_curve.is_monotonic(u_coefs):
            raise InvalidBehaviorParameters("The Bezier behavior does not describe a function. "
                                            "To define a multivalued curve, try Bezier2 instead.")

    def is_monotonic(self):
        f_coefs = np.array([0.0] + self._parameters['f_i'])
        return bezier_curve.is_monotonic(f_coefs)

    def _make(self):
        u_coefs = np.array([0.0] + self._parameters['u_i'])
        f_coefs = np.array([0.0] + self._parameters['f_i'])
        mode = self._parameters['mode']
        t = np.linspace(0, 1, self._sampling)
        u = bezier_curve.evaluate_poly(t, u_coefs)

        def fdu(_t, _):
            return bezier_curve.evaluate_poly(_t, f_coefs) * bezier_curve.evaluate_derivative_poly(_t,u_coefs)

        energy = solve_ivp(fun=fdu, t_span=[0.0, 1.0], y0=[0.0], t_eval=t).y[0, :]
        generalized_force = bezier_curve.evaluate_poly(t, f_coefs)
        df = bezier_curve.evaluate_derivative_poly(t, f_coefs)
        du = bezier_curve.evaluate_derivative_poly(t, u_coefs)
        generalized_stiffness = df / du


        u_start = u[0]
        k_start = generalized_stiffness[0]

        u_end = u[-1]
        e_end = energy[-1]
        f_end = generalized_force[-1]
        k_end = generalized_stiffness[-1]

        def energy_fun(uu):
            if mode == 0:
                e_in = interp1d(u, energy, kind='linear', bounds_error=False, fill_value=0.0)(np.abs(uu))
                e_out = e_end + f_end * (np.abs(uu) - u_end) + 0.5 * k_end * (np.abs(uu)- u_end) ** 2
                return (uu <= u_end) * e_in + (uu > u_end) * e_out
            if mode == 1:
                e_in = interp1d(u, energy, kind='linear', bounds_error=False, fill_value=0.0)(uu)
                e_beyond = e_end + f_end * (uu - u_end) + 0.5 * k_end * (uu - u_end) ** 2
                e_compression = + 0.5 * k_start * uu ** 2
                return (np.logical_and(uu <= u_end, uu > u_start) * e_in
                        + (uu > u_end) * e_beyond
                        + (uu < u_start) * e_compression)
            if mode == -1:
                e_in = interp1d(u, energy, kind='linear', bounds_error=False, fill_value=0.0)(-uu)
                e_beyond = e_end + f_end * (-uu - u_end) + 0.5 * k_end * (-uu - u_end) ** 2
                e_tension = + 0.5 * k_start * uu ** 2
                return (np.logical_and(-uu <= u_end, -uu > u_start) * e_in
                        + (-uu > u_end) * e_beyond
                        + (-uu < u_start) * e_tension)

        def gradient_fun(uu):
            tensile_f_fun = interp1d(u, generalized_force, kind='linear', bounds_error=False, fill_value='extrapolate')
            if mode == 0:
                return np.sign(uu) * tensile_f_fun(np.abs(uu))
            if mode == 1:
                return (uu > u_start) * tensile_f_fun(uu) + (uu < u_start) * k_start * uu
            if mode == -1:
                return (uu < u_start) * -tensile_f_fun(-uu) + (uu > u_start) * k_start * uu

        def hessian_fun(uu):
            tensile_k_fun = interp1d(u, generalized_stiffness, kind='linear', bounds_error=False, fill_value=k_end)
            if mode == 0:
                return tensile_k_fun(np.abs(uu))
            if mode == 1:
                return (uu >= u_start) * tensile_k_fun(uu) + (uu < u_start) * k_start
            if mode == -1:
                return (uu <= u_start) * tensile_k_fun(-uu) + (uu > u_start) * k_start

        self._energy = energy_fun
        self._gradient = gradient_fun
        self._hessian = hessian_fun

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._gradient(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._hessian(alpha - self._natural_measure),

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
            l = l0 + np.linspace(0.0, 1.1*u_i[-1])
            f, = bb.gradient_energy(l)
            fig, ax = plt.subplots()
            ax.plot(l - l0, f, '-', lw=5, alpha=0.5)
            ax.plot(*bb.get_control_points(), '-o')
            for fd_curve in force_displacement_curves:
                ax.plot(fd_curve[0], fd_curve[1], 'k-')
            plt.show()
        return u_i, f_i


class Bezier2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], mode: int = 0):
        super().__init__(natural_measure, mode=mode)
        self._parameters['u_i'] = u_i
        self._parameters['f_i'] = f_i
        self._a_coefs = np.array([0.0] + self._parameters['u_i'])
        self._b_coefs = np.array([0.0] + self._parameters['f_i'])
        self._check()
        self._make()

    def _a_fun(self, t):
        return bezier_curve.evaluate_poly(t, self._a_coefs)

    def _b_fun(self, t):
        return bezier_curve.evaluate_poly(t, self._b_coefs)

    def _da_fun(self, t):
        return bezier_curve.evaluate_derivative_poly(t, self._a_coefs)

    def _db_fun(self, t):
        return bezier_curve.evaluate_derivative_poly(t, self._b_coefs)

    def _d2a_fun(self, t):
        return bezier_curve.evaluate_second_derivative_poly(t, self._a_coefs)

    def _d2b_fun(self, t):
        return bezier_curve.evaluate_second_derivative_poly(t, self._b_coefs)

    def _d3a_fun(self, t):
        return bezier_curve.evaluate_third_derivative_poly(t, self._a_coefs)

    def _d3b_fun(self, t):
        return bezier_curve.evaluate_third_derivative_poly(t, self._b_coefs)

    def get_a_extrema(self) -> np.ndarray:
        return bezier_curve.get_extrema(self._a_coefs)

    def get_b_extrema(self) -> np.ndarray:
        return bezier_curve.get_extrema(self._b_coefs)

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        cp_x = np.array([0.0] + self._parameters['u_i'])
        cp_y = np.array([0.0] + self._parameters['f_i'])
        return cp_x, cp_y

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._a_coefs = np.array([0.0] + self._parameters['u_i'])
        self._b_coefs = np.array([0.0] + self._parameters['f_i'])
        self._check()
        if parameters:
            self._make()


class Spline2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], mode: int = 0):
        super().__init__(natural_measure, mode=mode)
        self._parameters['u_i'] = u_i
        self._parameters['f_i'] = f_i
        cp_t = np.linspace(0, 1, len(u_i) + 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)
        self._a_spl = make_interp_spline(cp_t, cp_u)
        self._b_spl = make_interp_spline(cp_t, cp_f)
        self._check()
        self._make()

    def _a_fun(self, t):
        return self._a_spl(t)

    def _b_fun(self, t):
        return self._b_spl(t)

    def _da_fun(self, t):
        return self._a_spl.derivative(1)(t)

    def _db_fun(self, t):
        return self._b_spl.derivative(1)(t)

    def _d2a_fun(self, t):
        return self._a_spl.derivative(2)(t)

    def _d2b_fun(self, t):
        return self._b_spl.derivative(2)(t)

    def _d3a_fun(self, t):
        return self._a_spl.derivative(3)(t)

    def _d3b_fun(self, t):
        return self._b_spl.derivative(3)(t)

    def get_a_extrema(self) -> np.ndarray:
        t_roots = PPoly.from_spline(self._a_spl.derivative()).roots()
        t_roots = np.real(t_roots[np.isreal(t_roots)])
        return np.sort(t_roots[np.logical_and(t_roots >= 0.0, t_roots <= 1.0)])

    def get_b_extrema(self) -> np.ndarray:
        t_roots = PPoly.from_spline(self._b_spl.derivative()).roots()
        t_roots = np.real(t_roots[np.isreal(t_roots)])
        return np.sort(t_roots[np.logical_and(t_roots >= 0.0, t_roots <= 1.0)])

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        cp_x = np.array([0.0] + self._parameters['u_i'])
        cp_y = np.array([0.0] + self._parameters['f_i'])
        return cp_x, cp_y

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        u_i = self._parameters['u_i']
        f_i = self._parameters['f_i']
        cp_t = np.linspace(0, 1, len(u_i) + 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)
        self._a_spl = make_interp_spline(cp_t, cp_u)
        self._b_spl = make_interp_spline(cp_t, cp_f)
        self._check()
        if parameters:
            self._make()


class PiecewiseBehavior(UnivariateBehavior, ControllableByPoints):
    def __init__(self, natural_measure, k_i, u_i, us, mode: int = 0):
        super().__init__(natural_measure, k_i=k_i, u_i=u_i, mode=mode, us=us)
        self._check()
        self._make()

    def _check(self):
        k, u, us = self._parameters['k_i'], self._parameters['u_i'], self._parameters['us']
        mode = self._parameters['mode']
        if mode not in (-1, 0, 1):
            raise InvalidBehaviorParameters('The mode should be -1, 0 or +1 (integer; no float number allowed).')
        if any(ui < 0 for ui in u):
            raise InvalidBehaviorParameters('Transition displacements must always be stricly positive. '
                                            'To make a piecewise behavior in compression only, use "mode = -1".')
        if us <= 0:
            raise InvalidBehaviorParameters('us must be strictly positive.')
        if u[0] - 0.0 < us or (len(u) > 1 and np.min(np.diff(u)) < 2 * us):
            raise InvalidBehaviorParameters(
                f'us ({us:.3E}) is too large for the piecewise intervals provided. '
                f'Should be less than {min(np.min(np.diff(u)) / 2, u[0] - 0.0):.3E}')
        if len(k) != len(u) + 1:
            raise InvalidBehaviorParameters(f'Expected {len(k) - 1} transitions, but only {len(k)} were provided.')

    def _make(self):
        k, u, us = self._parameters['k_i'], self._parameters['u_i'], self._parameters['us']
        mode = self._parameters['mode']
        self._u_i, self._f_i = spw.compute_piecewise_control_points(k, u, extra=4 * us)
        force_fun = spw.create_smooth_piecewise_function(k, u, us)
        stiffness_fun = spw.create_smooth_piecewise_derivative_function(k, u, us)
        if mode == 0:
            self._force_function = lambda uu: np.sign(uu) * force_fun(np.abs(uu))
            self._stiffness_function = lambda uu: stiffness_fun(np.abs(uu))
        elif mode == 1:
            self._force_function = force_fun
            self._stiffness_function = stiffness_fun
        elif mode == -1:
            self._force_function = lambda uu: -force_fun(-uu)
            self._stiffness_function = lambda uu: stiffness_fun(-uu)
        else:
            raise InvalidBehaviorParameters('This error should never be triggered '
                                            '(error: mode not -1, 0 or 1, while making behavior)')

    def is_monotonic(self):
        k = self._parameters['k_i']
        return all(kk > 0 for kk in k)


    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self._u_i.copy(), self._f_i.copy()

    def update_from_control_points(self, cp_x, cp_y):
        if (np.diff(cp_x) < 0).any():
            raise InvalidBehaviorParameters(f'Each control point must be on the right of the previous one.')
        k, u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_x, cp_y)
        self.update(k_i=k, u_i=u)

    def elastic_energy(self, alpha: float) -> float:
        return quad(self._force_function, 0.0, alpha - self._natural_measure)[0]

    def gradient_energy(self, alpha: float) -> tuple[float]:
        return self._force_function(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float) -> tuple[float]:
        return self._stiffness_function(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()


class ZigzagBehavior(UnivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], epsilon: float, mode: int = 0,
                 sampling: int = 100):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, mode=mode, epsilon=epsilon)
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
        mode = self._parameters['mode']
        if mode not in (-1, 0, 1):
            raise InvalidBehaviorParameters('The mode should be -1, 0 or +1 (integer; no float number allowed).')
        if any(ui < 0 for ui in u_i):
            raise InvalidBehaviorParameters('The u_i values must be strictly positive. '
                                            'To make a zigzag behavior in compression only, use "mode = -1".')
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')
        if len(u_i) != len(f_i):
            raise InvalidBehaviorParameters(f'u_i and f_i must contain the same number of elements')
        if (np.diff(u_i, prepend=0.0) <= 0.0).any():
            raise InvalidBehaviorParameters(
                f'u_i values should be positive and monotonically increasing. '
                f'For a multivalued curve, try Zigzag2 instead.')

    def _make(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        mode = self._parameters['mode']
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
        if mode == 0:
            self._energy = lambda uu: ((np.abs(uu) <= u_s[-1]) * e_inside(np.abs(uu))
                                       + (np.abs(uu) > u_s[-1]) * (e_s[-1] + f_s[-1] * (np.abs(uu) - u_s[-1])
                                                                    + 0.5 * k_s[-1] * (np.abs(uu) - u_s[-1]) ** 2))
            self._force = lambda uu: np.sign(uu) * interp1d(u_s, f_s, kind='linear', bounds_error=False,
                                                            fill_value='extrapolate')(np.abs(uu))
            self._stiffness = lambda uu: interp1d(u_s, k_s, kind='linear', bounds_error=False, fill_value=k_s[-1])(np.abs(uu))
        elif mode == 1:
            self._energy = lambda uu: (np.logical_and(uu <= u_s[-1], uu >=0) * e_inside(uu)
                                       + (uu > u_s[-1]) * (e_s[-1] + f_s[-1] * (uu - u_s[-1])
                                                                    + 0.5 * k_s[-1] * (uu - u_s[-1]) ** 2)
                                       + (uu < 0) * (0.5 * k_s[0] * uu ** 2))

            self._force = interp1d(u_s, f_s, kind='linear', bounds_error=False, fill_value='extrapolate')
            self._stiffness = interp1d(u_s, k_s, kind='linear', bounds_error=False, fill_value=(k_s[0], k_s[-1]))
        elif mode == -1:
            self._energy = lambda uu: (np.logical_and(uu >= -u_s[-1], uu <= 0) * e_inside(-uu)
                                       + (uu < -u_s[-1]) * (e_s[-1] + f_s[-1] * (u_s[-1] - uu)
                                                           + 0.5 * k_s[-1] * (uu - u_s[-1]) ** 2)
                                       + (uu > 0) * (0.5 * k_s[0] * uu ** 2))

            self._force = lambda uu: -interp1d(u_s, f_s, kind='linear', bounds_error=False, fill_value='extrapolate')(-uu)
            self._stiffness = lambda uu: interp1d(u_s, k_s, kind='linear', bounds_error=False, fill_value=(k_s[0], k_s[-1]))(-uu)
        else:
            raise InvalidBehaviorParameters('This error should never be triggered '
                                            '(error: mode not -1, 0 or 1, while making behavior)')


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

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], epsilon: float, mode: int = 0):
        super().__init__(natural_measure, mode)
        self._parameters['u_i'] = u_i
        self._parameters['f_i'] = f_i
        self._parameters['epsilon'] = epsilon

        n = len(u_i) + 1
        cp_t = np.arange(n) / (n - 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)

        # to update in the 'update' method
        self._delta = epsilon / (2 * (n - 1))
        self._k_u, self._x_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = spw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = spw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = spw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = spw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)

        self._check()
        self._make()

    def _a_fun(self, t):
        return self._raw_a_fun(t)

    def _b_fun(self, t):
        return self._raw_b_fun(t)

    def _da_fun(self, t):
        return self._raw_da_fun(t)

    def _db_fun(self, t):
        return self._raw_db_fun(t)

    def _d2a_fun(self, t):
        return self._raw_d2a_fun(t)

    def _d2b_fun(self, t):
        return self._raw_d2b_fun(t)

    def _d3a_fun(self, t):
        return np.zeros_like(t)

    def _d3b_fun(self, t):
        return np.zeros_like(t)

    def get_a_extrema(self) -> np.ndarray:
        return spw.get_extrema(self._k_u, self._x_u, self._delta)

    def get_b_extrema(self) -> np.ndarray:
        return spw.get_extrema(self._k_f, self._x_f, self._delta)

    def _check(self):
        u_i, f_i, epsilon = self._parameters['u_i'], self._parameters['f_i'], self._parameters['epsilon']
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')
        if len(u_i) != len(f_i):
            raise InvalidBehaviorParameters(f'u_i and f_i must contain the same number of elements')
        super()._check()

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0] + self._parameters['u_i']), np.array([0.0] + self._parameters['f_i'])

    def update_from_control_points(self, cp_x, cp_y):
        u_i = cp_x[1:].tolist()
        f_i = cp_y[1:].tolist()
        self.update(u_i=u_i, f_i=f_i)

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        u_i = self._parameters['u_i']
        f_i = self._parameters['f_i']
        epsilon = self._parameters['epsilon']

        n = len(u_i) + 1
        cp_t = np.arange(n) / (n - 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)
        self._delta = epsilon / (2 * (n - 1))
        self._k_u, self._x_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = spw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = spw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = spw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = spw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)

        self._check()
        if parameters:
            self._make()


class ContactBehavior(UnivariateBehavior):
    def __init__(self, natural_measure, f0, uc, delta):
        super().__init__(natural_measure, f0=f0, uc=uc, delta=delta)
        self._p = 3.0

    def elastic_energy(self, alpha: float) -> float:
        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        if alpha >= delta:
            return 0.0
        else:
            return f0 * uc / (p + 1) * ((delta - alpha) / uc) ** (p + 1)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float | np.ndarray]:
        if isinstance(alpha, np.ndarray):
            g = np.empty_like(alpha)
            for i, alpha_i in enumerate(alpha):
                g[i] = self.gradient_energy(alpha_i)[0]
            return g,

        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        if alpha >= delta:
            return 0.0,
        else:
            return -f0 * ((delta - alpha) / uc) ** p,

    def hessian_energy(self, alpha: float) -> tuple[float]:
        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        if alpha >= delta:
            return 0.0,
        else:
            return +f0 / uc * p * ((delta - alpha) / uc) ** (p - 1),


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


class IsothermalGas(IdealGas):

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
