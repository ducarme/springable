from collections.abc import Callable
from typing import Any

from ..utils import misc
from ..utils import bezier_curve
from ..utils import smooth_piecewise_linear_curve as spw
from ..utils import smoother_piecewise_linear_curve as sspw
from scipy.interpolate import interp1d, griddata, make_interp_spline, PPoly
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt

from ..utils.smooth_piecewise_linear_curve import create_smooth_piecewise_function


class MechanicalBehavior:
    _nb_dofs: int = NotImplemented

    def __init__(self, natural_measure, /, **parameters):
        self._natural_measure = natural_measure
        self._parameters = parameters

    def elastic_energy(self, *variables: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method is abstract.")

    def gradient_energy(self, *variables: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError("This method is abstract.")

    def hessian_energy(self, *variables: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError("This method is abstract.")

    @classmethod
    def get_nb_dofs(cls) -> int:
        return cls._nb_dofs

    def get_parameters(self) -> dict[str, Any]:
        return self._parameters

    def get_natural_measure(self) -> float:
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

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The method called is abstract.")

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        raise NotImplementedError("The method is abstract")

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        raise NotImplementedError("The method is abstract")


class BivariateBehavior(MechanicalBehavior):
    _nb_dofs: int = 2

    def __init__(self, natural_measure, mode):
        super().__init__(natural_measure, mode=mode)

        # will only be updated when calling _make()
        self.a: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.b: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.da: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.db: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.d2a: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.d2b: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.d3a: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.d3b: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.k: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.dk: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.d2k: Callable[[np.ndarray], np.ndarray] = self._not_made
        self.int_bda: Callable[[np.ndarray], np.ndarray] = self._not_made

        # to be overriden in subclasses
        self.tmax: float = NotImplemented

        # to update in the 'update' method
        self._hysteron_info = None
    
    def _not_made(self, *args) :
        raise RuntimeError('This function has not been initialized.')

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
        tmax = self.tmax

        if mode == 0:
            self.a = lambda t: np.sign(t) * self._a_fun(np.abs(t))
            self.b = lambda t: np.sign(t) *self._b_fun(np.abs(t))
            self.da = lambda t: self._da_fun(np.abs(t))
            self.db = lambda t: self._db_fun(np.abs(t))
            self.d2a = lambda t: np.sign(t) * self._d2a_fun(np.abs(t))
            self.d2b = lambda t: np.sign(t) * self._d2b_fun(np.abs(t))
            self.d3a = lambda t: self._d3a_fun(np.abs(t))
            self.d3b = lambda t: self._d3b_fun(np.abs(t))
            self.int_bda = lambda t: self._int_bda_fun(np.abs(t))

        elif mode == 1:
            self.a = self._a_fun
            self.b = self._b_fun
            self.da = self._da_fun
            self.db = self._db_fun
            self.d2a = self._d2a_fun
            self.d2b = self._d2b_fun
            self.d3a = self._d3a_fun
            self.d3b = self._d3b_fun
            self.int_bda = self._int_bda_fun

        elif mode == -1:
            self.a = lambda t: -self._a_fun(-t)
            self.b = lambda t: -self._b_fun(-t)
            self.da = lambda t: self._da_fun(-t)
            self.db = lambda t: self._db_fun(-t)
            self.d2a = lambda t: -self._d2a_fun(-t)
            self.d2b = lambda t: -self._d2b_fun(-t)
            self.d3a = lambda t: self._d3a_fun(-t)
            self.d3b = lambda t: self._d3b_fun(-t)
            self.int_bda = lambda t: self._int_bda_fun(-t)

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


        if mode == -1:
            all_t = np.linspace(-tmax, 0, round(100))
        else:
            all_t = np.linspace(0, tmax, round(100))
        da = self.da(all_t)
        db = self.db(all_t)
        db_da = db / da
        kmax = np.max(db_da[da > 0]) if np.any(da > 0.0) else np.inf
        kmin = np.min(db_da[da < 0]) if np.any(da < 0.0) else np.inf
        delta = 0.05 * kmax
        kstar = min(kmin - delta, kmax + delta)

        if kmin - kmax > 2 * delta:
            self._is_k_constant = True

            def k_fun(t):  # type: ignore
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
    
    def _int_bda_fun(self, t):
        raise NotImplementedError("This method is abstract.")

    def get_a_extrema(self) -> np.ndarray:
        raise NotImplementedError("This method is abstract.")

    def get_b_extrema(self) -> np.ndarray:
        raise NotImplementedError("This method is abstract.")

    def elastic_energy(self, alpha: np.ndarray, t: np.ndarray) -> np.ndarray:
        u = alpha - self._natural_measure
        return 0.5 * self.k(t) * (u - self.a(t)) ** 2 + self.b(t) * (u - self.a(t)) + self.int_bda(t)

    def gradient_energy(self, alpha: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = alpha - self._natural_measure
        dvdalpha = self.k(t) * (u - self.a(t)) + self.b(t)
        dvdt = (u - self.a(t)) * (self.db(t) - self.k(t) * self.da(t))
        if not self._is_k_constant:
            dvdt += 0.5 * (u - self.a(t)) ** 2 * self.dk(t)
        return dvdalpha, dvdt

    def hessian_energy(self, alpha: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = alpha - self._natural_measure
        d2vdalpha2 = self.k(t)
        d2vdalphadt = self.db(t) - self.k(t) * self.da(t)
        d2vdt2 = (u - self.a(t)) * (self.d2b(t) - self.k(t) * self.d2a(t)) + self.da(t) * (
                self.k(t) * self.da(t) - self.db(t))
        if not self._is_k_constant:
            d2vdalphadt += self.dk(t) * (u - self.a(t))
            d2vdt2 += (u - self.a(t)) * (0.5 * self.d2k(t) * (u - self.a(t)) - 2 * self.da(t) * self.dk(t))
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
            if mode == 0:  # symmetric mode
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
            if mode == 1:  # tensile mode
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
                self._hysteron_info = {'nb_stable_branches': nb_extrema // 2 + 1,
                                       'branch_intervals': branch_intervals,
                                       'is_branch_stable': is_branch_stable,
                                       'branch_ids': branch_ids}
            if mode == -1:  # compressive mode
                nb_extrema = extrema.shape[0]
                critical_t = np.hstack(([-np.inf], -extrema[::-1], [np.inf]))
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
                self._hysteron_info = {'nb_stable_branches': nb_extrema // 2 + 1,
                                       'branch_intervals': branch_intervals,
                                       'is_branch_stable': is_branch_stable,
                                       'branch_ids': branch_ids}

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._hysteron_info = None
        # TO BE EXTENDED FURTHER IN NECESSARY




class ControllableByPoints:
    """ Interface to implement for curves that can be defined by control points """

    def get_control_points(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update_from_control_points(self, cp_x: np.ndarray, cp_y: np.ndarray):
        raise NotImplementedError


class LinearBehavior(UnivariateBehavior):

    def __init__(self, natural_measure: float, k: float):
        super().__init__(natural_measure, k=k)

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        return 0.5 * self._parameters['k'] * (alpha - self._natural_measure) ** 2

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._parameters['k'] * (alpha - self._natural_measure),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._parameters['k'] * np.ones_like(alpha),

    def get_spring_constant(self) -> float:
        return self._parameters['k']


class LogarithmicBehavior(UnivariateBehavior):

    def __init__(self, natural_measure: float, k: float):
        super().__init__(natural_measure, k=k)

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        energy = np.where(alpha < 0.0, np.nan,
                          self._parameters['k'] * self._natural_measure * alpha * (np.log(alpha / self._natural_measure) - 1))
        return energy

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray]:
        force = np.where(alpha < 0.0, np.nan, self._parameters['k'] * self._natural_measure * np.log(alpha / self._natural_measure))
        return force,

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray]:
        stiffness = np.where(alpha < 0.0, np.nan, self._parameters['k'] * self._natural_measure / alpha)
        return stiffness,


class BezierBehavior(UnivariateBehavior, ControllableByPoints):
    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], mode: int = 0, sampling: int = 50):
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

        f_in = lambda t: bezier_curve.evaluate_poly(t, f_coefs)
        du_in = lambda t: bezier_curve.evaluate_derivative_poly(t, u_coefs)
        df_in = lambda t: bezier_curve.evaluate_derivative_poly(t, f_coefs)
        int_fdu_in = bezier_curve.create_antiderivative_of_parametric_bezier(f_coefs, u_coefs)
        # true inverse, but computationanly costly
        #inv_u_in = lambda uu: bezier_curve.evaluate_inverse_poly(uu, u_coefs)

        # approximate inverse, the f(u) is still infinitely smooth, except at the boundaries u=0 and u=u_coefs[-1]
        t_sampled = np.linspace(0, 1, round(self._sampling * len(u_coefs)))
        u_in_sampled = bezier_curve.evaluate_poly(t_sampled, u_coefs)
        inv_u_in = interp1d(u_in_sampled, t_sampled, kind='linear', bounds_error=False, fill_value=0.0)

        e = lambda uu: ( (uu <= 0) * 0.5 * (f_coefs[1] / u_coefs[1]) * uu**2
                        + np.logical_and(uu > 0, uu < u_coefs[-1]) * int_fdu_in(inv_u_in(uu))
                        + (uu >= u_coefs[-1]) * (int_fdu_in(1.0) + f_coefs[-1]*(uu-u_coefs[-1]) + (f_coefs[-1] - f_coefs[-2])/(u_coefs[-1] - u_coefs[-2])*(uu - u_coefs[-1])**2/2)
                        )
        f = lambda uu: ( (uu <= 0) * (f_coefs[1] / u_coefs[1]) * uu
                        + np.logical_and(uu > 0, uu < u_coefs[-1]) * f_in(inv_u_in(uu))
                        + (uu >= u_coefs[-1]) * (f_coefs[-1] + (f_coefs[-1] - f_coefs[-2])/(u_coefs[-1] - u_coefs[-2])*(uu - u_coefs[-1]))
                        )
        k = lambda uu: ( (uu <= 0) * (f_coefs[1] / u_coefs[1])
                        + np.logical_and(uu > 0, uu < u_coefs[-1]) * df_in(inv_u_in(uu)) / du_in(inv_u_in(uu))
                        + (uu >= u_coefs[-1]) * (f_coefs[-1] - f_coefs[-2])/(u_coefs[-1] - u_coefs[-2])
                        )

        if mode == 0:
            self._energy = lambda uu: e(np.abs(uu))
            self._force = lambda uu: np.sign(uu) * f(np.abs(uu))
            self._stiffness = lambda uu: k(np.abs(uu))
        elif mode == 1:
            self._energy = e
            self._force = f
            self._stiffness = k
        elif mode == -1:
            self._energy = lambda uu: e(-uu)
            self._force = lambda uu: -f(-uu)
            self._stiffness = lambda uu: k(-uu)
        else:
            raise InvalidBehaviorParameters('This error should never be triggered '
                                            '(error: mode not -1, 0 or 1, while making behavior)')

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._force(alpha - self._natural_measure),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._stiffness(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()

    @classmethod
    def compute_fitting_parameters(cls, force_displacement_curves: list[tuple], degree: int, clamp_last=False, show=False):
        umax = np.min([np.max(fd_curve[0]) for fd_curve in force_displacement_curves])
        fmax = np.min([np.max(fd_curve[1]) for fd_curve in force_displacement_curves])
        nb_samples = 500
        u_sampling = np.linspace(0.0, umax, nb_samples)

        def compute_mismatch(x: np.ndarray) -> float:
            natural_measure = 1.0
            u_i = x[::2].tolist()
            f_i = x[1::2].tolist()
            if clamp_last:
                u_i.append(umax)  # type: ignore
                f_i.append(fmax)  # type: ignore
            try:
                behavior = BezierBehavior(natural_measure, u_i=u_i, f_i=f_i)  # type: ignore
                f_fit = behavior.gradient_energy(natural_measure + u_sampling)[0]
            except InvalidBehaviorParameters as e:
                mismatch = nb_samples * fmax ** 2
                print(f"Could not define a proper mismatch measure based on the parameters u_i={u_i} and f_i={f_i}. "
                      f"{e.get_message()}. Mismatch is defaulted to {mismatch:.3E}.")
            else:
                mismatch = 0.0
                for fd_curve in force_displacement_curves:
                    u_data = fd_curve[0]
                    f_data = fd_curve[1]
                    f_exp = interp1d(u_data, f_data, fill_value='extrapolate')(u_sampling)  # type: ignore
                    mismatch += np.sum((f_exp - f_fit) ** 2)
            return mismatch


        u_guess = np.linspace(1 / degree, 1.0, degree) * umax
        f_guess = np.linspace(1 / degree, 1.0, degree) * fmax
        if clamp_last:
            u_guess = u_guess[:-1]
            f_guess = f_guess[:-1]

        initial_values = np.empty(u_guess.shape[0] * 2)
        initial_values[0::2] = u_guess
        initial_values[1::2] = f_guess

        bounds = [(0.0, None) if i % 2 == 0 else (None, None) for i in range(2 * degree)]
        bounds[1] = (0.0, None)  # first control point should be above f=0 axis
        if clamp_last:
            bounds = bounds[:-2]
            bounds[-2] = (0.0, umax)

        # control point abscissas should be monotonically increasing
        if not clamp_last:
            constraint_matrix = np.zeros((degree - 1, 2 * degree))
            for i in range(degree - 1):
                constraint_matrix[i, 2 * i] = -1.0
                constraint_matrix[i, 2 * i + 2] = 1.0
            lb = np.zeros(constraint_matrix.shape[0])
            ub = np.ones(constraint_matrix.shape[0]) * np.inf
        else:
            constraint_matrix = np.zeros((degree - 2, 2 * (degree-1)))
            for i in range(degree - 2):
                constraint_matrix[i, 2 * i] = -1.0
                constraint_matrix[i, 2 * i + 2] = 1.0
            lb = np.zeros(constraint_matrix.shape[0])
            ub = np.ones(constraint_matrix.shape[0]) * np.inf

        constraints = LinearConstraint(constraint_matrix, lb, ub)  # type: ignore

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
        if clamp_last:
            u_i.append(umax)
            f_i.append(fmax)

        if show:
            l0 = 1.0
            bb = BezierBehavior(l0, u_i, f_i)
            l = l0 + np.linspace(0.0, 1.1*u_i[-1])
            f, = bb.gradient_energy(l)
            _, ax = plt.subplots()
            ax.plot(l - l0, f, '-', lw=5, alpha=0.5, label='fit')
            ax.plot(*bb.get_control_points(), '-o')
            for fd_curve in force_displacement_curves:
                ax.plot(fd_curve[0], fd_curve[1], 'k-', label='exp')
            ax.legend()
            plt.show()

        return u_i, f_i


class Bezier2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], mode: int = 0):
        super().__init__(natural_measure, mode=mode)
        self._parameters['u_i'] = u_i
        self._parameters['f_i'] = f_i

        # to update in the update()
        self._n = len(self._parameters['u_i'])
        self.tmax = self._compute_tmax()
        self._a_coefs = np.array([0.0] + self._parameters['u_i'])
        self._b_coefs = np.array([0.0] + self._parameters['f_i'])
        self._int_bda_inside_fun = bezier_curve.create_antiderivative_of_parametric_bezier(self._b_coefs, self._a_coefs)

        self._check()
        self._make()
    
    def _compute_tmax(self) -> float:
        return np.sum(np.abs(np.diff(self._parameters['u_i'], prepend=0.0)))

    def _a_fun(self, t):
        return ((t <= 0) * self._n * self._a_coefs[1] * t / self.tmax
                + np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_poly(t/self.tmax, self._a_coefs)
                + (t > self.tmax) * (self._a_coefs[-1] + self._n * (self._a_coefs[-1] - self._a_coefs[-2]) * (t / self.tmax - 1))
                )
                
    def _b_fun(self, t):
        return ((t <= 0) * self._n * self._b_coefs[1] * t / self.tmax
                + np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_poly(t/self.tmax, self._b_coefs)
                + (t > self.tmax) * (self._b_coefs[-1] + self._n * (self._b_coefs[-1] - self._b_coefs[-2]) * (t / self.tmax - 1))
                )

    def _da_fun(self, t):
        return ((t <= 0) * self._n * self._a_coefs[1] / self.tmax
                + np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_derivative_poly(t/self.tmax, self._a_coefs)/self.tmax
                + (t > self.tmax) * self._n * (self._a_coefs[-1] - self._a_coefs[-2])  / self.tmax
                )

    def _db_fun(self, t):
        return ((t <= 0) * self._n * self._b_coefs[1] / self.tmax
                + np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_derivative_poly(t/self.tmax, self._b_coefs)/self.tmax
                + (t > self.tmax) * self._n * (self._b_coefs[-1] - self._b_coefs[-2])  / self.tmax
                )

    def _d2a_fun(self, t):
        return np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_second_derivative_poly(t/self.tmax, self._a_coefs)/self.tmax**2

    def _d2b_fun(self, t):
        return np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_second_derivative_poly(t/self.tmax, self._b_coefs)/self.tmax**2

    def _d3a_fun(self, t):
        return np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_third_derivative_poly(t/self.tmax, self._a_coefs)/self.tmax**3

    def _d3b_fun(self, t):
        return np.logical_and(t > 0, t<= self.tmax) * bezier_curve.evaluate_third_derivative_poly(t/self.tmax, self._b_coefs)/self.tmax**3
    
    def _int_bda_fun(self, t):
        u1 = self._a_coefs[1]
        f1 = self._b_coefs[1]
        un = self._a_coefs[-1]
        fn = self._b_coefs[-1]
        un_1 = self._a_coefs[-2]
        fn_1 = self._b_coefs[-2]
        return ((t <= 0) * 0.5 * self._n**2 * f1 * u1 * (t / self.tmax)**2  # yes, correct!
                + np.logical_and(t > 0, t<= self.tmax) * self._int_bda_inside_fun(t / self.tmax)
                + (t > self.tmax) * (self._int_bda_inside_fun(1.0)
                                     + fn * self._n * (un - un_1)  * (t / self.tmax - 1)
                                     + 0.5 * self._n**2 * (un - un_1) * (fn - fn_1) * ((t / self.tmax - 1) ** 2)
                                     )
                )
    
    def get_a_extrema(self) -> np.ndarray:
        return bezier_curve.get_extrema(self._a_coefs) * self.tmax

    def get_b_extrema(self) -> np.ndarray:
        return bezier_curve.get_extrema(self._b_coefs) * self.tmax

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
        self._n = len(self._parameters['u_i'])
        self._int_bda_inside_fun = bezier_curve.create_antiderivative_of_parametric_bezier(self._b_coefs, self._a_coefs)
        self.tmax = self._compute_tmax()
        self._check()
        if parameters:
            self._make()



class Spline2Behavior(BivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], deg: int = 3, mode: int = 0):
        super().__init__(natural_measure, mode=mode)
        self._parameters['u_i'] = u_i
        self._parameters['f_i'] = f_i
        self._parameters['deg'] = deg
        
        cp_t = np.linspace(0, 1, len(u_i) + 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)

        # to update in the 'update' method
        self.tmax = self._compute_tmax()
        self._a_spl = make_interp_spline(cp_t, cp_u, k=deg)
        self._b_spl = make_interp_spline(cp_t, cp_f, k=deg)

        self._check()
        self._make()
    
    def _compute_tmax(self) -> float:
        return np.sum(np.abs(np.diff(self._parameters['u_i'], prepend=0.0)))

    def _check(self):
        if len(self._parameters['u_i']) < self._parameters['deg']:
            raise InvalidBehaviorParameters(f'No enough control points. '
                                            f'At least {self._parameters['deg'] +1} are required.')
        super()._check()

    def _a_fun(self, t):
        return self._a_spl(t/self.tmax)

    def _b_fun(self, t):
        return self._b_spl(t/self.tmax)

    def _da_fun(self, t):
        return self._a_spl.derivative(1)(t/self.tmax)/self.tmax

    def _db_fun(self, t):
        return self._b_spl.derivative(1)(t/self.tmax)/self.tmax

    def _d2a_fun(self, t):
        return self._a_spl.derivative(2)(t/self.tmax)/self.tmax**2

    def _d2b_fun(self, t):
        return self._b_spl.derivative(2)(t/self.tmax)/self.tmax**2

    def _d3a_fun(self, t):
        return self._a_spl.derivative(3)(t/self.tmax)/self.tmax**3

    def _d3b_fun(self, t):
        return self._b_spl.derivative(3)(t/self.tmax)/self.tmax**3
    
    # to implement int_bda

    def get_a_extrema(self) -> np.ndarray:
        t_roots = PPoly.from_spline(self._a_spl.derivative()).roots()
        t_roots = np.real(t_roots[np.isreal(t_roots)])
        return np.sort(t_roots[np.logical_and(t_roots >= 0.0, t_roots <= 1.0)]) * self.tmax

    def get_b_extrema(self) -> np.ndarray:
        t_roots = PPoly.from_spline(self._b_spl.derivative()).roots()
        t_roots = np.real(t_roots[np.isreal(t_roots)])
        return np.sort(t_roots[np.logical_and(t_roots >= 0.0, t_roots <= 1.0)]) * self.tmax

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
        cp_t = np.linspace(0, 1, len(self._parameters['u_i']) + 1)
        cp_u = np.array([0.0] + self._parameters['u_i'])
        cp_f = np.array([0.0] + self._parameters['f_i'])

        # to update before calling make()
        self.tmax = self._compute_tmax()
        self._a_spl = make_interp_spline(cp_t, cp_u, k=self._parameters['deg'])
        self._b_spl = make_interp_spline(cp_t, cp_f, k=self._parameters['deg'])
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
        if u[0] - 0.0 < us:
            raise InvalidBehaviorParameters(f'us ({us:.3E}) is too large for the first piecewise interval. '
                                            f'Should be less than {u[0]:.3E}')
        if len(u) > 1 and np.min(np.diff(u)) < 2 * us:
            raise InvalidBehaviorParameters(
                f'us ({us:.3E}) is too large for the piecewise intervals provided. '
                f'Should be less than {min(np.min(np.diff(u)) / 2, u[0] - 0.0):.3E}')
        if len(k) != len(u) + 1:
            raise InvalidBehaviorParameters(f'Expected {len(k) - 1} transitions, but only {len(k)} were provided.')

    def _make(self):
        k, u, us = self._parameters['k_i'], self._parameters['u_i'], self._parameters['us']
        mode = self._parameters['mode']
        self._u_i, self._f_i = spw.compute_piecewise_control_points(k, u, extra=4 * us)
        energy_fun = spw.create_smooth_piecewise_antiderivative_function(k, u, us)
        force_fun = spw.create_smooth_piecewise_function(k, u, us)
        stiffness_fun = spw.create_smooth_piecewise_derivative_function(k, u, us)
        if mode == 0:
            self._energy_function = lambda uu: energy_fun(np.abs(uu))
            self._force_function = lambda uu: np.sign(uu) * force_fun(np.abs(uu))
            self._stiffness_function = lambda uu: stiffness_fun(np.abs(uu))
        elif mode == 1:
            self._energy_function = energy_fun
            self._force_function = force_fun
            self._stiffness_function = stiffness_fun
        elif mode == -1:
            self._energy_function = lambda uu: energy_fun(-uu)
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

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        return self._energy_function(alpha - self._natural_measure)


    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._force_function(alpha - self._natural_measure),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._stiffness_function(alpha - self._natural_measure),

    def update(self, natural_measure=None, /, **parameters):
        super().update(natural_measure, **parameters)
        self._check()
        if parameters:
            self._make()


class ZigzagBehavior(UnivariateBehavior, ControllableByPoints):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], epsilon: float, mode: int = 0):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, mode=mode, epsilon=epsilon)
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
        inv_u = spw.create_inverse_smooth_piecewise_function(slopes_u, transitions_u, delta)
        f = spw.create_smooth_piecewise_function(slopes_f, transitions_f, delta)
        du = spw.create_smooth_piecewise_derivative_function(slopes_u, transitions_u, delta)
        df = spw.create_smooth_piecewise_derivative_function(slopes_f, transitions_f, delta)
        int_fdu = spw.create_antiderivative_of_parametric_piecewise(slopes_u, slopes_f, transitions_u, delta)  # transitions_u == transitions_f

        if mode == 0:
            self._energy = lambda uu: int_fdu(inv_u(np.abs(uu)))
            self._force = lambda uu: np.sign(uu) * f(inv_u(np.abs(uu)))
            self._stiffness = lambda uu: df(inv_u(np.abs(uu))) / du(inv_u(np.abs(uu)))
        elif mode == 1:
            self._energy = lambda uu: int_fdu(inv_u(uu))
            self._force = lambda uu: f(inv_u(uu))
            self._stiffness = lambda uu: df(inv_u(uu)) / du(inv_u(uu))
        elif mode == -1:
            self._energy = lambda uu: int_fdu(inv_u(-uu))
            self._force = lambda uu: -f(inv_u(-uu))
            self._stiffness = lambda uu: df(inv_u(-uu)) / du(inv_u(-uu))
        else:
            raise InvalidBehaviorParameters('This error should never be triggered '
                                            '(error: mode not -1, 0 or 1, while making behavior)')


    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        return self._force(alpha - self._natural_measure),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
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

        # to check in 'update' method before _check()
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')

        # to update in the 'update' method
        self.tmax = self._compute_tmax()
        self._delta = epsilon / (2 * (n - 1))
        self._k_u, self._x_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = spw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = spw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = spw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = spw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_int_bda_fun = spw.create_antiderivative_of_parametric_piecewise(self._k_u, self._k_f, self._x_u, self._delta)

        self._check()
        self._make()

    def _compute_tmax(self):
        return np.sum(np.abs(np.diff(self._parameters['u_i'], prepend=0.0)))

    def _a_fun(self, t):
        return self._raw_a_fun(t/self.tmax)

    def _b_fun(self, t):
        return self._raw_b_fun(t/self.tmax)

    def _da_fun(self, t):
        return self._raw_da_fun(t/self.tmax)/self.tmax

    def _db_fun(self, t):
        return self._raw_db_fun(t/self.tmax)/self.tmax

    def _d2a_fun(self, t):
        return self._raw_d2a_fun(t/self.tmax)/self.tmax**2

    def _d2b_fun(self, t):
        return self._raw_d2b_fun(t/self.tmax)/self.tmax**2

    def _d3a_fun(self, t):
        return np.zeros_like(t)

    def _d3b_fun(self, t):
        return np.zeros_like(t)
    
    def _int_bda_fun(self, t):
        return self._raw_int_bda_fun(t)

    def get_a_extrema(self) -> np.ndarray:
        return spw.get_extrema(self._k_u, self._x_u, self._delta) * self.tmax

    def get_b_extrema(self) -> np.ndarray:
        return spw.get_extrema(self._k_f, self._x_f, self._delta) * self.tmax

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

        # to check in 'update' method before _check()
        if not 0.0 < epsilon < 1.0:
            raise InvalidBehaviorParameters(f'Parameter epsilon must be between 0 and 1 (current value: {epsilon:.3E})')

        n = len(u_i) + 1
        cp_t = np.arange(n) / (n - 1)
        cp_u = np.array([0.0] + u_i)
        cp_f = np.array([0.0] + f_i)
        self._delta = epsilon / (2 * (n - 1))
        self.tmax = self._compute_tmax()
        self._k_u, self._x_u = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = spw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = spw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = spw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = spw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = spw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = spw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_int_bda_fun = spw.create_antiderivative_of_parametric_piecewise(self._k_u, self._k_f, self._x_u, self._delta)

        self._check()
        if parameters:
            self._make()

class SmootherZigzag2Behavior(BivariateBehavior, ControllableByPoints):

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
        self.tmax = self._compute_tmax()
        self._delta = epsilon / (2 * (n - 1))
        self._k_u, self._x_u = sspw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = sspw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = sspw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = sspw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = sspw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = sspw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = sspw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = sspw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d3a_fun = sspw.create_smooth_piecewise_third_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d3b_fun = sspw.create_smooth_piecewise_third_derivative_function(self._k_f, self._x_f, self._delta)

        self._check()
        self._make()
    
    def _compute_tmax(self):
        return np.sum(np.abs(np.diff(self._parameters['u_i'], prepend=0.0)))

    def _a_fun(self, t):
        return self._raw_a_fun(t/self.tmax)

    def _b_fun(self, t):
        return self._raw_b_fun(t/self.tmax)

    def _da_fun(self, t):
        return self._raw_da_fun(t/self.tmax)/self.tmax

    def _db_fun(self, t):
        return self._raw_db_fun(t/self.tmax)/self.tmax

    def _d2a_fun(self, t):
        return self._raw_d2a_fun(t/self.tmax)/self.tmax**2

    def _d2b_fun(self, t):
        return self._raw_d2b_fun(t/self.tmax)/self.tmax**2

    def _d3a_fun(self, t):
        return self._raw_d3b_fun(t/self.tmax)/self.tmax**3

    def _d3b_fun(self, t):
        return self._raw_d3b_fun(t/self.tmax)/self.tmax**3

    def get_a_extrema(self) -> np.ndarray:
        return sspw.get_extrema(self._k_u, self._x_u, self._delta) * self.tmax

    def get_b_extrema(self) -> np.ndarray:
        return sspw.get_extrema(self._k_f, self._x_f, self._delta) * self.tmax

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

        # to update before check() and make()
        self.tmax = self._compute_tmax()
        self._delta = epsilon / (2 * (n - 1))
        self._k_u, self._x_u = sspw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_u)
        self._k_f, self._x_f = sspw.compute_piecewise_slopes_and_transitions_from_control_points(cp_t, cp_f)
        self._raw_a_fun = sspw.create_smooth_piecewise_function(self._k_u, self._x_u, self._delta)
        self._raw_b_fun = sspw.create_smooth_piecewise_function(self._k_f, self._x_f, self._delta)
        self._raw_da_fun = sspw.create_smooth_piecewise_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_db_fun = sspw.create_smooth_piecewise_derivative_function(self._k_f, self._x_f, self._delta)
        self._raw_d2a_fun = sspw.create_smooth_piecewise_second_derivative_function(self._k_u, self._x_u, self._delta)
        self._raw_d2b_fun = sspw.create_smooth_piecewise_second_derivative_function(self._k_f, self._x_f, self._delta)

        self._check()
        if parameters:
            self._make()


class ContactBehavior(UnivariateBehavior):
    def __init__(self, natural_measure, f0, uc, delta):
        super().__init__(natural_measure, f0=f0, uc=uc, delta=delta)
        self._p = 3.0

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        return (alpha < delta) * (f0 * uc / (p + 1) * ((delta - alpha) / uc) ** (p + 1))

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        return (alpha < delta) * (-f0 * ((delta - alpha) / uc) ** p),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        f0 = self._parameters['f0']
        uc = self._parameters['uc']
        delta = self._parameters['delta']
        p = self._p
        return (alpha < delta) * (f0 / uc * p * ((delta - alpha) / uc) ** (p - 1)),


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

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method is abstract")

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        raise NotImplementedError("This method is abstract")

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        raise NotImplementedError("This method is abstract")


class IsothermalGas(IdealGas):

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        v = alpha
        v0 = self._natural_measure
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT * (v / v0 - 1.0 - np.log(v / v0)))

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        v = alpha
        v0 = self._natural_measure
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT * (v - v0) / (v * v0)),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        v = alpha
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT / v ** 2),


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

    def elastic_energy(self, alpha: np.ndarray) -> np.ndarray:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT0 * (v / v0 - 1) + nRT0 / (gamma - 1) * ((v0 / v) ** (gamma - 1) - 1))

    def gradient_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT0 * (1 / v0 - 1 / v * (v0 / v) ** (gamma - 1))),

    def hessian_energy(self, alpha: np.ndarray) -> tuple[np.ndarray,]:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return np.where(v < 0, np.nan, nRT0 * gamma / v ** 2 * (v0 / v) ** (gamma - 1)),

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
