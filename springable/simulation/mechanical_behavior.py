from .math_utils.bezier_curve import *
from .math_utils.smooth_zigzag_curve import *
from scipy.interpolate import interp1d
from scipy.integrate import odeint, quad
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt


class MechanicalBehavior:
    _nb_dofs: int = None

    def __init__(self, **parameters):
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


class LinearBehavior(UnivariateBehavior):

    def __init__(self, spring_constant: float):
        super().__init__(k=spring_constant)

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return 0.5 * self._parameters['k'] * alpha ** 2

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._parameters['k'] * alpha,

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._parameters['k'],

    def get_spring_constant(self) -> float:
        return self._parameters['k']


class BezierBehavior(UnivariateBehavior):

    def __init__(self, u_i: list[float], f_i: list[float], sampling=100):
        super().__init__(u_i=u_i, f_i=f_i)
        u_coefs = np.array([0.0] + u_i)
        f_coefs = np.array([0.0] + f_i)
        t = np.linspace(0, 1, sampling)
        alpha = evaluate_poly(t, u_coefs)
        energy = odeint(lambda _, t_: evaluate_poly(t_, f_coefs) * evaluate_derivative_poly(t_, u_coefs), 0.0, t)[:, 0]
        generalized_force = evaluate_poly(t, f_coefs)
        generalized_stiffness = evaluate_derivative_poly(t, f_coefs) / evaluate_derivative_poly(t, u_coefs)
        self._energy = interp1d(alpha, energy, fill_value='extrapolate')
        self._first_der_energy = interp1d(alpha, generalized_force, fill_value='extrapolate')
        self._second_der_energy = interp1d(alpha, generalized_stiffness, fill_value='extrapolate')

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return self._energy(alpha)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._first_der_energy(alpha),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._second_der_energy(alpha),

    @classmethod
    def compute_fitting_parameters(cls, force_displacement_curves: list[tuple], degree: int):
        umax = np.min([np.max(fd_curve[0]) for fd_curve in force_displacement_curves])
        fmax = np.min([np.max(fd_curve[1]) for fd_curve in force_displacement_curves])
        nb_samples = 500
        u_sampling = np.linspace(0.0, umax, nb_samples)

        def compute_mismatch(x: np.ndarray) -> float:
            u_i = x[::2].tolist()
            f_i = x[1::2].tolist()
            behavior = cls(u_i=u_i, f_i=f_i)
            f_fit = behavior.gradient_energy(u_sampling)[0]
            mismatch = 0.0
            for fd_curve in force_displacement_curves:
                u_data = fd_curve[0]
                f_data = fd_curve[1]
                f_exp = interp1d(u_data, f_data, fill_value='extrapolate')(u_sampling)
                mismatch += np.sum((f_exp - f_fit) ** 2)
            return mismatch

        u_guess = np.linspace(1/degree, 1.0, degree) * umax
        f_guess = np.linspace(1/degree, 1.0, degree) * fmax

        initial_values = np.array([u_guess[i // 2] if i % 2 == 0 else f_guess[i // 2] for i in range(2 * degree)])
        bounds = [(0.0, None) if i % 2 == 0 else (None, None) for i in range(2 * degree)]
        bounds[1] = (0.0, None)  # first control point should be above f=0 axis

        # control point abscissas should be monotonically increasing
        constraint_matrix = np.zeros((degree - 1, 2 * degree))
        for i in range(degree - 1):
            constraint_matrix[i, 2*i:2*i+3] = [-1.0, 0.0, 1.0]
        lb = np.zeros(constraint_matrix.shape[0])
        ub = np.ones(constraint_matrix.shape[0]) * np.inf
        constraints = LinearConstraint(constraint_matrix, lb, ub)

        result = minimize(compute_mismatch,
                          x0=initial_values,
                          method='trust-constr',
                          bounds=bounds,
                          constraints=constraints,
                          options={'verbose': 1}
                          )
        optimal_parameters = result.x
        u_i = optimal_parameters[::2]
        f_i = optimal_parameters[1::2]
        return u_i.tolist(), f_i.tolist()


class Bezier2Behavior(BivariateBehavior):

    def __init__(self, u_i: list[float], f_i: list[float], n=100):
        super().__init__(u_i=u_i, f_i=f_i)
        self._a_coefs = np.array([0.0] + u_i)
        self._b_coefs = np.array([0.0] + f_i)
        self._a = lambda t: np.sign(t) * evaluate_poly(np.abs(t), self._a_coefs)
        self._b = lambda t: np.sign(t) * evaluate_poly(np.abs(t), self._b_coefs)
        self._da = lambda t: evaluate_derivative_poly(np.abs(t), self._a_coefs)
        self._db = lambda t: evaluate_derivative_poly(np.abs(t), self._b_coefs)
        self._d2a = lambda t: np.sign(t) * evaluate_second_derivative_poly(np.abs(t), self._a_coefs)
        self._d2b = lambda t: np.sign(t) * evaluate_second_derivative_poly(np.abs(t), self._b_coefs)
        self._dbda = lambda t: self._db(t) / self._da(t)
        adb = lambda t: self._db(t) * self._a(t)
        self._int_adb = lambda t: quad(adb, 0, t)[0]
        self._hysteron_info = None

        all_t = np.linspace(0, 1, 50)
        k_low = np.max(self._dbda(all_t)[self._da(all_t) > 0.0])
        k_high = np.min(self._dbda(all_t)[self._da(all_t) < 0.0]) if np.any(self._da(all_t) < 0.0) else + np.inf
        self._k = k_low * (1 + 0.1 * (1.0 - np.exp(-(k_high - k_low) / k_low)))
        # print(f'k low = {k_low}')
        # print(f'k high = {k_high}')
        # print(f'k = {self._k}')

    def elastic_energy(self, alpha: float, t: float) -> np.ndarray:
        return 0.5 * self._k * (alpha - self._a(t)) ** 2 + alpha * self._b(t) - self._int_adb(t)

    def gradient_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        dvdalpha = self._k * (alpha - self._a(t)) + self._b(t)
        dvdt = (alpha - self._a(t)) * (self._db(t) - self._k * self._da(t))
        return dvdalpha, dvdt

    def hessian_energy(self, alpha: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d2vdalpha2 = self._k
        d2vdalphadt = self._k * (alpha - self._a(t)) + self._db(t) - self._k * self._da(t)
        d2vdt2 = (alpha - self._a(t)) * (self._d2b(t) - self._k * self._d2a(t)) + self._da(t) * (
                self._k * self._da(t) - self._db(t))
        return d2vdalpha2, d2vdalphadt, d2vdt2

    def get_hysteron_info(self) -> dict[str, float]:
        if self._hysteron_info is None:
            self._compute_hysteron_info()
        return self._hysteron_info

    def _compute_hysteron_info(self):
        extrema = np.sort(get_extrema(self._a_coefs))
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


class ZigZagBehavior(UnivariateBehavior):
    def __init__(self, a, x, delta):
        if x[0] - 0.0 < delta or (len(x) > 1 and np.min(np.diff(x)) < 2 * delta):
            raise ValueError(f'Smoothing factor {delta} is too large for the zigzag intervals provided.')
        if len(a) != len(x) + 1:
            raise ValueError(f'Expected {len(a) - 1} transitions, but only {len(x)} were provided.')
        super().__init__(a=a, x=x, delta=delta)
        self._u_i, self._f_i = compute_zigzag_control_points(a, x)
        self._generalized_force_function = create_smooth_zigzag_function(a, x, delta)
        self._generalized_stiffness_function = create_smooth_zigzag_derivative_function(a, x, delta)

        # fig, axs = plt.subplots(3, 1)
        # t = np.linspace(-2*x[-1], 2*x[-1], 500)
        # axs[0].plot(t, self.elastic_energy(t))
        # axs[1].plot(t, self.gradient_energy(t)[0])
        # axs[2].plot(t, self.hessian_energy(t)[0])
        # plt.show()

    def elastic_energy(self, alpha: float) -> float:
        # if isinstance(alpha, np.ndarray):
        #     e = np.zeros_like(alpha)
        #     for i, alpha_i in enumerate(alpha):
        #         e[i] = self.elastic_energy(alpha_i)
        #     return e
        # else:
        #     return quad(self._generalized_force_function, 0.0, alpha)[0]
        return quad(self._generalized_force_function, 0.0, alpha)[0]

    def gradient_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_force_function(alpha),

    def hessian_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_stiffness_function(alpha),


class ContactBehavior(UnivariateBehavior):
    def __init__(self, k, d0):
        super().__init__(d0=d0, k=k)

    def elastic_energy(self, alpha: float) -> float:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0
        elif 0.0 < alpha < d0:
            return 0.5 * k * (d0 ** 2 - alpha ** 2) + 2 * k * d0 * (alpha - d0) - k * d0 ** 2 * np.log(alpha / d0)
        else:
            raise ValueError('Negative or zero value entered for a contact behavior')

    def gradient_energy(self, alpha: float) -> tuple[float]:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0,
        elif 0.0 < alpha < d0:
            return -k * (alpha - d0) ** 2 / alpha,
        else:
            raise ValueError('Negative or zero value entered for a contact behavior')

    def hessian_energy(self, alpha: float) -> tuple[float]:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0,
        elif 0.0 < alpha < d0:
            return k * ((d0 / alpha) ** 2 - 1),
        else:
            raise ValueError('Negative or zero value entered for a contact behavior')
