from .math_utils.bezier_curve import *
from .math_utils import smooth_zigzag_curve as szz
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

    def update(self, natural_measure, /, **parameters):
        self._natural_measure = natural_measure
        self._parameters = parameters
        # TO BE EXTENDED IN SUBCLASSES IF NECESSARY


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


class NaturalBehavior(UnivariateBehavior):

    def __init__(self, natural_measure: float, k: float):
        super().__init__(natural_measure, k=k)

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        energy = self._parameters['k'] * self._natural_measure * alpha * (np.log(alpha / self._natural_measure) - 1)
        if not np.isfinite(energy).any():
            raise NonfiniteBehavior
        return energy

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        force = self._parameters['k'] * self._natural_measure * np.log(alpha / self._natural_measure)
        if not np.isfinite(force).any():
            raise NonfiniteBehavior
        return force,

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        stiffness = self._parameters['k'] * self._natural_measure / alpha
        print(stiffness)
        if not np.isfinite(stiffness).any():
            raise NonfiniteBehavior
        return stiffness,


class BezierBehavior(UnivariateBehavior):
    def __init__(self, natural_measure, u_i: list[float], f_i: list[float],
                 sampling: int = 100, check_validity=True):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i)
        self._check_validity = check_validity
        self._sampling = sampling
        self._make()

    def _make(self):
        u_coefs = np.array([0.0] + self._parameters['u_i'])
        f_coefs = np.array([0.0] + self._parameters['f_i'])
        if self._check_validity and not is_monotonic(u_coefs):
            print(u_coefs)
            raise InvalidBehaviorParameters(
                "The Bezier behavior does not describe a function, try Bezier2 instead.")
        t = np.linspace(0, 1, self._sampling)
        u = evaluate_poly(t, u_coefs)

        def fdu(_t, _): return evaluate_poly(_t, f_coefs) * evaluate_derivative_poly(_t, u_coefs)

        energy = solve_ivp(fun=fdu, t_span=[0.0, 1.0], y0=[0.0], t_eval=t).y[0, :]
        self._energy = interp1d(u, energy, kind='linear', fill_value='extrapolate')
        generalized_force = evaluate_poly(t, f_coefs)
        self._first_der_energy = interp1d(u, generalized_force, kind='linear', fill_value='extrapolate')
        generalized_stiffness = evaluate_derivative_poly(t, f_coefs) / evaluate_derivative_poly(t, u_coefs)
        self._second_der_energy = interp1d(u, generalized_stiffness, kind='linear', fill_value='extrapolate')

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        return self._energy(alpha - self._natural_measure)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._first_der_energy(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        return self._second_der_energy(alpha - self._natural_measure),

    def update(self, natural_measure, /, **parameters):
        super().update(natural_measure, **parameters)
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
            behavior = cls(natural_measure, u_i=u_i, f_i=f_i, check_validity=False)
            f_fit = behavior._first_der_energy(u_sampling)
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


class Bezier2Behavior(BivariateBehavior):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float]):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i)
        self._make()

    def _make(self):
        self._a_coefs = np.array([0.0] + self._parameters['u_i'])
        self._b_coefs = np.array([0.0] + self._parameters['f_i'])

        # checking validity of behavior
        da0 = evaluate_derivative_poly(0.0, self._a_coefs)
        db0 = evaluate_derivative_poly(0.0, self._b_coefs)
        if da0 == 0.0 or db0 == 0.0:
            raise InvalidBehaviorParameters('The initial slope of the behavior'
                                            'cannot be perfectly horizontal or vertical')
        a_extrema = get_extrema(self._a_coefs)
        b_extrema = get_extrema(self._b_coefs)
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
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                    current_state = 'bottom-right'
                case 'bottom-right':
                    if transition == 'a_max':
                        current_state = 'bottom-left'
                    elif transition == 'b_min':
                        current_state = 'top-right'
                    else:
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case 'top-left':
                    if transition != 'b_max':
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                    current_state = 'bottom-left'
                case 'bottom-left':
                    if transition == 'a_min':
                        current_state = 'bottom-right'
                    elif transition == 'b_min':
                        current_state = 'top-left'
                    else:
                        raise InvalidBehaviorParameters('The curve cannot have such fold(s)')
                case _:
                    print('error')

        a1 = evaluate_poly(1.0, self._a_coefs)
        da1 = evaluate_derivative_poly(1.0, self._a_coefs)
        b1 = evaluate_poly(1.0, self._b_coefs)
        db1 = evaluate_derivative_poly(1.0, self._b_coefs)
        self._a = lambda t: ((np.abs(t) <= 1) * np.sign(t) * evaluate_poly(np.abs(t), self._a_coefs)
                             + (np.abs(t) > 1) * np.sign(t) * (a1 + da1 * (np.abs(t) - 1)))
        self._b = lambda t: ((np.abs(t) <= 1) * np.sign(t) * evaluate_poly(np.abs(t), self._b_coefs)
                             + (np.abs(t) > 1) * np.sign(t) * (b1 + db1 * (np.abs(t) - 1)))
        self._da = lambda t: ((np.abs(t) <= 1) * evaluate_derivative_poly(np.abs(t), self._a_coefs)
                              + (np.abs(t) > 1) * da1)
        self._db = lambda t: ((np.abs(t) <= 1) * evaluate_derivative_poly(np.abs(t), self._b_coefs)
                              + (np.abs(t) > 1) * db1)
        self._d2a = lambda t: (np.abs(t) <= 1) * np.sign(t) * evaluate_second_derivative_poly(np.abs(t), self._a_coefs)
        self._d2b = lambda t: (np.abs(t) <= 1) * np.sign(t) * evaluate_second_derivative_poly(np.abs(t), self._b_coefs)
        self._d3a = lambda t: (np.abs(t) <= 1) * evaluate_third_derivative_poly(np.abs(t), self._a_coefs)
        self._d3b = lambda t: (np.abs(t) <= 1) * evaluate_third_derivative_poly(np.abs(t), self._b_coefs)

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

    def update(self, natural_measure, /, **parameters):
        super().update(natural_measure, **parameters)
        self._make()

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

    def plot_energy_landscape(self, ax=None):
        n = 10_000
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

        ax.plot_surface(t_grid, y_grid, v_grid - 10 * y_grid, cmap='viridis')
        # ax.plot(t[stable], a[stable], int_bda[stable], 'bo', label='path')
        # ax.plot(t[stabilizable], a[stabilizable], int_bda[stabilizable], 'ko', label='path')
        # ax.plot(t[unstable], a[unstable], int_bda[unstable], 'ro', label='path')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\\y$')
        ax.set_zlabel('$U$')
        if make_new_fig:
            plt.show()

    def calculate_pseudoseries_force_displacement_curve(self):
        pass
        t_vector = np.linspace(0, 1, 100)
        eigvals_vector = np.zeros((t_vector.shape[0], 2))
        for i, t in enumerate(t_vector):
            h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + self._a(t), t)
            hessian_matrix = np.array([[h11, h12], [h12, h22]])
            eigvals, eigvects = np.linalg.eigh(hessian_matrix)
            eigvals_vector[i, :] = eigvals

        f0 = cumulative_trapezoid(eigvals_vector[:, 0], t_vector, initial=0)
        f1 = cumulative_trapezoid(eigvals_vector[:, 1], t_vector, initial=0)
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(t_vector, eigvals_vector[:, 0], '-o')
        ax[0].plot(t_vector, eigvals_vector[:, 1], '-o')
        ax[1].plot(t_vector, f0, '-o')
        ax[1].plot(t_vector, f1, '-o')
        plt.show()
        bb0 = BezierBehavior(1.0, *BezierBehavior.compute_fitting_parameters([(t_vector, f0)], 6, show=True))
        bb1 = BezierBehavior(1.0, *BezierBehavior.compute_fitting_parameters([(t_vector, f1)], 6, show=True))
        # bb0 = None

        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(t_vector, eigvals_vector[:, 0], '-o')
        # ax[0].plot(t_vector, eigvals_vector[:, 1], '-o')
        # ax[1].plot(t_vector, f0, '-o')
        # ax[1].plot(t_vector, f1, '-o')
        # plt.show()
        return bb0, bb1

    def compute_diagonal_mapping(self):
        n = 150

        tt = np.linspace(0, 1, n)
        y, t = np.meshgrid(np.linspace(np.min(self._a(tt)), np.max(self._a(tt)), n), tt)
        phi1 = np.zeros_like(t)  # y, t --> u
        phi2 = np.zeros_like(t)  # y, t --> v

        uu = np.linspace(-1.0, 1.0, n)
        vv = np.linspace(0.0, 1.75, n)
        u, v = np.meshgrid(uu, vv)
        ksi1 = np.zeros_like(t)  # u, v --> y
        ksi2 = np.zeros_like(t)  # u, v --> t

        for i in range(n):
            for j in range(n):
                h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y[i, j], t[i, j])
                rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
                lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
                lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
                v1 = np.array((-h12 / (h11 - lambda1), 1))
                v2 = np.array((-h12 / (h11 - lambda2), 1))
                nu1 = v1 / np.linalg.norm(v1)
                nu2 = v2 / np.linalg.norm(v2)
                p = np.array((nu1, nu2)).T
                phi1[i, j], phi2[i, j] = p @ np.array((y[i, j], t[i, j]))

        phi1_t = np.zeros_like(tt)  # y, t --> u
        phi2_t = np.zeros_like(tt)
        for i, ttt in enumerate(tt):
            h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + self._a(ttt), ttt)
            rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
            lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
            lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
            v1 = np.array((-h12 / (h11 - lambda1), 1))
            v2 = np.array((-h12 / (h11 - lambda2), 1))
            nu1 = v1 / np.linalg.norm(v1)
            nu2 = v2 / np.linalg.norm(v2)
            p = np.array((nu1, nu2)).T
            phi1_t[i], phi2_t[i] = p @ np.array((self._a(ttt), ttt))

        # points = np.array([phi1.ravel(), phi2.ravel()]).T  # Combine U, V into pairs

        # Interpolating to get the X and Y values on the (U, V) grid
        # ksi1 = griddata(points, y.ravel(), (u, v), method='cubic')
        # ksi2 = griddata(points, t.ravel(), (u, v), method='cubic')

        # for i in range(n):
        #     for j in range(n):
        #         mismatch = (phi1 - u[i,j])**2 + (phi2 - v[i,j])**2
        #         ii, jj = np.unravel_index(np.argmin(mismatch), mismatch.shape)
        #         h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y[ii, jj], t[ii, jj])
        #         rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
        #         lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
        #         lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
        #         v1 = np.array((-h12 / (h11 - lambda1), 1))
        #         v2 = np.array((-h12 / (h11 - lambda2), 1))
        #         nu1 = v1 / np.linalg.norm(v1)
        #         nu2 = v2 / np.linalg.norm(v2)
        #         p = np.array((nu1, nu2)).T
        #         ksi1[i, j], ksi2[i, j] = p.T @ np.array((u[i, j], v[i, j]))

        fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)
        # ax1.contourf(t, y, phi1)
        # ax2.contourf(t, y, phi2)
        # ax3.contourf(u, v, ksi1)
        # ax4.contourf(u, v, ksi2)
        # ax1.plot(tt, self._a(tt), 'r-', lw=3)
        # ax2.plot(tt, self._a(tt), 'r-', lw=3)

        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.plot_surface(t, y, phi1, cmap='viridis')
        ax2.plot_surface(t, y, phi2, cmap='viridis')
        # ax3.plot_surface(u, v, ksi1)
        # ax4.plot_surface(u, v, ksi2)
        ax3.plot_surface(phi1, phi2, y, alpha=0.3, cmap='viridis')
        ax4.plot_surface(phi1, phi2, t, alpha=0.3, cmap='viridis')
        ax3.scatter(phi1_t, phi2_t, self._a(tt), color='r')
        ax4.scatter(phi1_t, phi2_t, tt, color='r')
        # ax1.plot(tt, self._a(tt), 'r-', lw=3)
        # ax2.plot(tt, self._a(tt), 'r-', lw=3)
        plt.show()

    def compute_diagonal_mapping2(self):
        n = 25

        tt = np.linspace(0, 1.0, n)
        fig, ax = plt.subplots()
        ax.plot(tt, self._a(tt))
        ax.plot(tt, self._b(tt))
        plt.show()
        y, t = np.meshgrid(np.linspace(np.min(self._a(tt)), np.max(self._a(tt)), n), tt)

        nu1x = np.zeros_like(t)
        nu1y = np.zeros_like(t)
        nu2x = np.zeros_like(t)
        nu2y = np.zeros_like(t)

        for i in range(n):
            for j in range(n):
                h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y[i, j], t[i, j])
                rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
                lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
                lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
                v1 = np.array((-h12 / (h11 - lambda1), 1))
                v2 = np.array((-h12 / (h11 - lambda2), 1))
                nu1 = v1 / np.linalg.norm(v1)
                nu2 = v2 / np.linalg.norm(v2)
                nu1x[i, j] = nu1[0]
                nu1y[i, j] = nu1[1]
                nu2x[i, j] = nu2[0]
                nu2y[i, j] = nu2[1]

        fig, (ax0) = plt.subplots()
        ax0.quiver(y, t, nu1x, nu1y, color='tab:blue')
        ax0.quiver(y, t, nu2x, nu2y, color='tab:orange')
        ax0.plot(self._a(tt), tt, 'r-', lw=2)
        plt.show()

        uu = np.linspace(0.0, 1.0, n)
        vv = np.linspace(0.0, 1.5, n)
        u, v = np.meshgrid(uu, vv)

        def ode_system_uv(uv, xy, direction):
            x, y = xy
            h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y, x)
            rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
            lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
            lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
            v1 = np.array((-h12 / (h11 - lambda1), 1))
            v2 = np.array((-h12 / (h11 - lambda2), 1))
            nu1 = v1 / np.linalg.norm(v1)
            nu2 = v2 / np.linalg.norm(v2)
            if direction == 'u':
                return nu1  # Derivative in the direction of nu1
            elif direction == 'v':
                return nu2  # Derivative in the direction of nu2

        # Grid for u and v
        u_vals = np.linspace(0, 2.0, 30)  # Adjust grid resolution as needed
        v_vals = np.linspace(0, 2.0, 30)
        U, V = np.meshgrid(u_vals, v_vals)

        # Initialize arrays for X and Y over the (u, v) grid
        X = np.zeros(U.shape)
        Y = np.zeros(V.shape)

        # Starting point at (u, v) = (0, 0)
        x0, y0 = 0, 0

        # Populate X and Y on the grid
        for i in range(len(u_vals)):
            print(i)
            # Solve along u from (0, 0) to the current u, holding v = 0
            sol_u = solve_ivp(ode_system_uv, (0, u_vals[i]), [x0, y0], args=('u',))
            x_u, y_u = sol_u.y[:, -1]
            for j in range(len(v_vals)):
                # Solve along v from (0, 0) to the current v, holding u constant
                sol_v = solve_ivp(ode_system_uv, (0, v_vals[j]), [x_u, y_u], args=('v',))

                # Store the final (x, y) at (u, v) = (u_vals[i], v_vals[j])
                X[i, j], Y[i, j] = sol_v.y[:, -1]

        # Plot the surfaces
        fig = plt.figure(figsize=(12, 6))

        # X surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(U, V, X, cmap='viridis')
        ax1.set_title("X(u, v) Surface")
        ax1.set_xlabel("u")
        ax1.set_ylabel("v")
        ax1.set_zlabel("x")

        # Y surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(U, V, Y, cmap='plasma')
        ax2.set_title("Y(u, v) Surface")
        ax2.set_xlabel("u")
        ax2.set_ylabel("v")
        ax2.set_zlabel("y")

        plt.show()


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
        v0 = self._natural_measure
        v = alpha
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT * (v / v0 - 1.0 - np.log(v / v0))

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v0 = self._natural_measure
        v = alpha
        nRT = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT * (v - v0) / (v * v0),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v = alpha
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
        if gamma < 1.0:
            raise InvalidBehaviorParameters(f"The ratio of heat capacities 'gamma' must be strictly greater than 1."
                                            f"Current value = {gamma}")
        super().__init__(v0, n=n, R=R, T0=T0)
        self._parameters['gamma'] = gamma

    def elastic_energy(self, alpha: float | np.ndarray) -> float:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * (v / v0 - 1) + nRT0 / (gamma - 1) * ((v0 / v) ** (gamma - 1) - 1)

    def gradient_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * (1 / v0 - 1 / v * (v0 / v) ** (gamma - 1)),

    def hessian_energy(self, alpha: float | np.ndarray) -> tuple[float]:
        v0 = self._natural_measure
        gamma = self._parameters['gamma']
        v = alpha
        nRT0 = self._parameters['n'] * self._parameters['R'] * self._parameters['T0']
        return nRT0 * gamma / v ** 2 * (v0 / v) ** (gamma - 1),


class ZigZagBehavior(UnivariateBehavior):
    def __init__(self, natural_measure, a, x, delta):
        super().__init__(natural_measure, a=a, x=x, delta=delta)
        self._make()

    def _make(self):
        a, x, delta = self._parameters['a'], self._parameters['x'], self._parameters['delta']
        if x[0] - 0.0 < delta or (len(x) > 1 and np.min(np.diff(x)) < 2 * delta):
            raise InvalidBehaviorParameters(f'Smoothing factor {delta} is too large for the zigzag intervals provided.')
        if len(a) != len(x) + 1:
            raise InvalidBehaviorParameters(f'Expected {len(a) - 1} transitions, but only {len(x)} were provided.')
        self._u_i, self._f_i = szz.compute_zigzag_control_points(a, x)
        self._generalized_force_function = szz.create_smooth_zigzag_function(a, x, delta)
        self._generalized_stiffness_function = szz.create_smooth_zigzag_derivative_function(a, x, delta)

    def elastic_energy(self, alpha: float) -> float:
        return quad(self._generalized_force_function, 0.0, alpha - self._natural_measure)[0]

    def gradient_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_force_function(alpha - self._natural_measure),

    def hessian_energy(self, alpha: float) -> tuple[float]:
        return self._generalized_stiffness_function(alpha - self._natural_measure),

    def update(self, natural_measure, /, **parameters):
        super().update(natural_measure, **parameters)
        self._make()


class ZigZag2Behavior(BivariateBehavior):

    def __init__(self, natural_measure, u_i: list[float], f_i: list[float], delta):
        super().__init__(natural_measure, u_i=u_i, f_i=f_i, delta=delta)
        self._make()

    def _make(self):
        u_i, f_i, delta = self._parameters['u_i'], self._parameters['f_i'], self._parameters['delta']
        # checking validity of behavior
        if not 0.0 < delta < 1.0:
            raise InvalidBehaviorParameters(f'Parameter delta must be between 0 and 1 (current value: {delta}')
        if len(u_i) != len(f_i):
            raise InvalidBehaviorParameters(f'u_i and f_i must contain the same number of elements')
        self._cp_u = np.array([0.0] + u_i)
        self._cp_f = np.array([0.0] + f_i)

        n = len(u_i) + 1
        self._cp_t = np.arange(n) / (n - 1)
        self._delta = delta / (n - 1)
        slopes_u, transitions_u = szz.compute_zizag_slopes_and_transitions_from_control_points(self._cp_t, self._cp_u)
        slopes_f, transitions_f = szz.compute_zizag_slopes_and_transitions_from_control_points(self._cp_t, self._cp_f)

        self._a = szz.create_smooth_zigzag_function(slopes_u, transitions_u, self._delta)
        self._b = szz.create_smooth_zigzag_function(slopes_f, transitions_f, self._delta)
        self._da = szz.create_smooth_zigzag_derivative_function(slopes_u, transitions_u, self._delta)
        self._db = szz.create_smooth_zigzag_derivative_function(slopes_f, transitions_f, self._delta)
        self._d2a = szz.create_smooth_zigzag_second_derivative_function(slopes_u, transitions_u, self._delta)
        self._d2b = szz.create_smooth_zigzag_second_derivative_function(slopes_f, transitions_f, self._delta)
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

    def update(self, natural_measure, /, **parameters):
        super().update(natural_measure, **parameters)
        self._make()

    def get_hysteron_info(self) -> dict[str, float]:
        if self._hysteron_info is None:
            self._compute_hysteron_info()
        return self._hysteron_info

    def _compute_hysteron_info(self):
        extrema = np.sort(get_extrema(self._cp_t, self._cp_u, self._delta))
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

    def calculate_pseudoseries_force_displacement_curve(self):
        pass
        t_vector = np.linspace(0, 1, 100)
        eigvals_vector = np.zeros((t_vector.shape[0], 2))
        for i, t in enumerate(t_vector):
            h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + self._a(t), t)
            hessian_matrix = np.array([[h11, h12], [h12, h22]])
            eigvals, eigvects = np.linalg.eigh(hessian_matrix)
            eigvals_vector[i, :] = eigvals

        f0 = cumulative_trapezoid(eigvals_vector[:, 0], t_vector, initial=0)
        f1 = cumulative_trapezoid(eigvals_vector[:, 1], t_vector, initial=0)
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(t_vector, eigvals_vector[:, 0], '-o')
        ax[0].plot(t_vector, eigvals_vector[:, 1], '-o')
        ax[1].plot(t_vector, f0, '-o')
        ax[1].plot(t_vector, f1, '-o')
        plt.show()
        bb0 = BezierBehavior(1.0, *BezierBehavior.compute_fitting_parameters([(t_vector, f0)], 6, show=True))
        bb1 = BezierBehavior(1.0, *BezierBehavior.compute_fitting_parameters([(t_vector, f1)], 6, show=True))
        # bb0 = None

        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(t_vector, eigvals_vector[:, 0], '-o')
        # ax[0].plot(t_vector, eigvals_vector[:, 1], '-o')
        # ax[1].plot(t_vector, f0, '-o')
        # ax[1].plot(t_vector, f1, '-o')
        # plt.show()
        return bb0, bb1

    def compute_diagonal_mapping(self):
        n = 150

        tt = np.linspace(0, 1, n)
        y, t = np.meshgrid(np.linspace(np.min(self._a(tt)), np.max(self._a(tt)), n), tt)
        phi1 = np.zeros_like(t)  # y, t --> u
        phi2 = np.zeros_like(t)  # y, t --> v

        uu = np.linspace(-1.0, 1.0, n)
        vv = np.linspace(0.0, 1.75, n)
        u, v = np.meshgrid(uu, vv)
        ksi1 = np.zeros_like(t)  # u, v --> y
        ksi2 = np.zeros_like(t)  # u, v --> t

        for i in range(n):
            for j in range(n):
                h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y[i, j], t[i, j])
                rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
                lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
                lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
                v1 = np.array((-h12 / (h11 - lambda1), 1))
                v2 = np.array((-h12 / (h11 - lambda2), 1))
                nu1 = v1 / np.linalg.norm(v1)
                nu2 = v2 / np.linalg.norm(v2)
                p = np.array((nu1, nu2)).T
                phi1[i, j], phi2[i, j] = p @ np.array((y[i, j], t[i, j]))

        phi1_t = np.zeros_like(tt)  # y, t --> u
        phi2_t = np.zeros_like(tt)
        for i, ttt in enumerate(tt):
            h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + self._a(ttt), ttt)
            rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
            lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
            lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
            v1 = np.array((-h12 / (h11 - lambda1), 1))
            v2 = np.array((-h12 / (h11 - lambda2), 1))
            nu1 = v1 / np.linalg.norm(v1)
            nu2 = v2 / np.linalg.norm(v2)
            p = np.array((nu1, nu2)).T
            phi1_t[i], phi2_t[i] = p @ np.array((self._a(ttt), ttt))

        # points = np.array([phi1.ravel(), phi2.ravel()]).T  # Combine U, V into pairs

        # Interpolating to get the X and Y values on the (U, V) grid
        # ksi1 = griddata(points, y.ravel(), (u, v), method='cubic')
        # ksi2 = griddata(points, t.ravel(), (u, v), method='cubic')

        # for i in range(n):
        #     for j in range(n):
        #         mismatch = (phi1 - u[i,j])**2 + (phi2 - v[i,j])**2
        #         ii, jj = np.unravel_index(np.argmin(mismatch), mismatch.shape)
        #         h11, h12, h22 = self.hessian_energy(self.get_natural_measure() + y[ii, jj], t[ii, jj])
        #         rho = (h11 + h22) ** 2 - 4 * (h11 * h22 - h12 ** 2)
        #         lambda1 = (h11 + h22 - np.sqrt(rho)) / 2
        #         lambda2 = (h11 + h22 + np.sqrt(rho)) / 2
        #         v1 = np.array((-h12 / (h11 - lambda1), 1))
        #         v2 = np.array((-h12 / (h11 - lambda2), 1))
        #         nu1 = v1 / np.linalg.norm(v1)
        #         nu2 = v2 / np.linalg.norm(v2)
        #         p = np.array((nu1, nu2)).T
        #         ksi1[i, j], ksi2[i, j] = p.T @ np.array((u[i, j], v[i, j]))

        fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)
        # ax1.contourf(t, y, phi1)
        # ax2.contourf(t, y, phi2)
        # ax3.contourf(u, v, ksi1)
        # ax4.contourf(u, v, ksi2)
        # ax1.plot(tt, self._a(tt), 'r-', lw=3)
        # ax2.plot(tt, self._a(tt), 'r-', lw=3)

        # ax1 = fig.add_subplot(121, projection='3d')
        # ax2 = fig.add_subplot(122, projection='3d')
        ax3 = fig.add_subplot(121, projection='3d')
        ax4 = fig.add_subplot(122, projection='3d')
        # ax1.plot_surface(t, y, phi1, cmap='viridis')
        # ax2.plot_surface(t, y, phi2, cmap='viridis')
        # ax3.plot_surface(u, v, ksi1)
        # ax4.plot_surface(u, v, ksi2)
        ax3.plot_surface(phi1, phi2, y, alpha=0.3, cmap='viridis')
        ax4.plot_surface(phi1, phi2, t, alpha=0.3, cmap='viridis')
        ax3.scatter(phi1_t, phi2_t, self._a(tt), color='r')
        ax4.scatter(phi1_t, phi2_t, tt, color='r')
        # ax1.plot(tt, self._a(tt), 'r-', lw=3)
        # ax2.plot(tt, self._a(tt), 'r-', lw=3)
        plt.show()


class ContactBehavior(UnivariateBehavior):
    def __init__(self, k, d0):
        super().__init__(None, d0=d0, k=k)

    def elastic_energy(self, alpha: float) -> float:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0
        elif 0.0 < alpha < d0:
            return 0.5 * k * (d0 ** 2 - alpha ** 2) + 2 * k * d0 * (alpha - d0) - k * d0 ** 2 * np.log(alpha / d0)
        else:
            raise NonfiniteBehavior('Negative or zero value entered for a contact behavior')

    def gradient_energy(self, alpha: float) -> tuple[float]:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0,
        elif 0.0 < alpha < d0:
            return -k * (alpha - d0) ** 2 / alpha,
        else:
            raise NonfiniteBehavior('Negative or zero value entered for a contact behavior')

    def hessian_energy(self, alpha: float) -> tuple[float]:
        k = self._parameters['k']
        d0 = self._parameters['d0']
        if alpha >= d0:
            return 0.0,
        elif 0.0 < alpha < d0:
            return k * ((d0 / alpha) ** 2 - 1),
        else:
            raise NonfiniteBehavior('Negative or zero value entered for a contact behavior')


class InvalidBehaviorParameters(Exception):
    """ raise this when one attempts to create a mechanical behavior with invalid parameters"""


class NonfiniteBehavior(Exception):
    """ raise this when the behavior leads to a nonfinite value of energy, force or stiffness"""
