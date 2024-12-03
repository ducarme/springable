"""
Author: Paul Ducarme
Static Assembly Problem Solver class
"""
import warnings
import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve, lstsq
from scipy.optimize import minimize
from scipy.linalg import LinAlgWarning
import sys
import time
from .model import Model
from .node import Node
from .shape import IllDefinedShape
from .mechanical_behavior import NonfiniteBehavior


# warnings.filterwarnings("ignore")


class Result:
    def __init__(self, model: Model,
                 equilibrium_displacements, equilibrium_forces, equilibrium_stability, equilibrium_eigval_stats,
                 step_indices):
        self._model = model
        self._u = equilibrium_displacements
        self._f = equilibrium_forces
        self._stability = equilibrium_stability
        self._eigval_stats = equilibrium_eigval_stats
        self._step_indices = step_indices if step_indices is not None else np.zeros(equilibrium_displacements.shape[0],
                                                                                    dtype=int)

        nb_steps = len(self._model.get_loading())
        self._starting_index = None
        if self._u.ndim < 2 or self._u.shape[0] in (0, 1, 2):
            self._is_solution_unusable = True
            self._is_loading_solution_unusable = True
        else:
            self._is_solution_unusable = False
            if nb_steps == 1:
                self._is_loading_solution_unusable = False
                self._starting_index = 0
            else:
                index = np.argmax(step_indices == nb_steps - 1)
                if index == 0:
                    self._is_loading_solution_unusable = True
                elif self._u[index:, :].shape[0] in (0, 1, 2):
                    self._is_loading_solution_unusable = True
                else:
                    self._starting_index = index
                    self._is_loading_solution_unusable = False

    def get_model(self):
        return self._model

    def get_forces(self, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._f
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._f[self._starting_index:]

    def get_node_forces(self, _node: Node, direction: str, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._f[:, self._model.get_assembly().get_dof_index(_node, direction)]
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._f[self._starting_index:, self._model.get_assembly().get_dof_index(_node, direction)]

    def get_displacements(self, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._u
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._u[self._starting_index:]

    def get_node_displacements(self, _node: Node, direction: str, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._u[:, self._model.get_assembly().get_dof_index(_node, direction)]
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._u[self._starting_index:, self._model.get_assembly().get_dof_index(_node, direction)]

    def get_stability(self, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._stability
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._stability[self._starting_index:]

    def _get_eigval_stat(self, stat_column_index: int, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._eigval_stats[:, stat_column_index]
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._eigval_stats[self._starting_index:, stat_column_index]

    def get_eigval_stats(self, include_preloading=False, check_usability=True):
        if include_preloading:
            if self._is_solution_unusable and check_usability:
                raise UnusableSolution
            return self._eigval_stats
        else:
            if self._is_loading_solution_unusable and check_usability:
                raise UnusableSolution
            return self._eigval_stats[self._starting_index:, :]

    def get_lowest_eigval_in_force_control(self, include_preloading=False, check_usability=True):
        return self._get_eigval_stat(0, include_preloading, check_usability)

    def get_lowest_eigval_in_displacement_control(self, include_preloading=False, check_usability=True):
        return self._get_eigval_stat(1, include_preloading, check_usability)

    def get_nb_of_negative_eigval_in_force_control(self, include_preloading=False, check_usability=True):
        return np.round(self._get_eigval_stat(2, include_preloading, check_usability)).astype(int)

    def get_nb_of_negative_eigval_in_displacement_control(self, include_preloading=False, check_usability=True):
        return np.round(self._get_eigval_stat(3, include_preloading, check_usability)).astype(int)

    def get_step_indices(self):
        return self._step_indices

    def get_equilibrium_path(self):
        u = self.get_displacements()
        f = self.get_forces()
        m = self.get_model()
        n = m.get_force_vector() / np.linalg.norm(m.get_force_vector())  # force direction
        loaded_dof_indices = m.get_loaded_dof_indices()

        # projection of the displacement vector (relative to preload)
        # on the force direction final loading step
        displacement = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * n[loaded_dof_indices], axis=1)

        # projection of the applied vector force (relative to preload)
        # on the force direction prescribed in final loading step
        force = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * n[loaded_dof_indices], axis=1)
        return displacement, force

    def get_min_and_max_loading_displacement_and_force(self):
        u_load, f_load = self.get_equilibrium_path()
        return np.min(u_load), np.max(u_load), np.min(f_load), np.max(f_load)

    def get_starting_index(self):
        return self._starting_index


class UnusableSolution(Exception):
    """ raise this when one attempts to get a solution from a Result instance, but it is not usable """


def ignore_warnings(warning_class):
    """
    Decorator to ignore specific warnings in a function.

    Parameters:
    warning_class (Warning): The warning class to ignore.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warning_class)
                return func(*args, **kwargs)

        return wrapper

    return decorator


class StaticSolver:
    """ Class representing a static solver """
    STABLE = 'stable'  # stable under force and displacement control
    STABILIZABLE = 'stabilizable'  # stable under displacement-control only
    UNSTABLE = 'unstable'  # unstable under both force control and displacement control
    _DEFAULT_SOLVER_SETTINGS = {'reference_load_parameter': 0.05,
                                'radius': 0.05,
                                'detect_critical_points': True,
                                'bifurcate_at_simple_bifurcations': False,
                                'show_warnings': False,
                                'verbose': True,
                                'detail_verbose': False,
                                'i_max': 10e3,
                                'j_max': 20,
                                'convergence_value': 1e-6,
                                'alpha': 0.0,  # never larger than 0.5
                                'psi_p': 0.0,
                                'psi_c': 0.0,
                                'detect_mechanism': True,
                                }

    def __init__(self, model: Model, **solver_settings):
        self._model = model
        self._assembly = model.get_assembly()
        self._nb_dofs = self._assembly.get_nb_dofs()
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._nb_free_dofs = len(self._free_dof_indices)
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()
        self._loaded_dof_indices_step_list = (self._model.get_loaded_dof_indices_preloading_step_list()
                                              + [self._model.get_loaded_dof_indices()])
        self._solver_settings = StaticSolver._DEFAULT_SOLVER_SETTINGS.copy()
        self._solver_settings.update(solver_settings)

    def solve(self) -> Result:
        initial_coordinates = self._assembly.get_coordinates()
        step_force_vectors = self._model.get_force_vectors_preloading_step_list() + [self._model.get_force_vector()]
        max_displacement_map_step_list = self._model.get_max_displacement_map_preloading_step_list() + [
            self._model.get_max_displacement_map()]
        blocked_nodes_directions_step_list = self._model.get_blocked_nodes_directions_step_list()
        u, f, stability, eigval_stats, step_indices = self._solve_with_arclength(step_force_vectors,
                                                                                 max_displacement_map_step_list,
                                                                                 blocked_nodes_directions_step_list,
                                                                                 **self._solver_settings)
        if u.ndim == 2:
            self._assembly.set_coordinates(initial_coordinates)
        for blocked_nodes, directions in blocked_nodes_directions_step_list:
            self._assembly.release_nodes_along_directions(blocked_nodes, directions)
        return Result(self._model, u, f, stability, eigval_stats, step_indices)

    def guide_spring_assembly_to_natural_configuration(self):
        if not self._free_dof_indices:
            return
        initial_coordinates = self._assembly.get_coordinates()

        def elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_coordinates(coordinates)
            return self._assembly.compute_elastic_energy()

        def gradient_elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_coordinates(coordinates)
            return self._assembly.compute_elastic_force_vector()[self._free_dof_indices]

        def hessian_elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_coordinates(coordinates)
            return self._assembly.compute_structural_stiffness_matrix()[
                np.ix_(self._free_dof_indices, self._free_dof_indices)]

        try:
            result = minimize(elastic_energy, initial_coordinates[self._free_dof_indices],
                              jac=gradient_elastic_energy,
                              method='BFGS', tol=1e-6, options={'disp': False})
            natural_coordinates = np.empty(self._nb_dofs)
            natural_coordinates[self._free_dof_indices] = result.x
            natural_coordinates[self._fixed_dof_indices] = initial_coordinates[self._fixed_dof_indices]
            self._assembly.set_coordinates(natural_coordinates)
        except IllDefinedShape:
            self._assembly.set_coordinates(initial_coordinates)

    @ignore_warnings(LinAlgWarning)
    def _solve_with_arclength(self, force_vector_step_list, max_displacement_map_step_list,
                              blocked_nodes_direction_step_list,
                              show_warnings, detect_critical_points, bifurcate_at_simple_bifurcations,
                              reference_load_parameter, radius, i_max, j_max, convergence_value, verbose,
                              detail_verbose,
                              alpha, psi_p, psi_c, detect_mechanism):
        """
            Find equilibrium path using the arc-length method
        """
        start = time.time()
        nb_steps = len(force_vector_step_list)
        initial_coordinates = self._assembly.get_coordinates()
        equilibrium_forces = [np.zeros(self._nb_dofs)]
        equilibrium_displacements = [np.zeros(self._nb_dofs)]
        step_indices: list[int] = [0]
        self._assembly.block_nodes_along_directions(*blocked_nodes_direction_step_list[0])
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._nb_free_dofs = len(self._free_dof_indices)
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()

        try:
            ks = self._assembly.compute_structural_stiffness_matrix()
            stiffness_matrix_eval_counter = 1
            initial_loaded_dof_indices = self._loaded_dof_indices_step_list[0]
            equilibrium_eigval_stats = [self._compute_lowest_eigenvalues(ks, initial_loaded_dof_indices)]
        except IllDefinedShape:
            if verbose:
                reason = '--> aborted (the shape of an element is ill-defined)\r\n'
                update_progress(f'Solving progress (step 1/{nb_steps})', 0, 0, i_max, reason)
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        0, 0, 0, 0, 0, 0, 0)
                end = time.time()
                print(f"Solving duration: {end - start:.4f} s")
            return (np.array([np.nan]), np.array([np.nan]), np.array(['nan'], dtype=str),
                    np.array([np.nan]), np.array([0], dtype=int))

        equilibrium_stability = [self._assess_stability(ks, initial_loaded_dof_indices)]
        initial_eigval_magnitude = max(np.abs(equilibrium_eigval_stats[-1][0]), 0.1)
        linear_system_solving_counter = 0
        total_nb_increment_retries = 0
        total_nb_singular_matrices_avoided = 0
        nb_limit_points_detected = 0 if detect_critical_points else '?'
        nb_bifurcation_points_detected = 0 if detect_critical_points else '?'
        initial_radius_p = radius
        radius_p = initial_radius_p
        delta_s = 1.0
        f_ext = np.zeros(self._nb_dofs)
        u = np.zeros(self._nb_dofs)
        force_progress = 0.0
        current_step = None
        has_previously_bifurcated = False
        has_bifurcated = False

        i = 0
        try:
            for step, step_force_vector in enumerate(force_vector_step_list):
                current_step = step + 1
                self._assembly.block_nodes_along_directions(*blocked_nodes_direction_step_list[step])
                self._free_dof_indices = self._assembly.get_free_dof_indices()
                self._nb_free_dofs = len(self._free_dof_indices)
                self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()

                loaded_dof_indices = self._loaded_dof_indices_step_list[step]
                force_reached_at_previous_step = f_ext.copy()
                displacement_reached_at_previous_step = u.copy()
                delta_f = reference_load_parameter * step_force_vector
                previous_delta_u_inc = None
                previous_delta_lambda_inc = None
                increment_retries = 0
                force_progress = 0.0

                if step > 0:
                    equilibrium_displacements.append(equilibrium_displacements[-1])
                    equilibrium_forces.append(equilibrium_forces[-1])
                    equilibrium_stability.append(self._assess_stability(ks, loaded_dof_indices))
                    equilibrium_eigval_stats.append(self._compute_lowest_eigenvalues(ks, loaded_dof_indices))
                    step_indices.append(step)

                if (set(loaded_dof_indices) <= set(self._fixed_dof_indices)
                        or np.linalg.norm(step_force_vector[self._free_dof_indices]) == 0.0):
                    # all loaded degrees of freedom are constrained
                    # or the force applied of the unconstrained degrees of freedom has a magnitude of zero
                    update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max,
                                    status='--> skipped because trivial: no nonzero forces have been applied '
                                           'on the free degrees of freedom\r\n', stability=equilibrium_stability[-1])
                    continue

                if show_warnings:
                    if not set(loaded_dof_indices).isdisjoint(self._fixed_dof_indices):
                        print(f'For loading step {current_step}/{nb_steps}, '
                              f'one or more forces have been applied on fixed degrees of freedom. '
                              'It is probably not intended (these forces cannot affect the deformation)')
                    update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max,
                                    status='...', stability=equilibrium_stability[-1])

                k = self._get_reduced_stiffness_matrix(ks)
                g = self._get_reduced_vector(delta_f)
                norm_g = np.linalg.norm(g)
                while True:
                    has_increment_converged = False

                    # PREDICTOR PHASE FOR INCREMENT i
                    delta_u_inc = np.zeros(self._nb_free_dofs)
                    delta_lambda_inc = 0.0

                    # solve linear system
                    if i == 0 and detect_mechanism:
                        if cond(k, p=1) > 1e8:
                            raise MechanismDetected
                    try:
                        delta_u_hat = solve(k, g, assume_a='sym')
                    except np.linalg.LinAlgError:
                        # might happen on the first increment if detect_mechanism is False
                        # and the initial stiffness matrix is singular
                        k_perturbed = _perturb_singular_stiffness_matrix(k, 1e-5, show_message=show_warnings)
                        try:
                            delta_u_hat = solve(k_perturbed, g, assume_a='sym')
                        except np.linalg.LinAlgError:
                            raise MechanismDetected
                        total_nb_singular_matrices_avoided += 1
                    linear_system_solving_counter += 1

                    # Root computation and selection
                    delta_lambda_ite = radius_p / np.sqrt(np.inner(delta_u_hat, delta_u_hat) + psi_p ** 2)
                    root_choice_criteria = (previous_delta_lambda_inc is None
                                            or np.inner(delta_u_hat, previous_delta_u_inc)
                                            + (psi_p ** 2 * previous_delta_lambda_inc) >= 0)
                    root_sign = +1 if root_choice_criteria else -1
                    delta_lambda_ite *= root_sign

                    # Updating loads, displacements and structure + computation of the unbalanced forces
                    delta_u_ite = delta_lambda_ite * delta_u_hat
                    delta_u_inc += delta_u_ite
                    delta_lambda_inc += delta_lambda_ite
                    self._assembly.increment_coordinates(self._get_structural_displacements(delta_u_ite))
                    f_ext += (delta_lambda_ite * delta_f)
                    f_int = self._assembly.compute_elastic_force_vector()
                    r = f_int - f_ext
                    # Convergence check and preparation for next increment
                    if norm_g / reference_load_parameter < 1e-9:
                        # no force has been applied on the unconstrained degrees of freedom
                        has_increment_converged = True
                    elif np.linalg.norm(self._get_reduced_vector(r)) / norm_g < convergence_value:
                        has_increment_converged = True
                    # print(f'\ni: {i}')
                    # print(f'delta_lambda_inc: {delta_lambda_inc}')
                    # print(f'converged? {has_increment_converged}')

                    if not has_increment_converged:
                        # CORRECTOR PHASE (correcting using at most j_max iterations)
                        radius_c = np.sqrt(np.inner(delta_u_hat, delta_u_hat) / (
                                np.inner(delta_u_hat, delta_u_hat) + psi_p ** 2)) * radius_p
                        rhom_radius_c = (1 - alpha) * radius_c
                        rhom_u_inc = (1 - alpha) * delta_u_inc
                        rhom_lambda_inc = (1 - alpha) * delta_lambda_inc
                        for j in range(j_max):
                            ks = self._assembly.compute_structural_stiffness_matrix()
                            k = self._get_reduced_stiffness_matrix(ks)
                            stiffness_matrix_eval_counter += 1

                            # solve two linear systems
                            if i == 0 and detect_mechanism:
                                if cond(k, p=1) > 1e8:
                                    # this should probably never be executed, because, if there is a mechanism, it is
                                    # most likely detected during the predictor phase.
                                    raise MechanismDetected
                            try:
                                delta_u_hat = solve(k, g, assume_a='sym')
                                delta_u_bar = solve(k, -self._get_reduced_vector(r), assume_a='sym')
                            except np.linalg.LinAlgError:
                                k_perturbed = _perturb_singular_stiffness_matrix(k, 1e-5, show_message=show_warnings)
                                delta_u_hat = solve(k_perturbed, g, assume_a='sym')
                                delta_u_bar = solve(k_perturbed, -self._get_reduced_vector(r), assume_a='sym')
                                total_nb_singular_matrices_avoided += 1
                            linear_system_solving_counter += 2

                            # Root computation and selection
                            a0 = np.inner(delta_u_hat, delta_u_hat) + psi_c ** 2
                            b0 = 2 * np.inner(rhom_u_inc, delta_u_hat) + rhom_lambda_inc * psi_c ** 2
                            b1 = 2 * np.inner(delta_u_bar, delta_u_hat)
                            c0 = np.inner(rhom_u_inc, rhom_u_inc) + (rhom_lambda_inc * psi_c) ** 2 - rhom_radius_c ** 2
                            c1 = 2 * np.inner(rhom_u_inc, delta_u_bar)
                            c2 = np.inner(delta_u_bar, delta_u_bar)
                            a = a0
                            b = b0 + b1 * delta_s
                            c = c0 + c1 * delta_s + c2 * delta_s ** 2
                            rho = b ** 2 - 4 * a * c
                            if rho < 0:
                                # Resetting with a smaller radius
                                break
                            else:
                                root1 = (-b + np.sqrt(rho)) / (2 * a)
                                root2 = (-b - np.sqrt(rho)) / (2 * a)
                            t = np.inner(rhom_u_inc, delta_u_hat) + rhom_lambda_inc * psi_c ** 2
                            delta_lambda_ite = root1 if t * (root1 - root2) > 0 else root2

                            # Updating loads, displacements and structure + computation of the unbalanced forces
                            delta_u_ite = delta_s * delta_u_bar + delta_lambda_ite * delta_u_hat
                            delta_u_inc += delta_u_ite
                            delta_lambda_inc += delta_lambda_ite
                            # print(f'delta_lambda_inc: {delta_lambda_inc}')
                            rhom_u_inc += delta_u_ite
                            rhom_lambda_inc += delta_lambda_ite
                            self._assembly.increment_coordinates(self._get_structural_displacements(delta_u_ite))
                            f_int = self._assembly.compute_elastic_force_vector()
                            f_ext += delta_lambda_ite * delta_f
                            r = f_int - f_ext
                            # Convergence check and preparation for next increment
                            if (norm_g / reference_load_parameter < 1e-9 or
                                    np.linalg.norm(self._get_reduced_vector(r)) / norm_g < convergence_value):
                                has_increment_converged = True
                                break

                    # Preparation for the next increment
                    if has_increment_converged:
                        ks = self._assembly.compute_structural_stiffness_matrix()
                        k = self._get_reduced_stiffness_matrix(ks)
                        stiffness_matrix_eval_counter += 1

                        (fd_lowest_eigval,
                         ud_lowest_eigval,
                         fd_negative_eigval_count,
                         ud_negative_eigval_count) = self._compute_lowest_eigenvalues(ks, loaded_dof_indices)
                        _, _, previous_fd_negative_eigval_count, _ = equilibrium_eigval_stats[-1]

                        if detect_critical_points:
                            if previous_fd_negative_eigval_count < fd_negative_eigval_count:
                                multiplicity = fd_negative_eigval_count - previous_fd_negative_eigval_count
                                if detail_verbose:
                                    print(f'change in negative eigval'
                                          f'(from {previous_fd_negative_eigval_count} to {fd_negative_eigval_count})')
                                singular_eigvals, singular_modes = self._compute_singular_eigenmodes(ks, multiplicity)
                                if np.max(np.abs(singular_eigvals)) / initial_eigval_magnitude < 1e-3:
                                    # CRITICAL POINT REACHED
                                    if multiplicity == 1:
                                        singular_eigval = singular_eigvals[0]
                                        singular_mode = singular_modes[0]
                                    else:
                                        singular_eigval = min(singular_eigvals)
                                        ortho_load = _compute_vector_perpendicular_to(g / norm_g)
                                        singular_mode = np.zeros(self._nb_free_dofs)
                                        for sm in singular_modes:
                                            singular_mode += np.inner(ortho_load, sm) * sm
                                        singular_mode /= np.linalg.norm(singular_mode)

                                    if np.abs(np.inner(singular_mode, g / norm_g)) < 1e-3:
                                        if has_previously_bifurcated:
                                            if detail_verbose:
                                                print(f'looks like bifurcation point,'
                                                      f'but has recently bifurcated, so no.')
                                        else:
                                            # critical point is a bifurcation point
                                            nb_bifurcation_points_detected += 1
                                            if detail_verbose:
                                                print(f'bifurcation point')
                                                print(f'eigval: {singular_eigval}')
                                                print(f'v x f = {np.abs(np.inner(singular_mode, g / norm_g))}')

                                            if multiplicity == 1 and bifurcate_at_simple_bifurcations:
                                                # one bifurcates to the "buckled" branch
                                                self._assembly.set_coordinates(
                                                    initial_coordinates + equilibrium_displacements[-1])
                                                f_ext = equilibrium_forces[-1].copy()

                                                perturbation_magnitude = np.linalg.norm(equilibrium_displacements[-1]) / 100
                                                null_vector = self._get_structural_displacements(singular_mode)
                                                bifurcation_perturbation = perturbation_magnitude * null_vector

                                                self._assembly.increment_coordinates(bifurcation_perturbation)
                                                ks = self._assembly.compute_structural_stiffness_matrix()
                                                k = self._get_reduced_stiffness_matrix(ks)
                                                self._assembly.increment_coordinates(-bifurcation_perturbation)
                                                has_bifurcated = True
                                    else:
                                        # critical point is a limit point
                                        nb_limit_points_detected += 1
                                        if detail_verbose:
                                            print(f'limit point')
                                            print(f'eigval: {singular_eigval}')
                                            print(singular_mode)
                                            print(f'v x f = {np.abs(np.inner(singular_mode, g / norm_g))}')
                                    if has_bifurcated:
                                        has_bifurcated = False
                                        has_previously_bifurcated = True
                                        continue  # go to next increment
                                    else:
                                        # because limit point, or bifurcation point at which we do not want to bifurcate
                                        has_bifurcated = False
                                        has_previously_bifurcated = False
                                        # the equilibrium point does not need any special treatment,
                                        # the equilibrium will be saved
                                else:
                                    # the critical point was overshot, restart at previously converged increment
                                    # with smaller radius
                                    if radius_p < 1e-14:
                                        raise ConvergenceError
                                    if detail_verbose:
                                        print(f'eigval too large to be considered a singularity.'
                                              f' Restart increment {i} with smaller radius: {radius_p / 2}')
                                    self._assembly.set_coordinates(initial_coordinates + equilibrium_displacements[-1])
                                    f_ext = equilibrium_forces[-1].copy()
                                    ks = self._assembly.compute_structural_stiffness_matrix()
                                    k = self._get_reduced_stiffness_matrix(ks)
                                    radius_p /= 2.0
                                    has_bifurcated = False
                                    has_previously_bifurcated = False
                                    continue
                            else:
                                # no increase in nb of negative eigvals --> no sign of critical points
                                has_bifurcated = False
                                has_previously_bifurcated = False

                        # VALID EQUILIBRIUM POINT
                        i += 1
                        increment_retries = 0
                        previous_delta_u_inc = delta_u_inc.copy()
                        previous_delta_lambda_inc = delta_lambda_inc
                        u += self._get_structural_displacements(delta_u_inc)
                        radius_p = min(initial_radius_p, radius_p * 2)
                        stability_state = self._assess_stability(ks, loaded_dof_indices)
                        step_indices.append(step)
                        equilibrium_displacements.append(u.copy())
                        equilibrium_forces.append(f_ext.copy())
                        equilibrium_stability.append(stability_state)
                        equilibrium_eigval_stats.append([fd_lowest_eigval,
                                                         ud_lowest_eigval,
                                                         fd_negative_eigval_count,
                                                         ud_negative_eigval_count])
                    else:
                        increment_retries += 1
                        if increment_retries <= 5 and radius_p > 1e-14:
                            if show_warnings:
                                print(f"\nCorrection iterations did not converge for the increment {i + 1}"
                                      f"\t-> retry increment with smaller radius ({radius_p / 2.0:.3})"
                                      f"\t-> attempt {increment_retries}/{5}")
                            # Resetting structure to its previous incremental state with a smaller radius
                            total_nb_increment_retries += 1
                            self._assembly.set_coordinates(initial_coordinates + equilibrium_displacements[-1])
                            f_ext = equilibrium_forces[-1].copy()
                            ks = self._assembly.compute_structural_stiffness_matrix()
                            k = self._get_reduced_stiffness_matrix(ks)
                            radius_p /= 2.0
                            continue  # go to the next increment
                        else:
                            raise ConvergenceError

                    # Check if final force has been reached. If yes, we can go to the next loading step
                    step_f_ext = f_ext - force_reached_at_previous_step
                    step_f_ext_norm = np.linalg.norm(step_f_ext)
                    step_force_vector_norm = np.linalg.norm(step_force_vector)
                    f_vectors_aligned = np.inner(step_force_vector, step_f_ext) > 0
                    final_force_has_been_reached = step_force_vector_norm - step_f_ext_norm < 0.0 and f_vectors_aligned
                    force_progress = step_f_ext_norm / step_force_vector_norm if f_vectors_aligned else 0.0
                    if final_force_has_been_reached:
                        if verbose:
                            reason = f'--> final force for loading step {current_step} has been reached'
                            reason += '\r\n'
                            update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i,
                                            i_max, reason, stability=equilibrium_stability[-1])
                        break  # go to next loading step

                    # Checking if max displacement has been reached. If yes, we go to the next loading step.
                    try:
                        step_u = u - displacement_reached_at_previous_step
                        if max_displacement_map_step_list[step] is not None:
                            for dof_index, max_displacement in max_displacement_map_step_list[step].items():
                                if (abs(step_u[dof_index]) >= abs(max_displacement)
                                        and step_u[dof_index] * max_displacement >= 0.0):
                                    raise MaxDisplacementReached
                    except MaxDisplacementReached:
                        if verbose:
                            reason = f'--> max displacement for loading step {current_step} has been reached'
                            reason += '\r\n'
                            update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i,
                                            i_max, reason, stability=equilibrium_stability[-1])
                        break  # go to next loading step

                    # Checking if the max number of iteration has been reached. If yes, the solving process is aborted.
                    if i == i_max:
                        raise MaxNbIterationReached

                    if verbose:
                        update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max,
                                        '...', stability=equilibrium_stability[-1])

            if verbose:
                _print_message_with_final_solving_stats('Full equilibrium path was retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)

        except MaxNbIterationReached:
            if verbose:
                reason = '--> max nb of increments has been reached\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)
        except MechanismDetected:
            if verbose:
                reason = (f'--> aborted because the initial stiffness matrix is singular. '
                          'Boundary conditions most likely allow rigid-body modes.\r\n')
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)
        except ConvergenceError:
            if verbose:
                reason = '--> aborted (could not converge)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)
        except IllDefinedShape:
            if verbose:
                reason = '--> aborted (the shape of an element has become ill-defined)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)
        except NonfiniteBehavior:
            if verbose:
                reason = '--> aborted (the mechanical behavior of an element has produced a nonfinite value)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)

        end = time.time()
        if verbose:
            print(f"Solving duration: {end - start:.4f} s")

        return (np.array(equilibrium_displacements), np.array(equilibrium_forces),
                np.array(equilibrium_stability, dtype=str), np.array(equilibrium_eigval_stats),
                np.array(step_indices, dtype=int))

    def _compute_lowest_eigenvalues(self, ks, loaded_dof_indices):
        # force-driven case
        k_ = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
        k_ = np.delete(k_, self._fixed_dof_indices, axis=1)  # delete columns
        eigvals = np.linalg.eigvalsh(k_)
        if k_.shape[0] > 0:
            fd_lowest_eigval = eigvals[0]
            fd_negative_eigval_count = np.sum(eigvals < 0.0)
        else:
            fd_lowest_eigval = np.nan
            fd_negative_eigval_count = 0
        # displacement driven case
        k_ = np.delete(ks, self._fixed_dof_indices + loaded_dof_indices, axis=0)  # delete rows
        k_ = np.delete(k_, self._fixed_dof_indices + loaded_dof_indices, axis=1)  # delete columns
        if k_.shape[0] > 0:
            eigvals = np.linalg.eigvalsh(k_)
            ud_lowest_eigval = eigvals[0]
            ud_negative_eigval_count = np.sum(eigvals < 0.0)
        else:
            ud_lowest_eigval = np.nan
            ud_negative_eigval_count = 0
        return fd_lowest_eigval, ud_lowest_eigval, fd_negative_eigval_count, ud_negative_eigval_count

    def _compute_singular_eigenmodes(self, ks, nb_singularities):
        # force-driven case
        k_ = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
        k_ = np.delete(k_, self._fixed_dof_indices, axis=1)  # delete columns
        eigvals, eigvects = np.linalg.eigh(k_)
        sorting_indices = np.argsort(np.abs(eigvals))
        singular_mode_indices = sorting_indices[:nb_singularities]
        return ([eigvals[singular_mode_index] for singular_mode_index in singular_mode_indices],
                [eigvects[:, singular_mode_index] for singular_mode_index in singular_mode_indices])

    def _assess_stability(self, ks, loaded_dof_indices):
        try:
            k_ = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
            k_ = np.delete(k_, self._fixed_dof_indices, axis=1)  # delete columns
            np.linalg.cholesky(k_)
            # if no error is triggered, then matrix is positive definite
            return StaticSolver.STABLE
        except np.linalg.LinAlgError:
            k_ = np.delete(ks, self._fixed_dof_indices + loaded_dof_indices, axis=0)  # delete rows
            k_ = np.delete(k_, self._fixed_dof_indices + loaded_dof_indices, axis=1)  # delete columns
            try:
                np.linalg.cholesky(k_)
                # if no error is triggered, then matrix is positive definite
                return StaticSolver.STABILIZABLE
            except np.linalg.LinAlgError:
                return StaticSolver.UNSTABLE

    def _get_reduced_stiffness_matrix(self, ks):
        k = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
        k = np.delete(k, self._fixed_dof_indices, axis=1)  # delete columns
        return k

    def _get_reduced_vector(self, vs):
        """ Returns the vector without the entries for the fixed dofs """
        v = np.delete(vs, self._fixed_dof_indices)
        return v

    def _get_structural_displacements(self, u):
        us = np.zeros(self._nb_dofs)
        us[self._free_dof_indices] = u
        return us


class MechanismDetected(Exception):
    """ raise this when the stiffness matrix is singular at the start of the simulation """


class ConvergenceError(Exception):
    """ raise this when the algorithm cannot converge """


class MaxNbIterationReached(Exception):
    """ raise this when the max number of iteration is reached """


class MaxDisplacementReached(Exception):
    """ raise this when the max displacement is reached """


def _print_message_with_final_solving_stats(message,
                                            nb_increments,
                                            nb_stiffness_matrix_eval,
                                            nb_linear_system_resolutions,
                                            nb_increment_retries,
                                            nb_matrix_perturbations,
                                            nb_limits_points,
                                            nb_bifurcation_points):
    print(f"{message}: "
          f"increments = {nb_increments}"
          f" | stiffness matrix eval = {nb_stiffness_matrix_eval}"
          f" | linear system resolutions = {nb_linear_system_resolutions}"
          f" |\nincrement retries = {nb_increment_retries}"
          f" | singular stiffness matrices avoided = {nb_matrix_perturbations}"
          f" | limit points = {nb_limits_points}"
          f" | bifurcation points = {nb_bifurcation_points}")


def _perturb_singular_stiffness_matrix(k: np.ndarray, epsilon, show_message):
    frobenius_norm = np.linalg.norm(k, 'fro')
    perturbation = max(epsilon * frobenius_norm, epsilon * 1e-3)
    if show_message:
        print(f'\nStiffness matrix is exactly singular. '
              f'Applying a small perturbation ({perturbation}) '
              f'on the diagonal elements')
    return k + np.eye(*k.shape) * perturbation


def _compute_vector_perpendicular_to(v0):
    idx_max = np.argmax(np.abs(v0))
    v1 = np.zeros(v0.shape)
    v1[idx_max] = -v0[(idx_max + 1) % len(v0)] / v0[idx_max]
    v1[(idx_max + 1) % len(v0)] = 1
    return v1


def update_progress(title, progress, i, i_max, status, stability: str = None):
    if stability == StaticSolver.STABLE:
        symbol = '#'
    elif stability == StaticSolver.STABILIZABLE:
        symbol = 'o'
    elif stability == StaticSolver.UNSTABLE:
        symbol = 'x'
    else:
        symbol = '#'
    bar_length = 10
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    text = "\r{0}: [{1}] ({2}/{3}) {4}".format(title, f"{symbol}" * block + "-" * (bar_length - block), i, int(i_max),
                                               status)
    sys.stdout.write(text)
    sys.stdout.flush()
