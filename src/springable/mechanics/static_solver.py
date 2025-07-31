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
from dataclasses import asdict
import sys
import time
from .assembly import Assembly
from .model import Model
from .node import Node
from .shape import IllDefinedShape
from .default_solver_settings import SolverSettings
from .stability_states import StabilityStates


class Result:
    def __init__(self, model: Model,
                 equilibrium_displacements, equilibrium_forces, equilibrium_stability, equilibrium_eigval_stats,
                 step_indices, solving_process_info: dict | None = None):
        self._model = model
        self._u = equilibrium_displacements
        self._f = equilibrium_forces
        self._stability = equilibrium_stability
        self._eigval_stats = equilibrium_eigval_stats
        self._step_indices = step_indices if step_indices is not None else np.zeros(equilibrium_displacements.shape[0],
                                                                                    dtype=int)
        self._solving_process_info = solving_process_info

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

    def get_model(self) -> Model:
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

    def get_internal_force_from_element_index(self, element_index: int, include_preloading=False, check_usability=True):
        el = self.get_model().get_assembly().get_elements()[element_index]
        q0 = self.get_model().get_assembly().get_coordinates().copy()
        u = self.get_displacements(include_preloading, check_usability)
        internal_force = np.empty(u.shape[0])
        for i in range(u.shape[0]):
            self.get_model().get_assembly().set_coordinates(q0 + u[i, :])
            internal_force[i] = el.compute_generalized_force()
        self.get_model().get_assembly().set_coordinates(q0)
        return internal_force





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

    def get_loadstep_starting_index(self, loadstep_index: int):
        nb_steps = len(self._model.get_loading())
        if self._u.ndim < 2:
            return None

        if loadstep_index < 0:
            loadstep_index = nb_steps + loadstep_index

        if loadstep_index < 0:
            return None

        if loadstep_index == 0:
            return 0

        # load step index > 0
        index = np.argmax(self._step_indices == loadstep_index)
        if index == 0:
            return None
        else:
            return index

    def get_loadstep_end_index(self, loadstep_index: int):
        """ return the end index of the desired loadstep. """
        if self._u.ndim < 2:
            return None

        if loadstep_index < 0:
            nb_steps = len(self._model.get_loading())
            loadstep_index = nb_steps + loadstep_index

        if loadstep_index < 0:
            return None

        # load step index > 0
        index = self._u.shape[0] - np.argmax(self._step_indices[::-1] == loadstep_index) - 1
        if index == self._u.shape[0] - 1 and loadstep_index not in self._step_indices:
            return None
        return index

    def get_solving_process_info(self):
        return self._solving_process_info


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

    def __init__(self, model: Model, **solver_settings):
        self._model = model
        self._assembly = model.get_assembly()
        self._nb_dofs = self._assembly.get_nb_dofs()
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._nb_free_dofs = len(self._free_dof_indices)
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()
        self._loaded_dof_indices_step_list = (self._model.get_loaded_dof_indices_preloading_step_list()
                                              + [self._model.get_loaded_dof_indices()])
        self._solver_settings = SolverSettings()
        self._solver_settings.update(**solver_settings)

    def solve(self) -> Result:
        # this function assumes starting from an equilibrium point
        initial_coordinates = self._assembly.get_coordinates()
        step_force_vectors = self._model.get_force_vectors_preloading_step_list() + [self._model.get_force_vector()]
        max_displacement_map_step_list = self._model.get_max_displacement_map_preloading_step_list() + [
            self._model.get_max_displacement_map()]
        blocked_nodes_directions_step_list = self._model.get_blocked_nodes_directions_step_list()
        u, f, stability, eigval_stats, step_indices, info = self._solve_with_arclength(step_force_vectors,
                                                                                       max_displacement_map_step_list,
                                                                                       blocked_nodes_directions_step_list,
                                                                                       **asdict(self._solver_settings))
        if u.ndim == 2:
            self._assembly.set_coordinates(initial_coordinates)
        for blocked_nodes, directions in blocked_nodes_directions_step_list:
            self._assembly.release_nodes_along_directions(blocked_nodes, directions)
        return Result(self._model, u, f, stability, eigval_stats, step_indices, solving_process_info=info)

    def guide_spring_assembly_to_natural_configuration(self):
        if not self._free_dof_indices:
            return
        initial_coordinates = self._assembly.get_coordinates()

        def elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_coordinates(coordinates)
            energy = self._assembly.compute_elastic_energy()
            if not np.isfinite(energy):
                raise NonfiniteMechanicalQuantity
            return energy

        def gradient_elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_coordinates(coordinates)
            force_vector = self._get_reduced_vector(self._assembly.compute_elastic_force_vector())
            return force_vector

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
        except (IllDefinedShape, NonfiniteMechanicalQuantity):
            print('Could not guide the assembly to its natural configuration')
            self._assembly.set_coordinates(initial_coordinates)


    @ignore_warnings(LinAlgWarning)
    def _solve_with_arclength(self, force_vector_step_list, max_displacement_map_step_list,
                              blocked_nodes_direction_step_list,
                              show_warnings, detect_critical_points,
                              bifurcate_at_simple_bifurcations, critical_point_epsilon, bifurcation_perturbation_amplitude,
                              radius, i_max, j_max, convergence_value, verbose,
                              critical_point_detection_verbose, detect_mechanism):
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
            solving_process_info = {'duration (s)': 0,
                                    '# stiffness matrix evals': 1,
                                    '# linear system resolutions': 0,
                                    '# increment retries': 0,
                                    '# avoided singular stiffness matrices': 0,
                                    }
            return (np.array([np.nan]), np.array([np.nan]), np.array(['nan'], dtype=str),
                    np.array([np.nan]), np.array([0], dtype=int), solving_process_info)

        equilibrium_stability = [self._assess_stability(ks, initial_loaded_dof_indices)]
        initial_eigval_magnitude = max(np.abs(equilibrium_eigval_stats[-1][0]), 0.1)
        linear_system_solving_counter = 0
        total_nb_increment_retries = 0
        total_nb_singular_matrices_avoided = 0
        nb_critical_points_detected = 0 if detect_critical_points else '?'
        nb_limit_points_detected = 0 if detect_critical_points else '?'
        nb_bifurcation_points_detected = 0 if detect_critical_points else '?'
        initial_radius = radius
        f_ext = np.zeros(self._nb_dofs)
        u = np.zeros(self._nb_dofs)
        force_progress = 0.0
        current_step = None

        # only used if detect_critical_point is True
        searching_for_existing_critical_point = False
        multiplicity = None
        existing_critical_point_found = None
        is_critical = [False]

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
                norm_step_force_vector = np.linalg.norm(step_force_vector)
                delta_f = step_force_vector / norm_step_force_vector
                previous_delta_u_inc = None
                previous_delta_lambda_inc = None
                increment_retries = 0
                force_progress = 0.0
                bifurcation_perturbation = None
                bifurcation_should_be_induced = False

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
                    delta_lambda_ite = radius / np.sqrt(np.inner(delta_u_hat, delta_u_hat))
                    root_choice_criteria = (previous_delta_lambda_inc is None
                                            or np.inner(delta_u_hat, previous_delta_u_inc) >= 0)
                    root_sign = +1 if root_choice_criteria else -1
                    if previous_delta_u_inc is None and equilibrium_eigval_stats[0][0] < 0.0:
                        root_sign *= -1

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
                    if norm_step_force_vector < 1e-9:
                        # no force has been applied on the unconstrained degrees of freedom
                        has_increment_converged = True
                    elif np.linalg.norm(self._get_reduced_vector(r)) < convergence_value:
                        has_increment_converged = True
                    # print(f'\ni: {i}')
                    # print(f'delta_lambda_inc: {delta_lambda_inc}')
                    # print(f'converged? {has_increment_converged}')

                    if not has_increment_converged:
                        # CORRECTOR PHASE (correcting using at most j_max iterations)
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
                            a0 = np.inner(delta_u_hat, delta_u_hat)
                            b0 = 2 * np.inner(delta_u_inc, delta_u_hat)
                            b1 = 2 * np.inner(delta_u_bar, delta_u_hat)
                            c0 = np.inner(delta_u_inc, delta_u_inc) - radius ** 2
                            c1 = 2 * np.inner(delta_u_inc, delta_u_bar)
                            c2 = np.inner(delta_u_bar, delta_u_bar)
                            a = a0
                            b = b0 + b1
                            c = c0 + c1 + c2
                            rho = b ** 2 - 4 * a * c
                            if rho < 0:
                                # Resetting with a smaller radius
                                break
                            else:
                                root1 = (-b + np.sqrt(rho)) / (2 * a)
                                root2 = (-b - np.sqrt(rho)) / (2 * a)
                            delta_lambda_ite = root1 if b0 / a > 0 else root2

                            # Updating loads, displacements and structure + computation of the unbalanced forces
                            delta_u_ite = delta_u_bar + delta_lambda_ite * delta_u_hat
                            delta_u_inc += delta_u_ite
                            delta_lambda_inc += delta_lambda_ite
                            self._assembly.increment_coordinates(self._get_structural_displacements(delta_u_ite))
                            f_int = self._assembly.compute_elastic_force_vector()
                            f_ext += delta_lambda_ite * delta_f
                            r = f_int - f_ext
                            # Convergence check and preparation for next increment
                            if (norm_step_force_vector < 1e-9 or
                                    np.linalg.norm(self._get_reduced_vector(r)) < convergence_value):
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
                            if (previous_fd_negative_eigval_count < fd_negative_eigval_count
                                and not is_critical[-1]):
                                # The # of unstable modes increased, so we went pass a critical point. Let's reset to
                                # the previous equilibrium point and divide radius by two.
                                # We mark that we are now searching for a critical point of a certain multiplicity
                                # (this was already maybe the case)
                                searching_for_existing_critical_point = True
                                multiplicity = fd_negative_eigval_count - previous_fd_negative_eigval_count
                                if radius < 1e-14:
                                    raise ConvergenceError
                                if critical_point_detection_verbose:
                                    print(f'\nCritical point was passed. To search where it is, '
                                          f'increment {i} is restarted with smaller radius: {radius / 2:.3E}.')
                                total_nb_increment_retries += 1
                                self._assembly.set_coordinates(initial_coordinates + equilibrium_displacements[-1])
                                f_ext = equilibrium_forces[-1].copy()
                                ks = self._assembly.compute_structural_stiffness_matrix()
                                k = self._get_reduced_stiffness_matrix(ks)
                                radius /= 2.0
                                continue
                            else:
                                # no increase in # of unstable modes (or previous case was critical),
                                # we have two cases
                                if not searching_for_existing_critical_point:
                                    pass  # nothing special to do
                                else:
                                    # We are looking for a critical point, then let's check if we are close enough.
                                    # This point will be saved as valid eq point
                                    singular_eigvals, singular_modes = self._compute_singular_eigenmodes(ks, multiplicity)
                                    if not np.max(np.abs(singular_eigvals)) / initial_eigval_magnitude < critical_point_epsilon:
                                        # too far to be considered critical, nothing special to do
                                        pass
                                    else:
                                        # Close enough to the critical point, the current state lies just before the
                                        # increase in # of eigenvalues.
                                        nb_critical_points_detected += 1
                                        existing_critical_point_found = True
                                        if critical_point_detection_verbose:
                                            print(f'Critical point with multiplicity {multiplicity:.0f} reached '
                                                  f'(max |eigenvalues| = {np.max(np.abs(singular_eigvals)):.3E})')
                                        if multiplicity == 1:  # this is a simple critical point
                                            singular_mode = singular_modes[0]

                                            if np.abs(np.inner(singular_mode, g / norm_g)) < 1e-3:
                                                # This is a simple bifurcation point
                                                nb_bifurcation_points_detected += 1
                                                if critical_point_detection_verbose:
                                                    print(f'Simple bifurcation point')
                                                    print(f'Buckling mode = '
                                                          f'{self._get_structural_displacements(singular_mode)}')
                                                    print(f'v x f = {np.abs(np.inner(singular_mode, g / norm_g)): .3E}')

                                                if bifurcate_at_simple_bifurcations:
                                                    # mark that the stiffness matrix should be perturbed
                                                    # to induce bifurcation on the next increment,
                                                    # and prevent from continuing on the unstable branch
                                                    bifurcation_should_be_induced = True
                                                    perturbation_magnitude = np.linalg.norm(equilibrium_displacements[-1]) * bifurcation_perturbation_amplitude
                                                    null_vector = self._get_structural_displacements(singular_mode)
                                                    bifurcation_perturbation = perturbation_magnitude * null_vector


                                            else:  # This is a simple limit point
                                                nb_limit_points_detected += 1
                                                if critical_point_detection_verbose:
                                                    print(f'Simple limit point')
                                                    print(f'v x f = {np.abs(np.inner(singular_mode, g / norm_g)):.3E}')

                        # VALID EQUILIBRIUM POINT
                        i += 1
                        increment_retries = 0
                        previous_delta_u_inc = delta_u_inc.copy()
                        previous_delta_lambda_inc = delta_lambda_inc
                        u += self._get_structural_displacements(delta_u_inc)
                        radius = min(initial_radius, radius * 2)
                        stability_state = self._assess_stability(ks, loaded_dof_indices)
                        step_indices.append(step)
                        equilibrium_displacements.append(u.copy())
                        equilibrium_forces.append(f_ext.copy())
                        equilibrium_stability.append(stability_state)
                        equilibrium_eigval_stats.append([fd_lowest_eigval,
                                                         ud_lowest_eigval,
                                                         fd_negative_eigval_count,
                                                         ud_negative_eigval_count])
                        if detect_critical_points:
                            if searching_for_existing_critical_point and existing_critical_point_found:
                                searching_for_existing_critical_point = False
                                multiplicity = None
                                existing_critical_point_found = None
                                is_critical.append(True)
                            else:
                                is_critical.append(False)

                            if bifurcate_at_simple_bifurcations:
                                if bifurcation_should_be_induced:
                                    self._assembly.increment_coordinates(bifurcation_perturbation)
                                    ks = self._assembly.compute_structural_stiffness_matrix()
                                    k = self._get_reduced_stiffness_matrix(ks)

                                    # make sure assembly if not impacted (only stiffness matrix for next increment is perturbed)
                                    # and that bifurcation perturbation and flag are reset
                                    self._assembly.increment_coordinates(-bifurcation_perturbation)
                                    bifurcation_perturbation = None
                                    bifurcation_should_be_induced = False


                    else:   # if not converged
                        increment_retries += 1
                        if increment_retries <= 5 and radius > 1e-14:
                            if show_warnings:
                                print(f"\nCorrection iterations did not converge for the increment {i + 1}"
                                      f"\t-> retry increment with smaller radius ({radius / 2.0:.3E})"
                                      f"\t-> attempt {increment_retries}/{5}")
                            # Resetting structure to its previous incremental state with a smaller radius
                            total_nb_increment_retries += 1
                            self._assembly.set_coordinates(initial_coordinates + equilibrium_displacements[-1])
                            f_ext = equilibrium_forces[-1].copy()
                            ks = self._assembly.compute_structural_stiffness_matrix()
                            k = self._get_reduced_stiffness_matrix(ks)
                            radius /= 2.0
                            continue  # go to the next increment
                        else:
                            raise ConvergenceError

                    # Check if final force has been reached. If yes, we can go to the next loading step
                    step_f_ext = f_ext - force_reached_at_previous_step
                    step_f_ext_norm = np.linalg.norm(step_f_ext)
                    f_vectors_aligned = np.inner(step_force_vector, step_f_ext) > 0
                    final_force_has_been_reached = norm_step_force_vector - step_f_ext_norm < 0.0 and f_vectors_aligned
                    force_progress = step_f_ext_norm / norm_step_force_vector if f_vectors_aligned else 0.0
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
                reason = f'--> aborted (increment {i} could not converge)\r\n'
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
        except NonfiniteMechanicalQuantity:
            if verbose:
                reason = '--> aborted (the mechanical behavior of an element has produced a nonfinite value)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_singular_matrices_avoided,
                                                        nb_limit_points_detected, nb_bifurcation_points_detected)

        end = time.time()
        duration = end - start
        if verbose:
            print(f"Solving duration: {duration:.4f} s")

        solving_process_info = {'duration (s)': duration,
                                '# stiffness matrix evals': stiffness_matrix_eval_counter,
                                '# linear system resolutions': linear_system_solving_counter,
                                '# increment retries': total_nb_increment_retries,
                                '# avoided singular stiffness matrices': total_nb_singular_matrices_avoided,
                                }

        return (np.array(equilibrium_displacements), np.array(equilibrium_forces),
                np.array(equilibrium_stability, dtype=str), np.array(equilibrium_eigval_stats),
                np.array(step_indices, dtype=int), solving_process_info)

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
            return StabilityStates.STABLE
        except np.linalg.LinAlgError:
            k_ = np.delete(ks, self._fixed_dof_indices + loaded_dof_indices, axis=0)  # delete rows
            k_ = np.delete(k_, self._fixed_dof_indices + loaded_dof_indices, axis=1)  # delete columns
            try:
                np.linalg.cholesky(k_)
                # if no error is triggered, then matrix is positive definite
                return StabilityStates.STABILIZABLE
            except np.linalg.LinAlgError:
                return StabilityStates.UNSTABLE

    def _get_reduced_stiffness_matrix(self, ks):
        k = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
        k = np.delete(k, self._fixed_dof_indices, axis=1)  # delete columns
        if not np.isfinite(k).all():
            raise NonfiniteMechanicalQuantity
        return k

    def _get_reduced_vector(self, vs):
        """ Returns the vector without the entries for the fixed dofs """
        v = np.delete(vs, self._fixed_dof_indices)
        if not np.isfinite(v).all():
            raise NonfiniteMechanicalQuantity
        return v

    def _get_structural_displacements(self, u):
        us = np.zeros(self._nb_dofs)
        us[self._free_dof_indices] = u
        return us



class LoadStepSolver:
    def __init__(self, mdl: Model, max_nb_increments: int,
                 radius: float, convergence_tol: float,
                 max_nb_iterations: int, show_warnings: bool,
                 detect_mechanism):
        self._mdl = mdl
        self._asb = mdl.get_assembly()
        self._loaded_dof_indices_step_list = (mdl.get_loaded_dof_indices_preloading_step_list()
                                              + [mdl.get_loaded_dof_indices()])
        self._incr_index: int = 0
        self._step_index: int = 0
        self._f_ext : np.ndarray = np.zeros(self._asb.get_nb_dofs())
        self._u_path: list[np.ndarray] = [np.zeros(self._asb.get_nb_dofs())]  # absolute
        self._f_path: list[np.ndarray] = [self._f_ext.copy()]  # absolute

        # settings
        self._max_nb_incr = max_nb_increments
        self._radius = radius
        self._convergence_tol = convergence_tol
        self._max_nb_iterations = max_nb_iterations
        self._show_warnings = show_warnings

    
    def compute_next_loadstep(self):
        step_force_vector = self._loaded_dof_indices_step_list[self._step_index]
        step_force_norm  = np.linalg.norm(step_force_vector)
        step_force_unit_vector = step_force_vector / step_force_norm
        inc_solver = IncrementSolver(self._asb, self._f_ext, step_force_unit_vector,
                                     self._radius, self._convergence_tol, self._max_nb_iterations, self._show_warnings)
        while True:
            inc_solver.compute_next_equilibrium()



class IncrementSolver:
    """ Class to compute successive increments within a load step. You can also go back to a previously computed equilibrium.
    Automatically, a path is recorded, as you compute next increments. The object is created from an assembly and an external force vector assumed to be at equilibrium already (residual < tol).
    After running the 'compute_next_equilibrium()' method, the structure is moved to the next equilibrium, and the external force vector is updated accordingly.
    This object assumes that the boundary conditions of the underlying assembly are not changing, and that the external applied force does not change orientation
    (meaning that an object of this class is only valid within a single loadstep).
    If boundary conditions changes, a new instance must be created. """
    def __init__(self, asb: Assembly, f_ext: np.ndarray, unit_f: np.ndarray,
                 radius: float, convergence_tol: float,
                 max_nb_iterations: int, show_warnings: bool):
        """
        Args:
            asb (Assembly): assembly assumed to be at equilibrium (res < tol) under force f_ext
            f_ext (np.ndarray): current force vector acting on the structural coordinates
            unit_f (np.ndarray): step force unit vector
            radius (float): radius of the constraint equation (arclength setting)
            convergence_tol (float): norm of the residual under which equilibrium is assumed
            max_nb_iterations (int): maximum number of iterations during the correction phase. Beyond that number, raises MaxNbIterationsExceeded.
            show_warnings (bool): print warning?
        """
        self._asb = asb  # will be updated in place later
        self._nb_dofs = asb.get_nb_dofs()
        self._q0 = self._asb.get_coordinates()  # initial structural coordinates
        # self._u_path: list[np.ndarray] = [np.zeros(asb.get_nb_dofs())]  # structural displacement (relative to displacement to the starting state)
        # self._f_ext_path: list[np.ndarray] = [f_ext.copy()]  # absolute forces (structural coordinates)

        self._u = np.zeros(asb.get_nb_dofs())  # will be updated in place later
        self._f_ext = f_ext  # will be updated in place later
        self._UNIT_F = unit_f  # never changes
        self._radius = radius
        self._du_prev_inc = None  # increment previous to the latest increment
        self._CONVERGENCE_TOL = convergence_tol
        self._MAX_NB_ITERATIONS = max_nb_iterations
        self._show_warnings = show_warnings

    def set_radius(self, radius):
        self._radius = radius

    def _get_reduced_stiffness_matrix(self, ks):
        k = np.delete(ks, self._asb.get_fixed_dof_indices(), axis=0)  # delete rows
        k = np.delete(k, self._asb.get_fixed_dof_indices(), axis=1)  # delete columns
        if not np.isfinite(k).all():
            raise NonfiniteMechanicalQuantity
        return k

    def _get_reduced_vector(self, vs):
        """ Returns the vector without the entries for the fixed dofs """
        v = np.delete(vs, self._asb.get_fixed_dof_indices())
        if not np.isfinite(v).all():
            raise NonfiniteMechanicalQuantity
        return v

    def _get_structural_displacements(self, u):
        us = np.zeros(self._asb.get_nb_dofs())
        us[self._asb.get_free_dof_indices()] = u
        return us

    def reset_to_previous_equilibrium(self):
        """ If an exception is raised, the state did not change, else the state is set to the
        previous equilibrium point and the last equilibrium is deleted """
        if len(self._u_path) == 0:
            raise ValueError("This should never happens! Check code")
        if len(self._u_path) == 1:  
            raise NoPreviousEquilibrium
        
        if len(self._u_path) == 2:
            self._du_prev_inc = None
        else:  # at least 3 equilibrium points
            self._du_prev_inc = self._u_path[-2] - self._u_path[-3]

        del self._u_path[-1]
        del self._f_ext_path[-1]
        self._asb.set_coordinates(self._q0 + self._u_path[-1])
        self._f_ext = self._f_ext_path[-1].copy()
        




    def go_to_next_equilibrium(self):
        """ Tries to compute the next increment by driving the structure to the next equilibrium point,
        and computing the external force vector (parallel to df) at that new equilibrium point. If the increment
        is successful, a new equilibrium is recorded automatically. In case of an any event preventing from converging,
        the structure will be automatically reset to the configuration before running this method, no equilibrium is recorded,
        and the exception responsible for the failure is raised.
        """
        du_inc = np.zeros(self._nb_dofs)
        dl_inc = 0

        try:
            # PREDICTION
            # if the state of the structure hasn't been updated,
            # computing the stiffness matrix reuses the previously computed one.
            k = self._get_reduced_stiffness_matrix(self._asb.compute_structural_stiffness_matrix())
            try:
                du_hat = solve(k, self._UNIT_F, assume_a='sym')
            except np.linalg.LinAlgError:
                # might happen on the first increment if detect_mechanism is False
                # and the initial stiffness matrix is singular
                k_perturbed = _perturb_singular_stiffness_matrix(k, 1e-5, show_message=self._show_warnings)
                try:
                    du_hat = solve(k_perturbed, self._UNIT_F, assume_a='sym')
                except np.linalg.LinAlgError:
                    raise MechanismDetected
            

            dl_ite = self._radius / np.sqrt(np.inner(du_hat, du_hat))
            root_choice_criteria = (self._du_prev_inc is None
                                    or np.inner(du_hat, self._du_prev_inc) >= 0)
            root_sign = +1 if root_choice_criteria else -1
            if self._du_prev_inc is None:
                # if neg eigvenval --> switch direction
                root_sign *= -1
            dl_ite *= root_sign

            # Updating displacements, loads and structure + computation of the unbalanced forces
            du_ite = dl_ite * du_hat
            du_inc += du_ite
            dl_inc += dl_ite

            self._f_ext[self._asb.get_free_dof_indices()] += (dl_ite * self._UNIT_F)
            self._asb.increment_coordinates(self._get_structural_displacements(du_ite))
            f_int = self._asb.compute_elastic_force_vector()
            r = self._get_reduced_vector(f_int - self._f_ext)

            # Convergence check and preparation for next increment
            if np.linalg.norm(r) < self._CONVERGENCE_TOL:
                self._du_prev_inc = du_inc.copy()
                self._u_eq += self._get_structural_displacements(du_inc)
                self._f_eq = self._f_ext.copy()


            # CORRECTION
            for _ in range(self._MAX_NB_ITERATIONS):
                k = self._get_reduced_stiffness_matrix(self._asb.compute_structural_stiffness_matrix())
                try:
                    du_hat = solve(k, self._UNIT_F, assume_a='sym')
                    du_bar = solve(k, -r, assume_a='sym')
                except np.linalg.LinAlgError:
                    k_perturbed = _perturb_singular_stiffness_matrix(k, 1e-5, show_message=self._show_warnings)
                    try:
                        du_hat = solve(k_perturbed, self._UNIT_F, assume_a='sym')
                        du_bar = solve(k_perturbed, -r, assume_a='sym')
                    except np.linalg.LinAlgError:
                        raise MechanismDetected

                # Root computation and selection
                a0 = np.inner(du_hat, du_hat)
                b0 = 2 * np.inner(du_inc, du_hat)
                b1 = 2 * np.inner(du_bar, du_hat)
                c0 = np.inner(du_inc, du_inc) - self._radius ** 2
                c1 = 2 * np.inner(du_inc, du_bar)
                c2 = np.inner(du_bar, du_bar)
                a = a0
                b = b0 + b1
                c = c0 + c1 + c2
                rho = b ** 2 - 4 * a * c
                if rho < 0:
                    raise CorrectionIterationFailed
                else:
                    root1 = (-b + np.sqrt(rho)) / (2 * a)
                    root2 = (-b - np.sqrt(rho)) / (2 * a)
                dl_ite = root1 if b0 / a > 0 else root2

                # Updating loads, displacements and structure + computation of the unbalanced forces
                du_ite = du_bar + dl_ite * du_hat
                du_inc += du_ite
                dl_inc += dl_ite
                self._f_ext[self._asb.get_free_dof_indices()] += (dl_ite * self._UNIT_F)
                self._asb.increment_coordinates(self._get_structural_displacements(du_ite))
                f_int = self._asb.compute_elastic_force_vector()
                r = self._get_reduced_vector(f_int - self._f_ext)

                # Convergence check and saving the equilibrium
                if np.linalg.norm(r) < self._CONVERGENCE_TOL:
                    self._du_prev_inc = du_inc.copy()
                    self._u_eq += self._get_structural_displacements(du_inc)
                    self._f_eq = self._f_ext.copy()
            else:  # we reached the max number of iterations
                raise MaxNbCorrectionIterationsExceeded
        except Exception:  # something happened that prevented convergence, let reset to the previous eq point
            self._asb.set_coordinates(self._q0 + self._u_eq)
            self._f_ext = self._f_eq.copy()
            raise
        




class MechanismDetected(Exception):
    """ raise this when the stiffness matrix is singular at the start of the simulation """


class ConvergenceError(Exception):
    """ raise this when the algorithm cannot converge """


class MaxNbIterationReached(Exception):
    """ raise this when the max number of iteration is reached """


class MaxDisplacementReached(Exception):
    """ raise this when the max displacement is reached """


class NonfiniteMechanicalQuantity(Exception):
    """ raise this when the reduced force vector or the reduced stiffness matrix have nonfinite components
    or when the energy is nonfinite"""

class MaxNbCorrectionIterationsExceeded(Exception):
    """ raise this when the max number of iterations (for the correction phase of the arclength scheme) is reached """

class CorrectionIterationFailed(Exception):
    """ raise this when the root to the constraint equation (arclength scheme) cannot be found """

class NoPreviousEquilibrium(Exception):
    """ raise this when the one tries to set the structure to the previously computed equilibrium point, but there is none"""


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
    if stability == StabilityStates.STABLE:
        symbol = '#'
    elif stability == StabilityStates.STABILIZABLE:
        symbol = 'o'
    elif stability == StabilityStates.UNSTABLE:
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
