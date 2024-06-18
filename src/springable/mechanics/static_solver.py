"""
Author: Paul Ducarme
Static Assembly Problem Solver class
"""
import warnings
import numpy as np
from scipy.linalg import solve as solve_linear
from scipy.linalg import LinAlgWarning
from scipy.optimize import minimize
import sys
import time
from .model import Model
from .node import Node
from .shape import IllDefinedShape


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

    def get_min_and_max_loading_displacement_and_force(self):
        u = self.get_displacements()
        f = self.get_forces()
        loaded_dof_indices = self._model.get_loaded_dof_indices()
        f_goal = self._model.get_force_vector()
        f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
        f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
        u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
        return np.min(u_load), np.max(u_load), np.min(f_load), np.max(f_load)


class UnusableSolution(Exception):
    """ raise this when one attempts to get a solution from a Result instance, but it is not usable """


class StaticSolver:
    """ Class representing a static solver """
    STABLE = 'stable'  # stable under force and displacement control
    STABILIZABLE = 'stabilizable'  # stable under displacement-control only
    UNSTABLE = 'unstable'  # unstable under both force control and displacement control
    _DEFAULT_SOLVER_SETTINGS = {'reference_load_parameter': 0.05,
                                'radius': 0.05,
                                'show_warnings': False,
                                'verbose': True,
                                'i_max': 10e3,
                                'j_max': 20,
                                'convergence_value': 1e-6,
                                'alpha': 0.0,
                                'psi_p': 0.0,
                                'psi_c': 0.0
                                }

    def __init__(self, model: Model, **solver_settings):
        self._model = model
        self._assembly = model.get_assembly()
        self._nb_dofs = self._assembly.get_nb_dofs()
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()
        self._loaded_dof_indices_step_list = (self._model.get_loaded_dof_indices_preloading_step_list()
                                              + [self._model.get_loaded_dof_indices()])
        self._solver_settings = StaticSolver._DEFAULT_SOLVER_SETTINGS.copy()
        self._solver_settings.update(solver_settings)

    def solve(self) -> Result:
        step_force_vectors = self._model.get_force_vectors_preloading_step_list() + [self._model.get_force_vector()]
        max_displacement_map_step_list = self._model.get_max_displacement_map_preloading_step_list() + [
            self._model.get_max_displacement_map()]
        u, f, stability, eigval_stats, step_indices = self._solve_with_arclength(step_force_vectors,
                                                                                 max_displacement_map_step_list,
                                                                                 **self._solver_settings)
        if u.ndim == 2:
            self._assembly.increment_general_coordinates(-u[-1, :])
        return Result(self._model, u, f, stability, eigval_stats, step_indices)

    def guide_spring_assembly_to_natural_configuration(self):
        initial_coordinates = self._assembly.get_general_coordinates()

        def elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_general_coordinates(coordinates)
            return self._assembly.compute_elastic_energy()

        def gradient_elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_general_coordinates(coordinates)
            return self._assembly.compute_internal_nodal_forces()[self._free_dof_indices]

        def hessian_elastic_energy(free_coordinates):
            coordinates = initial_coordinates.copy()
            coordinates[self._free_dof_indices] = free_coordinates
            self._assembly.set_general_coordinates(coordinates)
            return self._assembly.compute_structural_stiffness_matrix()[
                np.ix_(self._free_dof_indices, self._free_dof_indices)]

        try:
            result = minimize(elastic_energy, initial_coordinates[self._free_dof_indices],
                              jac=gradient_elastic_energy,
                              method='BFGS', tol=1e-6, options={'disp': False})
            natural_coordinates = np.empty(self._nb_dofs)
            natural_coordinates[self._free_dof_indices] = result.x
            natural_coordinates[self._fixed_dof_indices] = initial_coordinates[self._fixed_dof_indices]
            self._assembly.set_general_coordinates(natural_coordinates)
        except IllDefinedShape:
            self._assembly.set_general_coordinates(initial_coordinates)

    def _solve_with_arclength(self, force_vector_step_list, max_displacement_map_step_list, show_warnings,
                              reference_load_parameter, radius, i_max, j_max, convergence_value, verbose,
                              alpha, psi_p, psi_c):
        """
            Find equilibrium path using the arc-length method
        """
        # warnings.filterwarnings('ignore', category=LinAlgWarning)
        start = time.time()
        nb_steps = len(force_vector_step_list)
        if verbose:
            update_progress(f'Solving progress (step {0}/{nb_steps})', 0.0, 0, i_max, status='...')

        equilibrium_forces = [np.zeros(self._nb_dofs)]
        equilibrium_displacements = [np.zeros(self._nb_dofs)]
        step_indices: list[int] = [0]
        try:
            initial_ks = self._assembly.compute_structural_stiffness_matrix()
            initial_loaded_dof_indices = self._loaded_dof_indices_step_list[0]
            equilibrium_eigval_stats = [self._compute_lowest_eigenvalues_and_count_negative_ones(initial_ks,
                                                                                                 initial_loaded_dof_indices)]
        except IllDefinedShape:
            return (np.array([np.nan]), np.array([np.nan]), np.array(['nan'], dtype=str),
                    np.array([np.nan]), np.array([0], dtype=int))

        equilibrium_stability = [self._assess_stability(initial_ks, initial_loaded_dof_indices)]
        stiffness_matrix_eval_counter = 1
        linear_system_solving_counter = 0
        total_nb_increment_retries = 0
        total_nb_matrix_perturbations = 0
        initial_radius_p = radius
        radius_p = initial_radius_p
        delta_s = 1.0
        f_ext = np.zeros(self._nb_dofs)
        u = np.zeros(self._nb_dofs)
        i = 0
        force_progress = 0.0
        current_step = None
        try:
            for step, step_force_vector in enumerate(force_vector_step_list):
                current_step = step + 1
                loaded_dof_indices = self._loaded_dof_indices_step_list[step]
                force_reached_at_previous_step = f_ext.copy()
                delta_f = reference_load_parameter * step_force_vector
                previous_delta_u_inc = None
                previous_delta_lambda_inc = None
                increment_retries = 0
                force_progress = 0.0
                if verbose:
                    update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max,
                                    status='...', stability=equilibrium_stability[-1])

                while True:
                    has_increment_converged = False

                    # Predictor phase of increment i
                    delta_u_inc = np.zeros(self._nb_dofs)
                    delta_lambda_inc = 0.0
                    ks = self._assembly.compute_structural_stiffness_matrix()
                    stiffness_matrix_eval_counter += 1
                    k = self._get_reduced_stiffness_matrix(ks)
                    g = self._get_reduced_vector(delta_f)
                    system_solved = False
                    delta_u_hat = None
                    while not system_solved:
                        try:
                            delta_u_hat = self._get_structural_displacements(solve_linear(k, g, assume_a='sym'))
                            system_solved = True
                        except np.linalg.LinAlgError:
                            if i == 0:
                                raise MechanismDetected
                            else:
                                k = _perturb_singular_stiffness_matrix(k, 1e-5, show_warnings)
                                total_nb_matrix_perturbations += 1
                    linear_system_solving_counter += 1
                    delta_u_bar = 0.0

                    # Root computation and selection
                    delta_lambda_ite = radius_p / np.sqrt(np.inner(delta_u_hat, delta_u_hat) + psi_p ** 2)
                    root_choice_criteria = previous_delta_lambda_inc is None \
                                           or np.inner(delta_u_hat, previous_delta_u_inc) + (
                                                   psi_p ** 2) * previous_delta_lambda_inc >= 0
                    root_sign = +1 if root_choice_criteria else -1
                    delta_lambda_ite *= root_sign

                    # Updating loads, displacements and structure + computation of the unbalanced forces
                    delta_u_ite = delta_s * delta_u_bar + delta_lambda_ite * delta_u_hat
                    delta_u_inc += delta_u_ite
                    delta_lambda_inc += delta_lambda_ite
                    self._assembly.increment_general_coordinates(delta_u_ite)
                    f_ext += (delta_lambda_ite * delta_f)
                    f_int = self._assembly.compute_internal_nodal_forces()
                    r = f_int - f_ext
                    # Convergence check and preparation for next increment
                    if np.linalg.norm(self._get_reduced_vector(r)) / np.linalg.norm(
                            self._get_reduced_vector(delta_f)) < convergence_value:
                        has_increment_converged = True

                    if not has_increment_converged:
                        # Corrector phase (iterations)
                        radius_c = np.sqrt(np.inner(delta_u_hat, delta_u_hat) / (
                                np.inner(delta_u_hat, delta_u_hat) + psi_p ** 2)) * radius_p
                        rhom_radius_c = (1 - alpha) * radius_c
                        rhom_u_inc = (1 - alpha) * delta_u_inc
                        rhom_lambda_inc = (1 - alpha) * delta_lambda_inc
                        for j in range(j_max):
                            ks = self._assembly.compute_structural_stiffness_matrix()
                            stiffness_matrix_eval_counter += 1
                            k = self._get_reduced_stiffness_matrix(ks)
                            g = self._get_reduced_vector(delta_f)
                            uf = self._get_reduced_vector(r)
                            system_solved = False
                            while not system_solved:
                                try:
                                    delta_u_hat = self._get_structural_displacements(solve_linear(k, g, assume_a='sym'))
                                    system_solved = True
                                except np.linalg.LinAlgError:
                                    k = _perturb_singular_stiffness_matrix(k, 1e-5, show_warnings)
                                    total_nb_matrix_perturbations += 1

                            delta_u_bar = self._get_structural_displacements(solve_linear(k, -uf, assume_a='sym'))

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
                            rhom_u_inc += delta_u_ite
                            rhom_lambda_inc += delta_lambda_ite
                            self._assembly.increment_general_coordinates(delta_u_ite)
                            f_int = self._assembly.compute_internal_nodal_forces()
                            f_ext += delta_lambda_ite * delta_f
                            r = f_int - f_ext
                            # Convergence check and preparation for next increment
                            if np.linalg.norm(self._get_reduced_vector(r)) / np.linalg.norm(
                                    self._get_reduced_vector(delta_f)) < convergence_value:
                                has_increment_converged = True
                                break

                    # Preparation for the next increment
                    if has_increment_converged:
                        i += 1
                        increment_retries = 0
                        previous_delta_u_inc = delta_u_inc.copy()
                        previous_delta_lambda_inc = delta_lambda_inc
                        u += delta_u_inc
                        radius_p = min(initial_radius_p, radius_p * 2.0)
                        stability_state = self._assess_stability(ks, loaded_dof_indices)
                        step_indices.append(step)
                        equilibrium_displacements.append(u.copy())
                        equilibrium_forces.append(f_ext.copy())
                        equilibrium_stability.append(stability_state)
                        equilibrium_eigval_stats.append(
                            self._compute_lowest_eigenvalues_and_count_negative_ones(ks, loaded_dof_indices))
                    else:
                        increment_retries += 1
                        if increment_retries <= 5:
                            if show_warnings:
                                print(f"\nCorrection iterations did not converge for the increment {i + 1}"
                                      f"\t-> retry increment with smaller radius ({radius_p / 2.0:.3})"
                                      f"\t-> attempt {increment_retries}/{5}")
                            # Resetting structure to its previous incremental state with a smaller radius
                            total_nb_increment_retries += 1
                            self._assembly.increment_general_coordinates(-delta_u_inc)
                            f_ext -= delta_lambda_inc * delta_f
                            radius_p /= 2.0
                            continue  # go to the next increment
                        else:
                            raise ConvergenceError

                    # Check if final force has been reached. If yes, we can go to the next loading step
                    step_f_ext = f_ext - force_reached_at_previous_step
                    step_f_ext_norm = np.linalg.norm(self._get_reduced_vector(step_f_ext))
                    step_force_vector_norm = np.linalg.norm(self._get_reduced_vector(step_force_vector))
                    f_vectors_aligned = np.inner(self._get_reduced_vector(step_force_vector),
                                                 self._get_reduced_vector(step_f_ext)) > 0
                    final_force_has_been_reached = step_force_vector_norm - step_f_ext_norm < 0.0 and f_vectors_aligned
                    force_progress = step_f_ext_norm / step_force_vector_norm if f_vectors_aligned else 0.0
                    if final_force_has_been_reached:
                        if verbose:
                            reason = f'--> final force for loading step {current_step} has been reached'
                            reason += '\r\n'
                            update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i,
                                            i_max, reason, stability=equilibrium_stability[-1])
                        break  # go to next loading step

                    # Checking if max displacement has been reached. If yes, the solving process is aborted.
                    if max_displacement_map_step_list[step] is not None:
                        for dof_index, max_displacement in max_displacement_map_step_list[step].items():
                            if abs(u[dof_index]) >= abs(max_displacement) and u[dof_index] * max_displacement >= 0.0:
                                raise MaxDisplacementReached

                    # Checking if the max number of iteration has been reached. If yes, the solving process is aborted.
                    if i == i_max:
                        raise MaxNbIterationReached

                    if verbose:
                        update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max,
                                        '...', stability=equilibrium_stability[-1])

            if verbose:
                _print_message_with_final_solving_stats('Full equilibrium path was retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)

        except MaxDisplacementReached:
            if verbose:
                reason = f'--> max displacement has been reached'
                reason += '\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was '
                                                        'only retrieved up to the maximum displacement',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)

        except MaxNbIterationReached:
            if verbose:
                reason = '--> max nb of increments has been reached\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)
        except MechanismDetected:
            if verbose:
                reason = ('--> aborted because initial stiffness matrix is singular. Boundary conditions allow '
                          'rigid-body modes, or the initial equilibrium configuration is at a critical point.\r\n')
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)
        except ConvergenceError:
            if verbose:
                reason = '--> aborted (could not converge)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)
        except IllDefinedShape:
            if verbose:
                reason = '--> aborted (the shape of an element has become ill-defined)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                _print_message_with_final_solving_stats('Full equilibrium path was not retrieved',
                                                        i, stiffness_matrix_eval_counter, linear_system_solving_counter,
                                                        total_nb_increment_retries, total_nb_matrix_perturbations)
        # except LinAlgWarning:
        #     pass

        end = time.time()
        if verbose:
            print(f"Solving duration: {end - start:.4f} s")

        return (np.array(equilibrium_displacements), np.array(equilibrium_forces),
                np.array(equilibrium_stability, dtype=str), np.array(equilibrium_eigval_stats),
                np.array(step_indices, dtype=int))

    def _compute_lowest_eigenvalues_and_count_negative_ones(self, ks, loaded_dof_indices):
        # force-driven case
        k_ = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
        k_ = np.delete(k_, self._fixed_dof_indices, axis=1)  # delete columns
        eigvals = np.linalg.eigvalsh(k_)
        fd_lowest_eigval = eigvals[0]
        fd_negative_eigval_count = np.sum(eigvals < 0.0)

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

    def _assess_stability(self, ks, loaded_dof_indices):
        try:
            k_ = np.delete(ks, self._fixed_dof_indices, axis=0)  # delete rows
            k_ = np.delete(k_, self._fixed_dof_indices, axis=1)  # delete columns
            np.linalg.cholesky(k_)
            # if no error is triggered, then matrix is positive definite
            return StaticSolver.STABLE
        except np.linalg.linalg.LinAlgError:
            k_ = np.delete(ks, self._fixed_dof_indices + loaded_dof_indices, axis=0)  # delete rows
            k_ = np.delete(k_, self._fixed_dof_indices + loaded_dof_indices, axis=1)  # delete columns
            try:
                np.linalg.cholesky(k_)
                # if no error is triggered, then matrix is positive definite
                return StaticSolver.STABILIZABLE
            except np.linalg.linalg.LinAlgError:
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
                                            nb_matrix_perturbations):
    print(f"{message}: "
          f"increments = {nb_increments}"
          f" | stiffness matrix eval = {nb_stiffness_matrix_eval}"
          f" | linear system resolutions = {nb_linear_system_resolutions}"
          f" | increment retries = {nb_increment_retries}"
          f" | stiffness matrix perturbations = {nb_matrix_perturbations}")


def _perturb_singular_stiffness_matrix(k, epsilon, show_message):
    frobenius_norm = np.linalg.norm(k, 'fro')
    epsilon_frobenius = epsilon * frobenius_norm
    if show_message:
        print(f'\nStiffness matrix was exactly singular.'
              f'\n\tApplying a small perturbation ({epsilon_frobenius})'
              f'\ton the diagonal elements')
    return k + np.eye(*k.shape) * epsilon_frobenius


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
