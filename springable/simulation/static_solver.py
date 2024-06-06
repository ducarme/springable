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
                 step_indices=None):
        self._model = model
        self._u = equilibrium_displacements
        self._f = equilibrium_forces
        self._stability = equilibrium_stability
        self._eigval_stats = equilibrium_eigval_stats
        self._step_indices = step_indices if step_indices is not None else np.zeros(equilibrium_displacements.shape[0],
                                                                                    dtype=int)

        nb_steps = len(self._model.get_loading())
        self._starting_index = None
        if nb_steps == 1:
            if self._u.shape[0] in (0, 1, 2):
                self._is_loading_solution_unusable = True
            else:
                self._is_loading_solution_unusable = False
                self._starting_index = 0
        else:
            index = np.argmax(step_indices == nb_steps - 1)
            if index == 0:
                self._is_loading_solution_unusable = True
            elif self._u[index:].shape[0] in (0, 1, 2):
                self._is_loading_solution_unusable = True
            else:
                self._starting_index = index
                self._is_loading_solution_unusable = False

    def get_model(self):
        return self._model

    def get_forces(self, include_preloading=False):
        if include_preloading:
            return self._f
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._f[self._starting_index:]

    def get_node_forces(self, _node: Node, direction: str, include_preloading=False):
        if include_preloading:
            return self._f[:, self._model.get_assembly().get_dof_index(_node, direction)]
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._f[self._starting_index:, self._model.get_assembly().get_dof_index(_node, direction)]

    def get_displacements(self, include_preloading=False):
        if include_preloading:
            return self._u
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._u[self._starting_index:]

    def get_node_displacements(self, _node: Node, direction: str, include_preloading=False):
        if include_preloading:
            return self._u[:, self._model.get_assembly().get_dof_index(_node, direction)]
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._u[self._starting_index:, self._model.get_assembly().get_dof_index(_node, direction)]

    def get_stability(self, include_preloading=False):
        if include_preloading:
            return self._stability
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._stability[self._starting_index:]

    def get_eigenval_stats(self, include_preloading=False):
        if include_preloading:
            return self._eigval_stats
        else:
            if self._is_loading_solution_unusable:
                raise UnusableLoadingSolution
            return self._eigval_stats[self._starting_index:]

    def get_step_indices(self):
        return self._step_indices


class UnusableLoadingSolution(Exception):
    """ raise this when one attempts to get a solution for the loading phase from a Result instance, but it is not usable
    (in case the preloading could not be completed,
    or that the first iteration or the loading phase failed) """


class StaticSolver:
    """ Class representing a static solver """
    STABLE_UNDER_DISPLACEMENT_CONTROL_ONLY = 'stable under displacement control only'
    STABLE = 'stable'
    UNSTABLE = 'unstable'
    _default_static_solver_parameters = {'method': 'ALM',
                                         'reference_load_parameter': 0.05,
                                         'radius': 0.05,
                                         'alpha': 0.0,
                                         'psi_p': 0.0,
                                         'psi_c': 0.0,
                                         'convergence_value': 1e-6,
                                         'i_max': 20e3,
                                         'j_max': 20,
                                         'verbose': True
                                         }

    def __init__(self, model: Model):
        self._solving_algorithms = {'ALM': self._solve_with_arclength}
        self._model = model
        self._assembly = model.get_assembly()
        self._nb_dofs = self._assembly.get_nb_dofs()
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()
        self._loaded_dof_indices_step_list = (self._model.get_loaded_dof_indices_preloading_step_list()
                                              + [self._model.get_loaded_dof_indices()])

    def solve(self, **solver_parameters) -> Result:
        solver_param = StaticSolver._default_static_solver_parameters.copy()
        solver_param.update(solver_parameters)
        method = solver_param.pop('method')
        _solve = self._solving_algorithms[method]
        method_parameters = solver_param

        step_force_vectors = self._model.get_force_vectors_preloading_step_list() + [self._model.get_force_vector()]
        max_displacement_map_step_list = self._model.get_max_displacement_map_preloading_step_list() + [
            self._model.get_max_displacement_map()]
        u, f, stability, eigval_stats, step_indices = _solve(step_force_vectors, max_displacement_map_step_list,
                                                             **method_parameters)
        self._assembly.increment_general_coordinates(-u[-1, :])
        return Result(self._model, u, f, stability, eigval_stats, step_indices)

    def guide_truss_to_natural_configuration(self):
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

        result = minimize(elastic_energy, initial_coordinates[self._free_dof_indices],
                          jac=gradient_elastic_energy,
                          method='BFGS', tol=1e-6, options={'disp': False})
        natural_coordinates = np.empty(self._nb_dofs)
        natural_coordinates[self._free_dof_indices] = result.x
        natural_coordinates[self._fixed_dof_indices] = initial_coordinates[self._fixed_dof_indices]
        self._assembly.set_general_coordinates(natural_coordinates)

    def _solve_with_arclength(self, force_vector_step_list, max_displacement_map_step_list,
                              reference_load_parameter=0.1,
                              radius=0.1,
                              alpha=0.0, psi_p=0.0, psi_c=0.0,
                              convergence_value=1e-6, i_max=2000, j_max=20, smart_stop=False,
                              verbose=True):
        """
            Find equilibrium path using the arc-length method
        """
        # warnings.filterwarnings('ignore', category=LinAlgWarning)
        start = time.time()

        equilibrium_forces = [np.zeros(self._nb_dofs)]
        equilibrium_displacements = [np.zeros(self._nb_dofs)]
        step_indices: list[int] = [0]
        initial_ks = self._assembly.compute_structural_stiffness_matrix()
        initial_loaded_dof_indices = self._loaded_dof_indices_step_list[0]
        equilibrium_eigval_stats = [self._compute_lowest_eigenvalues_and_count_negative_ones(initial_ks,
                                                                                             initial_loaded_dof_indices)]
        equilibrium_stability = [self._assess_stability(initial_ks, initial_loaded_dof_indices)]
        initial_coordinates = self._assembly.get_general_coordinates()
        stiffness_matrix_eval_counter = 1
        linear_system_solving_counter = 0
        initial_radius_p = radius
        radius_p = initial_radius_p
        delta_s = 1.0
        f_ext = np.zeros(self._nb_dofs)
        u = np.zeros(self._nb_dofs)
        i = 0
        force_progress = 0.0
        nb_steps = len(force_vector_step_list)
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
                    delta_u_hat = self._get_structural_displacements(solve_linear(k, g, assume_a='sym'))
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
                            delta_u_hat = self._get_structural_displacements(solve_linear(k, g, assume_a='sym'))
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
                            if verbose:
                                pass
                                print(f"\nCorrection iterations did not converge for the increment {i + 1}")
                                print(f"\t-> retry increment with smaller radius ({radius_p / 2.0:.3})")
                                print(f"\t-> attempt {increment_retries}/{5}")
                            # Resetting structure to its previous incremental state with a smaller radius
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
                print(f"Full equilibrium path was retrieved"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")

        except MaxDisplacementReached:
            if verbose:
                reason = f'--> max displacement has been reached'
                reason += '\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                print(f"Full equilibrium path was only retrieved up to the maximum displacement:"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")

        except MaxNbIterationReached:
            if verbose:
                reason = '--> max nb of increments has been reached\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                print(f"Full equilibrium path was not retrieved:"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")

        except np.linalg.LinAlgError:
            if verbose:
                reason = ('--> aborted because initial stiffness matrix is singular. Boundary conditions allow '
                          'rigid-body modes, or the initial equilibrium configuration is at a critical point.\r\n')
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                print(f"Full equilibrium path was not retrieved:"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")
        except ConvergenceError:
            if verbose:
                reason = '--> aborted (could not converge)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                print(f"Full equilibrium path was not retrieved:"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")
        except IllDefinedShape:
            if verbose:
                reason = '--> aborted (the shape of an element has become ill-defined)\r\n'
                update_progress(f'Solving progress (step {current_step}/{nb_steps})', force_progress, i, i_max, reason,
                                stability=equilibrium_stability[-1])
                print(f"Full equilibrium path was not retrieved:"
                      f" increments = {i}"
                      f" | stiffness matrix eval = {stiffness_matrix_eval_counter}"
                      f" | linear system resolutions = {linear_system_solving_counter}")
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
                return StaticSolver.STABLE_UNDER_DISPLACEMENT_CONTROL_ONLY
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


class ConvergenceError(Exception):
    """ raise this when the algorithm cannot converge """


class MaxNbIterationReached(Exception):
    """ raise this when the max number of iteration is reached """


class MaxDisplacementReached(Exception):
    """ raise this when the max displacement is reached """


def update_progress(title, progress, i, i_max, status, stability=None):
    if stability == StaticSolver.STABLE:
        symbol = '#'
    elif stability == StaticSolver.STABLE_UNDER_DISPLACEMENT_CONTROL_ONLY:
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
