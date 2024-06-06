"""
Author: Paul Ducarme
Static Truss Problem Solver class
"""
import numpy as np
from scipy.integrate import solve_ivp
from . import dynamics
import time as tm
import matplotlib.pyplot as plt


class DynamicSolver:
    """ Class representing a dynamic solver """

    _default_dynamic_solver_parameters = {'method': 'RK4',
                                          'rtol': 1e-6,
                                          'atol': 1e-6,
                                          't_max': 100,
                                          'verbose': True
                                          }

    def __init__(self, mdl: dynamics.DynamicModel):
        self._model = mdl
        self._external_force_function = mdl.get_force_vector_function()
        self._assembly = mdl.get_assembly()
        self._nb_dofs = self._assembly.get_nb_dofs()
        self._free_dof_indices = self._assembly.get_free_dof_indices()
        self._fixed_dof_indices = self._assembly.get_fixed_dof_indices()
        self._loaded_dof_indices = self._model.get_loaded_dof_indices_step_list()
        self._initial_positions = self._model.get_assembly().get_general_coordinates()
        self._initial_velocities = self._model.get_assembly().get_general_velocities()
        self._solving_algorithms = {'RK4': self._solve_with_runge_kutta4,
                                    'GA': self._solve_with_generalized_alpha}

    def to_structural_coordinates(self, free_coordinates):
        structural_coordinates = self._initial_positions.copy()
        structural_coordinates[self._free_dof_indices] = free_coordinates
        return structural_coordinates

    def to_structural_velocities(self, free_velocities):
        structural_velocities = self._initial_velocities.copy()
        structural_velocities[self._free_dof_indices] = free_velocities
        return structural_velocities

    def solve(self, **solver_parameters):
        _solver_parameters = DynamicSolver._default_dynamic_solver_parameters.copy()
        _solver_parameters.update(solver_parameters)
        method = _solver_parameters.pop('method')
        _solve = self._solving_algorithms[method]
        method_parameters = _solver_parameters
        time, q, dqdt, f = _solve(**method_parameters)
        return time, q, dqdt, f

    def _solve_with_runge_kutta4(self, t_max=10.0, rtol=1e-5, atol=1e-5, verbose=True):
        self._nb_free_dofs = len(self._free_dof_indices)
        self._masses = self._model.get_assembly().get_masses()[self._free_dof_indices]
        self._initial_positions = self._model.get_assembly().get_general_coordinates()
        self._initial_velocities = self._model.get_assembly().get_general_velocities()
        state0 = np.empty(2 * self._nb_free_dofs)
        state0[:self._nb_free_dofs] = self._initial_positions[self._free_dof_indices]
        state0[self._nb_free_dofs:] = self._initial_velocities[self._free_dof_indices]
        time0 = 0.0
        solution = solve_ivp(fun=self._ode_system, t_span=(time0, t_max), y0=state0, rtol=rtol, atol=atol)
        time = solution.t
        q = []
        dqdt = []
        f = []
        for i in range(len(time)):
            q.append(self.to_structural_coordinates(solution.y[:self._nb_free_dofs, i]))
            dqdt.append(self.to_structural_velocities(solution.y[self._nb_free_dofs:, i]))
            f.append(self._external_force_function(time[i]))

        self._model.get_assembly().set_general_coordinates(self._initial_positions)
        self._model.get_assembly().set_general_velocities(self._initial_velocities)
        return time, np.array(q), np.array(dqdt), np.array(f)

    def _ode_system(self, time, state):
        # apply state to model
        q = state[:self._nb_free_dofs]
        dqdt = state[self._nb_free_dofs:]
        self._model.get_assembly().set_general_coordinates(self.to_structural_coordinates(q))
        self._model.get_assembly().set_general_velocities(self.to_structural_velocities(dqdt))

        # compute forces from model
        internal_force_vector = self._model.get_assembly().compute_internal_nodal_forces()[self._free_dof_indices]
        external_force_vector = self._external_force_function(time)[self._free_dof_indices]

        # return derivative of state wrt time
        dsdt = np.empty(2 * self._nb_free_dofs)
        dsdt[:self._nb_free_dofs] = dqdt  # dqdt
        dsdt[self._nb_free_dofs:] = (external_force_vector - internal_force_vector) / self._masses  # d2qdt2
        return dsdt

    def _solve_with_generalized_alpha(self):
        pass
