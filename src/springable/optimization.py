from .mechanics import static_solver
from .readwrite import fileio as io
from .graphics.default_graphics_settings import DEFAULT_GENERAL_OPTIONS
from .graphics import animation
from .mechanics import model
from . import visualization
import numpy as np
import os.path
from scipy.optimize import minimize, direct, differential_evolution
import cma

def create_objective_function(model_path: str, design_parameter_names, a_funs: list[callable], b_fun: callable,
                              nb_samples: int):
    x = np.linspace(0, 1, nb_samples)
    a = [a_fun(x) for a_fun in a_funs]
    b = b_fun(x)

    def objective_function(design_parameter_values):
        design_parameters = {design_parameter_names[i]: design_parameter_values[i]
                             for i in range(len(design_parameter_names))}

        mdl = io.read_model(model_path, design_parameters)
        assmb = mdl.get_assembly()
        dof_indices = assmb.get_free_dof_indices()
        q0 = assmb.get_coordinates()
        error = 0.0
        for i in range(nb_samples):
            q = q0.copy()
            q[dof_indices] += [a[0][i], a[1][i]]
            assmb.set_coordinates(q)
            internal_force_vector = assmb.compute_elastic_force_vector()[dof_indices]
            external_force_vector = np.array([0.0, b[i]])
            oob_force_vector = (internal_force_vector - external_force_vector)
            # oob_force_vector[0] *= 0
            # error = oob_force_vector[0]**2
            #error += (nb_samples-i)*np.inner(oob_force_vector, oob_force_vector)
            error += np.inner(oob_force_vector, oob_force_vector)

        assmb.set_coordinates(q0)
        return error

    return objective_function


def optimize(model_path: str, a: list[callable], b: callable, nb_samples: int):
    _, design_parameters_data = io.read_parameters_from_model_file(model_path)
    design_parameter_names = list(design_parameters_data.keys())
    initial_values = np.array([design_parameters_data[name]['default value'] for name in design_parameter_names])
    lower_bounds = np.array([design_parameters_data[name]['lower bound'] for name in design_parameter_names])
    upper_bounds = np.array([design_parameters_data[name]['upper bound'] for name in design_parameter_names])
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(design_parameter_names))]

    objective_fun = create_objective_function(model_path, design_parameter_names, a, b, nb_samples)
    # Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, COBYLA, and COBYQA
    result = minimize(objective_fun, x0=initial_values, method='trust-constr', bounds=bounds, options={'verbose': 3,
        #'iprint':1,
        'disp': True,
    'maxiter': 10000})
    optimal_parameters = result.x
    return {design_parameter_names[i]: optimal_parameters[i] for i in range(len(design_parameter_names))}


def stochastically_optimize(model_path: str, a: list[callable], b: callable, nb_samples: int):
    _, design_parameters_data = io.read_parameters_from_model_file(model_path)
    design_parameter_names = list(design_parameters_data.keys())
    initial_values = np.array([design_parameters_data[name]['default value'] for name in design_parameter_names])
    lower_bounds = np.array([design_parameters_data[name]['lower bound'] for name in design_parameter_names])
    upper_bounds = np.array([design_parameters_data[name]['upper bound'] for name in design_parameter_names])
    objective_fun = create_objective_function(model_path, design_parameter_names, a, b, nb_samples)

    sigma = 1
    es = cma.CMAEvolutionStrategy(initial_values, sigma,
                                  {'bounds': [lower_bounds, upper_bounds],
                                   #'CSA_dampfac': 0.5,
                                   #'maxiter': 10e3,  # Very large number of iterations
                                   # 'maxfevals': 1e12,  # Very large number of function evaluations
                                   # 'tolfun': 1e-12,  # Extremely small tolerance for function value change
                                   # 'tolx': 1e-12,  # Extremely small tolerance for parameter change
                                   # 'tolfunhist': 1e-12  # Extremely small tolerance for function history

                                   }
                                  )
    es.optimize(objective_fun)

    # Retrieve the results
    best_solution = es.result.xbest
    print("Best solution found:", best_solution)
    print("Function value at best solution:", objective_fun(best_solution))
    return {design_parameter_names[i]: best_solution[i] for i in range(len(design_parameter_names))}


def globally_optimize(model_path: str, a: list[callable], b: callable, nb_samples: int):
    _, design_parameters_data = io.read_parameters_from_model_file(model_path)
    design_parameter_names = list(design_parameters_data.keys())
    initial_values = np.array([design_parameters_data[name]['default value'] for name in design_parameter_names])
    lower_bounds = np.array([design_parameters_data[name]['lower bound'] for name in design_parameter_names])
    upper_bounds = np.array([design_parameters_data[name]['upper bound'] for name in design_parameter_names])
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(design_parameter_names))]

    objective_fun = create_objective_function(model_path, design_parameter_names, a, b, nb_samples)
    result = differential_evolution(objective_fun, x0=initial_values, bounds=bounds, disp=True)

    optimal_parameters = result.x
    return {design_parameter_names[i]: optimal_parameters[i] for i in range(len(design_parameter_names))}
