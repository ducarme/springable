from . import static_solver
from ..readwrite import fileio as io
from ..graphics.graphic_settings import GeneralOptions as GO
from ..graphics import visuals
from ..graphics import plot
from ..simulation import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import os.path
import typing


def simulate(mdl: model.Model) -> static_solver.Result:
    slv = static_solver.StaticSolver(mdl)
    slv.guide_truss_to_natural_configuration()
    return slv.solve()


def _save(result: static_solver.Result, save_dir):
    io.write_results(result, save_dir)


def _make_all_graphics(result, save_dir, exp_fd_curve=None):
    if GO.generate_model_drawing:
        visuals.draw_model(result.get_model(), save_dir, 'model', show=False)
    if GO.generate_fd_plot:
        plot.force_displacement_curve(result, save_dir, 'fd_curve', exp_fd_curve=exp_fd_curve, show=GO.show_fd_plot)
    if GO.generate_animation:
        visuals.animate(result, save_dir, save_name='animation', show=GO.show_animation)


def simulate_model(model_path, save_dir, exp_fd_curve=None):
    mdl = io.read_model(model_path)
    if GO.generate_model_drawing and GO.show_model_drawing:
        visuals.draw_model(mdl, show=True)
    result = simulate(mdl)
    save_dir = io.mkdir(save_dir)
    io.copy_model_file(save_dir, model_path)
    _save(result, save_dir)
    try:
        _make_all_graphics(result, save_dir, exp_fd_curve)
    except static_solver.UnusableLoadingSolution:
        print("Cannot make the graphics, because the calculated equilibrium path is unusable")


def scan_parameter_space(model_path, save_dir, scan_parameters_one_by_one=True, exp_fd_curve=None):
    # CREATE MAIN DIRECTORY WHERE ALL SIMULATIONS WILL BE SAVED
    save_dir = io.mkdir(os.path.join(save_dir))
    io.copy_model_file(save_dir, model_path)

    # READ DEFAULT AND DESIGN PARAMETERS
    default_parameters, design_parameter_data = io.read_parameters_from_model_file(model_path)
    design_parameter_names = list(design_parameter_data.keys())
    design_parameter_vectors = []
    for design_parameter_name in design_parameter_names:
        if design_parameter_data[design_parameter_name]['is range parameter']:
            design_parameter_vector = np.linspace(design_parameter_data[design_parameter_name]['lower bound'],
                                                  design_parameter_data[design_parameter_name]['upper bound'],
                                                  design_parameter_data[design_parameter_name]['nb samples'])
        else:
            design_parameter_vector = design_parameter_data[design_parameter_name]['all possible values']
        design_parameter_vectors.append(design_parameter_vector)

    subsave_dirs = []
    sim_names = []
    dir_map = {}
    if scan_parameters_one_by_one:
        cnt = 0
        dir_map = {design_parameter_name: [] for design_parameter_name in design_parameter_names}
        sim_names_map = {design_parameter_name: [] for design_parameter_name in design_parameter_names}
        for design_parameter_index, design_parameter_name in enumerate(design_parameter_names):
            for design_parameter_value in design_parameter_vectors[design_parameter_index]:
                # running simulation with updated design parameters
                parameters = default_parameters.copy()
                parameters[design_parameter_name] = design_parameter_value
                mdl = io.read_model(model_path, parameters)
                res = simulate(mdl)

                # saving
                sim_name = f"sim{cnt}_{design_parameter_name}"
                subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))
                _save(res, subsave_dir)
                design_parameters = {design_parameter_name: parameters[design_parameter_name]
                                     for design_parameter_name in design_parameter_names}
                io.write_design_parameters(design_parameters, subsave_dir)

                # mapping design parameters to where results are saved
                dir_map[design_parameter_name].append(subsave_dir)
                sim_names_map[design_parameter_name].append(sim_name)
                subsave_dirs.append(subsave_dir)
                sim_names.append(sim_name)
                cnt += 1
    else:
        design_parameter_combinations = np.stack(np.meshgrid(*design_parameter_vectors), -1).reshape(-1,
                                                                                                     len(design_parameter_names))
        nb_combinations = design_parameter_combinations.shape[0]
        parameters = default_parameters.copy()
        for i in range(nb_combinations):
            # running simulaiton with updated parameters
            design_parameters = dict(zip(design_parameter_names, design_parameter_combinations[i, :]))
            parameters.update(design_parameters)
            mdl = io.read_model(model_path, parameters)
            res = simulate(mdl)

            # saving
            sim_name = f"sim{i}"
            subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))
            _save(res, subsave_dir)
            io.write_design_parameters(design_parameters, subsave_dir)

            # mapping design parameters to where results are saved
            subsave_dirs.append(subsave_dir)
            sim_names.append(sim_name)

    if scan_parameters_one_by_one:
        _, design_parameter_data = io.read_parameters_from_model_file(model_path)

        def colors(design_par_name):
            par_data = design_parameter_data[design_parameter_name]
            if par_data['is range parameter']:
                cmap = plt.get_cmap('viridis')
                lb = par_data['lower bound']
                ub = par_data['upper bound']
                cn = plt.Normalize(vmin=lb, vmax=ub, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cmap)
                for _dir in dir_map[design_par_name]:
                    design_par_value = io.read_design_parameters(_dir)[design_par_name]
                    yield sm.to_rgba(design_par_value)
            else:
                cmap = plt.get_cmap('tab10')
                for i, _dir in enumerate(dir_map[design_par_name]):
                    yield cmap(i)

        def results(design_par_name) -> typing.Iterator[static_solver.Result]:
            for _dir in dir_map[design_par_name]:
                yield io.read_results(_dir)

        for design_parameter_name in design_parameter_names:
            plot.force_displacement_curve(results(design_parameter_name), save_dir, f'fd_curve_{design_parameter_name}',
                                          colors(design_parameter_name), exp_fd_curve=exp_fd_curve)
