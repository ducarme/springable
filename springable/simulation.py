from .mechanics import static_solver
from .readwrite import fileio as io
from .graphics.default_graphics_settings import DEFAULT_GENERAL_OPTIONS, DEFAULT_PLOT_OPTIONS
from .graphics import visuals
from .graphics import plot
from .mechanics import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import os.path
import typing


def solve_model(_model: model.Model, **solver_settings) -> static_solver.Result:
    slv = static_solver.StaticSolver(_model, **solver_settings)
    slv.guide_truss_to_natural_configuration()
    return slv.solve()


def save_results(result: static_solver.Result, save_dir):
    io.write_results(result, save_dir)


def make_all_graphics(result, save_dir,
                      general_options: dict = None,
                      plot_options: dict = None,
                      animation_options: dict = None,
                      assembly_appearance: dict = None):
    go = DEFAULT_GENERAL_OPTIONS.copy()
    if general_options is not None:
        go.update(general_options)
    if plot_options is None:
        plot_options = {}
    if assembly_appearance is None:
        assembly_appearance = {}
    if animation_options is None:
        animation_options = {}
    if go['generate_model_drawing']:
        visuals.draw_model(result.get_model(), save_dir, 'model', show=False, **assembly_appearance)
    if go['generate_fd_plot']:
        plot.force_displacement_curve(result, save_dir, show=go['show_fd_plot'])
    if go['generate_animation']:
        visuals.animate(result, save_dir, show=go['show_animation'],
                        plot_options=plot_options, assembly_appearance=assembly_appearance,
                        **animation_options)


def simulate_model(model_path, save_dir, solver_settings_path=None, graphics_settings_path=None):
    # CREATE MAIN DIRECTORY WHERE RESULT WILL BE SAVED + COPY INPUT FILES
    save_dir = io.mkdir(save_dir)
    io.copy_model_file(save_dir, model_path)
    if solver_settings_path is not None:
        io.copy_solver_settings_file(save_dir, solver_settings_path)
    if graphics_settings_path is not None:
        io.copy_graphics_settings_file(save_dir, graphics_settings_path)

    mdl = io.read_model(model_path)
    if solver_settings_path is not None:
        solver_settings = io.read_solver_settings_file(solver_settings_path)
    else:
        solver_settings = {}

    general_options = DEFAULT_GENERAL_OPTIONS.copy()
    if graphics_settings_path is not None:
        graphics_settings = io.read_graphics_settings_file(graphics_settings_path)
        (custom_general_options,
         custom_plot_options,
         custom_animation_options,
         custom_assembly_appearance) = graphics_settings
    else:
        (custom_general_options,
         custom_plot_options,
         custom_animation_options,
         custom_assembly_appearance) = {}, {}, {}, {}
    general_options.update(custom_general_options)

    if general_options['generate_model_drawing']:
        visuals.draw_model(mdl, save_dir, show=general_options['show_model_drawing'], **custom_assembly_appearance)

    result = solve_model(mdl, **solver_settings)

    save_results(result, save_dir)
    try:
        make_all_graphics(result, save_dir,
                          general_options=custom_general_options,
                          plot_options=custom_plot_options,
                          animation_options=custom_animation_options,
                          assembly_appearance=custom_assembly_appearance)
    except static_solver.UnusableLoadingSolution:
        print("Cannot make the graphics, because the calculated equilibrium path is unusable")


def scan_parameter_space(model_path, save_dir, scan_parameters_one_by_one=True,
                         solver_settings_path=None, graphics_settings_path=None):
    if solver_settings_path is not None:
        solver_settings = io.read_solver_settings_file(solver_settings_path)
    else:
        solver_settings = {}

    general_options = DEFAULT_GENERAL_OPTIONS.copy()
    if graphics_settings_path is not None:
        graphics_settings = io.read_graphics_settings_file(graphics_settings_path)
        (custom_general_options,
         custom_plot_options,
         custom_animation_options,
         custom_assembly_appearance) = graphics_settings
    else:
        (custom_general_options,
         custom_plot_options,
         custom_animation_options,
         custom_assembly_appearance) = {}, {}, {}, {}
    general_options.update(custom_general_options)

    # CREATE MAIN DIRECTORY WHERE ALL SIMULATIONS WILL BE SAVED
    # + SUBDIRECTORIES FOR REQUIRED GRAPHICS
    # + COPY INPUT FILES
    save_dir = io.mkdir(os.path.join(save_dir))
    model_drawings_dir = None
    if general_options['generate_all_model_drawings']:
        model_drawings_dir = io.mkdir(os.path.join(save_dir, 'all_model_drawings'))
    fd_plots_dir = None
    if general_options['generate_all_fd_plots']:
        fd_plots_dir = io.mkdir(os.path.join(save_dir, 'all_fd_plots'))
    animations_dir = None
    if general_options['generate_all_animations']:
        animations_dir = io.mkdir(os.path.join(save_dir, 'all_animations'))
    io.copy_model_file(save_dir, model_path)
    if solver_settings_path is not None:
        io.copy_solver_settings_file(save_dir, solver_settings_path)
    if graphics_settings_path is not None:
        io.copy_graphics_settings_file(save_dir, graphics_settings_path)

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
                # prepare saving folder
                sim_name = f"sim{cnt}_{design_parameter_name}"
                subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))

                # running simulation with updated design parameters
                parameters = default_parameters.copy()
                parameters[design_parameter_name] = design_parameter_value
                mdl = io.read_model(model_path, parameters)

                if general_options['generate_all_model_drawings']:
                    visuals.draw_model(mdl, model_drawings_dir, save_name=sim_name,
                                       show=general_options['show_all_model_drawings'], **custom_assembly_appearance)

                # run simulation
                res = solve_model(mdl, **solver_settings)

                # save results and extra information
                save_results(res, subsave_dir)
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
            # running simulation with updated parameters
            design_parameters = dict(zip(design_parameter_names, design_parameter_combinations[i, :]))
            parameters.update(design_parameters)
            mdl = io.read_model(model_path, parameters)
            res = solve_model(mdl, **solver_settings)

            # saving
            sim_name = f"sim{i}"
            subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))
            save_results(res, subsave_dir)
            io.write_design_parameters(design_parameters, subsave_dir)

            # mapping design parameters to where results are saved
            subsave_dirs.append(subsave_dir)
            sim_names.append(sim_name)

    # MAKE GRAPHICS
    if scan_parameters_one_by_one:

        if general_options['generate_parametric_fd_plots']:
            _, design_parameter_data = io.read_parameters_from_model_file(model_path)

            def make_color_converter(design_par_name: str, dir_map: dict) -> mcm.ScalarMappable:
                par_data = design_parameter_data[design_par_name]
                if par_data['is range parameter']:
                    cmap = plt.get_cmap(custom_plot_options.get('range_parameter_scan_colormap',
                                                                DEFAULT_PLOT_OPTIONS['range_parameter_scan_colormap']))
                    lb = par_data['lower bound']
                    ub = par_data['upper bound']
                    cn = plt.Normalize(vmin=lb, vmax=ub, clip=True)
                else:
                    cmap = plt.get_cmap(custom_plot_options.get('discrete_parameter_scan_colormap',
                                                                DEFAULT_PLOT_OPTIONS[
                                                                    'discrete_parameter_scan_colormap']))
                    cn = plt.Normalize(vmin=0, vmax=par_data['nb samples'])
                return mcm.ScalarMappable(norm=cn, cmap=cmap)

            def colors(color_converter: mcm.ScalarMappable, design_par_name: str):
                par_data = design_parameter_data[design_par_name]
                if par_data['is range parameter']:
                    for _dir in dir_map[design_par_name]:
                        design_par_value = io.read_design_parameters(_dir)[design_par_name]
                        yield color_converter.to_rgba(design_par_value)
                else:
                    for i, _dir in enumerate(dir_map[design_par_name]):
                        yield color_converter.to_rgba(i)

            def need_labels(design_par_name):
                par_data = design_parameter_data[design_par_name]
                max_nb_values = custom_plot_options.get('max_nb_legend_entries_for_discrete_parameter',
                                                        DEFAULT_PLOT_OPTIONS[
                                                            'max_nb_legend_entries_for_discrete_parameter'])
                return not par_data['is range parameter'] and len(dir_map[design_par_name]) <= max_nb_values

            def labels(design_par_name):
                par_data = design_parameter_data[design_par_name]
                for i, _dir in enumerate(dir_map[design_par_name]):
                    yield f'{design_par_name} = {(par_data["all possible values"][i]):.4f}'

            def results(design_par_name) -> typing.Iterator[static_solver.Result]:
                for _dir in dir_map[design_par_name]:
                    yield io.read_results(_dir)

            for design_parameter_name in design_parameter_names:
                color_converter = make_color_converter(design_parameter_name, dir_map)
                par_data = design_parameter_data[design_parameter_name]
                if par_data['is range parameter']:
                    cbar_info = {'title': design_parameter_name, 'scalar_mappable': color_converter}
                else:
                    cbar_info = None
                plot.force_displacement_curve(results(design_parameter_name), save_dir,
                                              f'fd_curve_{design_parameter_name}',
                                              colors(color_converter, design_parameter_name),
                                              labels(design_parameter_name) if need_labels(design_parameter_name) else None,
                                              cbar_info=cbar_info
                                              )
