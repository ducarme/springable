from .mechanics import static_solver
from .readwrite import fileio as io
from .graphics.default_graphics_settings import GeneralOptions
from .graphics import animation
from .mechanics import model
from . import visualization
import numpy as np
import os.path


def solve_model(_model: model.Model | str, solver_settings: dict | str = None) -> static_solver.Result:
    if not isinstance(_model, model.Model):
        _model = io.read_model(_model)
        if not isinstance(_model, model.Model):
            raise ValueError('Invalid first argument. It should be the model file path or a Model object.')
    if solver_settings is not None:
        if isinstance(solver_settings, str):
            solver_settings = io.read_solver_settings_file(solver_settings)
        if not isinstance(solver_settings, dict):
            ValueError('Invalid second argument. When specified, it should be the solver-settings file path, '
                       'or a dictionary of settings')
    else:
        solver_settings = {}
    slv = static_solver.StaticSolver(_model, **solver_settings)
    slv.guide_spring_assembly_to_natural_configuration()
    return slv.solve()


def save_results(result: static_solver.Result, save_dir):
    io.write_results(result, save_dir)


def simulate_model(model_path, save_dir=None, solver_settings_path=None, graphics_settings_path=None,
                   postprocessing=None):
    # CREATE MAIN DIRECTORY WHERE RESULT WILL BE SAVED + COPY INPUT FILES
    if save_dir is None:
        save_dir = io.mkdir(os.path.splitext(os.path.basename(model_path))[0])
    else:
        save_dir = io.mkdir(save_dir)
    io.copy_model_file(save_dir, model_path)
    if solver_settings_path is not None:
        io.copy_solver_settings_file(save_dir, solver_settings_path)
    if graphics_settings_path is not None:
        io.copy_graphics_settings_file(save_dir, graphics_settings_path)

    # READ INPUT FILES
    mdl = io.read_model(model_path)
    general_options = GeneralOptions()
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
    general_options.update(**custom_general_options)

    if general_options.generate_model_drawing:
        if general_options.show_model_drawing:
            print("Model is drawn in new window. Close the window to continue with the simulation...")
        animation.draw_model(mdl, save_dir, show=general_options.show_model_drawing, **custom_assembly_appearance)

    result = solve_model(mdl, solver_settings_path)
    save_results(result, save_dir)
    print(f"Simulation results have been saved in {save_dir}.")
    custom_general_options['generate_model_drawing'] = False
    custom_general_options['show_model_drawing'] = False
    visualization.visualize_result(result, save_dir,
                                   graphics_settings=[custom_general_options,
                                                      custom_plot_options,
                                                      custom_animation_options,
                                                      custom_assembly_appearance],
                                   postprocessing=postprocessing)

    return save_dir


def scan_parameter_space(model_path, save_dir=None, scan_parameters_one_by_one=True,
                         solver_settings_path=None, graphics_settings_path=None, postprocessing=None):
    if solver_settings_path is not None:
        solver_settings = io.read_solver_settings_file(solver_settings_path)
    else:
        solver_settings = {}

    general_options = GeneralOptions()
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
    general_options.update(**custom_general_options)

    # CREATE MAIN DIRECTORY WHERE ALL SIMULATIONS WILL BE SAVED
    # + SUBDIRECTORY TO STORE MODEL DRAWINGS
    # + COPY INPUT FILES
    if save_dir is None:
        save_dir = io.mkdir(os.path.splitext(os.path.basename(model_path))[0])
    else:
        save_dir = io.mkdir(save_dir)
    model_drawings_dir = None
    if general_options.generate_all_model_drawings:
        model_drawings_dir = io.mkdir(os.path.join(save_dir, 'all_model_drawings'))
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

    all_sim_names = []
    par_name_to_sim_names = {}
    if scan_parameters_one_by_one:
        cnt = 0
        nb_simulations = sum([design_parameter_data[par_name]['nb samples']] for par_name in design_parameter_names)
        par_name_to_sim_names = {design_parameter_name: [] for design_parameter_name in design_parameter_names}
        for design_parameter_index, design_parameter_name in enumerate(design_parameter_names):
            for design_parameter_value in design_parameter_vectors[design_parameter_index]:
                print(f'Simulation {cnt+1}/{nb_simulations}')
                # prepare saving folder
                sim_name = f"sim{cnt}_{design_parameter_name}"
                subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))

                # running simulation with updated design parameters
                parameters = default_parameters.copy()
                parameters[design_parameter_name] = design_parameter_value
                mdl = io.read_model(model_path, parameters)

                if general_options.generate_all_model_drawings:
                    animation.draw_model(mdl, model_drawings_dir, save_name=sim_name,
                                         show=general_options.show_all_model_drawings, **custom_assembly_appearance)

                # run simulation
                res = solve_model(mdl, solver_settings)

                # save results and extra information
                save_results(res, subsave_dir)
                design_parameters = {design_parameter_name: parameters[design_parameter_name]
                                     for design_parameter_name in design_parameter_names}
                io.write_design_parameters(design_parameters, subsave_dir)

                # mapping design parameters to where results are saved
                par_name_to_sim_names[design_parameter_name].append(sim_name)
                all_sim_names.append(sim_name)
                cnt += 1
    else:
        design_parameter_combinations = np.stack(np.meshgrid(*design_parameter_vectors), -1).reshape(-1,
                                                                                                     len(design_parameter_names))
        nb_combinations = design_parameter_combinations.shape[0]
        parameters = default_parameters.copy()
        for i in range(nb_combinations):
            print(f'Simulation {i+1}/{nb_combinations}')

            # preparing for saving
            sim_name = f"sim{i}"
            subsave_dir = io.mkdir(os.path.join(save_dir, sim_name))

            # running simulation with updated parameters
            design_parameters = dict(zip(design_parameter_names, design_parameter_combinations[i, :]))
            parameters.update(design_parameters)
            mdl = io.read_model(model_path, parameters)

            if general_options.generate_all_model_drawings:
                animation.draw_model(mdl, model_drawings_dir, save_name=sim_name,
                                     show=general_options.show_all_model_drawings, **custom_assembly_appearance)

            # run simulation
            res = solve_model(mdl, solver_settings)

            # saving
            save_results(res, subsave_dir)
            io.write_design_parameters(design_parameters, subsave_dir)

            # mapping design parameters to where results are saved
            all_sim_names.append(sim_name)

    general_info = {
        'MODEL_FILENAME': os.path.basename(model_path),
        'ALL_SIM_NAMES': all_sim_names,
        'PARAMETERS_SCANNED_ONE_BY_ONE': 'yes' if scan_parameters_one_by_one else 'no',
        'PARAMETER_NAME_TO_SIM_NAMES_MAPPING': par_name_to_sim_names
    }
    io.write_scanning_general_info(general_info, save_dir)

    # make graphics
    custom_general_options['general_all_model_drawings'] = False
    visualization.visualize_scan_results(save_dir, save_dir, [custom_general_options,
                                                              custom_plot_options,
                                                              custom_animation_options,
                                                              custom_assembly_appearance],
                                         postprocessing)
