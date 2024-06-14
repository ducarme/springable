from .graphics.default_graphics_settings import DEFAULT_GENERAL_OPTIONS
from .graphics import plot, animation
from .readwrite import fileio as io
from .mechanics import static_solver, model
import sys
import numpy as np
import os.path


def _results(save_dirs: list[str]):
    for save_dir in save_dirs:
        yield io.read_results(save_dir)


def _print_progress(title, index, length):
    progress = float(index + 1) / length
    bar_length = 10
    if progress < 0.0:
        progress = 0.0
    if progress >= 1:
        progress = 1.0
    block = int(round(bar_length * progress))
    text = "\r{0}: [{1}] ({2}/{3})".format(title, "#" * block + "-" * (bar_length - block), index + 1, length)
    sys.stdout.write(text)
    sys.stdout.flush()


def _load_graphics_settings(graphics_settings):
    if graphics_settings is not None:
        if isinstance(graphics_settings, str):
            graphics_settings = io.read_graphics_settings_file(graphics_settings)
        valid = isinstance(graphics_settings, (tuple, list))
        if valid:
            for options in graphics_settings:
                if not isinstance(options, dict):
                    valid = False
                    break
        if not valid:
            raise ValueError("Incorrect graphics settings specification. If specified, the last argument must"
                             "be the path to the graphic-settings file, or an already loaded graphic-settings"
                             "variable, that is a list or tuple of 4 dictionaries")
    else:
        graphics_settings = {}, {}, {}, {}
    return graphics_settings


def _load_result(_result):
    if isinstance(_result, str):
        _result = io.read_results(_result)
    if not isinstance(_result, static_solver.Result):
        raise ValueError("Incorrect result specification. The first argument should be the directory where the result "
                         "files are stored, or an already loaded Result object")
    return _result


def visualize_scan_results(scan_results_dir: str, save_dir: str = '',
                           graphics_settings: list[dict] | tuple[dict] | str = None):
    general_options, plot_options, animation_options, assembly_appearance = _load_graphics_settings(graphics_settings)
    go = DEFAULT_GENERAL_OPTIONS.copy()
    go.update(general_options)

    # Somehow, the variables herein below should be automatically read from scan_results_directory
    general_info = io.read_scanning_general_info(scan_results_dir)
    model_path = os.path.join(scan_results_dir, general_info['MODEL_FILENAME'])
    all_sim_names = general_info['ALL_SIM_NAMES']
    scan_parameters_one_by_one = general_info['PARAMETERS_SCANNED_ONE_BY_ONE'] == 'yes'

    # MAKE GRAPHICS
    if scan_parameters_one_by_one:
        par_name_to_sim_names = general_info['PARAMETER_NAME_TO_SIM_NAMES_MAPPING']
        if go['generate_parametric_fd_plots']:
            _, par_name_to_par_data = io.read_parameters_from_model_file(model_path)

            for design_parameter_name, sim_names in par_name_to_sim_names.items():
                subsave_dirs = [os.path.join(scan_results_dir, sim_name) for sim_name in sim_names]
                parameter_values = [io.read_design_parameters(subsave_dir)[design_parameter_name]
                                    for subsave_dir in subsave_dirs]

                plot.parametric_force_displacement_curve(_results(subsave_dirs), design_parameter_name,
                                                         par_name_to_par_data[design_parameter_name], parameter_values,
                                                         save_dir, save_name=f'fd_curve_{design_parameter_name}',
                                                         show=go['show_parametric_fd_plots'], **plot_options)

    # first scanning of result folders to obtain axes limits for force-displacement plots
    min_u, max_u, min_f, max_f = [None] * 4
    if go['generate_all_fd_plots']:
        min_u, max_u, min_f, max_f = +np.inf, -np.inf, +np.inf, -np.inf
        for sim_name in all_sim_names:
            res = io.read_results(os.path.join(scan_results_dir, sim_name))
            try:
                extrema = res.get_min_and_max_loading_displacement_and_force()
                min_u = min(min_u, extrema[0])
                max_u = max(max_u, extrema[1])
                min_f = min(min_f, extrema[2])
                max_f = max(max_f, extrema[3])
            except static_solver.UnusableSolution:
                pass

    # scanning result folders to make necessary graphics
    if go['generate_all_model_drawings'] or go['generate_all_fd_plots'] or go['generate_all_animations']:
        print('Generating all graphics...')
        drawings_dir = io.mkdir(os.path.join(save_dir, 'all_drawings')) if go['generate_all_model_drawings'] else None
        fd_plots_dir = io.mkdir(os.path.join(save_dir, 'all_fd_plots')) if go['generate_all_fd_plots'] else None
        animations_dir = io.mkdir(os.path.join(save_dir, 'all_animations')) if go['generate_all_animations'] else None

        for i, sim_name in enumerate(all_sim_names):
            try:
                res = io.read_results(os.path.join(scan_results_dir, sim_name))
                if drawings_dir is not None:
                    animation.draw_model(res.get_model(), drawings_dir,
                                         save_name=sim_name, show=False, **assembly_appearance)
                if fd_plots_dir is not None:
                    plot.force_displacement_curve(res, fd_plots_dir, sim_name, show=False,
                                                  xlim=(min_u, max_u), ylim=(min_f, max_f), **plot_options)
                if animations_dir is not None:
                    animation.animate(res, animations_dir, sim_name, show=False,
                                      plot_options=plot_options, assembly_appearance=assembly_appearance,
                                      **animation_options)
            except static_solver.UnusableSolution:
                pass
            print(f'Postprocessed simulations: {i + 1}/{len(all_sim_names)}')
        print('All graphics generated successfully')


def visualize_result(result: static_solver.Result | str, save_dir: str = '',
                     graphics_settings: list | tuple | str = None):
    result = _load_result(result)
    general_options, plot_options, animation_options, assembly_appearance = _load_graphics_settings(graphics_settings)

    go = DEFAULT_GENERAL_OPTIONS.copy()
    go.update(general_options)
    try:
        if go['generate_model_drawing']:
            animation.draw_model(result.get_model(), save_dir, 'model', show=go['show_model_drawing'],
                                 **assembly_appearance)
        if go['generate_fd_plot']:
            plot.force_displacement_curve(result, save_dir, show=go['show_fd_plot'], **plot_options)
        if go['generate_animation']:
            animation.animate(result, save_dir, show=go['show_animation'],
                              plot_options=plot_options, assembly_appearance=assembly_appearance,
                              **animation_options)
    except static_solver.UnusableSolution:
        print("Cannot make the graphics, because the calculated equilibrium path is unusable")


def make_animation(result, save_dir, show=True, graphics_settings=None, **animation_options):
    result = _load_result(result)
    _, plot_options, anim_options, assembly_appearance = _load_graphics_settings(graphics_settings)
    anim_options.update(animation_options)
    animation.animate(result, save_dir, show=show,
                      plot_options=plot_options, assembly_appearance=assembly_appearance,
                      **anim_options)


def make_force_displacement_plot(result, save_dir, show=True, graphics_settings=None, **plot_options):
    result = _load_result(result)
    _, p_options, _, _ = _load_graphics_settings(graphics_settings)
    p_options.update(plot_options)
    plot.force_displacement_curve(result, save_dir, show=show, **p_options)


def make_drawing(mdl, save_dir, show=True, graphics_settings=None, **assembly_appearance):
    if isinstance(mdl, str):
        mdl = io.read_model(mdl)
    if not isinstance(mdl, model.Model):
        raise ValueError("Incorrect result specification. The first argument should be the model file path,"
                         "or an already loaded Model object")
    _, _, _, a_appearance = _load_graphics_settings(graphics_settings)
    a_appearance.update(assembly_appearance)
    animation.draw_model(mdl, save_dir, show=show, **a_appearance)
