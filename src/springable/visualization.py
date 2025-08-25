from .graphics.default_graphics_settings import GeneralOptions
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


def _load_graphics_settings(graphics_settings: str | tuple | list):
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
            raise ValueError("Incorrect graphics settings specification. If specified, the last argument must "
                             "be the path to the graphic-settings file, or an already loaded graphic-settings "
                             "variable, that is a list or tuple of 4 dictionaries")
    else:
        graphics_settings = {}, {}, {}, {}
    return graphics_settings


def load_result(_result: str | static_solver.Result) -> static_solver.Result:
    if isinstance(_result, str):
        _result = io.read_results(_result)
    if not isinstance(_result, static_solver.Result):
        raise ValueError("Incorrect result specification. The first argument should be the directory where the result "
                         "files are stored, or an already loaded Result object")
    return _result


def visualize_scan_results(scan_results_dir: str, save_dir: str = '',
                           graphics_settings: list[dict] | tuple[dict] | str = None, postprocessing = None):
    general_options, plot_options, animation_options, assembly_appearance = _load_graphics_settings(graphics_settings)
    go = GeneralOptions()
    go.update(**general_options)

    # Somehow, the variables herein below should be automatically read from scan_results_directory
    general_info = io.read_scanning_general_info(scan_results_dir)
    model_path = os.path.join(scan_results_dir, general_info['MODEL_FILENAME'])
    all_sim_names = general_info['ALL_SIM_NAMES']
    scanning_mode = general_info['SCANNING_MODE']

    # MAKE GRAPHICS
    if scanning_mode == 'separate':
        par_name_to_sim_names = general_info['PARAMETER_NAME_TO_SIM_NAMES_MAPPING']
        if go.generate_parametric_fd_plots or postprocessing is not None:
            _, par_name_to_par_data = io.read_parameters_from_model_file(model_path)

            for design_parameter_name, sim_names in par_name_to_sim_names.items():
                subsave_dirs = [os.path.join(scan_results_dir, sim_name) for sim_name in sim_names]
                parameter_values = [io.read_design_parameters(subsave_dir)[design_parameter_name]
                                    for subsave_dir in subsave_dirs]

                plot.parametric_force_displacement_curve(_results(subsave_dirs), design_parameter_name,
                                                         par_name_to_par_data[design_parameter_name], parameter_values,
                                                         save_dir, save_name=f'fd_curve_{design_parameter_name}',
                                                         show=go.show_parametric_fd_plots, **plot_options)
                if postprocessing is not None:
                    for pp in postprocessing:
                        plot.parametric_curve(pp['postprocessing_fun'], _results(subsave_dirs), design_parameter_name,
                                              par_name_to_par_data[design_parameter_name], parameter_values,
                                              save_dir, save_name=f'{pp["save_name"]}_{design_parameter_name}',
                                              xlabel=pp['xlabel'], ylabel=pp['ylabel'],
                                              show=go.show_parametric_custom_plots, **plot_options
                                              )
    if scanning_mode == 'together':
        if go.generate_parametric_fd_plots or postprocessing is not None:
            subsave_dirs = [os.path.join(scan_results_dir, sim_name) for sim_name in all_sim_names]
            dummy_parameter_data = {'default value': 0,
                                    'lower bound': 0,
                                    'upper bound': len(all_sim_names)-1,
                                    'nb samples': len(all_sim_names),
                                    'is numeric parameter': True,
                                    'is range parameter': True}
            plot.parametric_force_displacement_curve(_results(subsave_dirs), 'sim #',
                                                     dummy_parameter_data, np.arange(0, len(all_sim_names)).tolist(),
                                                     save_dir, save_name=f'parametric_fd_curve',
                                                     show=go.show_parametric_fd_plots, **plot_options)
            if postprocessing is not None:
                for pp in postprocessing:
                    plot.parametric_curve(pp['postprocessing_fun'], _results(subsave_dirs), 'sim #',
                                          dummy_parameter_data, np.arange(0, len(all_sim_names)).tolist(),
                                          save_dir, save_name=f'{pp["save_name"]}',
                                          xlabel=pp['xlabel'], ylabel=pp['ylabel'],
                                          show=go.show_parametric_custom_plots, **plot_options
                                          )

    # first scanning of result folders to obtain axes limits for force-displacement plots
    min_u, max_u, min_f, max_f = [None] * 4
    if go.generate_all_fd_plots:
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
    if go.generate_all_model_drawings or go.generate_all_fd_plots or go.generate_all_animations:
        print('Generating all graphics...')
        drawings_dir = io.mkdir(os.path.join(save_dir, 'all_drawings')) if go.generate_all_model_drawings else None
        fd_plots_dir = io.mkdir(os.path.join(save_dir, 'all_fd_plots')) if go.generate_all_fd_plots else None
        animations_dir = io.mkdir(os.path.join(save_dir, 'all_animations')) if go.generate_all_animations else None

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
            print(f'Post-processed simulations: {i + 1}/{len(all_sim_names)}')
        print('All graphics generated successfully')


def visualize_result(result: static_solver.Result | str, save_dir: str = '',
                     graphics_settings: list | tuple | str = None, postprocessing = None):
    result = load_result(result)
    general_options, plot_options, animation_options, assembly_appearance = _load_graphics_settings(graphics_settings)

    go = GeneralOptions()
    go.update(**general_options)
    print(f"Post-processing starts...")
    try:
        at_least_one = False
        if go.generate_model_drawing:
            at_least_one = True
            if go.show_model_drawing:
                print("Spring model is drawn in a new window. Close the window to continue...")
            animation.draw_model(result.get_model(), save_dir, 'model', show=go.show_model_drawing,
                                 **assembly_appearance)
        if go.generate_fd_plot:
            at_least_one = True
            if go.show_fd_plot:
                print("Force-displacement curve is drawn in a new window. Close the window to continue...")
            plot.force_displacement_curve(result, save_dir, show=go.show_fd_plot, **plot_options)

        if postprocessing is not None:
            for pp in postprocessing:
                at_least_one = True
                if go.show_custom_plot:
                    print(f"Custom '{pp['save_name']}' plot is drawn in a new window. Close the window to continue...")
                plot.curve(pp['postprocessing_fun'], result, save_dir, save_name=pp['save_name'],
                           show=go.show_custom_plot, xlabel=pp['xlabel'], ylabel=pp['ylabel'], **plot_options)

        if go.generate_animation:
            at_least_one = True
            animation.animate(result, save_dir, show=go.show_animation,
                              plot_options=plot_options, assembly_appearance=assembly_appearance,
                              **animation_options)
        if at_least_one:
            print(f"Post-processed results have been saved in {save_dir}.")
        else:
            print("Nothing has been post-processed, as no post-processed output has been requested.")
    except static_solver.UnusableSolution:
        print("Cannot make the graphics, because the calculated equilibrium path is unusable")


def make_animation(result, save_dir, save_name: str = None, show=True, assembly_span=None, characteristic_length=None,
                   extra_init=None, extra_update=None, graphics_settings=None, **animation_options):
    result = load_result(result)
    _, plot_options, anim_options, assembly_appearance = _load_graphics_settings(graphics_settings)
    anim_options.update(animation_options)
    animation.animate(result, save_dir, save_name=save_name, show=show,
                      extra_init=extra_init, extra_update=extra_update,
                      characteristic_length=characteristic_length, assembly_span=assembly_span,
                      plot_options=plot_options, assembly_appearance=assembly_appearance,
                      **anim_options)


def make_force_displacement_plot(result, save_dir, show=True,
                                 graphics_settings=None, xlim=None, ylim=None, **plot_options):
    result = load_result(result)
    _, p_options, _, _ = _load_graphics_settings(graphics_settings)
    p_options.update(plot_options)
    plot.force_displacement_curve(result, save_dir, show=show, xlim=xlim, ylim=ylim, **p_options)

def make_custom_plot(processing_fun, result, save_dir, show=True,
                     graphics_settings=None, xlim=None, ylim=None, **plot_options):
    result = load_result(result)
    _, p_options, _, _ = _load_graphics_settings(graphics_settings)
    p_options.update(plot_options)
    plot.curve(processing_fun, result, save_dir, show=show, xlim=xlim, ylim=ylim, **p_options)


def make_model_drawing(mdl: str | model.Model, save_dir,
                 save_name='model', show=True, characteristic_length=None,
                 xlim: tuple[float, float] = None, ylim: tuple[float, float] = None,
                 graphics_settings=None, **assembly_appearance):
    if isinstance(mdl, str):
        mdl = io.read_model(mdl)
    if not isinstance(mdl, model.Model):
        raise ValueError("Incorrect model specification. The first argument should be the model file path, "
                         "or an already loaded Model object")
    _, _, _, a_appearance = _load_graphics_settings(graphics_settings)
    a_appearance.update(assembly_appearance)
    animation.draw_model(mdl, save_dir, save_name, show=show,
                         characteristic_length=characteristic_length, xlim=xlim, ylim=ylim,
                         **a_appearance)
    
def make_model_construction_animation(mdl: str | model.Model, save_dir, duration_per_node=0.5, duration_per_element=0.5, duration_per_loadstep=0.5, inbetween_duration=0.0,
                                      end_duration=0.0,
                                      fps=50, rate_fun='none', save_as_gif=True, save_as_mp4=False, save_name="model_construction_animation", show=True,
                                      assembly_span=None, characteristic_length=None, xlim: tuple[float, float] = None, ylim: tuple[float, float] = None,
                                      graphics_settings=None, **assembly_appearance):
    if isinstance(mdl, str):
        mdl = io.read_model(mdl)
    if not isinstance(mdl, model.Model):
        raise ValueError("Incorrect model specification. The first argument should be the model file path, "
                         "or an already loaded Model object")
    _, _, _, a_appearance = _load_graphics_settings(graphics_settings)
    a_appearance.update(assembly_appearance)
    animation.animate_model_construction(mdl, save_dir, duration_per_node, duration_per_element, duration_per_loadstep, inbetween_duration,
                                         end_duration,
                                         fps, rate_fun, save_as_gif, save_as_mp4, show, save_name,
                                         assembly_span, characteristic_length, xlim, ylim, **a_appearance)

def make_equilibrium_state_drawing(result, save_dir,
                                   state_index=None,
                                   start_of_loadstep_index=None,
                                   end_of_loadstep_index=None,
                                   save_name='equilibrium_state',
                                   show=True,
                                   assembly_span: float = None,
                                   characteristic_length: float = None,
                                   xlim: tuple[float, float] = None,
                                   ylim: tuple[float, float] = None,
                                   graphics_settings=None, **assembly_appearance):
    result = load_result(result)
    _, _, _, a_appearance = _load_graphics_settings(graphics_settings)
    a_appearance.update(assembly_appearance)
    animation.draw_equilibrium_state(result, state_index, start_of_loadstep_index, end_of_loadstep_index,
                                     save_dir, save_name, show, assembly_span, characteristic_length, xlim, ylim,
                                     **a_appearance)
