from ..mechanics.static_solver import Result
from ..mechanics import model
from ..mechanics.result_processing import extract_loading_path, extract_unloading_path, LoadingPathEmpty
from .drawing import ModelDrawing
from . import visual_helpers, plot
from . import figure_formatting as ff
from .default_graphics_settings import AssemblyAppearanceOptions, AnimationOptions, PlotOptions
from ..readwrite import fileio as io
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_model(mdl: model.Model, save_dir=None, save_name='model',
               show=True,
               assembly_span: float = None,
               characteristic_length: float = None,
               xlim: tuple[float, float] = None,
               ylim:  tuple[float, float] = None,
               **assembly_appearance):
    aa = AssemblyAppearanceOptions()
    aa.update(**assembly_appearance)

    with plt.style.context(aa.stylesheet):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        xmin, ymin, xmax, ymax = mdl.get_assembly().get_dimensional_bounds()
        if assembly_span is None:
            assembly_span = max(xmax - xmin, ymax - ymin)


        ModelDrawing(ax, mdl, aa, assembly_span=assembly_span, characteristic_length=characteristic_length)
        canvas_span = 1.25 * assembly_span
        midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if xlim is None:
            ax.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
        else:
            ax.set_xlim(*xlim)
        if ylim is None:
            ax.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
        else:
            ax.set_ylim(*ylim)

        ff.adjust_spines([ax], 0, ['bottom', 'top', 'left', 'right'] if aa.show_axes else [])
        ff.adjust_figure_layout(fig, aa.drawing_fig_width, aa.drawing_fig_height, pad=0.1)
        if save_dir is not None:
            ff.save_fig(fig, save_dir, save_name, ['png', 'pdf'], transparent=aa.transparent, dpi=aa.drawing_dpi)
        if show:
            plt.show()
        else:
            plt.close(fig=fig)


def draw_equilibrium_state(res: Result,
                           state_index: int = None,
                           start_of_loadstep_index: int  = None,
                           end_of_loadstep_index: int = None,
                           save_dir=None, save_name='state',
                           show=True,
                           assembly_span: float = None,
                           characteristic_length: float = None,
                           xlim: tuple[float, float] = None,
                           ylim:  tuple[float, float] = None,
                           **assembly_appearance):


    aa = AssemblyAppearanceOptions()
    aa.update(**assembly_appearance)

    u = res.get_displacements(include_preloading=True)
    nb_states = u.shape[0]
    if state_index is None:
        if start_of_loadstep_index is not None:
            state_index = res.get_loadstep_starting_index(start_of_loadstep_index)
        elif end_of_loadstep_index is not None:
            state_index = res.get_loadstep_end_index(end_of_loadstep_index)

    if state_index is None or state_index > nb_states - 1:
        print('Cannot draw equilibrium state, because the state cannot be found. Try a different index.')
        return

    mdl = res.get_model()
    loadstep_index = res.get_step_indices()[state_index]
    initial_coordinates = mdl.get_assembly().get_coordinates().copy()
    u  = res.get_displacements(include_preloading=True)

    mdl.get_assembly().set_coordinates(initial_coordinates + u[state_index, :])
    blocked_nodes_directions_step_list = mdl.get_blocked_nodes_directions_step_list()
    for ls_ix, blocked_nodes_directions in enumerate(blocked_nodes_directions_step_list):
        if ls_ix <= loadstep_index:
            mdl.get_assembly().block_nodes_along_directions(*blocked_nodes_directions)

    (bounds,
     characteristic_length_,
     element_color_handler,
     element_opacity_handler,
     force_color_handler,
     all_force_amounts,
     all_preforce_amounts) = visual_helpers.scan_result_and_compute_quantities_for_animations(res, aa)

    if all_force_amounts is not None:
        force_amounts = {loaded_node: amounts[0] for loaded_node, amounts in all_force_amounts.items()}
    else:
        force_amounts = None

    if all_preforce_amounts is not None:
        preforce_amounts = {loaded_node: amounts[0] for loaded_node, amounts in all_preforce_amounts.items()}
    else:
        preforce_amounts = None

    forces_after_preloading = np.sum(mdl.get_force_vectors_step_list()[:loadstep_index], axis=0)


    with plt.style.context(aa.stylesheet):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        characteristic_length = characteristic_length if characteristic_length is not None else characteristic_length_
        xmin, ymin, xmax, ymax = bounds
        assembly_span = assembly_span if assembly_span is not None else max(xmax - xmin, ymax - ymin)
        _model_drawing = ModelDrawing(ax, mdl, aa, characteristic_length, assembly_span,
                                      element_color_handler=element_color_handler,
                                      element_opacity_handler=element_opacity_handler,
                                      force_color_handler=force_color_handler, force_amounts=force_amounts,
                                      force_vector_after_preloading=forces_after_preloading,
                                      preforce_amounts=preforce_amounts)

        xmin, ymin, xmax, ymax = bounds
        canvas_span = 1.25 * assembly_span
        midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if xlim is None:
            ax.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
        else:
            ax.set_xlim(*xlim)
        if ylim is None:
            ax.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
        else:
            ax.set_ylim(*ylim)


        ff.adjust_spines([ax], 0, ['bottom', 'top', 'left', 'right'] if aa.show_axes else [])
        ff.adjust_figure_layout(fig, aa.drawing_fig_width, aa.drawing_fig_height, pad=0.1)
        if save_dir is not None:
            ff.save_fig(fig, save_dir, f'{save_name}_{state_index}',
                        ['png', 'pdf'], transparent=aa.transparent, dpi=aa.drawing_dpi)
        if show:
            plt.show()
        else:
            plt.close(fig=fig)

        mdl.get_assembly().set_coordinates(initial_coordinates)
        for blocked_nodes_directions in blocked_nodes_directions_step_list:
            mdl.get_assembly().release_nodes_along_directions(*blocked_nodes_directions)




def animate(_result: Result, save_dir, save_name: str = None, show=True,
            extra_init=None, extra_update=None,
            plot_options: dict = None, assembly_appearance: dict = None, **animation_options):
    ao = AnimationOptions()
    ao.update(**animation_options)

    aa = AssemblyAppearanceOptions()
    if assembly_appearance is not None:
        aa.update(**assembly_appearance)

    po = PlotOptions()
    if plot_options is not None:
        po.update(**plot_options)
    po.drive_mode = ao.drive_mode
    po.loading_sequence = 'cycle' if ao.cycling else 'loading'

    with plt.style.context(ao.stylesheet):
        if ao.side_plot_mode != 'none':
            fig = plt.figure(figsize=(ao.animation_width, ao.animation_height))
            grid = plt.GridSpec(1, 2, wspace=0.20, hspace=0.01, bottom=0.15, left=0.01)
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[0, 1])
            spines = []
            spines += ['left'] if po.show_left_spine else []
            spines += ['right'] if po.show_right_spine else []
            spines += ['top'] if po.show_top_spine else []
            spines += ['bottom'] if po.show_bottom_spine else []
            ff.adjust_spines(ax2, po.spine_offset, spines)
        else:
            fig, ax1 = plt.subplots()
            ax2 = None

        if extra_init is not None:
            fig, ax1, ax2, extra = extra_init(fig, ax1, ax2)

        ax1.axis('off')
        (bounds, characteristic_length,
         element_color_handler,
         element_opacity_handler,
         force_color_handler,
         all_force_amounts,
         all_preforce_amounts) = visual_helpers.scan_result_and_compute_quantities_for_animations(_result, aa)
        xmin, ymin, xmax, ymax = bounds
        assembly_span = max(xmax - xmin, ymax - ymin)
        canvas_span = 1.25 * assembly_span
        midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
        ax1.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
        ax1.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
        ax1.set_aspect('equal', 'box')

        _model = _result.get_model()
        u = _result.get_displacements()
        forces_after_preloading = _result.get_forces()[0]
        _natural_coordinates = _model.get_assembly().get_coordinates()
        _model.get_assembly().set_coordinates(_natural_coordinates + u[0, :])
        blocked_nodes_directions_step_list = _model.get_blocked_nodes_directions_step_list()
        for blocked_nodes_directions in blocked_nodes_directions_step_list:
            _model.get_assembly().block_nodes_along_directions(*blocked_nodes_directions)

        if all_force_amounts is not None:
            force_amounts = {loaded_node: amounts[0] for loaded_node, amounts in all_force_amounts.items()}
        else:
            force_amounts = None

        if all_preforce_amounts is not None:
            preforce_amounts = {loaded_node: amounts[0] for loaded_node, amounts in all_preforce_amounts.items()}
        else:
            preforce_amounts = None
        _model_drawing = ModelDrawing(ax1, _model, aa, characteristic_length, assembly_span,
                                      element_color_handler=element_color_handler,
                                      element_opacity_handler=element_opacity_handler,
                                      force_color_handler=force_color_handler, force_amounts=force_amounts,
                                      force_vector_after_preloading=forces_after_preloading,
                                      preforce_amounts=preforce_amounts)

        deformation, force = _result.get_equilibrium_path()
        if ao.drive_mode != 'none':
            try:
                loading_path_indices, _, _ = extract_loading_path(_result, ao.drive_mode)
                unloading_path_indices = None
                if ao.cycling:
                    unloading_path_indices, _, _ = extract_unloading_path(_result, ao.drive_mode,
                                                                               starting_index=loading_path_indices[-1])

                if ao.drive_mode == 'force':
                    if ao.cycling:
                        loading_nb_frames = ao.nb_frames // 2
                        unloading_nb_frames = ao.nb_frames - loading_nb_frames
                        loading_driving_force = np.linspace(force[loading_path_indices[0]],
                                                            force[loading_path_indices[-1]],
                                                            loading_nb_frames)
                        unloading_driving_force = np.linspace(force[unloading_path_indices[0]],
                                                              force[unloading_path_indices[-1]], unloading_nb_frames)
                        loading_frame_indices = interp1d(force[loading_path_indices], loading_path_indices,
                                                         kind='nearest')(
                            loading_driving_force).astype(int)
                        unloading_frame_indices = interp1d(force[unloading_path_indices], unloading_path_indices,
                                                           kind='nearest')(unloading_driving_force).astype(int)
                        frame_indices = np.hstack((loading_frame_indices, unloading_frame_indices))
                    else:
                        driving_force = np.linspace(force[loading_path_indices[0]], force[loading_path_indices[-1]],
                                                    ao.nb_frames)
                        frame_indices = interp1d(force[loading_path_indices], loading_path_indices, kind='nearest')(
                            driving_force).astype(int)

                elif ao.drive_mode == 'displacement':
                    if ao.cycling:
                        loading_nb_frames = ao.nb_frames // 2
                        unloading_nb_frames = ao.nb_frames - loading_nb_frames
                        loading_driving_displacement = np.linspace(deformation[loading_path_indices[0]],
                                                                   deformation[loading_path_indices[-1]],
                                                                   loading_nb_frames)
                        unloading_driving_displacement = np.linspace(deformation[unloading_path_indices[0]],
                                                                     deformation[unloading_path_indices[-1]],
                                                                     unloading_nb_frames)
                        loading_frame_indices = interp1d(deformation[loading_path_indices], loading_path_indices,
                                                         kind='nearest')(loading_driving_displacement).astype(int)
                        unloading_frame_indices = interp1d(deformation[unloading_path_indices], unloading_path_indices,
                                                           kind='nearest')(unloading_driving_displacement).astype(int)
                        frame_indices = np.hstack((loading_frame_indices, unloading_frame_indices))
                    else:
                        driving_displacement = np.linspace(0.0, deformation[loading_path_indices[-1]], ao.nb_frames)
                        frame_indices = interp1d(deformation[loading_path_indices], loading_path_indices,
                                                 kind='nearest')(
                            driving_displacement).astype(int)
                else:
                    raise ValueError(f'unknown drive mode {ao.drive_mode}')
            except LoadingPathEmpty:
                print(f"Cannot make the animation in {ao.drive_mode}-driven mode, "
                      f"because no stable points have been found under these loading conditions")
                return
        else:
            frame_indices = np.round(np.linspace(0, u.shape[0] - 1, ao.nb_frames)).astype(int)

        dot = None
        if ao.side_plot_mode == "force_displacement_curve":
            plot.force_displacement_curve_in_ax(_result, ax2, po)
            dot = ax2.plot([deformation[0]], [force[0]],
                           'o', color=ao.animated_equilibrium_point_color,
                           markersize=ao.animated_equilibrium_point_size * po.default_markersize,
                           zorder=1.1)[0]
            ax2.set_xlabel(po.default_xlabel)
            ax2.set_ylabel(po.default_ylabel)
            if po.hide_ticklabels:
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
            if ((((po.show_stability_legend and po.color_mode == 'stability') or
                  (po.show_stability_legend and po.plot_style == 'line')) and
                 not (po.show_driven_path and po.driven_path_only))
                or (po.show_driven_path
                    and po.show_driven_path_legend
                    and po.drive_mode in ('force', 'displacement'))):
                ax2.legend(numpoints=5, markerscale=1.5)

        def update(i):
            # update assembly
            _model.get_assembly().set_coordinates(_natural_coordinates + u[i, :])

            # update external forces
            # /!\ force_amounts dict should not be overridden
            if force_amounts is not None:
                for loaded_node in force_amounts.keys():
                    force_amounts[loaded_node] = all_force_amounts[loaded_node][i]

            if preforce_amounts is not None:
                for preloaded_node in preforce_amounts.keys():
                    preforce_amounts[preloaded_node] = all_preforce_amounts[preloaded_node][i]

            _model_drawing.update()
            if extra_update is not None:
                extra_update(i, fig, ax1, ax2, extra)
            if ao.side_plot_mode == 'force_displacement_curve':
                dot.set_xdata([deformation[i]])
                dot.set_ydata([force[i]])

        if save_name is None:
            save_name = ao.default_animation_name

        if ao.save_frames_as_png:
            print('Generating PNG frames...')
            os.mkdir(os.path.join(save_dir, f'{save_name}_frames'))
            for frame_cnt, increment in enumerate(frame_indices):
                update(increment)
                frame_count_text = f"{frame_cnt}".zfill(4)
                frame_name = f"frame-{frame_count_text}.png"
                plt.savefig(os.path.join(save_dir, f'{save_name}_frames', frame_name), dpi=ao.dpi,
                            transparent=False,
                            bbox_inches='tight')
                visual_helpers.print_progress(frame_cnt, frame_indices.shape[0])
            _model.get_assembly().set_coordinates(_natural_coordinates)
            for blocked_nodes_directions in blocked_nodes_directions_step_list:
                _model.get_assembly().release_nodes_along_directions(*blocked_nodes_directions)

            print('\nPNG frames saved successfully')

        filepath = None
        format_type = None
        if ao.save_as_gif or ao.save_as_transparent_mov or ao.save_as_mp4:
            ani = FuncAnimation(fig, update, frames=frame_indices)
            if ao.save_as_gif:
                format_type = 'image'
                print('Generating GIF animation...')
                filepath = os.path.join(save_dir, f'{save_name}.gif')
                ani.save(filepath, fps=ao.fps, dpi=ao.dpi,
                         progress_callback=visual_helpers.print_progress)
                print('\nGIF animation saved successfully')
                _model.get_assembly().set_coordinates(_natural_coordinates)
                for blocked_nodes_directions in blocked_nodes_directions_step_list:
                    _model.get_assembly().release_nodes_along_directions(*blocked_nodes_directions)

            if ao.save_as_transparent_mov:
                format_type = 'video'
                fig.patch.set_visible(False)
                if ao.side_plot_mode != 'none':
                    ax2.patch.set_visible(False)
                print('Generating transparent MOV animation...')
                filepath = os.path.join(save_dir, f'{save_name}.mov')
                ani.save(
                    filepath,
                    codec="png",
                    dpi=ao.dpi,
                    fps=ao.fps,
                    bitrate=-1,
                    savefig_kwargs={"transparent": True, "facecolor": "none"},
                    progress_callback=visual_helpers.print_progress
                )
                print('\nMOV transparent animation saved successfully')
                _model.get_assembly().set_coordinates(_natural_coordinates)
                for blocked_nodes_directions in blocked_nodes_directions_step_list:
                    _model.get_assembly().release_nodes_along_directions(*blocked_nodes_directions)

            if ao.save_as_mp4:
                format_type = 'video'
                fig.patch.set_visible(True)
                if ao.side_plot_mode != 'none':
                    ax2.patch.set_visible(True)
                print('Generating MP4 animation...')
                filepath = os.path.join(save_dir, f'{save_name}.mp4')
                ani.save(filepath, codec='h264', fps=ao.fps, dpi=ao.dpi,
                         progress_callback=visual_helpers.print_progress)
                print('\nMP4 animation saved successfully')
                _model.get_assembly().set_coordinates(_natural_coordinates)
                for blocked_nodes_directions in blocked_nodes_directions_step_list:
                    _model.get_assembly().release_nodes_along_directions(*blocked_nodes_directions)
            try:
                plt.close(fig=fig)
            except AttributeError:
                pass
                # Some user reported an attribute error after saving animations, when using Visual Code Studio
                # It is hard to reproduce and clarify the cause, so this is a quick and dirty solution.
            if show and filepath is not None:
                if io.is_notebook():
                    io.play_media_in_notebook_if_possible(filepath, format_type=format_type)
                else:
                    try:
                        io.open_file_with_default_os_app(filepath)
                    except OSError:
                        print('Cannot open animation automatically. Open the result folder instead to check out the '
                              'animation.')
