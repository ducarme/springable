from ..mechanics.static_solver import Result
from ..readwrite import fileio as io
from ..mechanics import model
from .drawing import ModelDrawing
from . import plot, visual_helpers
from .figure_utils import figure_formatting as ff
from .default_graphics_settings import (DEFAULT_ANIMATION_OPTIONS,
                                        DEFAULT_ASSEMBLY_APPEARANCE,
                                        DEFAULT_PLOT_OPTIONS)
from ..readwrite import fileio as io
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_model(mdl: model.Model, save_dir=None, save_name='model', show=True, **assembly_appearance):
    aa = DEFAULT_ASSEMBLY_APPEARANCE.copy()
    aa.update(assembly_appearance)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ModelDrawing(ax, mdl, aa)
    xmin, ymin, xmax, ymax = mdl.get_assembly().get_dimensional_bounds()
    assembly_span = max(xmax - xmin, ymax - ymin)
    canvas_span = 1.25 * assembly_span
    midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
    ax.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
    if save_dir is not None:
        ff.save_fig(fig, save_dir, save_name, ['png', 'pdf'])
    if show:
        plt.show()
    plt.close()


def animate(_result: Result, save_dir, save_name: str = None, show=True,
            extra_init=None, extra_update=None,
            plot_options: dict = None, assembly_appearance: dict = None, **animation_options):
    ao = DEFAULT_ANIMATION_OPTIONS.copy()
    ao.update(animation_options)

    aa = DEFAULT_ASSEMBLY_APPEARANCE.copy()
    if assembly_appearance is not None:
        aa.update(assembly_appearance)

    po = DEFAULT_PLOT_OPTIONS.copy()
    if plot_options is not None:
        po.update(plot_options)
    po['drive_mode'] = ao['drive_mode']
    po['loading_sequence'] = 'cycle' if ao['cycling'] else 'loading'

    if ao['side_plot_mode'] != 'none':
        fig = plt.figure(figsize=(8, 4.5))
        grid = plt.GridSpec(1, 2, wspace=0.20, hspace=0.01, bottom=0.15, left=0.01)
        ax1 = fig.add_subplot(grid[0, 0])
        with plt.style.context(ao['plot_stylesheet']):
            ax2 = fig.add_subplot(grid[0, 1])
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
            ff.adjust_spines(ax2)
    else:
        fig, ax1 = plt.subplots()
        ax2 = None

    if extra_init is not None:
        with plt.style.context(ao['plot_stylesheet']):
            extra = extra_init(fig, ax1, ax2)

    ax1.axis('off')
    bounds, characteristic_length, color_handler, opacity_handler = visual_helpers.compute_requirements_for_animation(
        _result, aa)
    xmin, ymin, xmax, ymax = bounds
    assembly_span = max(xmax - xmin, ymax - ymin)
    canvas_span = 1.25 * assembly_span
    midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax1.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
    ax1.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
    ax1.set_aspect('equal', 'box')
    _model = _result.get_model()
    fext = _result.get_forces()

    force_vector = _model.get_force_vector()
    force_direction = force_vector / np.linalg.norm(force_vector)
    loaded_dof_indices = _model.get_loaded_dof_indices()

    fext_i = fext[0, :].copy()
    u = _result.get_displacements()
    _natural_coordinates = _model.get_assembly().get_general_coordinates()
    _model.get_assembly().set_general_coordinates(_natural_coordinates + u[0, :])

    _model_drawing = ModelDrawing(ax1, _model, aa, fext_i, characteristic_length, assembly_span, color_handler,
                                  opacity_handler, None)

    # projection of the displacement vector (relative to preload)
    # on the force direction final loading step
    deformation = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * force_direction[loaded_dof_indices],
                         axis=1)

    # projection of the applied vector force (relative to preload)
    # on the force direction prescribed in final loading step
    force = np.sum((fext[:, loaded_dof_indices] - fext[0, loaded_dof_indices]) * force_direction[loaded_dof_indices],
                   axis=1)

    if ao['drive_mode'] != 'none':
        loading_path_indices, _, _ = plot.extract_loading_path(_result, ao['drive_mode'])
        unloading_path_indices = None
        if ao['cycling']:
            unloading_path_indices, _, _ = plot.extract_unloading_path(_result, ao['drive_mode'],
                                                                       starting_index=loading_path_indices[-1])

        if ao['drive_mode'] == 'force':
            if ao['cycling']:
                loading_nb_frames = ao['nb_frames'] // 2
                unloading_nb_frames = ao['nb_frames'] - loading_nb_frames
                loading_driving_force = np.linspace(force[loading_path_indices[0]], force[loading_path_indices[-1]],
                                                    loading_nb_frames)
                unloading_driving_force = np.linspace(force[unloading_path_indices[0]],
                                                      force[unloading_path_indices[-1]], unloading_nb_frames)
                loading_frame_indices = interp1d(force[loading_path_indices], loading_path_indices, kind='nearest')(
                    loading_driving_force).astype(int)
                unloading_frame_indices = interp1d(force[unloading_path_indices], unloading_path_indices,
                                                   kind='nearest')(unloading_driving_force).astype(int)
                frame_indices = np.hstack((loading_frame_indices, unloading_frame_indices))
            else:
                driving_force = np.linspace(force[loading_path_indices[0]], force[loading_path_indices[-1]],
                                            ao['nb_frames'])
                frame_indices = interp1d(force[loading_path_indices], loading_path_indices, kind='nearest')(
                    driving_force).astype(int)

        elif ao['drive_mode'] == 'displacement':
            if ao['cycling']:
                loading_nb_frames = ao['nb_frames'] // 2
                unloading_nb_frames = ao['nb_frames'] - loading_nb_frames
                loading_driving_displacement = np.linspace(deformation[loading_path_indices[0]],
                                                           deformation[loading_path_indices[-1]], loading_nb_frames)
                unloading_driving_displacement = np.linspace(deformation[unloading_path_indices[0]],
                                                             deformation[unloading_path_indices[-1]],
                                                             unloading_nb_frames)
                loading_frame_indices = interp1d(deformation[loading_path_indices], loading_path_indices,
                                                 kind='nearest')(loading_driving_displacement).astype(int)
                unloading_frame_indices = interp1d(deformation[unloading_path_indices], unloading_path_indices,
                                                   kind='nearest')(unloading_driving_displacement).astype(int)
                frame_indices = np.hstack((loading_frame_indices, unloading_frame_indices))
            else:
                driving_displacement = np.linspace(0.0, deformation[loading_path_indices[-1]], ao['nb_frames'])
                frame_indices = interp1d(deformation[loading_path_indices], loading_path_indices, kind='nearest')(
                    driving_displacement).astype(int)
        else:
            raise ValueError(f'unknown drive mode {ao["drive_mode"]}')
    else:
        frame_indices = np.round(np.linspace(0, u.shape[0] - 1, ao['nb_frames'])).astype(int)

    dot = None
    if ao['side_plot_mode'] != "none":
        plot.force_displacement_curve_in_ax(_result, ax2, po)
        dot = ax2.plot([deformation[0]], [force[0]], 'o', color='tab:red', markersize=10)[0]
        ax2.set_xlabel('displacement')
        ax2.set_ylabel('force')
        if ((po['show_stability_legend'] and po['color_mode'] == 'stability')
                or (po['show_driven_path'] and po['show_driven_path_legend'] and po['drive_mode'] in (
                        'force', 'displacement'))):
            ax2.legend(numpoints=5, markerscale=1.5)

    def update(i):
        _model.get_assembly().set_general_coordinates(_natural_coordinates + u[i, :])
        fext_i[:] = fext[i, :]
        _model_drawing.update()
        if extra_update is not None:
            extra_update(fig, ax1, ax2, extra)
        if ao['side_plot_mode'] != 'none':
            dot.set_xdata([deformation[i]])
            dot.set_ydata([force[i]])

    if save_name is None:
        save_name = ao['default_animation_name']

    if ao['save_frames_as_png']:
        print('Generating PNG frames...')
        os.mkdir(os.path.join(save_dir, f'{save_name}_frames'))
        for frame_cnt, increment in enumerate(frame_indices):
            update(increment)
            frame_count_text = f"{frame_cnt}".zfill(4)
            frame_name = f"frame-{frame_count_text}.png"
            plt.savefig(os.path.join(save_dir, f'{save_name}_frames', frame_name), dpi=ao['dpi'],
                        transparent=False,
                        bbox_inches='tight')
            visual_helpers.print_progress(frame_cnt, frame_indices.shape[0])
        _model.get_assembly().set_general_coordinates(_natural_coordinates)
        print('\nPNG frames saved successfully')

    filepath = None
    format_type = None
    if ao['save_as_gif'] or ao['save_as_transparent_mov'] or ao['save_as_mp4']:
        ani = FuncAnimation(fig, update, frames=frame_indices)
        if ao['save_as_gif']:
            format_type = 'image'
            print('Generating GIF animation...')
            filepath = os.path.join(save_dir, f'{save_name}.gif')
            ani.save(filepath, fps=ao['fps'], dpi=ao['dpi'],
                     progress_callback=visual_helpers.print_progress)
            print('\nGIF animation saved successfully')
            _model.get_assembly().set_general_coordinates(_natural_coordinates)

        if ao['save_as_transparent_mov']:
            format_type = 'video'
            fig.patch.set_visible(False)
            if ao['side_plot_mode'] != 'none':
                ax2.patch.set_visible(False)
            print('Generating transparent MOV animation...')
            filepath = os.path.join(save_dir, f'{save_name}.mov')
            ani.save(
                filepath,
                codec="png",
                dpi=ao['dpi'],
                fps=ao['fps'],
                bitrate=-1,
                savefig_kwargs={"transparent": True, "facecolor": "none"},
                progress_callback=visual_helpers.print_progress
            )
            print('\nMOV transparent animation saved successfully')
            _model.get_assembly().set_general_coordinates(_natural_coordinates)
        if ao['save_as_mp4']:
            format_type = 'video'
            fig.patch.set_visible(True)
            if ao['side_plot_mode'] != 'none':
                ax2.patch.set_visible(True)
            print('Generating MP4 animation...')
            filepath = os.path.join(save_dir, f'{save_name}.mp4')
            ani.save(filepath, codec='h264', fps=ao['fps'], dpi=ao['dpi'],
                     progress_callback=visual_helpers.print_progress)
            print('\nMP4 animation saved successfully!')
            _model.get_assembly().set_general_coordinates(_natural_coordinates)
        plt.close()
        if show and filepath is not None:
            if io.is_notebook():
                io.play_media_in_notebook_if_possible(filepath, format_type=format_type)
            else:
                try:
                    io.open_file_with_default_os_app(filepath)
                except OSError:
                    print('Cannot open animation automatically. Open the result folder instead to check out the '
                          'animation.')
