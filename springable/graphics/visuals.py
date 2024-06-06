from ..simulation.static_solver import Result
from ..simulation import model
from .graphic_settings import AnimationOptions as AO
from .drawing import ModelDrawing
from . import plot, visual_helpers
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

# matplotlib.rcParams['animation.ffmpeg_path'] = (
#     r'C:\\Users\\ducarme\\PycharmProjects\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\bin\\ffmpeg.exe')



def draw_model(mdl: model.Model, save_dir=None, save_name='model', show=True):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ModelDrawing(ax, mdl)
    if save_dir is not None:
        plot.save_fig(fig, save_dir, save_name, ['png', 'pdf'])
    if show:
        plt.show()
    plt.close()


def animate(_result: Result, save_dir, save_name, show=True):
    if AO.side_plot_mode != 0:
        fig = plt.figure(figsize=(8, 4.5))
        grid = plt.GridSpec(1, 2, wspace=0.20, hspace=0.01, bottom=0.15, left=0.01)
        ax1 = fig.add_subplot(grid[0, 0])
        with plt.style.context(AO.plotstylesheet):
            ax2 = fig.add_subplot(grid[0, 1])
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
            plot.adjust_spines(ax2)
    else:
        fig, ax1 = plt.subplots()
        ax2 = None
    ax1.axis('off')
    bounds, characteristic_length, color_handler, opacity_handler = visual_helpers.compute_requirements_for_animation(
        _result)
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

    _model_drawing = ModelDrawing(ax1, _model, fext_i, characteristic_length, assembly_span, color_handler,
                                  opacity_handler, None)

    # projection of the displacement vector (relative to preloading)
    # on the force direction final loading step
    deformation = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * force_direction[loaded_dof_indices],
                         axis=1)

    # projection of the applied vector force (relative to preloading)
    # on the force direction prescribed in final loading step
    force = np.sum((fext[:, loaded_dof_indices] - fext[0, loaded_dof_indices]) * force_direction[loaded_dof_indices],
                   axis=1)

    if AO.drive_mode is not None:
        loading_path_indices, _, _ = plot.extract_loading_path(_result, AO.drive_mode)
        unloading_path_indices = None
        if AO.cycle:
            if AO.drive_mode == plot.DriveModes.FORCE:
                load_after_loading = force[loading_path_indices[-1]]
            elif AO.drive_mode == plot.DriveModes.DISPLACEMENT:
                load_after_loading = deformation[loading_path_indices[-1]]
            else:
                raise ValueError('unknown drive mode')
            unloading_path_indices, _, _ = plot.extract_unloading_path(_result, AO.drive_mode,
                                                                       starting_load=load_after_loading)

        if AO.drive_mode == plot.DriveModes.FORCE:
            if AO.cycle:
                loading_nb_frames = AO.nb_frames // 2
                unloading_nb_frames = AO.nb_frames - loading_nb_frames
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
                                            AO.nb_frames)
                frame_indices = interp1d(force[loading_path_indices], loading_path_indices, kind='nearest')(
                    driving_force).astype(int)

        elif AO.drive_mode == plot.DriveModes.DISPLACEMENT:
            if AO.cycle:
                loading_nb_frames = AO.nb_frames // 2
                unloading_nb_frames = AO.nb_frames - loading_nb_frames
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
                driving_displacement = np.linspace(0.0, deformation[loading_path_indices[-1]], AO.nb_frames)
                frame_indices = interp1d(deformation[loading_path_indices], loading_path_indices, kind='nearest')(
                    driving_displacement).astype(int)
        else:
            raise ValueError('unknown drive mode')
    else:
        frame_indices = np.round(np.linspace(0, u.shape[0] - 1, AO.nb_frames)).astype(int)

    dot = None
    if AO.side_plot_mode != 0:
        plot.force_displacement_curve_in_ax(_result, ax2, marker='o', driven_path_only=False, drive_mode=AO.drive_mode,
                                            cycle=AO.cycle, show_snapping_arrows=True)
        dot = ax2.plot([deformation[0]], [force[0]], 'o', color='tab:red', markersize=10)[0]
        ax2.set_xlabel('displacement')
        ax2.set_ylabel('force')


    def update(i):
        _model.get_assembly().set_general_coordinates(_natural_coordinates + u[i, :])
        fext_i[:] = fext[i, :]
        _model_drawing.update()
        if AO.side_plot_mode != 0:
            dot.set_xdata([deformation[i]])
            dot.set_ydata([force[i]])

    if save_name is None:
        save_name = AO.default_animation_name

    if AO.save_frames_as_png:
        print('Generating PNG frames...')
        os.mkdir(os.path.join(save_dir, f'{save_name}_frames'))
        for frame_cnt, increment in enumerate(frame_indices):
            update(increment)
            frame_count_text = f"{frame_cnt}".zfill(4)
            frame_name = f"frame-{frame_count_text}.png"
            plt.savefig(os.path.join(save_dir, f'{save_name}_frames', frame_name), dpi=AO.dpi,
                        transparent=False,
                        bbox_inches='tight')
            visual_helpers.print_progress(frame_cnt, frame_indices.shape[0])
        _model.get_assembly().set_general_coordinates(_natural_coordinates)
        print('\nPNG frames saved successfully')

    filepath = None
    ani = FuncAnimation(fig, update, frames=frame_indices)
    if AO.save_as_gif:
        print('Generating GIF animation...')
        filepath = os.path.join(save_dir, f'{save_name}.gif')
        ani.save(filepath, fps=AO.fps, dpi=AO.dpi,
                 progress_callback=visual_helpers.print_progress)
        print('\nGIF animation saved successfully')
    if AO.save_as_transparent_mov:
        fig.patch.set_visible(False)
        if AO.side_plot_mode != 0:
            ax2.patch.set_visible(False)
        print('Generating transparent MOV animation...')
        filepath = os.path.join(save_dir, f'{save_name}.mov')
        ani.save(
            filepath,
            codec="png",
            dpi=AO.dpi,
            fps=AO.fps,
            bitrate=-1,
            savefig_kwargs={"transparent": True, "facecolor": "none"},
            progress_callback=visual_helpers.print_progress
        )
        print('\nMOV transparent animation saved successfully')
    if AO.save_as_mp4:
        fig.patch.set_visible(True)
        if AO.side_plot_mode != 0:
            ax2.patch.set_visible(True)
        print('Generating MP4 animation...')
        filepath = os.path.join(save_dir, f'{save_name}.mp4')
        ani.save(filepath, codec='h264', fps=AO.fps, dpi=AO.dpi,
                 progress_callback=visual_helpers.print_progress)
        print('\nMP4 animation saved successfully')
    _model.get_assembly().set_general_coordinates(_natural_coordinates)

    if show:
        if filepath is not None:
            os.startfile(filepath)
        else:
            plt.show()
    plt.close()
