from ..io_handling.graphic_settings import AnimationOptions as AO
from ..io_handling.graphic_settings import AssemblyAppearance as AA
from ..io_handling.drawing import ModelDrawing
from . import dynamics
from . import animation_helpers
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def animate(mdl: dynamics.DynamicModel, time, q_sim, dqdt_sim, f_sim, save_dir, save_name, show=True):
    AA.show_forces = False
    fig = plt.figure(figsize=(12, 5))
    grid = plt.GridSpec(1, 1, wspace=0.3, hspace=0.1, bottom=0.15, left=0.15)
    ax1 = fig.add_subplot(grid[0, 0])
    t = np.linspace(time[0], time[-1], AO.nb_frames)
    q = interp1d(time, q_sim, axis=0)(t)
    dqdt = interp1d(time, dqdt_sim, axis=0)(t)
    f = interp1d(time, f_sim, axis=0)(t)

    bounds, characteristic_length, color_handler, opacity_handler = animation_helpers.compute_requirements_for_animation(
        mdl, time, q, dqdt, f)
    xmin, ymin, xmax, ymax = bounds
    assembly_span = max(xmax - xmin, ymax - ymin)
    canvas_span = 1.25 * assembly_span
    midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax1.set_xlim(midx - canvas_span / 2, midx + canvas_span / 2)
    ax1.set_ylim(midy - canvas_span / 2, midy + canvas_span / 2)
    ax1.set_aspect('equal', 'box')
    _initial_coordinates = mdl.get_assembly().get_general_coordinates()
    _model_drawing = ModelDrawing(ax1, mdl, None, characteristic_length, assembly_span, color_handler,
                                  opacity_handler, None)
    final_force_vector = mdl.get_force_vectors_step_list()
    final_force_direction = final_force_vector / np.linalg.norm(final_force_vector)

    def update(i):
        mdl.get_assembly().set_general_coordinates(q[i, :])
        mdl.get_assembly().set_general_velocities(q[i, :])
        _model_drawing.update()

    if save_name is None:
        save_name = AO.default_animation_name

    frame_indices = np.round(np.linspace(0, q.shape[0] - 1, AO.nb_frames)).astype(int)
    if AO.save_frames_as_png:
        if AO.animation_verbose:
            print('Generating PNG frames...')
        os.mkdir(os.path.join(save_dir, f'{save_name}_frames'))
        for frame_cnt, increment in enumerate(frame_indices):
            update(increment)
            frame_count_text = f"{frame_cnt}".zfill(4)
            frame_name = f"frame-{frame_count_text}.png"
            plt.savefig(os.path.join(save_dir, f'{save_name}_frames', frame_name), dpi=AO.dpi,
                        transparent=False,
                        bbox_inches='tight')
        mdl.get_assembly().set_general_coordinates(_initial_coordinates)
        if AO.animation_verbose:
            print('PNG frames saved successfully')

    filepath = None
    ani = FuncAnimation(fig, update, frames=np.round(np.linspace(0, q.shape[0] - 1, AO.nb_frames)).astype(int))
    if AO.save_as_gif:
        if AO.animation_verbose:
            print('Generating GIF animation...')
        filepath = os.path.join(save_dir, f'{save_name}.gif')
        ani.save(filepath, fps=AO.fps, dpi=AO.dpi)
        if AO.animation_verbose:
            print('GIF animation saved successfully')
    if AO.save_as_transparent_mov:
        fig.patch.set_visible(False)
        if AO.animation_verbose:
            print('Generating transparent MOV animation...')
        filepath = os.path.join(save_dir, f'{save_name}.mov')
        ani.save(
            filepath,
            codec="png",
            dpi=AO.dpi,
            fps=AO.fps,
            bitrate=-1,
            savefig_kwargs={"transparent": True, "facecolor": "none"},
        )
        if AO.animation_verbose:
            print('MOV transparent animation saved successfully')
    if AO.save_as_mp4:
        fig.patch.set_visible(True)
        if AO.animation_verbose:
            print('Generating MP4 animation...')
        filepath = os.path.join(save_dir, f'{save_name}.mp4')
        ani.save(filepath, codec='h264', fps=AO.fps, dpi=AO.dpi)
        if AO.animation_verbose:
            print('MP4 animation saved successfully')
    mdl.get_assembly().set_general_coordinates(_initial_coordinates)

    if show:
        if filepath is not None:
            os.startfile(filepath)
        else:
            plt.show()
    plt.close()
