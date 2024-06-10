from ..simulation import static_solver
from .default_graphics_settings import DEFAULT_PLOT_OPTIONS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import pathlib
import os
import types
import typing


def save_fig(fig, save_dir, save_name, formats):
    if save_dir:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    if not isinstance(formats, list):
        formats = [formats]
    for format in formats:
        fig.savefig(os.path.join(save_dir, save_name + '.' + format))


def adjust_figure_layout(fig, fig_width=None, fig_height=None, pad=0.0):
    if fig_width is not None:
        fig.set_figwidth(fig_width)
    if fig_height is not None:
        fig.set_figheight(fig_height)
    fig.tight_layout(pad=pad)


def _adjust_spines(ax, spines=("left", "bottom"), outward=True):
    for loc, spine in ax.spines.items():
        if loc in spines:
            ax.spines[loc].set_visible(True)
            if outward:
                spine.set_position(("outward", 12))  # outward by 18 points
        else:
            ax.spines[loc].set_visible(False)  # don't draw spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    elif "right" in spines:
        ax.yaxis.set_ticks_position("right")
    else:
        # no yaxis ticks
        ax.yaxis.set_visible(False)

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    elif "top" in spines:
        ax.xaxis.set_ticks_position("top")
    else:
        # no xaxis ticks
        ax.xaxis.set_visible(False)


def adjust_spines(axs):
    only_one_axis = not isinstance(axs, list)
    if only_one_axis:
        axs = [axs]
    for ax in axs:
        _adjust_spines(ax)


class DriveModes:
    FORCE = 0
    DISPLACEMENT = 1


def extract_loading_path(result: static_solver.Result, drive_mode: int, starting_load=0.0):
    mdl = result.get_model()
    u = result.get_displacements()
    f = result.get_forces()
    stability = result.get_stability()
    loaded_dof_indices = mdl.get_loaded_dof_indices()

    f_goal = mdl.get_force_vector()
    f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
    f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    match drive_mode:
        case DriveModes.FORCE:
            path_indices = [0]
            current_load = starting_load
            for index in range(1, f_load.shape[0]):
                load = f_load[index]
                stable = stability[index] == static_solver.StaticSolver.STABLE
                if load > current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case DriveModes.DISPLACEMENT:
            path_indices = [0]
            current_load = starting_load
            for index in range(1, u_load.shape[0]):
                load = u_load[index]
                stable = stability[index] != static_solver.StaticSolver.UNSTABLE
                if load > current_load and stable:
                    current_load = u_load[index]
                    path_indices.append(index)
        case _:
            raise ValueError('unknown drive mode')
    is_restabilization = np.diff(path_indices, prepend=[0]) > 1
    restabilization_indices = np.array(path_indices)[is_restabilization]
    tmp = [is_restabilization[index + 1] == True for index in range(len(is_restabilization) - 1)]
    tmp.append(False)
    critical_indices = np.array(path_indices)[tmp]
    return path_indices, critical_indices.tolist(), restabilization_indices.tolist()


def extract_unloading_path(result: static_solver.Result, drive_mode: int, starting_load=np.inf):
    mdl = result.get_model()
    u = result.get_displacements()
    f = result.get_forces()
    stability = result.get_stability()

    loaded_dof_indices = mdl.get_loaded_dof_indices()
    f_goal = mdl.get_force_vector()
    f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
    f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    match drive_mode:
        case DriveModes.FORCE:
            path_indices = []
            current_load = starting_load
            for index in range(f_load.shape[0] - 1, -1, -1):
                load = f_load[index]
                stable = stability[index] == static_solver.StaticSolver.STABLE
                if load < current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case DriveModes.DISPLACEMENT:
            path_indices = []
            current_load = starting_load
            for index in range(u_load.shape[0] - 1, -1, -1):
                load = u_load[index]
                stable = stability[index] != static_solver.StaticSolver.UNSTABLE
                if load < current_load and stable:
                    current_load = u_load[index]
                    path_indices.append(index)
        case _:
            raise ValueError('unknown drive mode')
    is_restabilization = np.diff(path_indices, prepend=path_indices[0]) < -1
    restabilization_indices = np.array(path_indices)[is_restabilization]
    tmp = [is_restabilization[index + 1] for index in range(len(is_restabilization) - 1)]
    tmp.append(False)
    critical_indices = np.array(path_indices)[tmp]
    return path_indices, critical_indices.tolist(), restabilization_indices.tolist()


def force_displacement_curve_in_ax(result: static_solver.Result, ax: plt.Axes, plot_options,
                                   color=None, label=None):
    po = plot_options
    mdl = result.get_model()
    u = result.get_displacements()
    f = result.get_forces()
    loaded_dof_indices = mdl.get_loaded_dof_indices()

    f_goal = mdl.get_force_vector()
    f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
    f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    stability_markersizes = [po['size_stable_points'],
                             po['size_stabilizable_points'],
                             po['size__unstable_points']]
    if not po['driven_path_only']:
        stability = result.get_stability()
        ax.plot(u_load, f_load, 'k-', linewidth=0.5, zorder=1.1)
        if color is None:  # then color is determined by 'color_mode' set in the plot options

            if po['marker_mode'] == 0:
                # then each point is colored by its stability
                stability_colors = [po['color_for_stable_points'],
                                    po['color_for_stabilizable_points'],
                                    po['color_for_unstable_points']]
                zorder = 2.5
                for i, stability_state in enumerate([static_solver.StaticSolver.STABLE,
                                                     static_solver.StaticSolver.STABILIZABLE,
                                                     static_solver.StaticSolver.UNSTABLE]):
                    current_stability = stability == stability_state
                    if po['show_stability_legend'] and label is None:
                        lbl_i = stability_state
                    elif label is not None and i == 0:
                        lbl_i = label
                    else:
                        lbl_i = ''
                    ax.plot(u_load[current_stability], f_load[current_stability], ls='',
                            color=stability_colors[i],
                            marker=po['default_marker'],
                            markersize=po['default_markersize'] * stability_markersizes[i],
                            zorder=zorder,
                            label=lbl_i)
                    zorder -= 0.1

            elif po['color_mode'] in (1, 2):
                # then each point is colored by the lowest eigenvalue in the stiffness matrix
                cm = po['lowest_eigval_colormap']
                lowest_eigval = (result.get_lowest_eigval_in_force_control()
                                 if po['color_mode'] == 1 else
                                 result.get_lowest_eigval_in_displacement_control())
                max_eigval_magnitude = np.max(np.abs(lowest_eigval))
                cn = plt.Normalize(vmin=-max_eigval_magnitude, vmax=max_eigval_magnitude, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(u_load, f_load, c=sm.to_rgba(lowest_eigval), s=po['default_markersize'],
                           marker=po['default_marker'], label=label if label is not None else '')
            elif po['color_mode'] in (3, 4):
                # then each point is colored by the nb of negative eigenvalues in the stiffness matrix
                nb_negative_eigval = (result.get_nb_of_negative_eigval_in_force_control()
                                      if po['color_mode'] == 3 else
                                      result.get_nb_of_negative_eigval_in_displacement_control())
                cm = 'plasma'
                cn = plt.Normalize(vmin=0, vmax=np.max(nb_negative_eigval), clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(u_load, f_load, c=sm.to_rgba(nb_negative_eigval), s=po['default_markersize'],
                           marker=po['default_marker'], label=label if label is not None else '')

            else:  # 'color_mode' is None
                # then the default color is used
                ax.plot(u_load, f_load, po['default_marker'], color=po['default_color'],
                        label=label if label is not None else '')
        else:  # a color has been specified as input
            ax.plot(u_load, f_load, po['default_marker'], color=color,
                    label=label if label is not None else '')

    if po['drive_mode'] is not None:
        (loading_path_indices,
         loading_critical_indices,
         loading_restabilization_indices) = extract_loading_path(result, po['drive_mode'])
        if po['cycle']:
            (unloading_path_indices,
             unloading_critical_indices,
             unloading_restabilization_indices) = extract_unloading_path(result, po['drive_mode'])
            path_indices = loading_path_indices + unloading_path_indices
            critical_indices = loading_critical_indices + unloading_critical_indices
            restabilization_indices = loading_restabilization_indices + unloading_restabilization_indices
        else:
            path_indices = loading_path_indices
            critical_indices = loading_critical_indices
            restabilization_indices = loading_restabilization_indices

        if po['show_driven_path_legend'] and label is None:
            lbl = f'{"force" if po["drive_mode"] == 0 else "displacement"}-driven path'
        elif label is None:
            lbl = ''
        else:
            lbl = label

        ax.plot(u_load[path_indices], f_load[path_indices], ls='',
                markersize=po['default_markersize'] if po['driven_path_only'] else po['default_markersize'] / 2,
                marker=po['default_marker'],
                color=color if color is not None else po['driven_path_color'],
                label=lbl,
                zorder=1.1, alpha=0.75)
        nb_transitions = min(len(critical_indices), len(restabilization_indices))
        if po['show_snapping_arrows']:
            match po['drive_mode']:
                case DriveModes.FORCE:
                    for i in range(nb_transitions):
                        arrow = mpatches.FancyArrowPatch((u_load[critical_indices[i]],
                                                          f_load[critical_indices[i]]),
                                                         (u_load[restabilization_indices[i]],
                                                          f_load[critical_indices[i]]),
                                                         edgecolor='none',
                                                         facecolor=po[
                                                             'snapping_arrow_color'] if color is None else color,
                                                         mutation_scale=15,
                                                         alpha=po['snapping_arrow_opacity'],
                                                         zorder=1.2)
                        ax.add_patch(arrow)
                case DriveModes.DISPLACEMENT:
                    for i in range(nb_transitions):
                        arrow = mpatches.FancyArrowPatch((u_load[critical_indices[i]],
                                                          f_load[critical_indices[i]]),
                                                         (u_load[critical_indices[i]],
                                                          f_load[restabilization_indices[i]]),
                                                         edgecolor='none',
                                                         facecolor=po[
                                                             'snapping_arrow_color'] if color is None else color,
                                                         mutation_scale=15,
                                                         alpha=po['snapping_arrow_opacity'],
                                                         zorder=1.2)
                        ax.add_patch(arrow)
    elif po['driven_path_only']:
        raise ValueError('Inconsistent plot options: "driven_path_only = True", and "drive_mode = None"')


def force_displacement_curve(
        result: static_solver.Result | list[static_solver.Result] | typing.Iterator[static_solver.Result],
        save_dir, save_name, color=None, label=None, show=True, xlim=None, ylim=None,
        **plot_options):
    po = DEFAULT_PLOT_OPTIONS.copy()
    po.update(plot_options)

    with plt.style.context(po['stylesheet_path']):
        fig, ax = plt.subplots(figsize=(po['figure_width'], po['figure_height']))
        if isinstance(result, list):
            for i, r in enumerate(result):
                force_displacement_curve_in_ax(r, ax, po,
                                               color=color[i] if color is not None else None,
                                               label=label[i] if label is not None else None)
        elif isinstance(result, types.GeneratorType):
            for r in result:
                force_displacement_curve_in_ax(r, ax, po,
                                               color=next(color) if color is not None else None,
                                               label=next(label) if label is not None else None)
        else:
            force_displacement_curve_in_ax(result, ax, plot_options, color=color, label=label)

        if (label is not None
                or (color is None and po['show_stability_legend'] and po['color_mode'] == 0)
                or (color is None and po['show_driven_path_legend'] and po['drive_mode'] in (0, 1))):
            ax.legend(numpoints=5, markerscale=1.5)

        ax.set_xlabel('displacement (mm)')
        ax.set_ylabel('force (N)')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        adjust_spines(ax)
        adjust_figure_layout(fig)
        save_fig(fig, save_dir, save_name, ["png", "pdf"])
        if show:
            plt.show()
        else:
            plt.close()
