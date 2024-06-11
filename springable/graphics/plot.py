from ..simulation import static_solver
from .default_graphics_settings import DEFAULT_PLOT_OPTIONS
from .figure_utils import figure_formatting as ff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import types
import typing


class DriveModes:
    FORCE = 0
    DISPLACEMENT = 1


def extract_loading_path(result: static_solver.Result, drive_mode: int, starting_index: int = 0):
    mdl = result.get_model()
    u = result.get_displacements()
    f = result.get_forces()
    stability = result.get_stability()
    loaded_dof_indices = mdl.get_loaded_dof_indices()
    if starting_index < 0:
        starting_index = u.shape[0] + starting_index

    f_goal = mdl.get_force_vector()
    f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
    f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    match drive_mode:
        case DriveModes.FORCE:
            path_indices = [0]
            current_load = f_load[starting_index]
            for index in range(starting_index, f_load.shape[0]):
                load = f_load[index]
                stable = stability[index] == static_solver.StaticSolver.STABLE
                if load > current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case DriveModes.DISPLACEMENT:
            path_indices = [0]
            current_load = u_load[starting_index]
            for index in range(starting_index, u_load.shape[0]):
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


def extract_unloading_path(result: static_solver.Result, drive_mode: int, starting_index: int = -1):
    mdl = result.get_model()
    u = result.get_displacements()
    f = result.get_forces()
    stability = result.get_stability()
    if starting_index < 0:
        starting_index = u.shape[0] + starting_index

    loaded_dof_indices = mdl.get_loaded_dof_indices()
    f_goal = mdl.get_force_vector()
    f_goal_normalized = f_goal[loaded_dof_indices] / np.linalg.norm(f_goal[loaded_dof_indices])
    f_load = np.sum((f[:, loaded_dof_indices] - f[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    u_load = np.sum((u[:, loaded_dof_indices] - u[0, loaded_dof_indices]) * f_goal_normalized, axis=1)
    match drive_mode:
        case DriveModes.FORCE:
            path_indices = []
            current_load = f_load[starting_index]
            for index in range(starting_index, -1, -1):
                load = f_load[index]
                stable = stability[index] == static_solver.StaticSolver.STABLE
                if load < current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case DriveModes.DISPLACEMENT:
            path_indices = []
            current_load = u_load[starting_index]
            for index in range(starting_index, -1, -1):
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
    stability_markersizes = [po['size_for_stable_points'],
                             po['size_for_stabilizable_points'],
                             po['size_for_unstable_points']]
    if not po['driven_path_only']:
        stability = result.get_stability()
        ax.plot(u_load, f_load, 'k-', linewidth=0.5, zorder=3.0)
        if color is None:  # then color is determined by 'color_mode' set in the plot options

            if po['color_mode'] == 0:
                # then each point is colored by its stability
                stability_colors = [po['color_for_stable_points'],
                                    po['color_for_stabilizable_points'],
                                    po['color_for_unstable_points']]
                zorder = 1.0
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
                           marker=po['default_marker'], label=label if label is not None else '', zorder=1)
                cbar = plt.colorbar(sm, cax=None, ax=ax)
                if po['color_mode'] == 1:
                    cbar.ax.set_title('$\\lambda_{\\text{min}}$', loc='left')
                else:
                    cbar.ax.set_title('$\\bar{\\lambda}_{\\text{min}}$', loc='left')
            elif po['color_mode'] in (3, 4):
                # then each point is colored by the nb of negative eigenvalues in the stiffness matrix
                nb_negative_eigval = (result.get_nb_of_negative_eigval_in_force_control()
                                      if po['color_mode'] == 3 else
                                      result.get_nb_of_negative_eigval_in_displacement_control())

                cm = plt.get_cmap(po['nb_negative_eigval_colormap'], np.max(nb_negative_eigval) + 1)
                cn = plt.Normalize(vmin=0 - 0.5, vmax=np.max(nb_negative_eigval) + 0.5, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(u_load, f_load, c=sm.to_rgba(nb_negative_eigval), s=po['default_markersize'],
                           marker=po['default_marker'], label=label if label is not None else '', zorder=1)

                cbar = plt.colorbar(sm, cax=None, ax=ax, ticks=np.arange(0, np.max(nb_negative_eigval) + 1))
                if po['color_mode'] == 3:
                    cbar.ax.set_title('$\\sum_i (\\lambda_i < 0)$', loc='left')
                else:
                    cbar.ax.set_title('$\\sum_i (\\bar{\\lambda}_i < 0)$', loc='left')

            else:  # 'color_mode' is None
                # then the default color is used
                ax.plot(u_load, f_load, po['default_marker'], color=po['default_color'],
                        label=label if label is not None else '', zorder=1)
        else:  # a color has been specified as input
            zorder = 1.0
            for i, stability_state in enumerate([static_solver.StaticSolver.STABLE,
                                                 static_solver.StaticSolver.STABILIZABLE,
                                                 static_solver.StaticSolver.UNSTABLE]):
                if label is not None and i == 0:
                    lbl_i = label
                else:
                    lbl_i = ''
                current_stability = stability == stability_state
                ax.plot(u_load[current_stability], f_load[current_stability], ls='',
                        color=color,
                        marker=po['default_marker'],
                        markersize=po['default_markersize'] * stability_markersizes[i],
                        zorder=zorder,
                        label=lbl_i)
                zorder -= 0.1

    if po['drive_mode'] is not None:
        (loading_path_indices,
         loading_critical_indices,
         loading_restabilization_indices) = extract_loading_path(result, po['drive_mode'])
        if po['loading_sequence'] in ('cycle', 'loading_unloading'):
            unloading_start_index = loading_path_indices[-1] if po['loading_sequence'] == 'cycle' else -1
            (unloading_path_indices,
             unloading_critical_indices,
             unloading_restabilization_indices) = extract_unloading_path(result, po['drive_mode'],
                                                                         starting_index=unloading_start_index)

            path_indices = loading_path_indices + unloading_path_indices
            critical_indices = loading_critical_indices + unloading_critical_indices
            restabilization_indices = loading_restabilization_indices + unloading_restabilization_indices
        else:
            path_indices = loading_path_indices
            critical_indices = loading_critical_indices
            restabilization_indices = loading_restabilization_indices

        if label is None:
            if po['show_driven_path_legend']:
                lbl = f'{"force" if po["drive_mode"] == 0 else "displacement"}-driven path'
            else:
                lbl = ''
        else:
            lbl = label if po['driven_path_only'] else ''

        if po['show_driven_path']:
            ax.plot(u_load[path_indices], f_load[path_indices], ls='',
                    markersize=po['size_for_driven_path'] * po['default_markersize'],
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


def parametric_force_displacement_curve(results: list[static_solver.Result] | typing.Iterator[static_solver.Result],
                                        parameter_name: str, parameter_data: dict, parameter_values: list[float | str],
                                        save_dir, save_name=None, show=True, xlim=None, ylim=None, **plot_options):
    po = DEFAULT_PLOT_OPTIONS.copy()
    po.update(plot_options)

    if parameter_data['is range parameter']:
        cmap = plt.get_cmap(po['range_parameter_scan_colormap'])
        lb = parameter_data['lower bound']
        ub = parameter_data['upper bound']
        cn = plt.Normalize(vmin=lb, vmax=ub, clip=True)
        scalar_mappable = mcm.ScalarMappable(norm=cn, cmap=cmap)
        colors = [scalar_mappable.to_rgba(val) for val in parameter_values]
        labels = None
    else:
        cmap = plt.get_cmap(po['discrete_parameter_scan_colormap'])
        cn = plt.Normalize(vmin=0, vmax=parameter_data['nb samples'])
        scalar_mappable = mcm.ScalarMappable(norm=cn, cmap=cmap)
        colors = [scalar_mappable.to_rgba(i) for i in range(len(parameter_values))]
        if len(parameter_values) <= po['max_nb_legend_entries_for_discrete_parameter']:
            if parameter_data['is numeric parameter']:
                labels = [f'{parameter_name} = {val:.4f}' for val in parameter_values]
            else:
                labels = [f'{parameter_name} = {val}' for val in parameter_values]
        else:
            labels = None

    with plt.style.context(po['stylesheet_path']):
        fig, ax = plt.subplots(figsize=(po['figure_width'], po['figure_height']))
        i = 0
        for res in results:
            try:
                force_displacement_curve_in_ax(res, ax, po,
                                               color=colors[i],
                                               label=labels[i] if labels is not None else None)
            except static_solver.UnusableSolution:
                pass
            i += 1

        if labels is not None:
            ax.legend(numpoints=5, markerscale=1.5)

        if parameter_data['is range parameter']:
            cbar = fig.colorbar(scalar_mappable, cax=None, ax=ax)
            cbar.ax.set_title(parameter_name, loc='left')

        ax.set_xlabel('displacement (mm)')
        ax.set_ylabel('force (N)')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ff.adjust_spines(ax)
        ff.adjust_figure_layout(fig)
        if save_name is None:
            save_name = po['default_plot_name']
        ff.save_fig(fig, save_dir, save_name, ["png", "pdf"])
        if show:
            plt.show()
        else:
            plt.close()


def force_displacement_curve(result: static_solver.Result, save_dir, save_name=None, color=None, label=None, show=True,
                             xlim=None, ylim=None, **plot_options):
    po = DEFAULT_PLOT_OPTIONS.copy()
    po.update(plot_options)

    with plt.style.context(po['stylesheet_path']):
        fig, ax = plt.subplots(figsize=(po['figure_width'], po['figure_height']))
        force_displacement_curve_in_ax(result, ax, po, color=color, label=label)

        if (label is not None
                or (po['show_stability_legend'] and po['color_mode'] == 0)
                or (po['show_driven_path_legend'] and po['drive_mode'] in (0, 1))):
            ax.legend(numpoints=5, markerscale=1.5)

        ax.set_xlabel('displacement (mm)')
        ax.set_ylabel('force (N)')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ff.adjust_spines(ax)
        ff.adjust_figure_layout(fig)
        if save_name is None:
            save_name = po['default_plot_name']
        ff.save_fig(fig, save_dir, save_name, ["png", "pdf"])
        if show:
            plt.show()
        else:
            plt.close()
