from ..mechanics.static_solver import Result, UnusableSolution
from ..mechanics.result_processing import (extract_branches, extract_loading_path, extract_unloading_path,
                                           LoadingPathEmpty, DiscontinuityInTheSolutionPath, LoadingPathIsNotDescribedBySolution)
from ..mechanics.stability_states import StabilityStates
from .default_graphics_settings import PlotOptions
from . import figure_formatting as ff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.patches import ArrowStyle
import typing


def curve_in_ax(processing_fun, result: Result, ax: plt.Axes, plot_options: PlotOptions, color, label, zorder=1.0):
    po = plot_options
    x, y = processing_fun(result)
    stability_colors = [po.color_for_stable_points,
                        po.color_for_stabilizable_points,
                        po.color_for_unstable_points]

    stability_styles = [po.style_for_stable_branches,
                        po.style_for_stabilizable_branches,
                        po.style_for_unstable_branches]

    stability_markersizes = [po.size_for_stable_points,
                             po.size_for_stabilizable_points,
                             po.size_for_unstable_points]

    stability_labels = [po.label_for_stable_points,
                        po.label_for_stabilizable_points,
                        po.label_for_unstable_points]

    if not po.driven_path_only:
        branches = extract_branches(result)

        if po.plot_style == 'points':
            ax.plot(x, y, 'k-', linewidth=0.5, zorder=zorder)

        if color is None:  # then color is determined by 'color_mode' set in the plot options

            if po.color_mode == 'stability':
                # then each point is colored by its stability
                for i, stability_state in enumerate([StabilityStates.STABLE,
                                                     StabilityStates.STABILIZABLE,
                                                     StabilityStates.UNSTABLE]):
                    if po.show_stability_legend and label is None:
                        lbl_i = stability_labels[i]
                    elif label is not None and i == 0:
                        lbl_i = label
                    else:
                        lbl_i = ''
                    for j, branch in enumerate(branches[stability_state]):
                        zorder -= 0.01
                        ax.plot(x[branch], y[branch],
                                ls='' if po.plot_style == 'points' else stability_styles[i],
                                lw=po.default_linewidth,
                                solid_capstyle='round',
                                color=stability_colors[i],
                                marker=po.default_marker if po.plot_style == 'points' else None,
                                markersize=po.default_markersize * stability_markersizes[i],
                                zorder=zorder,
                                label=lbl_i if j == 0 else '')

            elif po.color_mode in ('min_eigval_fd', 'min_eigval_ud'):
                # then each point is colored by the lowest eigenvalue in the stiffness matrix
                cm = po.lowest_eigval_colormap
                lowest_eigval = (result.get_lowest_eigval_in_force_control()
                                 if po.color_mode == 'min_eigval_fd' else
                                 result.get_lowest_eigval_in_displacement_control())
                max_eigval_magnitude = np.max(np.abs(lowest_eigval))
                cn = plt.Normalize(vmin=-max_eigval_magnitude, vmax=max_eigval_magnitude, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(x, y, c=sm.to_rgba(lowest_eigval), s=po.default_markersize,
                           marker=po.default_marker, label=label if label is not None else '', zorder=1)
                cbar = plt.colorbar(sm, cax=None, ax=ax)
                if po.color_mode == 'min_eigval_fd':
                    cbar.ax.set_title('$\\lambda_{\\text{min}}$', loc='left')
                else:
                    cbar.ax.set_title('$\\bar{\\lambda}_{\\text{min}}$', loc='left')
            elif po.color_mode in ('nb_neg_eigval_fd', 'nb_neg_eigval_ud'):
                # then each point is colored by the nb of negative eigenvalues in the stiffness matrix
                nb_negative_eigval = (result.get_nb_of_negative_eigval_in_force_control()
                                      if po.color_mode == 'nb_neg_eigval_fd' else
                                      result.get_nb_of_negative_eigval_in_displacement_control())

                cm = plt.get_cmap(po.nb_negative_eigval_colormap, np.max(nb_negative_eigval) + 1)
                cn = plt.Normalize(vmin=0 - 0.5, vmax=np.max(nb_negative_eigval) + 0.5, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(x, y, c=sm.to_rgba(nb_negative_eigval), s=po.default_markersize,
                           marker=po.default_marker, label=label if label is not None else '', zorder=1)

                cbar = plt.colorbar(sm, cax=None, ax=ax, ticks=np.arange(0, np.max(nb_negative_eigval) + 1))
                if po.color_mode == 'nb_neg_eigval_fd':
                    cbar.ax.set_title('$\\sum_i (\\lambda_i < 0)$', loc='left')
                else:
                    cbar.ax.set_title('$\\sum_i (\\bar{\\lambda}_i < 0)$', loc='left')
            elif po.color_mode == 'energy':
                # then each point is colored by the elastic energy stored at that state
                a = result.get_model().get_assembly()
                x0 = a.get_coordinates()
                u = result.get_displacements()
                energies = []
                for i in range(u.shape[0]):
                    xi = x0 + u[i, :]
                    a.set_coordinates(xi)
                    energies.append(a.compute_elastic_energy())
                a.set_coordinates(x0)
                energies = np.array(energies)

                max_energy = np.max(energies)
                min_energy = np.min(energies)
                cm = po.energy_colormap
                cn = plt.Normalize(vmin=min_energy, vmax=max_energy, clip=True)
                sm = mcm.ScalarMappable(norm=cn, cmap=cm)
                ax.scatter(x, y, c=sm.to_rgba(energies), s=po.default_markersize,
                           marker=po.default_marker, label=label if label is not None else '', zorder=1)
                cbar = plt.colorbar(sm, cax=None, ax=ax)
                cbar.ax.set_title('energy', loc='left')

            else:  # 'color_mode' is 'none' or something else
                # then the default color is used
                if po.plot_style == 'points':
                    ax.plot(x, y, po.default_marker,
                            color=po.default_color,
                            alpha=po.default_opacity,
                            markersize=po.default_markersize,
                            label=label if label is not None else '', zorder=1)
                else:
                    for i, stability_state in enumerate([StabilityStates.STABLE,
                                                         StabilityStates.STABILIZABLE,
                                                         StabilityStates.UNSTABLE]):
                        for j, branch in enumerate(branches[stability_state]):
                            ax.plot(x[branch], y[branch],
                                    ls=stability_styles[i],
                                    lw=po.default_linewidth,
                                    solid_capstyle='round',
                                    color=po.default_color,
                                    alpha=po.default_opacity,
                                    zorder=1,
                                    label=stability_labels[i] if j == 0 else '')

        else:  # a color has been specified as input
            zorder = 1.0
            for i, stability_state in enumerate([StabilityStates.STABLE,
                                                 StabilityStates.STABILIZABLE,
                                                 StabilityStates.UNSTABLE]):
                if label is not None and i == 0:
                    lbl_i = label
                else:
                    lbl_i = ''
                for j, branch in enumerate(branches[stability_state]):
                    ax.plot(x[branch], y[branch],
                            ls='' if po.plot_style == 'points' else stability_styles[i],
                            lw=po.default_linewidth,
                            solid_capstyle='round',
                            color=color,
                            marker=po.default_marker if po.plot_style == 'points' else None,
                            markersize=po.default_markersize * stability_markersizes[i],
                            zorder=zorder,
                            label=lbl_i if j == 0 else '')
                    zorder -= 0.1

    if po.drive_mode != 'none' and (po.show_driven_path or po.show_snapping_arrows):
        try:
            (loading_path_indices,
             loading_critical_indices,
             loading_restabilization_indices) = extract_loading_path(result, po.drive_mode)
            if po.loading_sequence in ('cycle', 'loading_unloading'):
                unloading_start_index = loading_path_indices[-1] if po.loading_sequence == 'cycle' else -1
                (unloading_path_indices,
                 unloading_critical_indices,
                 unloading_restabilization_indices) = extract_unloading_path(result, po.drive_mode,
                                                                             starting_index=unloading_start_index)

                path_indices = loading_path_indices + unloading_path_indices
                critical_indices = loading_critical_indices + unloading_critical_indices
                restabilization_indices = loading_restabilization_indices + unloading_restabilization_indices
            else:
                path_indices = loading_path_indices
                critical_indices = loading_critical_indices
                restabilization_indices = loading_restabilization_indices

            if label is None:
                if po.show_driven_path_legend:
                    lbl = f'{po.drive_mode}-driven path'
                else:
                    lbl = ''
            else:
                lbl = label if po.driven_path_only else ''

            if po.show_driven_path:
                if po.plot_style == 'points':
                    ax.plot(x[path_indices], y[path_indices], ls='',
                            markersize=po.size_for_driven_path * po.default_markersize,
                            marker=po.default_marker,
                            color=color if color is not None else po.driven_path_color,
                            label=lbl,
                            zorder=1.1)
                else:
                    nb_transitions = min(len(critical_indices), len(restabilization_indices))
                    if nb_transitions == 0:
                        ax.plot(x[path_indices], y[path_indices],
                                ls='-',
                                lw=po.default_linewidth * po.size_for_driven_path,
                                color=color if color is not None else po.driven_path_color,
                                label=lbl,
                                zorder=1.1)
                    else:  # at least one snapping event
                        ax.plot(x[path_indices[0]:critical_indices[0]],
                                y[path_indices[0]:critical_indices[0]],
                                ls='-',
                                lw=po.default_linewidth * po.size_for_driven_path,
                                color=color if color is not None else po.driven_path_color,
                                label=lbl,
                                zorder=1.1)
                        for i in range(1, nb_transitions):
                            ax.plot(x[restabilization_indices[i-1]:critical_indices[i]],
                                    y[restabilization_indices[i-1]:critical_indices[i]],
                                    ls='-',
                                    lw=po.default_linewidth * po.size_for_driven_path,
                                    color=color if color is not None else po.driven_path_color,
                                    label=lbl,
                                    zorder=1.1)
                        ax.plot(x[restabilization_indices[-1]:path_indices[-1]],
                                y[restabilization_indices[-1]:path_indices[-1]],
                                ls='-',
                                lw=po.default_linewidth * po.size_for_driven_path,
                                color=color if color is not None else po.driven_path_color,
                                label=lbl,
                                zorder=1.1)

            if po.show_snapping_arrows:
                if po.drive_mode in ('force', 'displacement'):
                    nb_transitions = min(len(critical_indices), len(restabilization_indices))
                    for i in range(nb_transitions):
                        start = np.array((x[critical_indices[i]], y[critical_indices[i]]))
                        end = np.array((x[restabilization_indices[i]], y[restabilization_indices[i]]))

                        ax.annotate(
                            "",
                            xy=start,
                            xytext=end,
                            arrowprops=dict(
                                arrowstyle=ArrowStyle.CurveA(head_length=po.snapping_arrow_headlength,
                                                             head_width=po.snapping_arrow_headwidth),
                                color=po.snapping_arrow_color,
                                alpha=po.snapping_arrow_opacity,
                                linewidth=po.snapping_arrow_width,
                                ls=po.snapping_arrow_style,
                                shrinkA=0.0, shrinkB=0.0,
                            ),
                            annotation_clip=False)
        except LoadingPathEmpty:
            print(f"Cannot draw the {po.drive_mode}-driven path, "
                  f"because not stable points have been found under these loading conditions")
            pass
        except DiscontinuityInTheSolutionPath:
            print(f"Cannot draw the {po.drive_mode}-driven path, "
                  f"because discontinuities have been detected in the solution path. "
                  f"Run a more refined simulation to find a valid solution"
                  f" (use a smaller 'radius' value in the solver settings)."
            )
            pass
        except LoadingPathIsNotDescribedBySolution:
            extra_info = ""
            if po.drive_mode == 'displacement':
                extra_info += ("To extract the displacement-driven path from a simulation, "
                "only a single degree of freedom can be loaded for the final loadstep, "
                "as the solver treats multiple loads (within a loadstep) as evolving proportionally in force, not displacement.\n"
                "TIP: you might want to split your multi-load final loadstep into multiple loadsteps with a single load in the final loadstep, "
                "using the 'then' keyword in the LOADING section. See documentation for more details:\n"
                "https://paulducarme.com/springable/creating_the_spring_model_csv_file/#the-loading-section"
                "\nIf this explanation is unclear, feel free to send an email to the author at paulducarme@hotmail.com, who is going to "
                "do his best to answer quickly.")
            print(f"Cannot draw the {po.drive_mode}-driven path or snapping arrows, "
                  f"because the model does not describe a {po.drive_mode}-driven loading. " + extra_info)
            pass
    elif po.driven_path_only:
        raise ValueError('Inconsistent plot options: "driven_path_only == True", and "drive_mode == "none"')


def force_displacement_curve_in_ax(result: Result, ax: plt.Axes, plot_options,
                                   color=None, label=None):
    def processing_fun(res: Result):
        return res.get_equilibrium_path()

    curve_in_ax(processing_fun, result, ax, plot_options, color, label)


def parametric_curve(processing_fun: callable,
                     results: list[Result] | typing.Iterator[Result],
                     parameter_name: str, parameter_data: dict, parameter_values: list[float | str],
                     save_dir=None, save_name=None, show=True, xlim=None, ylim=None, xlabel=None, ylabel=None,
                     **plot_options):
    if save_dir is None and show is None:
        print("Plot-making cancelled because no save directory and show = False")
        return
    po = PlotOptions()
    po.update(**plot_options)

    if parameter_data['is range parameter']:
        cmap = plt.get_cmap(po.range_parameter_scan_colormap)
        lb = parameter_data['lower bound']
        ub = parameter_data['upper bound']
        cn = plt.Normalize(vmin=lb, vmax=ub, clip=True)
        scalar_mappable = mcm.ScalarMappable(norm=cn, cmap=cmap)
        colors = [scalar_mappable.to_rgba(val) for val in parameter_values]
        labels = None
    else:
        cmap = plt.get_cmap(po.discrete_parameter_scan_colormap)
        cn = plt.Normalize(vmin=0, vmax=parameter_data['nb samples'])
        scalar_mappable = mcm.ScalarMappable(norm=cn, cmap=cmap)
        colors = [scalar_mappable.to_rgba(i) for i in range(len(parameter_values))]
        if len(parameter_values) <= po.max_nb_legend_entries_for_discrete_parameter:
            if parameter_data['is numeric parameter']:
                labels = [f'{parameter_name} = {val:.4f}' for val in parameter_values]
            else:
                labels = [f'{parameter_name} = {val}' for val in parameter_values]
        else:
            labels = None

    with plt.style.context(po.stylesheet):
        fig, ax = plt.subplots(figsize=(po.figure_width, po.figure_height))
        ax.set_box_aspect(po.axis_box_aspect)
        i = 0
        for res in results:
            try:
                curve_in_ax(processing_fun, res, ax, po, color=colors[i],
                            label=labels[i] if labels is not None else None, zorder=i+1)
            except UnusableSolution:
                pass
            i += 1

        if labels is not None:
            ax.legend(numpoints=5, markerscale=1.5)

        if parameter_data['is range parameter']:
            cbar = fig.colorbar(scalar_mappable, cax=None, ax=ax)
            cbar.ax.set_title(parameter_name, loc='left')

        ax.set_xlabel(xlabel if xlabel is not None else po.default_xlabel)
        ax.set_ylabel(ylabel if ylabel is not None else po.default_ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        elif po.enforce_xlim:
            ax.set_xlim((po.xmin, po.xmax))
        if ylim is not None:
            ax.set_ylim(ylim)
        elif po.enforce_ylim:
            ax.set_ylim((po.ymin, po.ymax))

        spines = []
        spines += ['left'] if po.show_left_spine else []
        spines += ['right'] if po.show_right_spine else []
        spines += ['top'] if po.show_top_spine else []
        spines += ['bottom'] if po.show_bottom_spine else []
        if po.hide_ticklabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        ff.adjust_spines(ax, po.spine_offset, spines)
        ff.adjust_figure_layout(fig)

        if save_dir is not None:
            if save_name is None:
                save_name = po.default_plot_name
            ff.save_fig(fig, save_dir, save_name, ["png", "pdf"], transparent=po.transparent, dpi=po.dpi)
        if show:
            plt.show()
        else:
            plt.close(fig)


def parametric_force_displacement_curve(results: list[Result] | typing.Iterator[Result],
                                        parameter_name: str, parameter_data: dict, parameter_values: list[float | str],
                                        save_dir=None, save_name=None, show=True, xlim=None, ylim=None, **plot_options):
    def processing_fun(res: Result):
        return res.get_equilibrium_path()

    parametric_curve(processing_fun, results, parameter_name, parameter_data, parameter_values,
                     save_dir, save_name, show, xlim, ylim, **plot_options)


def force_displacement_curve(result: Result, save_dir=None, save_name=None, color=None, label=None,
                             show=True, xlim=None, ylim=None, preplot=None, afterplot=None,
                             **plot_options):
    def processing_fun(res: Result):
        return res.get_equilibrium_path()

    curve(processing_fun, result, save_dir, save_name, color, label, show,
          xlim=xlim, ylim=ylim, preplot=preplot, afterplot=afterplot, **plot_options)


def curve(processing_fun: callable, result: Result,
          save_dir=None, save_name=None, color=None, label=None, show=True,
          xlabel=None, ylabel=None, xlim=None, ylim=None, preplot=None, afterplot=None,
          **plot_options):
    if save_dir is None and show is None:
        print("Plot-making cancelled because no save directory and show = False")
        return
    po = PlotOptions()
    po.update(**plot_options)

    with plt.style.context(po.stylesheet):
        result.check_if_solution_usable()
        
        fig, ax = plt.subplots(figsize=(po.figure_width, po.figure_height))
        ax.set_box_aspect(po.axis_box_aspect)
        if preplot is not None:
            preplot(fig, ax)

        curve_in_ax(processing_fun, result, ax, po, color=color, label=label)

        if (label is not None
                or (po.show_stability_legend and po.color_mode == 'stability')
                or (po.show_stability_legend and po.color_mode == 'none' and po.plot_style == 'line')
                or (po.show_driven_path_legend and po.show_driven_path and po.drive_mode in ('force', 'displacement'))):
            ax.legend(numpoints=5, markerscale=1.5)

        ax.set_xlabel(xlabel if xlabel is not None else po.default_xlabel)
        ax.set_ylabel(ylabel if ylabel is not None else po.default_ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        elif po.enforce_xlim:
            ax.set_xlim((po.xmin, po.xmax))
        if ylim is not None:
            ax.set_ylim(ylim)
        elif po.enforce_ylim:
            ax.set_ylim((po.ymin, po.ymax))
        spines = []
        spines += ['left'] if po.show_left_spine else []
        spines += ['right'] if po.show_right_spine else []
        spines += ['top'] if po.show_top_spine else []
        spines += ['bottom'] if po.show_bottom_spine else []
        if po.hide_ticklabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ff.adjust_spines(ax, po.spine_offset, spines)
        ff.adjust_figure_layout(fig)
        if afterplot is not None:
            afterplot(fig, ax)
        if save_dir is not None:
            if save_name is None:
                save_name = po.default_plot_name
            ff.save_fig(fig, save_dir, save_name, ["png", "pdf"], transparent=po.transparent, dpi=po.dpi)
        if show:
            plt.show()
        else:
            plt.close(fig)
