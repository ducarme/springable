from dataclasses import dataclass
from ..utils.dataclass_utils import Updatable


@dataclass
class GeneralOptions(Updatable):
    generate_model_drawing: bool = True
    show_model_drawing: bool = True
    generate_fd_plot: bool = True
    show_fd_plot: bool = True
    show_custom_plot: bool = True
    generate_animation: bool = True
    show_animation: bool = True
    generate_parametric_fd_plots: bool = True
    show_parametric_fd_plots: bool = True
    show_parametric_custom_plots: bool = True
    generate_all_model_drawings: bool = False
    show_all_model_drawings: bool = False
    generate_all_fd_plots: bool = False
    generate_all_animations: bool = False


@dataclass
class PlotOptions(Updatable):
    show_driven_path: bool = False
    driven_path_only: bool = False
    loading_sequence: str = 'cycle'
    show_snapping_arrows: bool = True
    plot_style: str = 'points'
    drive_mode: str = 'none'
    color_mode: str = 'stability'
    show_stability_legend: bool = True
    show_driven_path_legend: bool = True
    figure_width: float = 4.
    figure_height: float = 4.
    transparent: bool = False
    dpi: float = 300
    range_parameter_scan_colormap: str = 'viridis'
    discrete_parameter_scan_colormap: str = 'tab10'
    lowest_eigval_colormap: str = 'PuOr'
    nb_negative_eigval_colormap: str = 'plasma'
    energy_colormap: str = 'viridis'
    max_nb_legend_entries_for_discrete_parameter: int = 4
    default_plot_name: str = 'fd_curve'
    default_xlabel: str = 'displacement'
    default_ylabel: str = 'force'
    default_color: str = '#a0a0a0'
    default_opacity: float = 1.0,
    default_marker: str = 'o'
    default_markersize: float = 2.5
    default_linewidth: float = 2.0
    color_for_stable_points: str = '#86A7FC'
    color_for_stabilizable_points: str = '#FFDD95'
    color_for_unstable_points: str = '#FF9843'
    style_for_stable_branches: str = '-'
    style_for_stabilizable_branches: str = '--'
    style_for_unstable_branches: str = ':'
    label_for_stable_points: str = 'stable'
    label_for_stabilizable_points: str = 'stabilizable'
    label_for_unstable_points: str = 'unstable'
    driven_path_color: str = '#444444'
    size_for_driven_path: float = 0.4
    snapping_arrow_color: str = '#aaaaaa'
    snapping_arrow_opacity: float = 0.35
    snapping_arrow_style: str = '-'
    snapping_arrow_width: float = 2.5
    size_for_stable_points: float = 1.
    size_for_stabilizable_points: float = .66
    size_for_unstable_points: float = .33
    stylesheet: str = 'default'
    spine_offset: float = 10.
    show_top_spine: bool = False
    show_bottom_spine: bool = True
    show_left_spine: bool = True
    show_right_spine: bool = False
    hide_ticklabels: bool = False


@dataclass
class AnimationOptions(Updatable):
    fps: int = 25
    nb_frames: int = 100
    dpi: int = 200
    drive_mode: str = 'none'
    cycling: bool = False
    default_animation_name: str = 'animation'
    save_as_mp4: bool = False
    save_as_transparent_mov: bool = False
    save_as_gif: bool = True
    save_frames_as_png: bool = False
    side_plot_mode: str = 'force_displacement_curve'
    stylesheet: str = 'default'
    spine_offset: float = 10.
    animated_equilibrium_point_color: str = '#ff0000'
    animated_equilibrium_point_size: float = 2.
    animation_width: float = 8
    animation_height: float = 4.5


@dataclass
class AssemblyAppearanceOptions(Updatable):
    drawing_fig_width: float = 3
    drawing_fig_height: float = 3
    drawing_dpi: float = 300
    show_axes: bool = False
    stylesheet: str = 'default'
    transparent: bool = False
    coloring_mode: str = 'generalized_force'
    color_elements: bool = True
    color_forces: bool = False
    colormap: str = 'coolwarm'
    show_state_of_hysterons: bool = True
    hysteron_state_label_size: float = 10.
    hysteron_state_bg_color: str = '#E0E0E0'
    hysteron_state_txt_color: str = '#101010'
    show_forces: bool = True
    hide_low_preloading_forces: bool = False
    low_preloading_force_threshold: float = 0.0
    spring_style: str = 'simple'
    spring_nb_coils: int = 6
    spring_aspect: float = 0.4
    spring_linewidth: float = 2.
    spring_width_scaling: float = 1.0
    spring_default_opacity: float = 1.0
    spring_default_color: str = '#CECECE'
    angular_spring_style: str = 'line'
    angular_spring_nb_coils: int = 6
    angular_spring_radii_ratio: float = 6
    angular_spring_aspect: float = 0.4
    angular_spring_linewidth: float = 2.
    angular_spring_radius_scaling: float = 1.0
    angular_spring_default_opacity: float = 1.0
    angular_spring_default_color: str = '#CECECE'
    area_spring_default_opacity: float = 0.5
    area_spring_default_color: str = '#CECECE'
    line_spring_linewidth: float = 2.
    line_spring_default_opacity: float = 1.0
    line_spring_default_color: str = '#CECECE'
    line_spring_dot_color: str = "#ffffff"
    line_spring_dot_opacity: float = 1.0
    distance_spring_line_linewidth: float = 1.0
    distance_spring_line_default_color: str = '#CECECE'
    distance_spring_line_default_opacity: float = 0.7
    node_style: str = 'simple'
    node_size: float = 3.
    node_color: str = '#101010'
    node_edgecolor: str = '#101010'
    node_edgewidth: float = 1.0
    show_node_numbers: bool = False
    node_nb_color: str = '#CECECE'
    node_nb_shift_x: float = 5
    node_nb_shift_y: float = 5
    node_nb_fontsize: float = 10
    force_vector_style: str ='basic'
    preload_force_opacity: float = 0.65
    preload_force_default_color: str = '#cecece'
    force_default_color: str = '#000000'
    force_vector_scaling: float = 1.0
    force_vector_connection: str = 'tail'
