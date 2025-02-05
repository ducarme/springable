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
    drive_mode: str = 'none'
    color_mode: str = 'stability'
    show_stability_legend: bool = True
    show_driven_path_legend: bool = True
    figure_width: float = 4.
    figure_height: float = 4.
    transparent: bool = False
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
    default_marker: str = 'o'
    default_markersize: float = 2.5
    color_for_stable_points: str = '#86A7FC'
    color_for_stabilizable_points: str = '#FFDD95'
    color_for_unstable_points: str = '#FF9843'
    driven_path_color: str = '#444444'
    size_for_driven_path: float = 0.4
    snapping_arrow_color: str = '#aaaaaa'
    snapping_arrow_opacity: float = 0.35
    size_for_stable_points: float = 1.
    size_for_stabilizable_points: float = .66
    size_for_unstable_points: float = .33
    stylesheet: str = 'default'
    spine_offset: float = 10.


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


@dataclass
class AssemblyAppearanceOptions(Updatable):
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
    force_vector_scaling: float = 1.0
    spring_style: str = 'simple'
    spring_nb_coils: int = 6
    spring_aspect: float = 0.4
    spring_linewidth: float = 2.
    spring_width_scaling: float = 1.0
    spring_default_opacity: float = 1.0
    spring_default_color: str = '#CECECE'
    rotation_spring_style: str = 'line'
    rotation_spring_nb_coils: int = 6
    rotation_spring_radii_ratio: float = 6
    rotation_spring_aspect: float = 0.4
    rotation_spring_linewidth: float = 2.
    rotation_spring_radius_scaling: float = 1.0
    rotation_spring_default_opacity: float = 1.0
    rotation_spring_default_color: str = '#CECECE'
    area_spring_default_opacity: float = 0.5
    area_spring_default_color: str = '#CECECE'
    line_spring_linewidth: float = 2.
    line_spring_default_opacity: float = 1.0
    line_spring_default_color: str = '#CECECE'
    distance_spring_line_linewidth: float = 1.0
    distance_spring_line_default_color: str = '#CECECE'
    distance_spring_line_default_opacity: float = 0.7
    node_size: float = 3.
    node_color: str = '#101010'
    show_node_numbers: bool = False
    node_nb_color: str = '#CECECE'
    force_inner_color: str = '#010101'
    preload_force_inner_color: str = '#A0A0A0'
    force_default_outer_color: str = '#CECECE'
    force_vector_length_scaling: float = 1.0
    force_vector_connection: str = 'tail'
