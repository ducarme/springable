DEFAULT_GENERAL_OPTIONS = {
    "generate_model_drawing": True,
    "show_model_drawing": True,
    "generate_fd_plot": True,
    "show_fd_plot": True,
    "generate_animation": True,
    "show_animation": True
}

DEFAULT_PLOT_OPTIONS = {
    "drive_mode": 0,
    "cycle": False,
    "driven_path_only": False,
    "show_snapping_arrows": False,
    "show_stability_with_color": True,
    "show_stability_with_marker": False,
    "color_for_stable_points": "#86A7FC",
    "color_for_stabilizable_points": "#FFDD95",
    "color_for_unstable_points": "#FF9843",
    "marker_for_stable_points": "o",
    "marker_for_stabilizable_points": "_",
    "marker_for_unstable_points": "|",

    "stylesheet_path": "springable//graphics//figure_utils//default.mplstyle",
    "figure_width": 4,
    "figure_height": 4
}

DEFAULT_ANIMATION_OPTIONS = {
    "fps": 30,
    "nb_frames": 120,
    "dpi": 200,
    "drive_mode": 1,
    "cycle": False,
    "default_animation_name": "animation",
    "save_as_mp4": False,
    "save_as_transparent_mov": False,
    "save_as_gif": True,
    "save_frames_as_png": False,
    "side_plot_mode": 1,
    "plot_stylesheet_path": "springable//graphics//figure_utils//default.mplstyle"
}

DEFAULT_ASSEMBLY_APPEARANCE = {
    "element_coloring_mode": 2,
    "force_coloring_mode": 1,
    "show_state_of_hysterons": True,
    "show_forces": True,
    "hysteron_state_label_size": 20,
    "hysteron_state_bg_color": "#E0E0E0",
    "hysteron_state_txt_color": "#101010",
    "nb_spring_coils": 4,
    "spring_linewidth": 3,
    "spring_width_scaling": 1.0,
    "spring_default_opacity": 1.0,
    "spring_default_color": "#CECECE",
    "rotation_spring_linewidth": 3,
    "rotation_spring_radius_scaling": 1.0,
    "rotation_spring_default_opacity": 1.0,
    "rotation_spring_default_color": "#CECECE",
    "area_spring_default_opacity": 0.3,
    "area_spring_default_color": "#CECECE",
    "line_spring_linewidth": 3,
    "line_spring_default_opacity": 1.0,
    "line_spring_default_color": "#CECECE",
    "distance_spring_line_linewidth": 1.0,
    "distance_spring_line_default_color": "#CECECE",
    "distance_spring_line_default_opacity": 0.7,
    "node_size": 3,
    "node_color": "#101010",
    "show_node_numbers": False,
    "node_nb_color": "#CECECE",
    "force_default_inner_color": "#101010",
    "force_default_outer_color": "#CECECE",
    "force_vector_length_scaling": 1.0
}
