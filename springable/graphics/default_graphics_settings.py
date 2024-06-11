DEFAULT_GENERAL_OPTIONS = {
    "generate_model_drawing": True,
    "show_model_drawing": True,
    "generate_fd_plot": True,
    "show_fd_plot": True,
    "generate_animation": True,
    "show_animation": True,
    # herein below the desired outputs when scanning parameters
    "generate_parametric_fd_plots": True,
    "show_parametric_fd_plots": True,
    "generate_all_model_drawings": False,
    "show_all_model_drawings": False,
    "generate_all_fd_plots": False,
    "generate_all_animations": False,
}

DEFAULT_PLOT_OPTIONS = {
    "show_driven_path": False,
    "driven_path_only": False,
    "loading_sequence": 'cycle',  # loading, cycle, loading_unloading
    "show_snapping_arrows": True,
    "drive_mode": None,

    "color_mode": 0,  # is *not* used when showing multiple equilibrium paths after a parameter scan
    # None (default color), 0 (by stability), 1 (by lowest force-driven eig. val.), 2 (by lowest displacement-driven
    # eig. val.), 3 (by nb force-driven eig. vals < 0), 4 (by nb displacement-driven eig. val < 0)
    "show_stability_legend": True,  # only has an effect if 'color_mode' = 0
    "show_driven_path_legend": True,  # only has an effect if 'drive_mode' is 0 or 1 and 'drive_mode_only' is False

    "figure_width": 4,
    "figure_height": 4,

    # herein below, colormaps. Must be named colormaps from matplotlib
    # (visit https://matplotlib.org/stable/users/explain/colors/colormaps.html for more info)
    "range_parameter_scan_colormap": 'viridis',
    "discrete_parameter_scan_colormap": 'tab10',
    "lowest_eigval_colormap": 'PuOr',  # used when 'color_mode' 1 or 2 is used, diverging colormap recommended
    "nb_negative_eigval_colormap": 'plasma',  # used when 'color_mode' 3 or 4 is used

    "max_nb_legend_entries_for_discrete_parameter": 4,

    "default_plot_name": 'fd_curve',
    "default_color": "#a0a0a0",
    "default_marker": 'o',
    "default_markersize": 2.5,

    # herein below colors to indicate the stability of an equilibrium point,
    # only used when 'color_mode' 0 is used.
    "color_for_stable_points": "#86A7FC",
    "color_for_stabilizable_points": "#FFDD95",
    "color_for_unstable_points": "#FF9843",

    "driven_path_color": "#444444",
    "size_for_driven_path": 0.4,  # scaling of the default markersize for driven path
    "snapping_arrow_color": "#aaaaaa",  # only has an effect if 'show_snapping_arrow' is True
    "snapping_arrow_opacity": 0.35,  # only has an effect if 'show_snapping_arrow' is True

    # herein below scaling of the markersize to indicate stability,
    # has no effect in case the 'color_mode' 1, 2, 3 or 4 is used
    "size_for_stable_points": 1,
    "size_for_stabilizable_points": .66,
    "size_for_unstable_points": .33,
    "stylesheet": "default",

}

DEFAULT_ANIMATION_OPTIONS = {
    "fps": 25,
    "nb_frames": 100,
    "dpi": 200,
    "drive_mode": None,
    "cycling": False,
    "default_animation_name": "animation",
    "save_as_mp4": False,
    "save_as_transparent_mov": False,
    "save_as_gif": True,
    "save_frames_as_png": False,
    "side_plot_mode": 1,
    "plot_stylesheet": "default"
}

DEFAULT_ASSEMBLY_APPEARANCE = {
    "element_coloring_mode": 2,
    "force_coloring_mode": 1,
    "show_state_of_hysterons": True,
    "hysteron_state_label_size": 10,
    "hysteron_state_bg_color": "#E0E0E0",
    "hysteron_state_txt_color": "#101010",
    "show_forces": True,
    "force_vector_scaling": 1.0,
    "nb_spring_coils": 4,
    "spring_linewidth": 3,
    "spring_width_scaling": 1.0,
    "spring_default_opacity": 1.0,
    "spring_default_color": "#CECECE",
    "rotation_spring_linewidth": 3,
    "rotation_spring_radius_scaling": 1.0,
    "rotation_spring_default_opacity": 1.0,
    "rotation_spring_default_color": "#CECECE",
    "area_spring_default_opacity": 0.5,
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
