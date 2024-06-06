class GeneralOptions:
    generate_model_drawing = True
    show_model_drawing = True
    generate_fd_plot = True
    show_fd_plot = True
    generate_animation = True
    show_animation = True


class PlotOptions:
    drive_mode = 0  # None (no drive mode), 0 (force-driven), 1 (displacement-driven)
    cycle = False  # True: loading then unloading, False: just loading. No effect when 'drive_mode' is not 'None'
    driven_path_only = False
    show_snapping_arrows = False
    stability_colors = {'stable': '#86A7FC',
                        'stable under displacement control only': '#FFDD95',
                        'unstable': '#FF9843'}
    stability_markers = {'stable': 'o',
                         'stable under displacement control only': '_',
                         'unstable': '|'}
    experiment_color = '#ff00ff'
    stylesheet = 'springable//graphics//figure_utils//default.mplstyle'
    figsize = (4, 4)


class AnimationOptions:
    # Animation options
    fps = 30
    nb_frames = 120
    dpi = 200
    drive_mode = 1  # None (follows entire equilibrium path), 0 (force-driven), 1 (displacement-driven)
    cycle = False  # True: loading then unloading, False: just loading. No effect when 'drive_mode' is not 'None'
    default_animation_name = 'animation'
    save_as_mp4 = False
    save_as_transparent_mov = False
    save_as_gif = True
    save_frames_as_png = False
    # Side-plot mode, i.e. to play an animated plot next to the structural animation
    # 0 = no side plot
    # 1 = force-displacement plot
    # 2 = custom-plot
    side_plot_mode = 1
    plotstylesheet = 'springable//graphics//figure_utils//default.mplstyle'


class AssemblyAppearance:
    # General
    # Coloring mode for elements in the assembly
    # 0 = color elements using their default color
    # 1 = color elements by elastic energy
    # 2 = color elements by derivative of elastic energy (force for springs,
    # torque for rotational springs, etc)
    # 3 = color elements by second derivative of elastic energy (axial stiffness for springs,
    # rotational stiffness for rotational springs, etc)
    element_coloring_mode = 2

    # Coloring mode for forces in the assembly
    # 0 = color force arrows using their default color
    # 1 = color force arrows using the current force value
    force_coloring_mode = 1

    show_state_of_hysterons = True
    show_forces = True
    hysteron_state_label_size = 20
    hysteron_state_bg_color = '#E0E0E0'
    hysteron_state_txt_color = '#101010'

    # Longitudinal spring appearance
    nb_spring_coils = 4
    spring_linewidth = 3
    spring_width_scaling = 1.0
    spring_default_opacity = 1.0
    spring_default_color = '#CECECE'

    # Rotation spring appearance
    rotation_spring_linewidth = 3
    rotation_spring_radius_scaling = 1.0
    rotation_spring_default_opacity = 1.0
    rotation_spring_default_color = '#CECECE'

    # Area spring appearance
    area_spring_default_opacity = 0.3
    area_spring_default_color = '#CECECE'

    # Line spring appearance
    line_spring_linewidth = 3
    line_spring_default_opacity = 1.0
    line_spring_default_color = '#CECECE'

    # Distance/gap spring appearance
    distance_spring_line_linewidth = 1.0
    distance_spring_line_default_color = '#CECECE'
    distance_spring_line_default_opacity = 0.7

    # Node appearance
    node_size = 3
    node_color = '#101010'
    show_node_numbers = False
    node_nb_color = '#CECECE'

    # Forces appearance
    force_inner_color = '#101010'
    force_default_outer_color = '#CECECE'
    force_vector_length_scaling = 1.0
