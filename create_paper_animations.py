from src.springable.simulation import solve_model
from src.springable.readwrite.fileio import write_results, read_results
from src.springable.visualization import make_model_construction_animation, make_animation, _load_graphics_settings
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from src.springable.mechanics.mechanical_behavior import BezierBehavior
from src.springable.visualization import make_model_drawing
from src.springable.readwrite.fileio import write_behavior, read_behavior, read_results, read_model, read_experimental_force_displacement_data
from src.springable.simulation import simulate_model
from src.springable.mechanics.static_solver import Result
from src.springable.graphics.plot import force_displacement_curve_in_ax, curve_in_ax
from src.springable.graphics.default_graphics_settings import PlotOptions
from scipy.interpolate import interp1d

main_folder = 'MODELS_FROM_SI'
main_graphics_filepath = os.path.join(main_folder, 'article_movie_models.toml')
main_result_dir = 'out'

def make_animations(subfig, specific_graphics, specific_graphics_mdl_construction, remake_anim, remake_anim_construction, rerun,
                    extra_init=None, extra_update=None, specific_plot_options=None):
    save_dir = os.path.join(main_result_dir, f'{subfig}_results')
    os.makedirs(save_dir, exist_ok=True)
    if rerun:
        res = solve_model(os.path.join(main_folder, subfig.upper(), f'{subfig}_model.csv'),
                        solver_settings=os.path.join(main_folder, subfig.upper(), f'{subfig}_ss.toml'))
        write_results(res, save_dir)
    else:
        res = read_results(save_dir)

    if remake_anim:
        go, po, ao , aa = _load_graphics_settings(main_graphics_filepath)
        aa.update(specific_graphics)
        if specific_plot_options is not None:
            po.update(specific_plot_options)
        make_animation(res, main_result_dir, f'{subfig}_anim', graphics_settings=(go, po, ao, aa),
                       extra_init=extra_init, extra_update=extra_update)

    if remake_construction:
        make_model_construction_animation(res.get_model(), main_result_dir, save_as_gif=False, save_as_mp4=True,
                                        duration_per_node=0.2, duration_per_element=0.2, duration_per_loadstep=0.2,
                                        inbetween_duration=0.8, end_duration=2.5, fps=10, save_name=f'{subfig}_model_anim',
                                        graphics_settings=main_graphics_filepath,
                                        **specific_graphics_mdl_construction)

subfig = 'fig1a'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig1b'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)


subfig = 'fig1e'
specific_graphics = {'force_vector_connection':'head'}
specific_graphics_mdl_construction = {'force_vector_connection':'head',
                                      'show_node_numbers': False, 'spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig1f'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig3a'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)


subfig = 'fig3b'
specific_graphics = {'force_vector_connection':'tail'}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig3c'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig3d'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)


subfig = 'fig5atop'
specific_graphics = {'force_vector_connection':'head'}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2,
                                      'force_vector_connection':'head'}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)


subfig = 'fig5abottom'
specific_graphics = {'force_vector_connection':'head'}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2,
                                      'force_vector_connection':'head'}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)


subfig = 'fig5b1'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun, specific_plot_options={'axis_box_aspect': 1/3})

subfig = 'fig5b2'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun, specific_plot_options={'axis_box_aspect': 1/3})

subfig = 'fig5cleft'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig5cright'
specific_graphics = {}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2}
remake_anim = False
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)

subfig = 'fig5d'
specific_graphics = {'force_vector_connection': 'head'}
specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2, 'force_vector_connection': 'head'}
remake_anim = True
remake_construction = False
rerun = False
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun, specific_plot_options={'axis_box_aspect': 1.0})










# SUPER DIRTY: GRAPHICS OPTIONS MANUALLY CHANGED, ANIMATION METHOD MANUALLY CHANGED 
# unbalanced_force = 0.08
# def _get_sequence_indices_no_pause(u):
#     """ assumes no pause after unloading sequence """
#     start_index = np.argmax(u > 1e-5)
#     loading_start_indices = []
#     loading_end_indices = []
#     unloading_start_indices = []
#     unloading_end_indices = []
#     loading = True
#     unloading = False
#     latest_index = start_index
#     loading_start_indices.append(start_index)
#     while True:
#         if loading:
#             relative_index = np.argmax(np.diff(u[latest_index:]) < -1e-5)
#             if relative_index == 0:
#                 loading_end_indices.append(u.shape[0] - 1)
#                 break
#             loading_end_indices.append(latest_index + relative_index)
#             unloading_start_indices.append(latest_index + relative_index)
#             loading = False
#             unloading = True
#             latest_index += relative_index
#             continue
#         if unloading:
#             relative_index = np.argmax(np.diff(u[latest_index:]) > -1e-9)
#             if relative_index == 0:
#                 unloading_end_indices.append(u.shape[0] - 1)
#                 break
#             unloading_end_indices.append(latest_index + relative_index)
#             loading_start_indices.append(latest_index + relative_index)
#             loading = True
#             unloading = False
#             latest_index += relative_index
#             continue
#     return (loading_start_indices, loading_end_indices,
#             unloading_start_indices, unloading_end_indices)

# def extract_loading_sequence(u, loading_sequence_index):
#     loading_start_indices, loading_end_indices, _, _ = _get_sequence_indices_no_pause(u)
#     start_index = loading_start_indices[loading_sequence_index]
#     end_index = loading_end_indices[loading_sequence_index]
#     return start_index, end_index


# def extract_unloading_sequence(u, unloading_sequence_index):
#     _, _, unloading_start_indices, unloading_end_indices = _get_sequence_indices_no_pause(u)
#     start_index = unloading_start_indices[unloading_sequence_index]
#     end_index = unloading_end_indices[unloading_sequence_index]
#     return start_index, end_index


# u, f = read_experimental_force_displacement_data(os.path.join('FIGURES/FIGURE_3/two_units_45', 'Specimen_RawData_1.csv'),
#                                                  displacement_column_index=1, force_column_index=2, delimiter=';')

# start_loading, end_loading = extract_loading_sequence(u, 1)
# start_unloading, end_unloading = extract_unloading_sequence(u, 1)
# branch_start_indices = [start_loading, 16244, start_unloading, 23043]
# branch_end_indices = [16235, end_loading, 23035, end_unloading]




# # fig, ax1, ax2, extra = extra_init(fig, ax1, ax2)
# def extra_init(fig, ax1: plt.Axes, ax2: plt.Axes):
#     extra = [0, 0, 0]
#     extra[0] = ax2.plot([], [], 'o', color='#F8C8B1', markersize=6, zorder=0.0)[0]
#     extra[1] = ax2.plot([], [], 'o', color='#F78C63', markersize=5, zorder=0.1)[0]
#     extra[2] = True
#     return fig, ax1, ax2, extra


# # extra_update(i, fig, ax1, ax2, extra)
# def extra_update(i, fig, ax1: plt.Axes, ax2: plt.Axes, extra, deformation):
#     if extra[2]:
#         data_index = np.argmax(u[start_loading:] >= deformation[i])
#         if data_index != 0 or i == 0:
#             data_index += start_loading
#             extra[0].set_xdata(u[start_loading: min(data_index, end_loading)] )
#             extra[0].set_ydata(f[start_loading: min(data_index, end_loading)] + 0.08)
#         elif data_index == 0:
#             extra[2] = False

#         if i == 2541:
#             extra[2] = False
#             print('YES')
#     else:
#         data_index = np.argmax(u[start_unloading:] <= deformation[i])
#         if data_index != 0 or i == 0:
#             data_index += start_unloading
#             extra[1].set_xdata(u[start_unloading: min(data_index, end_unloading)])
#             extra[1].set_ydata(f[start_unloading: min(data_index, end_unloading)] + 0.08)




# subfig = 'fig4d'
# specific_graphics = {'preload_force_opacity': 0.4}
# specific_graphics_mdl_construction = {'show_node_numbers': False, 'spring_linewidth': 2, 'angular_spring_linewidth': 2,
#                                       'preload_force_opacity': 0.4}
# remake_anim = True
# remake_construction = False
# rerun = False
# make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
#                 remake_anim, remake_construction, rerun, extra_init, extra_update)








