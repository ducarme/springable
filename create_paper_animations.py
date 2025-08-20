from src.springable.simulation import solve_model
from src.springable.readwrite.fileio import write_results, read_results
from src.springable.visualization import make_model_construction_animation, make_animation, _load_graphics_settings
import os

main_folder = 'MODELS_FROM_SI'
main_graphics_filepath = os.path.join(main_folder, 'article_movie_models.toml')
main_result_dir = 'out'

def make_animations(subfig, specific_graphics, specific_graphics_mdl_construction, remake_anim, remake_anim_construction, rerun):
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
        make_animation(res, main_result_dir, f'{subfig}_anim', graphics_settings=(go, po, ao, aa))

    if remake_construction:
        make_model_construction_animation(res.get_model(), main_result_dir, save_as_gif=False, save_as_mp4=True,
                                        inbetween_duration=0.5, end_duration=1.0, fps=5, save_name=f'{subfig}_model_anim',
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
remake_anim = True
remake_construction = True
rerun = True
make_animations(subfig, specific_graphics, specific_graphics_mdl_construction,
                remake_anim, remake_construction, rerun)








