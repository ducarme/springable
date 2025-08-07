from src.springable.simulation import solve_model
from src.springable.readwrite.fileio import write_results, read_results
from src.springable.visualization import make_model_construction_animation, make_animation
import os

main_folder = 'MODELS_FROM_SI'
main_graphics_filepath = os.path.join(main_folder, 'article_movie_models.toml')
main_result_dir = 'out'

subfig = 'fig1a'
specific_graphics = {}
rerun = True
save_dir = os.path.join(main_result_dir, f'{subfig}_results')
os.makedirs(save_dir, exist_ok=True)
if rerun:
    res = solve_model(os.path.join(main_folder, subfig.upper(), f'{subfig}_model.csv'),
                      solver_settings=os.path.join(main_folder, subfig.upper(), f'{subfig}_ss.toml'))
    write_results(res, save_dir)
else:
    res = read_results(save_dir)

make_animation(res, main_result_dir, f'{subfig}_anim', graphics_settings=main_graphics_filepath, **specific_graphics)
make_model_construction_animation(res.get_model(), main_result_dir, save_as_gif=False, save_as_mp4=True,
                                  inbetween_duration=0.5, fps=5, save_name=f'{subfig}_model_anim',
                                  graphics_settings=main_graphics_filepath, **specific_graphics)




