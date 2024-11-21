from src.springable.simulation import simulate_model
from src.springable.visualization import make_animation
from src.springable.readwrite import fileio

simulate_model('models/presentation/simple_area.csv', 'results/presentation/simple_area',
               solver_settings_path='custom_solver_settings.toml',
               graphics_settings_path='custom_graphics_settings.toml')