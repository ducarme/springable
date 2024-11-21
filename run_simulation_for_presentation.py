from src.springable.simulation import simulate_model
from src.springable.visualization import make_animation
from src.springable.readwrite import fileio

simulate_model('models/presentation/simple_countersnapping.csv', 'results/presentation/simple_cs',
               solver_settings_path='custom_solver_settings.toml',
               graphics_settings_path='custom_graphics_settings.toml')