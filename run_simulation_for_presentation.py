from src.springable.simulation import simulate_model, scan_parameter_space
from src.springable.visualization import make_animation
from src.springable.readwrite import fileio

simulate_model('models/presentation/two_nonlinear_springs_in_series.csv', 'results/presentation/tnls',
               solver_settings_path='custom_solver_settings.toml',
               graphics_settings_path='custom_graphics_settings.toml')