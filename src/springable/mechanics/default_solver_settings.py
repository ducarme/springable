from dataclasses import dataclass
from ..utils.dataclass_utils import Updatable


@dataclass
class SolverSettings(Updatable):
    reference_load_parameter: float = 0.05
    radius: float = 0.05
    detect_critical_points: bool = False
    critical_point_epsilon: float = 1e-3
    bifurcate_at_simple_bifurcations: bool = False
    bifurcation_perturbation_amplitude: float = 1e-2
    show_warnings: bool = False
    verbose: bool = True
    critical_point_detection_verbose: bool = False
    i_max: float = 5e3
    j_max: float = 15
    convergence_value: float = 1e-6
    detect_mechanism: bool = True
