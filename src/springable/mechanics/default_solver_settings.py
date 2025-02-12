from dataclasses import dataclass
from ..utils.dataclass_utils import Updatable


@dataclass
class SolverSettings(Updatable):
    reference_load_parameter: float = 0.05
    radius: float = 0.05
    detect_critical_points: bool = False
    critical_point_epsilon: float = 1e-3
    bifurcate_at_simple_bifurcations: bool = False
    show_warnings: bool = False
    verbose: bool = True
    detail_verbose: bool = False
    i_max: float = 5e3
    j_max: float = 15
    convergence_value: float = 1e-6
    alpha: float = 0.0  # positive and never larger than 0.5
    psi_p: float = 0.0
    psi_c: float = 0.0
    detect_mechanism: bool = True
