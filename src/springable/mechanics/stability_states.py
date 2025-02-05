from dataclasses import dataclass


@dataclass
class StabilityStates:
    STABLE: str = 'stable'  # stable under force and displacement control
    STABILIZABLE: str = 'stabilizable'  # stable under displacement-control only
    UNSTABLE: str = 'unstable'  # unstable under both force control and displacement control
