from ..mechanics.mechanical_behavior import SmootherZigzag2Behavior
from ..readwrite.keywords import usable_behaviors
from ..mechanics.mechanical_behavior import *

DEBUG = False
DEFAULT_NATURAL_MEASURE = 1.0
LIST_DEFAULT_BEHAVIORS: list[MechanicalBehavior] = [
    LinearBehavior(DEFAULT_NATURAL_MEASURE, k=1.0),
    LogarithmicBehavior(DEFAULT_NATURAL_MEASURE, k=1.0),
    BezierBehavior(DEFAULT_NATURAL_MEASURE, u_i=[1.0, 2.0, 3.0], f_i=[1.0, -1.0, 1.0]),
    PiecewiseBehavior(DEFAULT_NATURAL_MEASURE, k_i=[1.0, -1.0, 1.0], u_i=[1, 2], us=0.25),
    ZigzagBehavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0], 0.5),
    ContactBehavior(DEFAULT_NATURAL_MEASURE, f0=10.0, uc=0.5, delta=0.5),
    IsothermalGas(DEFAULT_NATURAL_MEASURE, n=1.0, R=1.0, T0=1.0),
    IsentropicGas(DEFAULT_NATURAL_MEASURE, n=1.0, R=1.0, T0=1.0, gamma=3.0),

    Bezier2Behavior(DEFAULT_NATURAL_MEASURE, u_i=[1.0, 2.0, 3.0], f_i=[1.0, -1.0, 1.0]),
    Zigzag2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0], 0.5),
    # Spline2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0]),
    # SmootherZigzag2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0], 0.5),
]
INITIALLY_SELECTED_BEHAVIOR = Bezier2Behavior

DEFAULT_BEHAVIORS = {usable_behaviors.type_to_name[type(b)]: b for b in LIST_DEFAULT_BEHAVIORS}
INITIALLY_SELECTED_BEHAVIOR_INDEX = next(i for i, b in enumerate(DEFAULT_BEHAVIORS.values()) if isinstance(b, INITIALLY_SELECTED_BEHAVIOR))

ZOOMING_SCROLL_RATE = 1.1

NB_SAMPLES = 400
XLIM = (-1., 5.)
YLIM = (-2., 2.5)

MAX_NB_CP = 18 # max number of control points,
# for behaviors controllable by control points

# EXPERIMENTAL DATA (csv files, headers will automatically be ignored)
DISPLACEMENT_COLUMN_INDEX = 0
FORCE_COLUMN_INDEX = 1
DELIMITER = ','

# RESPONSE CURVE
FMAX = 0.5
SAMPLING = 150
OOB_TOL = 0.2e-3

