from ..readwrite.keywords import usable_behaviors
from ..mechanics.mechanical_behavior import *

DEBUG = False

DEFAULT_NATURAL_MEASURE = 1.0
DEFAULT_BEHAVIORS: dict[str, MechanicalBehavior] = dict()

DEFAULT_BEHAVIORS['LINEAR'] = LinearBehavior(DEFAULT_NATURAL_MEASURE, k=1.0)
DEFAULT_BEHAVIORS['LOGARITHM'] = LogarithmBehavior(DEFAULT_NATURAL_MEASURE, k=1.0)
DEFAULT_BEHAVIORS['BEZIER'] = BezierBehavior(DEFAULT_NATURAL_MEASURE, u_i=[1.0, 2.0, 3.0], f_i=[1.0, -1.0, 1.0])
DEFAULT_BEHAVIORS['BEZIER2'] = Bezier2Behavior(DEFAULT_NATURAL_MEASURE, u_i=[1.0, 2.0, 3.0], f_i=[1.0, -1.0, 1.0])
DEFAULT_BEHAVIORS['PIECEWISE'] = PiecewiseBehavior(DEFAULT_NATURAL_MEASURE, k_i=[1.0, -1.0, 1.0], u_i=[1, 2], us=0.25)
DEFAULT_BEHAVIORS['ZIGZAG'] = ZigzagBehavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0], 0.5)
DEFAULT_BEHAVIORS['ZIGZAG2'] = Zigzag2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0], [1.0, -5.0, 1.0], 0.5)
DEFAULT_BEHAVIORS['CONTACT'] = ContactBehavior(0.0, f0=10.0, uc=0.5, delta=0.5)
DEFAULT_BEHAVIORS['ISOTHERMAL_GAS'] = IsothermalGas(DEFAULT_NATURAL_MEASURE, n=1.0, R=1.0, T0=1.0)
DEFAULT_BEHAVIORS['ISENTROPIC_GAS'] = IsentropicGas(DEFAULT_NATURAL_MEASURE, n=1.0, R=1.0, T0=1.0, gamma=3.0)
NB_SAMPLES = 400
XLIM = (-1., 5.)
YLIM = (-2., 2.5)

# EXPERIMENTAL DATA (csv files, headers will automatically ignored)
DISPLACEMENT_COLUMN_INDEX = 1
FORCE_COLUMN_INDEX = 2
DELIMITER = ','



