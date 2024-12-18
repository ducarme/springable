from ..readwrite.keywords import usable_behaviors
from ..mechanics.mechanical_behavior import *

DEFAULT_NATURAL_MEASURE = 1.0
DEFAULT_BEHAVIORS: dict[str, MechanicalBehavior] = {'LINEAR': LinearBehavior(DEFAULT_NATURAL_MEASURE, 1.0),
                                                    'LOGARITHM': LogarithmBehavior(DEFAULT_NATURAL_MEASURE, 1.0),
                                                    'BEZIER': BezierBehavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0],
                                                                             [1.0, -1.0, 1.0]),
                                                    'BEZIER2': Bezier2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0],
                                                                               [1.0, -1.0, 1.0]),
                                                    'PIECEWISE': PiecewiseBehavior(DEFAULT_NATURAL_MEASURE, [1.0, -1.0, 1.0],
                                                                                [1.0, 2.0], 0.2),
                                                    'ZIGZAG2': ZigZag2Behavior(DEFAULT_NATURAL_MEASURE, [1.0, 2.0, 3.0],
                                                                               [1.0, -5.0, 1.0], 0.5),
                                                    'CONTACT': ContactBehavior(0.0, 10.0, 0.5),
                                                    'ISOTHERMIC_GAS': IsothermicGas(DEFAULT_NATURAL_MEASURE, 1.0, 1.0,
                                                                                    1.0),
                                                    'ISENTROPIC_GAS': IsentropicGas(DEFAULT_NATURAL_MEASURE, 1.0, 1.0,
                                                                                    1.0, 3.0)}
NB_SAMPLES = 150
XLIM = (-1., 5.)
YLIM = (-2., 2.5)


