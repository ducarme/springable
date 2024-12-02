from ..readwrite.keywords import usable_behaviors
from ..mechanics.mechanical_behavior import *

default_natural_measure = 1.0
DEFAULT_BEHAVIORS: dict[str, MechanicalBehavior] = {'LINEAR': LinearBehavior(default_natural_measure, 1.0),
                                                    'NATURAL': NaturalBehavior(default_natural_measure, 1.0),
                                                    'BEZIER': BezierBehavior(default_natural_measure, [1.0, 2.0, 3.0],
                                                                             [1.0, -1.0, 1.0]),
                                                    'BEZIER2': Bezier2Behavior(default_natural_measure, [1.0, 2.0, 3.0],
                                                                               [1.0, -1.0, 1.0]),
                                                    'ZIGZAG': ZigZagBehavior(default_natural_measure, [1.0, -1.0, 1.0],
                                                                             [1.0, 2.0], 0.2),
                                                    'ZIGZAG2': ZigZag2Behavior(default_natural_measure, [1.0, 2.0, 3.0],
                                                                               [1.0, -1.0, 1.0], 0.2),
                                                    'CONTACT': ContactBehavior(5.0, 0.1),
                                                    'ISOTHERMIC_GAS': IsothermicGas(default_natural_measure, 1.0, 1.0,
                                                                                    1.0),
                                                    'ISENTROPIC_GAS': IsentropicGas(default_natural_measure, 1.0, 1.0,
                                                                                    1.0, 3.0)}
