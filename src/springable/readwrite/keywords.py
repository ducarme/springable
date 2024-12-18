from ..mechanics import shape
from ..mechanics import mechanical_behavior
from .keywordmapping import KeywordMapping

usable_shapes = KeywordMapping({shape.Segment: '',
                                shape.Area: 'AREA',
                                shape.Path: 'LINE',
                                shape.Angle: 'ROTATION',
                                shape.SignedDistancePointLine: 'DISTANCE',
                                shape.SquaredDistancePointSegment: 'GAP',
                                })

usable_behaviors = KeywordMapping({mechanical_behavior.LinearBehavior: 'LINEAR',
                                   mechanical_behavior.LogarithmBehavior: 'LOGARITHM',
                                   mechanical_behavior.BezierBehavior: 'BEZIER',
                                   mechanical_behavior.Bezier2Behavior: 'BEZIER2',
                                   mechanical_behavior.PiecewiseBehavior: 'PIECEWISE',
                                   mechanical_behavior.ZigZag2Behavior: 'ZIGZAG2',
                                   mechanical_behavior.ContactBehavior: 'CONTACT',
                                   mechanical_behavior.IsothermicGas: 'ISOTHERMIC_GAS',
                                   mechanical_behavior.IsentropicGas: 'ISENTROPIC_GAS',
                                   })

