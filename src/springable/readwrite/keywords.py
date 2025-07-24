from ..mechanics import shape
from ..mechanics import mechanical_behavior
from ..utils.one2one_mapping import One2OneMapping

usable_shapes = One2OneMapping({shape.SegmentLength: 'LONGITUDINAL',
                                shape.Angle: 'ANGULAR',
                                shape.Area: 'AREA',
                                shape.PathLength: 'PATH',
                                shape.SignedDistancePointLine: 'DISTANCE',
                                shape.SignedXDist: 'X DISTANCE',
                                shape.SignedYDist: 'Y DISTANCE',
                                # shape.SquaredDistancePointSegment: 'GAP',  # experimental
                                })

usable_behaviors = One2OneMapping({mechanical_behavior.LinearBehavior: 'LINEAR',
                                   mechanical_behavior.LogarithmicBehavior: 'LOGARITHMIC',
                                   mechanical_behavior.BezierBehavior: 'BEZIER',
                                   mechanical_behavior.Bezier2Behavior: 'BEZIER2',
                                   mechanical_behavior.PiecewiseBehavior: 'PIECEWISE',
                                   mechanical_behavior.ZigzagBehavior: 'ZIGZAG',
                                   mechanical_behavior.Zigzag2Behavior: 'ZIGZAG2',
                                   mechanical_behavior.ContactBehavior: 'CONTACT',
                                   mechanical_behavior.IsothermalGas: 'ISOTHERMAL',
                                   mechanical_behavior.IsentropicGas: 'ISENTROPIC',
                                   # mechanical_behavior.SmootherZigzag2Behavior: 'SZIGZAG2',  # experimental
                                   # mechanical_behavior.Spline2Behavior: 'SPLINE2',  # experimental
                                   })

