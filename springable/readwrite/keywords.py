from ..mechanics import shape
from ..mechanics import mechanical_behavior
from .keywordmapping import KeywordMapping

usable_shapes = KeywordMapping({shape.Segment: '',
                                shape.Area: 'AREA',
                                shape.Path: 'LINE',
                                shape.Angle: 'ROTATION',
                                shape.DistancePointLine: 'DISTANCE',
                                shape.SquaredDistancePointSegment: 'GAP'})

usable_shape_operations = KeywordMapping({shape.Sum: 'SUM',
                                          shape.Negative: 'NEG'})

usable_behaviors = KeywordMapping({mechanical_behavior.LinearBehavior: 'LINEAR',
                                   mechanical_behavior.BezierBehavior: 'BEZIER',
                                   mechanical_behavior.Bezier2Behavior: 'BEZIER2',
                                   mechanical_behavior.ZigZagBehavior: 'ZIGZAG',
                                   mechanical_behavior.ContactBehavior: 'CONTACT'})

