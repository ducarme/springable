from .gui.behavior_creator import BehaviorCreatorGUI
from .mechanics.mechanical_behavior import ZigZagBehavior, BezierBehavior, Bezier2Behavior

def start_behavior_creation():
    BehaviorCreatorGUI(Bezier2Behavior([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))