from .gui.behavior_creator import BehaviorCreatorGUI
from .mechanics.mechanical_behavior import ZigZagBehavior, BezierBehavior, Bezier2Behavior
import numpy as np
_u_i = np.array([0.0,
                 3.6469744295621784,
                 1.7703266451373811,
                 2.16166551961459,
                 0.8453438509185234,
                 1.1210598761183754,
                 1.3078352480279523,
                 3.6025041029170413])
_f_i = np.array([0.0,
                 8.726919339164237,
                 -1.1127308066083579,
                 -0.9608843537414966,
                 11.439909297052154,
                 -2.448979591836734,
                 -2.9956268221574343,
                 6.692176870748298])

_u_i = np.array([
                 1.0,
                 2.0,
                 3.0, 4.0, 5.0,6.0, 7.0, 8.0])
_f_i = np.array([
                 1.0,
                 2.0,
                 3.0, 4.0, 5.0,6.0, 7.0, 8.0])



def start_behavior_creation():
    BehaviorCreatorGUI(Bezier2Behavior(_u_i.tolist(), _f_i.tolist()))