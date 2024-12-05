from dataclasses import dataclass
from ..mechanics.mechanical_behavior import *
from ..readwrite.interpreting import behavior_to_text
from ..readwrite.keywords import usable_behaviors
from .default_behaviors import DEFAULT_BEHAVIORS
import numpy as np


@dataclass
class BehaviorInfo:
    behavior_type: str
    behavior_parameters: dict
    behavior_natural_measure: float


class GUIEventHandler:
    def __init__(self):
        self._behaviors: dict[str, MechanicalBehavior] = {}
        self._behavior_notebook = None
        self._drawing_space = None

    def print_behaviors(self):
        for name, behavior in self._behaviors.items():
            print(f'{name}: {behavior_to_text(behavior, fmt='.2E', full_name=True, specify_natural_measure=True)}')

    def connect_to_notebook(self, behavior_notebook):
        self._behavior_notebook = behavior_notebook

    def connect_to_drawing_space(self, drawing_space):
        self._drawing_space = drawing_space

    def remove_behavior(self, tab_name):
        print(f'Notebook GUI sent event to handler to handle the removal of the behavior named {tab_name}')
        # to implement
        try:
            self._behaviors.pop(tab_name)
        except KeyError:
            pass
        self._drawing_space.remove_curve(tab_name)
        self.print_behaviors()

    def add_behavior(self, tab_name):
        print(f'Notebook GUI sent event to handler to handle the addition of a new behavior named {tab_name}')

        behavior_type_name = self._behavior_notebook.get_behavior_type(tab_name)
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        notebook_parameters = self._behavior_notebook.get_behavior_parameters(tab_name)
        behavior = DEFAULT_BEHAVIORS[behavior_type_name]
        self._behaviors[tab_name] = behavior
        behavior.update(natural_measure, **notebook_parameters)
        if isinstance(behavior, (BezierBehavior, Bezier2Behavior, ZigZagBehavior, ZigZag2Behavior)):
            cp_x, cp_y = behavior.get_control_points()
            if isinstance(behavior, UnivariateBehavior):
                umin, umax = -5., 5
                nb_samples = 100
                u = np.linspace(umin, umax, nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
            elif isinstance(behavior, BivariateBehavior):
                nb_samples = 100
                t = np.linspace(0, 1, nb_samples)
                u = behavior._a(t)
                f = behavior._b(t)
            self._drawing_space.add_curve(tab_name, u, f, True, cp_x, cp_y)
        else:
            umin, umax = -5., 5
            nb_samples = 100
            u = np.linspace(umin, umax, nb_samples)
            f = behavior.gradient_energy(natural_measure + u)[0]
            self._drawing_space.add_curve(tab_name, u, f, False)





        #
        # # if control points were already defined, those will be used;
        # # otherwise the control points derived from the default behavior will be used
        # # (should be the behavior of the function called herein below)
        # curve_interactor_parameters = self._curve_interactor.get_behavior_parameters(tab_name)
        #
        curve_interactor_parameters = {}
        behavior_parameters = notebook_parameters | curve_interactor_parameters
        behavior_type: type[MechanicalBehavior] = usable_behaviors.name_to_type[behavior_type_name]



        self.print_behaviors()

    def change_behavior_type(self, tab_name):
        print(f'Notebook GUI sent event to handler to handle'
              f'the change of behavior type of the behavior named {tab_name}')

        new_behavior_type_name = self._behavior_notebook.get_behavior_type(tab_name)
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        notebook_parameters = self._behavior_notebook.get_behavior_parameters(tab_name)

        curve_interactor_parameters = {}
        behavior_parameters = notebook_parameters | curve_interactor_parameters

        behavior_type: type[MechanicalBehavior] = usable_behaviors.name_to_type[new_behavior_type_name]
        self._behaviors[tab_name] = behavior_type(natural_measure, **behavior_parameters)

        umin, umax = -5., 5
        nb_samples = 100
        u = np.linspace(umin, umax, nb_samples)
        f = self._behaviors[tab_name].gradient_energy(natural_measure + u)[0]
        self._drawing_space.update_curve(tab_name, u, f)
        self.print_behaviors()

    def update_behavior_parameter(self, tab_name, parameter_name):
        print(f'Notebook GUI sent event to handler to handle the update of '
              f'the parameter {parameter_name} of the behavior named {tab_name}')

        par_val = self._behavior_notebook.get_behavior_parameter(tab_name, parameter_name)
        self._behaviors[tab_name].update(**{parameter_name: par_val})

        natural_measure = self._behaviors[tab_name].get_natural_measure()
        umin, umax = -5., 5
        nb_samples = 100
        u = np.linspace(umin, umax, nb_samples)
        f = self._behaviors[tab_name].gradient_energy(natural_measure + u)[0]
        self._drawing_space.update_curve(tab_name, u, f)
        self.print_behaviors()
        # to be extended

    def update_behavior_parameter_from_control_points(self, name):
        print(f'Drawing space GUI sent event to handler to handle the update of the behavior named {name}')
        behavior = self._behaviors[name]
        natural_measure = self._behaviors[name].get_natural_measure()

        if isinstance(behavior, (BezierBehavior, Bezier2Behavior, ZigZagBehavior, ZigZag2Behavior)):
            cp_x, cp_y = self._drawing_space.get_control_points(name)
            behavior.update_from_control_points(cp_x, cp_y)
            if isinstance(behavior, UnivariateBehavior):
                umin, umax = -5., 5
                nb_samples = 100
                u = np.linspace(umin, umax, nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
            elif isinstance(behavior, BivariateBehavior):
                nb_samples = 100
                t = np.linspace(0, 1, nb_samples)
                u = behavior._a(t)
                f = behavior._b(t)
            else:
                raise ValueError
            self._drawing_space.update_curve(name, u, f)
        self.print_behaviors()

    def update_behavior_natural_measure(self, tab_name):
        print(f'Notebook GUI sent event to handler to handle the update of '
              f'the natural measure of the behavior named {tab_name}')
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        self._behaviors[tab_name].update(natural_measure)
        umin, umax = -5., 5
        nb_samples = 100
        u = np.linspace(umin, umax, nb_samples)
        f = self._behaviors[tab_name].gradient_energy(natural_measure + u)[0]
        self._drawing_space.update_curve(tab_name, u, f)
        self.print_behaviors()
        # to be extended

    def get_behavior_text(self, tab_name: str, specify_natural_measure: bool) -> str:
        return behavior_to_text(self._behaviors[tab_name],
                                fmt='.2E', full_name=True, specify_natural_measure=specify_natural_measure)
