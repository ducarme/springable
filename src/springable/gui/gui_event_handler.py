from dataclasses import dataclass
from ..mechanics.mechanical_behavior import MechanicalBehavior
from ..readwrite.interpreting import behavior_to_text
from ..readwrite.keywords import usable_behaviors


@dataclass
class BehaviorInfo:
    behavior_type: str
    behavior_parameters: dict
    behavior_natural_measure: float


class GUIEventHandler:
    def __init__(self):
        self._behaviors: dict[str, MechanicalBehavior] = {}
        self._behavior_notebook = None
        self._curve_interactor = None


    def print_behaviors(self):
        for name, behavior in self._behaviors.items():
            print(f'{name}: {behavior_to_text(behavior, fmt='.2E', full_name=True, specify_natural_measure=True)}')

    def connect_to_notebook(self, behavior_notebook):
        self._behavior_notebook = behavior_notebook

    def remove_behavior(self, tab_name):
        print(f'Notebook GUI sent event to handler to handle the removal of the behavior named {tab_name}')
        # to implement
        try:
            self._behaviors.pop(tab_name)
        except KeyError:
            pass
        self.print_behaviors()
    def add_behavior(self, tab_name, behavior_type_name: str, behavior_parameters, natural_measure):
        print(f'Notebook GUI sent event to handler to handle the addition of a new behavior named')
        # behavior_type_name = self._behavior_notebook.get_behavior_type(tab_name)
        # natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        # notebook_parameters = self._behavior_notebook.get_behavior_parameters(tab_name)
        #
        # # if control points were already defined, those will be used;
        # # otherwise the control points derived from the default behavior will be used
        # # (should be the behavior of the function called herein below)
        # curve_interactor_parameters = self._curve_interactor.get_behavior_parameters(tab_name)
        #
        # behavior_parameters = notebook_parameters | curve_interactor_parameters

        behavior_type: type[MechanicalBehavior] = usable_behaviors.name_to_type[behavior_type_name]
        self._behaviors[tab_name] = behavior_type(natural_measure, **behavior_parameters)
        self.print_behaviors()

    def change_behavior_type(self, tab_name, new_behavior_type_name, parameters, natural_measure):
        print(
            f'Notebook GUI sent event to handler to handle the change of behavior type of the behavior named {tab_name}, '
            f'info {new_behavior_type_name}, with parameters {parameters}')

        behavior_type: type[MechanicalBehavior] = usable_behaviors.name_to_type[new_behavior_type_name]
        self._behaviors[tab_name] = behavior_type(natural_measure, **parameters)
        self.print_behaviors()

    def update_behavior_parameters(self, tab_name, parameters):
        print(
            f'Notebook GUI sent event to handler to handle the update of the parameters of the behavior named {tab_name}, '
            f'new parameters {parameters}')
        self._behaviors[tab_name].update(**parameters)
        self.print_behaviors()
        # to be extended

    def update_behavior_natural_measure(self, tab_name, natural_measure):
        print(
            f'Notebook GUI sent event to handler to handle the update of the natural measure of the behavior named {tab_name},'
            f' new natural measure: {natural_measure}')
        self._behaviors[tab_name].update(natural_measure)
        self.print_behaviors()
        # to be extended

    def get_behavior_text(self, tab_name: str, specify_natural_measure: bool) -> str:
        return behavior_to_text(self._behaviors[tab_name],
                                fmt='.2E', full_name=True, specify_natural_measure=specify_natural_measure)
