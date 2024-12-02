from dataclasses import dataclass


@dataclass
class BehaviorInfo:
    behavior_type: str
    specify_alpha0: bool
    behavior_parameters: dict


class GUIEventHandler:
    def __init__(self):
        self._behavior_info: dict[str, BehaviorInfo] = {}

    def remove_behavior(self, tab_name):
        print(f'GUI sent event to handler to handle the removal of the behavior named {tab_name}')
        # to implement
        try:
            self._behavior_info.pop(tab_name)
        except KeyError:
            pass

    def add_behavior(self, tab_name, behavior_type, behavior_parameters):
        print(f'GUI sent event to handler to handle the addition of a new behavior named {tab_name}, '
              f'of type {behavior_type}, with initial parameters {behavior_parameters}')
        self._behavior_info[tab_name] = BehaviorInfo(behavior_type, specify_alpha0, behavior_parameters)

    def change_behavior_type(self, tab_name, new_behavior_type):
        print(f'GUI sent event to handler to handle the change of behavior type of the behavior named {tab_name}, '
              f'info {new_behavior_type}')
        # to implement

    def update_behavior_parameters(self, tab_name, parameters):
        print(f'GUI sent event to handler to handle the update of the parameters of the behavior named {tab_name}, '
              f'new parameters {parameters}')
        # to implement
