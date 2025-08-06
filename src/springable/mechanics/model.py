import numpy as np
from .assembly import Assembly
from .node import Node
from .load import LoadStep


class Model:

    def __init__(self, assembly: Assembly, loadsteps: list[LoadStep]):
        self._assembly = assembly
        self._loadsteps = loadsteps
        self._force_vector_step_list: list[np.ndarray] = []
        self._dof_to_load_step_list = []
        self._loaded_nodes_step_list: list[set[Node]] = []
        self._blocked_nodes_directions_step_list: list[tuple[list[Node], list[str]]] = []
        for _loadstep in loadsteps:
            dof_to_load = {}
            loaded_nodes = set()
            for nodal_load in _loadstep.get_nodal_loads():
                _node = nodal_load.get_node()
                _direction = nodal_load.get_direction()
                _force = nodal_load.get_force()
                _max_displacement = nodal_load.get_max_displacement()
                index = self._assembly.get_dof_index(_node, _direction)
                dof_to_load[index] = {'force': _force, 'max_displacement': _max_displacement}
                loaded_nodes.add(_node)
            self._dof_to_load_step_list.append(dof_to_load)
            self._loaded_nodes_step_list.append(loaded_nodes)
            self._force_vector_step_list.append(self._compute_step_force_vector(dof_to_load))
            self._blocked_nodes_directions_step_list.append(_loadstep.get_blocked_nodes_directions())

        self._loaded_dof_indices_step_list = []
        for dof_to_load in self._dof_to_load_step_list:
            self._loaded_dof_indices_step_list.append(sorted(dof_to_load.keys()))

        self._max_displacement_map_step_list = []
        for dof_to_load in self._dof_to_load_step_list:
            displacement_map = {}
            for dof_index, load in dof_to_load.items():
                max_displacement = load['max_displacement']
                if max_displacement is not None:
                    displacement_map[dof_index] = max_displacement
            self._max_displacement_map_step_list.append(displacement_map if displacement_map else None)

    def get_loaded_dof_indices_preloading_step_list(self):
        return self._loaded_dof_indices_step_list[:-1]

    def get_loaded_dof_indices(self):
        return self._loaded_dof_indices_step_list[-1]

    def get_loaded_nodes_preloading_step_list(self):
        return self._loaded_nodes_step_list[:-1]

    def get_loaded_nodes(self) -> set[Node]:
        if len(self._loaded_nodes_step_list) > 0:
            return self._loaded_nodes_step_list[-1]
        else:
            return set()

    def get_preloaded_nodes(self) -> set[Node]:
        preloaded_nodes = set()
        for loaded_nodes in self._loaded_nodes_step_list[:-1]:
            preloaded_nodes |= loaded_nodes
        return preloaded_nodes

    def get_force_vectors_preloading_step_list(self):
        return self._force_vector_step_list[:-1]

    def get_force_vector(self):
        if len(self._force_vector_step_list) > 0:
            return self._force_vector_step_list[-1]
        else:
            return None


    def get_force_vectors_step_list(self):
        return self._force_vector_step_list

    def get_preforce_vector(self):
        return np.sum(self._force_vector_step_list[:-1], axis=0)

    def get_max_displacement_map_preloading_step_list(self):
        return self._max_displacement_map_step_list[:-1]

    def get_max_displacement_map(self):
        return self._max_displacement_map_step_list[-1]

    def _compute_step_force_vector(self, dof_to_load) -> np.ndarray:
        step_force_vector = np.zeros(self._assembly.get_nb_dofs())
        for dof_index, load in dof_to_load.items():
            step_force_vector[dof_index] = load['force']
        return step_force_vector

    def get_blocked_nodes_directions_step_list(self) -> list[tuple[list[Node], list[str]]]:
        return self._blocked_nodes_directions_step_list

    def get_assembly(self) -> Assembly:
        return self._assembly

    def get_loading(self) -> list[LoadStep]:
        return self._loadsteps
