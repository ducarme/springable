import numpy as np
from ..mechanics import node, element, assembly, model, mechanical_behavior, shape, load


class DampeningElement(element.Element):
    def __init__(self, _shape: shape.Shape,
                 behavior: mechanical_behavior.UnivariateBehavior, element_name=None):
        super().__init__(_shape, behavior, element_name)

    def compute_energy(self) -> float:
        """ cannot be used """
        raise NotImplementedError("cannot calculate the energy of a dampening element")

    def compute_generalized_stiffness(self) -> float:
        """ cannot be used """
        raise NotImplementedError("cannot calculate the stiffness of a dampening element")

    def compute_stiffness_matrix(self) -> np.ndarray:
        """ cannot be used """
        raise NotImplementedError("cannot calculate the stiffness matrix of a dampening element")


class DynamicAssembly:
    def __init__(self, static_assembly: assembly.Assembly,
                 initial_velocities: dict[tuple[node.Node, str], float],
                 masses: dict[node.Node, float],
                 dampening_elements: list[DampeningElement],
                 moving_nodes: dict[tuple[node.Node, str], callable],
                 active_behaviors: dict[element.Element, callable]):
        self._static_assembly = static_assembly
        self._masses = masses
        self._dampening_elements = dampening_elements
        self._driven_nodes = moving_nodes
        self._active_behaviors = active_behaviors

        self._nodes = self._static_assembly.get_nodes()
        self._elastic_elements = self._static_assembly.get_elements()
        self._elements = self._elastic_elements + self._dampening_elements
        self._nb_internal_dofs = np.sum([_el.get_nb_internal_dofs() for _el in self._elements]) if self._elements else 0
        self._nb_dofs = 2 * len(self._nodes) + self._nb_internal_dofs

        # assigns a unique number to each element (elastic elements retain their number)
        for element_number, _element in enumerate(self._elements):
            _element.set_element_nb(element_number)

        # dictionaries to keep track of the degree-of-freedom indices for each node dofs, element internal dof and
        # elements external dofs
        self._nodes_dof_indices = self._static_assembly.get_nodes_dof_indices()
        self._elements_internal_dof_indices = {}
        dof_counter = 2 * len(self._nodes)
        for _element in self._elements:
            nb_internal_dofs = _element.get_nb_internal_dofs()
            self._elements_internal_dof_indices[_element.get_element_nb()] = [dof_counter + i
                                                                              for i in range(nb_internal_dofs)]
            dof_counter += nb_internal_dofs

        self._elements_dof_indices = {}
        for _element in self._elements:
            indices = []
            nodes_in_element = _element.get_nodes()
            for _node in nodes_in_element:
                indices += self._nodes_dof_indices[_node.get_node_nb()]
            indices += self._elements_internal_dof_indices[_element.get_element_nb()]
            self._elements_dof_indices[_element.get_element_nb()] = indices

        self._velocities = np.zeros(self._nb_dofs)
        for (nd, direction), velocity in initial_velocities:
            dof_index = self.get_dof_index(nd, direction)
            self._velocities[dof_index] = velocity

        self._driven_dofs_functions = {}
        for (driven_node, direction), fun in self._driven_nodes.items():
            dof_index = self.get_dof_index(driven_node, direction)
            self._driven_dofs_functions[dof_index] = fun

    def get_driven_dofs_functions(self) -> dict[int, callable]:
        return self._driven_dofs_functions

    def set_coordinates(self, coordinates: np.ndarray):
        """ Sets the values of the nodal and internal coordinates """
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            _node.set_position(coordinates[indices])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                _element.set_internal_coordinates(coordinates[indices])

    def set_velocities(self, velocities: np.ndarray):
        """ Sets the values of the nodal and internal coordinates """
        for nd in self._nodes:
            nd_indices = self._nodes_dof_indices[nd.get_node_nb()]
            nd.set_velocity(velocities[nd_indices])
        for el in self._elements:
            internal_dof_indices = self._elements_internal_dof_indices[el.get_element_nb()]
            if internal_dof_indices:
                el.set_internal_coordinate_velocity(velocities)

    def compute_elemental_generalized_forces(self):
        energy_derivatives = {}
        for _element in self._elements:
            energy_derivatives[_element] = _element.compute_generalized_force()
        return energy_derivatives

    def compute_elastic_energy(self) -> float:
        """ Returns the elastic energy currently stored in the assembly """
        return self._static_assembly.compute_elastic_energy()

    def compute_elastic_force_vector(self) -> np.ndarray:
        """ Computes the net elastic force for each degree of freedom """
        return self._static_assembly.compute_elastic_force_vector()

    def compute_dampening_force_vector(self) -> np.ndarray:
        """ Computes the net dampening force for each degree of freedom """
        return self._static_assembly.compute_elastic_force_vector()

    def get_coordinates(self) -> np.ndarray:
        coordinates = np.empty(self._nb_dofs)
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            coordinates[indices] = np.array([_node.get_x(), _node.get_y()])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                coordinates[indices] = _element.get_internal_coordinates()
        return coordinates

    def get_velocities(self) -> np.ndarray:
        velocities = np.empty(self._nb_dofs)
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            velocities[indices] = np.array([_node.get_vx(), _node.get_vy()])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                velocities[indices] = _element.get_internal_coordinate_velocity()
        return velocities

    def get_nb_dofs(self) -> int:
        return self._nb_dofs

    def get_nodes_dof_indices(self) -> dict[int, list[int]]:
        return self._static_assembly.get_nodes_dof_indices()

    def get_dof_index(self, _node: node.Node, direction: str) -> int:
        return self._static_assembly.get_dof_index(_node, direction)

    def get_nodes(self) -> set[node.Node]:
        return self._nodes

    def get_elements(self) -> list[element.Element]:
        return self._elements

    def determine_free_and_fixed_dof_indices(self) -> tuple[list[int], list[int]]:
        free_dof_indices = []
        fixed_dof_indices = []
        for _node in self._nodes:
            node_indices = self._nodes_dof_indices[_node.get_node_nb()]
            node_free_dof_indices = []
            node_fixed_dof_indices = []
            if _node.is_fixed_horizontally() or ((_node, 'X') in self._driven_nodes):
                node_fixed_dof_indices.append(node_indices[0])
            else:
                node_free_dof_indices.append(node_indices[0])
            if _node.is_fixed_vertically() or ((_node, 'Y') in self._driven_nodes):
                node_fixed_dof_indices.append(node_indices[1])
            else:
                node_free_dof_indices.append(node_indices[1])
            free_dof_indices += node_free_dof_indices
            fixed_dof_indices += node_fixed_dof_indices
        for _element in self._elements:
            element_internal_dof_indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if element_internal_dof_indices:
                free_dof_indices += element_internal_dof_indices
        free_dof_indices.sort()
        fixed_dof_indices.sort()
        return free_dof_indices, fixed_dof_indices


class DynamicSolver:
    def __init__(self, dynamic_assembly: DynamicAssembly):
        self._dynamic_assembly = dynamic_assembly
        self._free_dof_indices, _ = dynamic_assembly.determine_free_and_fixed_dof_indices()
        self._nb_free_dofs = len(self._free_dof_indices)
        self._q0 = dynamic_assembly.get_coordinates()
        self._dq0dt = dynamic_assembly.get_velocities()

    def _ode_system(self, time, state):
        q = self._q0.copy()
        dqdt = self._dq0dt.copy()

        # set free dofs from state
        q[self._free_dof_indices] = state[:self._nb_free_dofs]
        dqdt[self._free_dof_indices] = state[self._nb_free_dofs:]

        # set driven dofs from functions
        for driven_dof, displacement_function in self._dynamic_assembly.get_driven_dofs_functions().items():
            q[driven_dof] = self._q0[driven_dof] + displacement_function(time)
            dqdt[driven_dof] = displacement_function(time)

        # update assembly accordingly
        self._dynamic_assembly.set_coordinates(q)
        self._dynamic_assembly.set_velocities(dqdt)





        # update behaviors
        # update positions
        # update velocities
        # update forces

        # compute forces
        # elastic force vector
        # damping force vector (together with the one earlier easier)
        # inertia force vector -> Ma
        # external force vectors

        # ode system
        # dqdt = dqdt
        # d2qdt2 = M^-1 * (external_forces - elastic_forces - damping_forces)

        # return derivative of state wrt time
        dsdt = np.empty(2 * self._nb_free_dofs)
        dsdt[:self._nb_free_dofs] = dqdt  # dqdt
        dsdt[self._nb_free_dofs:] = (external_force_vector - internal_force_vector) / self._masses  # d2qdt2
        return dsdt
