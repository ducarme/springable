import numpy as np
from .node import Node
from .element import Element
from . import shape


class Assembly:
    """ Class representing an assembly of elements connecting nodes """

    def __init__(self, nodes: set[Node], elements: list[Element], auto_node_numbering=True):
        """ Initializes the assembly class """
        self._nodes = nodes
        self._elements = elements
        self._nb_internal_dofs = np.sum([_el.get_nb_internal_dofs() for _el in self._elements]) if self._elements else 0
        self._nb_dofs = 2 * len(nodes) + self._nb_internal_dofs

        if auto_node_numbering:
            # assigns a unique number to each node
            for node_number, _node in enumerate(self._nodes):
                _node.set_node_nb(node_number)

        # assigns a unique number to each element
        for element_number, _element in enumerate(elements):
            _element.set_element_nb(element_number)

        # creates dictionaries to keep track of the degree-of-freedom indices for each node and element
        self._nodes_dof_indices = {}
        for _node in self._nodes:
            node_nb = _node.get_node_nb()
            self._nodes_dof_indices[node_nb] = [2 * node_nb, 2 * node_nb + 1]

        self._elements_internal_dof_indices = {}
        dof_counter = 2 * len(self._nodes)
        for _element in elements:
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

        self._free_dof_indices, self._fixed_dof_indices = self._determine_free_and_fixed_dof_indices()

        # print("Node: DOF indices dictionary")
        # for node_nb, indices in self._nodes_dof_indices.items():
        #     print(f'{node_nb}: {indices}')
        # print("Element: internal DOF indices dictionary")
        # for el_nb, indices in self._elements_internal_dof_indices.items():
        #     print(f'{el_nb}: {indices}')
        # print("Element: DOF indices dictionary")
        # for el_nb, indices in self._elements_dof_indices.items():
        #     print(f'{el_nb}: {indices}')
        # print("Free dof indices")
        # print(self._free_dof_indices)
        # print("Fixed dof indices")
        # print(self._fixed_dof_indices)

    def block_nodes_along_directions(self, nodes: list[Node], directions: list[str]):
        for node, direction in zip(nodes, directions):
            match direction:
                case 'X':
                    node.block_horizontally()
                case 'Y':
                    node.block_vertically()
                case _:
                    raise ValueError(f'{direction} is an unknown direction')
        self._free_dof_indices, self._fixed_dof_indices = self._determine_free_and_fixed_dof_indices()

    def release_nodes_along_directions(self, nodes: list[Node], directions: list[str]):
        for node, direction in zip(nodes, directions):
            match direction:
                case 'X':
                    node.release_horizontally()
                case 'Y':
                    node.release_vertically()
                case _:
                    raise ValueError(f'{direction} is an unknown direction')
        self._free_dof_indices, self._fixed_dof_indices = self._determine_free_and_fixed_dof_indices()

    def increment_coordinates(self, coordinate_increments: np.ndarray):
        """ Updates the values of the general coordinates by applying an increment """
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            _node.displace(coordinate_increments[indices])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                _element.increment_internal_coordinates(coordinate_increments[indices])

    def set_coordinates(self, coordinates: np.ndarray):
        """ Sets the values of the nodal and internal coordinates """
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            _node.set_position(coordinates[indices])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                _element.set_internal_coordinates(coordinates[indices])

    def compute_elemental_energies(self):
        energies = {}
        for _element in self._elements:
            energies[_element] = _element.compute_energy()
        return energies

    def compute_elemental_generalized_forces(self):
        energy_derivatives = {}
        for _element in self._elements:
            energy_derivatives[_element] = _element.compute_generalized_force()
        return energy_derivatives

    def compute_elemental_generalized_stiffnesses(self):
        energy_second_derivatives = {}
        for _element in self._elements:
            energy_second_derivatives[_element] = _element.compute_generalized_stiffness()
        return energy_second_derivatives

    def compute_elastic_energy(self) -> float:
        """ Returns the elastic energy currently stored in the assembly """
        energy = 0.0
        for _element in self._elements:
            energy += _element.compute_energy()
        return energy

    def compute_elastic_force_vector(self) -> np.ndarray:
        """ Computes the net internal forces for each degree of freedom """
        elastic_force_vector = np.zeros(self._nb_dofs)
        for _element in self._elements:
            indices = self._elements_dof_indices[_element.get_element_nb()]
            elastic_force_vector[indices] += _element.compute_force_vector()
        return elastic_force_vector

    def compute_structural_stiffness_matrix(self) -> np.ndarray:
        """ Computes the global stiffness matrix of the current assem structure"""
        ks = np.zeros((self._nb_dofs, self._nb_dofs))
        for _element in self._elements:
            ke = _element.compute_stiffness_matrix()
            indices = self._elements_dof_indices[_element.get_element_nb()]
            ks[np.ix_(indices, indices)] += ke
        return ks

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

    def get_nb_dofs(self) -> int:
        return self._nb_dofs

    def get_nodes_dof_indices(self) -> dict[int, list[int]]:
        return self._nodes_dof_indices

    def get_fixed_dof_indices(self) -> list[int]:
        return self._fixed_dof_indices

    def get_free_dof_indices(self) -> list[int]:
        return self._free_dof_indices

    def get_dof_index(self, _node: Node, direction: str) -> int:
        dof_indices = self._nodes_dof_indices[_node.get_node_nb()]
        match direction:
            case 'X':
                index = dof_indices[0]
            case 'Y':
                index = dof_indices[1]
            case _:
                raise ValueError(f'{direction} is an unknown direction')
        return index

    def get_nodes(self) -> set[Node]:
        return self._nodes

    def get_elements(self) -> list[Element]:
        return self._elements

    def _determine_free_and_fixed_dof_indices(self) -> tuple[list[int], list[int]]:
        free_dof_indices = []
        fixed_dof_indices = []
        for _node in self._nodes:
            node_indices = self._nodes_dof_indices[_node.get_node_nb()]
            node_free_dof_indices = []
            node_fixed_dof_indices = []
            if _node.is_fixed_horizontally():
                node_fixed_dof_indices.append(node_indices[0])
            else:
                node_free_dof_indices.append(node_indices[0])
            if _node.is_fixed_vertically():
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

    def get_dimensional_bounds(self):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for _node in self._nodes:
            x, y = _node.get_x(), _node.get_y()
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
        return xmin, ymin, xmax, ymax

    def compute_characteristic_length(self):
        all_spring_lengths = []
        for _element in self._elements:
            s = _element.get_shape()
            if isinstance(s, shape.Segment):
                all_spring_lengths.append((s.compute(shape.Shape.MEASURE)))
        if all_spring_lengths:  # at least 1 longitudinal spring in assembly
            characteristic_length = np.quantile(all_spring_lengths, .75)
        else:  # no longitudinal elements in the assembly
            xmin, ymin, xmax, ymax = self.get_dimensional_bounds()
            characteristic_length = np.mean([xmax - xmin, ymax - ymin])
        return characteristic_length

    @staticmethod
    def get_node_from_set(nodes: set[Node], node_nb: int):
        for _node in nodes:
            if node_nb == _node.get_node_nb():
                return _node
        else:
            raise ValueError("No node corresponds to this node number")
