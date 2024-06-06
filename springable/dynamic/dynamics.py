import numpy as np

from simulation.node import Node
from ..simulation import node, element, assembly, model, mechanical_behavior, shape, loadmap


class DynamicNode(node.Node):

    def __init__(self, x, y, is_fixed_horizontally, is_fixed_vertically, mass, vx, vy, node_nb=None):
        super().__init__(x, y, is_fixed_horizontally, is_fixed_vertically, node_nb=node_nb)
        self._mass = mass
        self._vx = vx
        self._vy = vy

    def get_mass(self):
        return self._mass

    def set_velocities(self, v):
        self._vx = v[0]
        self._vy = v[1]

    def get_vx(self):
        return self._vx

    def get_vy(self):
        return self._vy


class DynamicElement(element.Element):

    def __init__(self, _shape: shape.Shape, natural_measure: float, _behavior: mechanical_behavior.MechanicalBehavior,
                 damping_factor):
        super().__init__(_shape, natural_measure, _behavior)
        self._delta = damping_factor
        self._v = np.zeros(self._behavior.get_nb_dofs() - 1)  # values of velocities for the internal degrees of freedom

    def get_nodal_velocities(self):
        nodal_velocities = []
        for _node in self.get_shape().get_nodes():
            nodal_velocities += [_node.get_vx(), _node.get_vy()]
        return np.array(nodal_velocities)

    def set_internal_velocities(self, v: np.ndarray):
        if self._behavior.get_nb_dofs() == 1:
            raise ValueError("The element does not have any internal degree of freedom")
        else:
            self._v = v.copy()

    def compute_force_vector(self) -> np.ndarray:
        elastic_force_vector = super().compute_force_vector()
        measure, jacobian = self._shape.compute(shape.Shape.MEASURE_AND_JACOBIAN)
        damping_force_vector = self._delta * np.inner(self.get_nodal_velocities(), jacobian) * jacobian
        total_force = np.zeros(self.get_nb_dofs())
        total_force[:self.get_nb_external_dofs()] = damping_force_vector
        return damping_force_vector + elastic_force_vector


class DynamicAssembly(assembly.Assembly):
    def get_general_velocities(self) -> np.ndarray:
        velocities = np.empty(self._nb_dofs)
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            velocities[indices] = np.array([_node.get_vx(), _node.get_vy()])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                velocities[indices] = 0.0 * _element.get_internal_coordinates()
        return velocities

    def set_general_velocities(self, velocities_values: np.ndarray):
        """ Sets the values of the general coordinates """
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            _node.set_velocities(velocities_values[indices])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                _element.set_internal_coordinates(velocities_values[indices])

    def get_masses(self):
        masses = np.empty(self._nb_dofs)
        for _node in self._nodes:
            indices = self._nodes_dof_indices[_node.get_node_nb()]
            masses[indices] = np.array([_node.get_mass(), _node.get_mass()])
        for _element in self._elements:
            indices = self._elements_internal_dof_indices[_element.get_element_nb()]
            if indices:
                masses[indices] = 0.0 * _element.get_internal_coordinates()
        return masses


class DynamicModel(model.Model):
    def __init__(self, assembly: DynamicAssembly, loadmap: loadmap.LoadStep):
        super().__init__(assembly, loadmap)
        self._force_vector_function = None

    def set_force_vector_function(self, force_vector_function):
        self._force_vector_function = force_vector_function

    def get_force_vector_function(self):
        return self._force_vector_function