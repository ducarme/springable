import numpy as np
from .node import Node
from . import shape
from .mechanical_behavior import (MechanicalBehavior,
                                  UnivariateBehavior,
                                  BivariateBehavior,
                                  LinearBehavior)


class Element:
    """ Class describing an element"""

    def __init__(self, _shape: shape.Shape, natural_measure: float, behavior: MechanicalBehavior, element_name=None):
        self._shape = _shape
        self._natural_measure = natural_measure
        self._behavior = behavior
        self._t = np.zeros(self._behavior.get_nb_dofs() - 1)  # values of the internal degrees of freedom
        self._el_nb = None
        self._element_name = element_name

    def get_nodes(self) -> tuple[Node, ...]:
        return self._shape.get_nodes()

    def get_natural_measure(self) -> float:
        return self._natural_measure

    def get_shape(self) -> shape.Shape:
        return self._shape

    def get_behavior(self) -> MechanicalBehavior:
        return self._behavior

    def get_nb_dofs(self) -> int:
        return self.get_nb_external_dofs() + self.get_nb_internal_dofs()

    def get_nb_external_dofs(self) -> int:
        return self._shape.get_nb_dofs()

    def get_nb_internal_dofs(self) -> int:
        return self._behavior.get_nb_dofs() - 1

    def get_element_nb(self) -> int:
        return self._el_nb

    def set_element_nb(self, element_number: int):
        self._el_nb = element_number

    def increment_internal_coordinates(self, u: np.ndarray):
        if self._behavior.get_nb_dofs() == 1:
            raise ValueError("The element does not have any internal degree of freedom")
        else:
            self._t += u

    def set_internal_coordinates(self, t: np.ndarray):
        if self._behavior.get_nb_dofs() == 1:
            raise ValueError("The element does not have any internal degree of freedom")
        else:
            self._t = t.copy()

    def get_internal_coordinates(self) -> np.ndarray:
        if self._behavior.get_nb_dofs() == 1:
            raise ValueError("The element does not have any internal degree of freedom")
        return self._t

    def compute_energy(self) -> float:
        """ Computes and returns the elastic energy currently stored in the element """
        alpha = self._shape.compute(shape.Shape.MEASURE) - self._natural_measure
        return self._behavior.elastic_energy(alpha, *self._t)

    def compute_energy_derivative(self) -> float:
        """ Computes and returns the value of
        the first (partial) derivative of the elastic energy with respect to alpha (elemental reference system) """
        alpha = self._shape.compute(shape.Shape.MEASURE) - self._natural_measure
        return self._behavior.gradient_energy(alpha, *self._t)[0]

    def compute_energy_second_derivative(self) -> float:
        """ Computes and returns the value of
        the second (partial) derivative of the elastic energy with respect to alpha (elemental reference system) """
        alpha = self._shape.compute(shape.Shape.MEASURE) - self._natural_measure
        return self._behavior.hessian_energy(alpha, *self._t)[0]

    def compute_force_vector(self) -> np.ndarray:
        """ Computes and returns the gradient of the elastic energy with respect to the general coordinates (global
        reference system)"""
        shape_measure, jacobian = self._shape.compute(shape.Shape.MEASURE_AND_JACOBIAN)
        alpha = shape_measure - self._natural_measure

        if self._behavior.get_nb_dofs() == 1:
            force_vector = self._behavior.gradient_energy(alpha)[0] * jacobian
            return force_vector

        if self._behavior.get_nb_dofs() == 2:
            n = self.get_nb_dofs()
            force_vector = np.empty(n)
            dvdalpha, dvdt = self._behavior.gradient_energy(alpha, *self._t)
            force_vector[:-1] = dvdalpha * jacobian
            force_vector[-1] = dvdt
            return force_vector
        else:
            raise NotImplementedError("Not implementation to compute stiffness matrix with"
                                      "a mechanical behavior with more than 2 DOFS ")

    def compute_stiffness_matrix(self) -> np.ndarray:
        """ Computes and returns the matrix of the second derivatives of the elastic energy with respect to the general
        coordinates (global reference system)"""
        shape_measure, jacobian, hessian = self._shape.compute(shape.Shape.MEASURE_JACOBIAN_AND_HESSIAN)
        alpha = shape_measure - self._natural_measure

        if self._behavior.get_nb_dofs() == 1:
            stiffness_matrix = (self._behavior.hessian_energy(alpha)[0] * np.outer(jacobian, jacobian)
                                + self._behavior.gradient_energy(alpha)[0] * hessian)
            return stiffness_matrix

        if self._behavior.get_nb_dofs() == 2:
            n = self.get_nb_dofs()
            stiffness_matrix = np.empty((n, n))
            alpha = shape_measure - self._natural_measure
            dvdalpha, dvdt = self._behavior.gradient_energy(alpha, *self._t)
            d2vdalpha2, d2vdalphadt, d2vdt2 = self._behavior.hessian_energy(alpha, *self._t)
            stiffness_matrix[:-1, :-1] = (d2vdalpha2 * np.outer(jacobian, jacobian)
                                          + dvdalpha * hessian)
            stiffness_matrix[-1, -1] = d2vdt2
            stiffness_matrix[-1, :-1] = stiffness_matrix[:-1, -1] = d2vdalphadt * jacobian
            return stiffness_matrix
        else:
            raise NotImplementedError(
                "Not implementation to compute stiffness matrix with mechanical behavior with more than 2 DOFS ")

