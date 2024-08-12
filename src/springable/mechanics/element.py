import numpy as np
from .node import Node
from . import shape
from .mechanical_behavior import (MechanicalBehavior)


class Element:
    """ Class describing an element"""

    def __init__(self, _shape: shape.Shape, behavior: MechanicalBehavior, element_name=None):
        self._shape = _shape
        self._behavior = behavior
        self._t = np.zeros(self._behavior.get_nb_dofs() - 1)  # values of the internal degrees of freedom
        self._el_nb = None
        self._element_name = element_name

    def get_nodes(self) -> tuple[Node, ...]:
        return self._shape.get_nodes()

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
        alpha = self._shape.compute(shape.Shape.MEASURE)
        return self._behavior.elastic_energy(alpha, *self._t)

    def compute_generalized_force(self) -> float:
        """ Computes and returns the value of the generalized force with respect to alpha (elemental reference
        system)"""
        alpha = self._shape.compute(shape.Shape.MEASURE)
        return self._behavior.gradient_energy(alpha, *self._t)[0]

    def compute_generalized_stiffness(self) -> float:
        """ Computes and returns the value of the generalized stiffness with respect to alpha """
        alpha = self._shape.compute(shape.Shape.MEASURE)
        hessian = self._behavior.hessian_energy(alpha, *self._t)
        if len(hessian) == 1:  # for behavior with 0 hidden variable (Univariate behavior)
            return hessian[0]
        if len(hessian) == 3:  # for behavior with 1 hidden variable (Bivariate behavior)
            return (hessian[0] * hessian[2] - hessian[1] ** 2) / hessian[2]
        else:  # for behavior with more than 1 hidden variable (Trivariate behavior for example)
            hessian_size = int(round((np.sqrt(1 + 8 * len(hessian)) - 1) / 2))
            hessian_matrix = np.zeros((hessian_size, hessian_size))
            index = 0
            for i in range(hessian_size):
                for j in range(i, hessian_size):
                    hessian_matrix[i, j] = hessian[index]
                    index += 1
            for i in range(1, hessian_size):
                for j in range(i):
                    hessian_matrix[i, j] = hessian_matrix[j, i]
            return np.linalg.det(hessian_matrix) / np.linalg.det(hessian_matrix[1:hessian_size, 1:hessian_size])

    def compute_force_vector(self) -> np.ndarray:
        """ Computes and returns the gradient of the elastic energy with respect to the general coordinates (global
        reference system)"""
        shape_measure, jacobian = self._shape.compute(shape.Shape.MEASURE_AND_JACOBIAN)
        alpha = shape_measure

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
        alpha = shape_measure

        if self._behavior.get_nb_dofs() == 1:
            stiffness_matrix = (self._behavior.hessian_energy(alpha)[0] * np.outer(jacobian, jacobian)
                                + self._behavior.gradient_energy(alpha)[0] * hessian)
            return stiffness_matrix

        if self._behavior.get_nb_dofs() == 2:
            n = self.get_nb_dofs()
            stiffness_matrix = np.empty((n, n))
            alpha = shape_measure
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

