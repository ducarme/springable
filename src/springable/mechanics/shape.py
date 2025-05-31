from .node import Node
import numpy as np
from collections import OrderedDict


class Shape:
    # output modes for 'compute()' method
    MEASURE = 0
    MEASURE_AND_JACOBIAN = 1
    MEASURE_JACOBIAN_AND_HESSIAN = 2

    def __init__(self, *nodes: Node):
        self._nodes = nodes

    def get_nodal_coordinates(self) -> list[float]:
        nodal_coordinates = []
        for node in self._nodes:
            nodal_coordinates += [node.get_x(), node.get_y()]
        return nodal_coordinates

    def get_nodes(self) -> tuple[Node, ...]:
        return self._nodes

    def get_nb_nodes(self) -> int:
        return len(self._nodes)

    def get_nb_dofs(self) -> int:
        return 2 * self.get_nb_nodes()

    def compute(self, output_mode) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError("This method is abstract.")

    def __add__(self, another_shape):
        return Sum(self, another_shape)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return self + (-other)


class SegmentLength(Shape):
    MIN_LENGTH_ALLOWED = 1e-6

    # constant matrix used to compute the hessian of the transformation
    _a = np.array([[1., 0., -1., 0.],
                   [0., 1., 0., -1.],
                   [-1., 0., 1., 0.],
                   [0., -1., 0., 1.]])

    def __init__(self, node1: Node, node2: Node):
        super().__init__(node1, node2)

    def compute(self, output_mode: str) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = self.get_nodal_coordinates()
        length = SegmentLength.calculate_length(x1, y1, x2, y2)
        if output_mode == Shape.MEASURE:
            return length
        if length < SegmentLength.MIN_LENGTH_ALLOWED:
            raise IllDefinedShape(
                f'Length between node {self._nodes[0].get_node_nb()} and {self._nodes[1].get_node_nb()}'
                f' is lower than {SegmentLength.MIN_LENGTH_ALLOWED}.')
        dx = x1 - x2
        dy = y1 - y2

        # jacobian: derivatives of the length with respect to the nodal coordinates
        # if length == 0.0:
        #     dldq = np.ones([1, -1, 1, -1]) / np.sqrt(2)
        # else:
        dldq = 1 / length * np.array([dx, dy, -dx, -dy])
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return length, dldq
            # hessian: matrix of the second derivatives of the length with respect to the nodal coordinates
        d2ldq2 = 1 / length * (SegmentLength._a - np.outer(dldq, dldq))
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return length, dldq, d2ldq2
        raise ValueError("Unknown mode")

    @staticmethod
    def calculate_length(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class DistanceBetweenTwoSegments(Shape):
    pass


class Angle(Shape):
    MIN_ANGLE_ALLOWED = 0.05 * np.pi / 180.0
    MAX_ANGLE_ALLOWED = 2 * np.pi - MIN_ANGLE_ALLOWED

    # constant matrices used to compute the hessian of the transformation
    _d2Xdq2 = np.array([[0, 0, -1, 0, 1, 0],
                        [0, 0, 0, -1, 0, 1],
                        [-1, 0, 2, 0, -1, 0],
                        [0, -1, 0, 2, 0, -1],
                        [1, 0, -1, 0, 0, 0],
                        [0, 1, 0, -1, 0, 0]])

    _d2Ydq2 = np.array([[0, 0, 0, -1, 0, 1],
                        [0, 0, 1, 0, -1, 0],
                        [0, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 1, 0],
                        [0, -1, 0, 1, 0, 0],
                        [1, 0, -1, 0, 0, 0]])

    def __init__(self, node0: Node, node1: Node, node2: Node):
        super().__init__(node0, node1, node2)

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        x0, y0, x1, y1, x2, y2 = self.get_nodal_coordinates()
        try:
            theta = Angle.calculate_angle(x0, y0, x1, y1, x2, y2)
        except ValueError:
            raise IllDefinedShape
        if output_mode == Shape.MEASURE:
            return theta

        if not Angle.MIN_ANGLE_ALLOWED <= theta <= Angle.MAX_ANGLE_ALLOWED:
            raise IllDefinedShape

        cross = Angle._calculate_cross_product(x0, y0, x1, y1, x2, y2)  # Y
        dot = Angle._calculate_dot_product(x0, y0, x1, y1, x2, y2)  # X
        d0dY = dot / (dot ** 2 + cross ** 2)
        d0dX = -cross / (dot ** 2 + cross ** 2)

        dYdx0 = y2 - y1
        dYdy0 = x1 - x2
        dYdx1 = y0 - y2
        dYdy1 = x2 - x0
        dYdx2 = y1 - y0
        dYdy2 = x0 - x1

        dXdx0 = x2 - x1
        dXdy0 = y2 - y1
        dXdx1 = 2 * x1 - x0 - x2
        dXdy1 = 2 * y1 - y0 - y2
        dXdx2 = x0 - x1
        dXdy2 = y0 - y1

        dYdq = np.array([dYdx0, dYdy0, dYdx1, dYdy1, dYdx2, dYdy2])
        dXdq = np.array([dXdx0, dXdy0, dXdx1, dXdy1, dXdx2, dXdy2])

        # jacobian: derivatives of the angle theta with respect to the nodal coordinates
        d0dq = d0dY * dYdq + d0dX * dXdq
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return theta, d0dq

        d20dY2 = -2 * dot * cross / (dot ** 2 + cross ** 2) ** 2
        d20dYdX = (cross ** 2 - dot ** 2) / (dot ** 2 + cross ** 2) ** 2
        d20dX2 = +2 * dot * cross / (dot ** 2 + cross ** 2) ** 2

        # hessian: matrix of the second derivatives of the angle theta with respect to the nodal coordinates
        d20dq2 = (d0dY * Angle._d2Ydq2
                  + d0dX * Angle._d2Xdq2
                  + d20dY2 * np.outer(dYdq, dYdq)
                  + d20dYdX * (np.outer(dYdq, dXdq) + np.outer(dXdq, dYdq))
                  + d20dX2 * np.outer(dXdq, dXdq)
                  )
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return theta, d0dq, d20dq2
        raise ValueError("Unknown mode")

    @staticmethod
    def calculate_angle(x0, y0, x1, y1, x2, y2) -> float:
        cross = Angle._calculate_cross_product(x0, y0, x1, y1, x2, y2)
        dot = Angle._calculate_dot_product(x0, y0, x1, y1, x2, y2)
        if cross == dot == 0.0:
            raise ValueError
        return np.arctan2(cross, dot) % (2 * np.pi)

    @staticmethod
    def _calculate_cross_product(x0, y0, x1, y1, x2, y2) -> float:
        cross_product = (x1 - x0) * (y1 - y2) - (y1 - y0) * (x1 - x2)
        return cross_product

    @staticmethod
    def _calculate_dot_product(x0, y0, x1, y1, x2, y2) -> float:
        dot_product = (x1 - x0) * (x1 - x2) + (y1 - y0) * (y1 - y2)
        return dot_product


class SignedArea(Shape):
    def __init__(self, *nodes):
        if len(nodes) < 3:
            raise ValueError('At least three nodes are required to define an area')
        super().__init__(*nodes)
        n = len(nodes)
        self._hessian_signed_area = np.zeros(shape=(2 * n, 2 * n))
        for k in range(n):
            self._hessian_signed_area[2 * k, (2 * k + 3) % (2 * n)] = +0.5
            self._hessian_signed_area[2 * k, 2 * k - 1] = -0.5
            self._hessian_signed_area[2 * k + 1, (2 * k + 2) % (2 * n)] = -0.5
            self._hessian_signed_area[2 * k + 1, 2 * k - 2] = +0.5

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        coordinates = self.get_nodal_coordinates()
        signed_area = Area.calculate_signed_area(coordinates)
        if output_mode == Shape.MEASURE:
            return signed_area
        n = int(len(coordinates) / 2)  # nb of nodes
        x, y = coordinates[::2], coordinates[1::2]
        jacobian_signed_area = np.empty(2 * n)
        jacobian_signed_area[0::2] = 0.5 * np.array([y[(k + 1) % n] - y[k - 1] for k in range(n)])
        jacobian_signed_area[1::2] = 0.5 * np.array([x[k - 1] - x[(k + 1) % n] for k in range(n)])
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return signed_area, jacobian_signed_area
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return signed_area, jacobian_signed_area, self._hessian_signed_area
        else:
            raise ValueError('Unknown mode')

    @staticmethod
    def calculate_signed_area(coordinates):
        n = int(len(coordinates) / 2)  # nb of nodes
        x, y = coordinates[::2], coordinates[1::2]
        return 0.5 * np.sum([x[k] * y[(k + 1) % n] - x[(k + 1) % n] * y[k] for k in range(n)])


class Area(SignedArea):

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:

        computed_metrics = super().compute(output_mode)
        if output_mode == Shape.MEASURE:
            signed_area = computed_metrics
            return np.abs(signed_area)
        elif output_mode == Shape.MEASURE_AND_JACOBIAN:
            signed_area = computed_metrics[0]
            jacobian_signed_area = computed_metrics[1]
            return np.abs(signed_area), np.sign(signed_area) * jacobian_signed_area
        elif output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            signed_area = computed_metrics[0]
            jacobian_signed_area = computed_metrics[1]
            hessian_signed_area = computed_metrics[2]
            return (np.abs(signed_area),
                    np.sign(signed_area) * jacobian_signed_area,
                    np.sign(signed_area) * hessian_signed_area)
        else:
            raise ValueError('Unknown mode')


class SquaredDistancePointSegment(Shape):
    MIN_SQUARED_DIST_ALLOWED = 1e-12

    _d2v10v10_dq2 = np.array([[2.0, 0.0, -2.0, 0.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0, -2.0, 0.0, 0.0],
                              [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                              [0.0, -2.0, 0.0, 2.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    _d2v20v20_dq2 = np.array([[2.0, 0.0, 0.0, 0.0, -2.0, 0.0],
                              [0.0, 2.0, 0.0, 0.0, 0.0, -2.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [-2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                              [0.0, -2.0, 0.0, 0.0, 0.0, 2.0]])

    _d2v12v12_dq2 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 2.0, 0.0, -2.0, 0.0],
                              [0.0, 0.0, 0.0, 2.0, 0.0, -2.0],
                              [0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                              [0.0, 0.0, 0.0, -2.0, 0.0, 2.0]])

    _d2v10v12_dq2 = np.array([[0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
                              [-1.0, 0.0, 2.0, 0.0, -1.0, 0.0],
                              [0.0, -1.0, 0.0, 2.0, 0.0, -1.0],
                              [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])

    def __init__(self, node0: Node, node1: Node, node2: Node):
        super().__init__(node0, node1, node2)

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        x0, y0, x1, y1, x2, y2 = self.get_nodal_coordinates()
        v10 = np.array([x0 - x1, y0 - y1])
        v12 = np.array([x2 - x1, y2 - y1])
        v20 = np.array([x0 - x2, y0 - y2])
        if np.inner(v12, v20) >= 0.0:
            dist2 = np.inner(v20, v20)
            if output_mode == Shape.MEASURE:
                return dist2
                # return dist2 ** 0.5
            jacobian = np.array([2 * (x0 - x2), 2 * (y0 - y2), 0.0, 0.0, -2 * (x0 - x2), -2 * (y0 - y2)])
            if output_mode == Shape.MEASURE_AND_JACOBIAN:
                return dist2, jacobian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian
            hessian = self._d2v20v20_dq2
            if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
                return dist2, jacobian, hessian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian, 0.5 * dist2 ** -0.5 * hessian + -0.25 * dist2
                # ** (-3./2) * np.outer(jacobian, jacobian)
            else:
                raise ValueError('Unknown mode')

        elif -np.inner(v12, v10) > 0.0:
            dist2 = np.inner(v10, v10)
            if output_mode == Shape.MEASURE:
                return dist2
                # return dist2 ** 0.5

            jacobian = np.array([-2 * (x1 - x0), -2 * (y1 - y0), 2 * (x1 - x0), 2 * (y1 - y0), 0.0, 0.0])
            if output_mode == Shape.MEASURE_AND_JACOBIAN:
                return dist2, jacobian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian

            hessian = self._d2v10v10_dq2
            if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
                return dist2, jacobian, hessian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian, 0.5 * dist2 ** -0.5 * hessian + -0.25 * dist2 ** (-3./2) * np.outer(jacobian, jacobian)

            else:
                raise ValueError('Unknown mode')
        else:
            v10v10 = np.inner(v10, v10)
            v10v12 = np.inner(v10, v12)
            num = v10v12 ** 2
            v12v12 = np.inner(v12, v12)
            dist2 = v10v10 - num / v12v12
            if output_mode == Shape.MEASURE:
                return dist2
                # return dist2 ** 0.5

            dv10v10_dq = np.array([-2 * (x1 - x0), -2 * (y1 - y0), 2 * (x1 - x0), 2 * (y1 - y0), 0.0, 0.0])
            dv10v12_dq = np.array([x2 - x1, y2 - y1, 2 * x1 - x0 - x2, 2 * y1 - y0 - y2, x0 - x1, y0 - y1])
            dv12v12_dq = np.array([0.0, 0.0, -2 * (x2 - x1), -2 * (y2 - y1), 2 * (x2 - x1), 2 * (y2 - y1)])
            dnum_dq = 2 * v10v12 * dv10v12_dq
            jacobian = dv10v10_dq - dnum_dq / v12v12 + num / v12v12 ** 2 * dv12v12_dq
            if output_mode == Shape.MEASURE_AND_JACOBIAN:
                return dist2, jacobian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian

            d2num_dq2 = 2 * (np.outer(dv10v12_dq, dv10v12_dq) + v10v12 * self._d2v10v12_dq2)
            hessian = (self._d2v10v10_dq2
                       - d2num_dq2 / v12v12
                       + num / v12v12 ** 2 * self._d2v12v12_dq2
                       + (np.outer(dnum_dq, dv12v12_dq) + np.outer(dv12v12_dq, dnum_dq)) / v12v12 ** 2
                       - 2 * num / v12v12 ** 3 * np.outer(dv12v12_dq, dv12v12_dq))
            if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
                return dist2, jacobian, hessian
                # return dist2 ** 0.5, 0.5 * dist2 ** -0.5 * jacobian, 0.5 * dist2 ** -0.5 * hessian + -0.25 * dist2 ** (-3./2) * np.outer(jacobian, jacobian)

            else:
                raise ValueError('Unknown mode')


class CompoundShape(Shape):

    def __init__(self, *shapes: Shape):
        self._shapes = shapes
        self._shape_local_dof_indices = OrderedDict()
        node_local_dof_indices = OrderedDict()
        index = 0
        for _shape in shapes:
            self._shape_local_dof_indices[_shape] = []
            for _node in _shape.get_nodes():
                if _node not in node_local_dof_indices:
                    node_local_dof_indices[_node] = [index, index + 1]
                    index += 2
                self._shape_local_dof_indices[_shape] += node_local_dof_indices[_node]
        super().__init__(*list(node_local_dof_indices.keys()))

    def get_shapes(self):
        return self._shapes

class SignedXDist(Shape):
    def __init__(self, node0: Node, node1: Node):
        super().__init__(node0, node1)

    def compute(self, output_mode) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = self.get_nodal_coordinates()
        x_dist = x0 - x1
        if output_mode == Shape.MEASURE:
            return x_dist

        jacobian = np.array([1, 0, -1, 0])
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return x_dist, jacobian

        hessian = np.zeros((4, 4))
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return x_dist, jacobian, hessian

        else:
            raise ValueError('Unknown mode')

class SignedYDist(Shape):
    def __init__(self, node0: Node, node1: Node):
        super().__init__(node0, node1)

    def compute(self, output_mode) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = self.get_nodal_coordinates()
        y_dist = y0 - y1
        if output_mode == Shape.MEASURE:
            return y_dist

        jacobian = np.array([0, 1, 0, -1])
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return y_dist, jacobian

        hessian = np.zeros((4, 4))
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return y_dist, jacobian, hessian

        else:
            raise ValueError('Unknown mode')

class DistancePointLine(CompoundShape):

    def __init__(self, node0: Node, node1: Node, node2: Node):
        super().__init__(Area(node0, node1, node2), SegmentLength(node1, node2))

    def compute(self, output_mode) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        triangle, line = self.get_shapes()
        triangle_metric = triangle.compute(output_mode)
        line_metric = line.compute(output_mode)
        if not isinstance(triangle_metric, tuple):
            area = triangle_metric
            length = line_metric
        else:
            area = triangle_metric[0]
            length = line_metric[0]
        distance = 2 * area / length
        if output_mode == Shape.MEASURE:
            return distance

        area_local_indices = self._shape_local_dof_indices[triangle]
        line_local_indices = self._shape_local_dof_indices[line]
        jacobian_area = np.zeros(self.get_nb_dofs())
        jacobian_line = np.zeros(self.get_nb_dofs())
        jacobian_area[area_local_indices] = triangle_metric[1]
        jacobian_line[line_local_indices] = line_metric[1]
        jacobian = 2 * (length * jacobian_area - area * jacobian_line) / length ** 2
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return distance, jacobian

        hessian_area = np.zeros((self.get_nb_dofs(), self.get_nb_dofs()))
        hessian_line = np.zeros((self.get_nb_dofs(), self.get_nb_dofs()))
        hessian_area[np.ix_(area_local_indices, area_local_indices)] = triangle_metric[2]
        hessian_line[np.ix_(line_local_indices, line_local_indices)] = line_metric[2]

        hessian = 2 * (
                hessian_area / length
                - (np.outer(jacobian_area, jacobian_line) + np.outer(jacobian_line, jacobian_area)) / length ** 2
                - area * hessian_line / length ** 2
                + 2 * area * np.outer(jacobian_line, jacobian_line) / length ** 3
        )
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return distance, jacobian, hessian

        else:
            raise ValueError('Unknown mode')


class SignedDistancePointLine(CompoundShape):
    """ Shape metric that computes the signed distance between a point and a line.
    The line is specified as two nodes forming a vector. If the point is on the left of the vector,
    the distance is positive else negative. """

    def __init__(self, node0: Node, node1: Node, node2: Node):
        super().__init__(SignedArea(node0, node1, node2), SegmentLength(node1, node2))

    def compute(self, output_mode) \
            -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        triangle, line = self.get_shapes()
        triangle_metric = triangle.compute(output_mode)
        line_metric = line.compute(output_mode)
        if not isinstance(triangle_metric, tuple):
            area = triangle_metric
            length = line_metric
        else:
            area = triangle_metric[0]
            length = line_metric[0]
        distance = 2 * area / length
        if output_mode == Shape.MEASURE:
            return distance

        area_local_indices = self._shape_local_dof_indices[triangle]
        line_local_indices = self._shape_local_dof_indices[line]
        jacobian_area = np.zeros(self.get_nb_dofs())
        jacobian_line = np.zeros(self.get_nb_dofs())
        jacobian_area[area_local_indices] = triangle_metric[1]
        jacobian_line[line_local_indices] = line_metric[1]
        jacobian = 2 * (length * jacobian_area - area * jacobian_line) / length ** 2
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return distance, jacobian

        hessian_area = np.zeros((self.get_nb_dofs(), self.get_nb_dofs()))
        hessian_line = np.zeros((self.get_nb_dofs(), self.get_nb_dofs()))
        hessian_area[np.ix_(area_local_indices, area_local_indices)] = triangle_metric[2]
        hessian_line[np.ix_(line_local_indices, line_local_indices)] = line_metric[2]

        hessian = 2 * (
                hessian_area / length
                - (np.outer(jacobian_area, jacobian_line) + np.outer(jacobian_line, jacobian_area)) / length ** 2
                - area * hessian_line / length ** 2
                + 2 * area * np.outer(jacobian_line, jacobian_line) / length ** 3
        )
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return distance, jacobian, hessian

        else:
            raise ValueError('Unknown mode')


class Sum(CompoundShape):

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        shape_metrics = {}
        for _shape in self._shapes:
            shape_metrics[_shape] = _shape.compute(output_mode)

        measure = 0.0
        for shape_metric in shape_metrics.values():
            if isinstance(shape_metric, tuple):
                shape_measure = shape_metric[0]
            else:
                shape_measure = shape_metric
            measure += shape_measure
        if output_mode == Shape.MEASURE:
            return measure

        jacobian = np.zeros(self.get_nb_dofs())
        for _shape, shape_metric in shape_metrics.items():
            shape_jacobian = shape_metric[1]
            local_indices = self._shape_local_dof_indices[_shape]
            jacobian[local_indices] += shape_jacobian
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return measure, jacobian

        hessian = np.zeros((self.get_nb_dofs(), self.get_nb_dofs()))
        for _shape, shape_metric in shape_metrics.items():
            shape_hessian = shape_metric[2]
            local_indices = self._shape_local_dof_indices[_shape]
            hessian[np.ix_(local_indices, local_indices)] += shape_hessian
        return measure, jacobian, hessian


class Negative(CompoundShape):
    def __init__(self, _shape: Shape):
        self._shape = _shape
        super().__init__(_shape)

    def compute(self, output_mode) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        shape_metric = self._shape.compute(output_mode)
        if output_mode == Shape.MEASURE:
            return -shape_metric
        if output_mode == Shape.MEASURE_AND_JACOBIAN:
            return -shape_metric[0], -shape_metric[1]
        if output_mode == Shape.MEASURE_JACOBIAN_AND_HESSIAN:
            return -shape_metric[0], -shape_metric[1], -shape_metric[2]




class Path(Sum):

    def __init__(self, *nodes: Node):
        segments = []
        for i in range(len(nodes) - 1):
            segments.append(SegmentLength(nodes[i], nodes[i + 1]))
        super().__init__(*segments)


class HoleyArea(Sum):
    def __init__(self, *areas: Area):
        signed_areas = [areas[i] if i == 0 else -areas[i] for i in range(len(areas))]
        super().__init__(*signed_areas)
        self._bulk_area = areas[0]
        self._holes = tuple(areas[1:])

    def get_bulk_area(self) -> Area:
        return self._bulk_area

    def get_holes(self) -> tuple[Area, ...]:
        return self._holes


class IllDefinedShape(ValueError):
    """ raise this when the shape is ill-defined """
