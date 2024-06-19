from ..mechanics.static_solver import Result
from ..mechanics.element import Element
from ..mechanics import shape
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import sys

shape_unit_dimensions: dict[type[shape], int] = {shape.Segment: 1,
                                                 shape.Area: 2,
                                                 shape.Angle: 0,
                                                 shape.Path: 1,
                                                 shape.DistancePointLine: -1,
                                                 shape.SquaredDistancePointSegment: -2}


class PropertyHandler:
    def __init__(self, high_value, mode):
        self._high_value = high_value
        self._mode = mode
        self._mapper = self._make_mapper()

    def _make_mapper(self):
        raise NotImplemented

    def determine_property_value(self, _element: Element):
        match self._mode:
            case 'energy':
                return self._mapper(_element.compute_energy())
            case 'generalized_force':
                return self._mapper[shape_unit_dimensions[type(_element.get_shape())]](
                    _element.compute_energy_derivative())
            case 'generalized_stiffness':
                return self._mapper[shape_unit_dimensions[type(_element.get_shape())]](
                    _element.compute_energy_second_derivative())
            case _:
                raise ValueError(f"Unknown mode {self._mode}")


class ColorHandler(PropertyHandler):
    cmap = plt.get_cmap('coolwarm')

    def _make_mapper(self):
        if self._mode == 'energy':
            h_val = max(1e-4, self._high_value)
            cn = plt.Normalize(vmin=-h_val, vmax=h_val, clip=True)
            mapper = lambda value, norm=cn: mcm.ScalarMappable(norm=norm, cmap=ColorHandler.cmap).to_rgba(value)
        elif self._mode not in ('none', 'energy'):
            mapper = {}
            for dim, hv in self._high_value.items():
                h_val = max(1e-4, hv)
                cn = plt.Normalize(vmin=-h_val, vmax=h_val, clip=True)
                mapper[dim] = lambda value, norm=cn: mcm.ScalarMappable(norm=norm, cmap=ColorHandler.cmap).to_rgba(
                    value)
        else:
            raise NotImplemented(f'Cannot make a color handler with mode {self._mode}')
        return mapper


class OpacityHandler(PropertyHandler):

    def _make_mapper(self):
        if self._mode == 'energy':
            h_val = max(1e-4, self._high_value)
            mapper = lambda value, high_val=h_val: \
                (float(interp1d([0.0, high_val], [0.0, 1.0], bounds_error=False, fill_value=(0.0, 1.0))(abs(value))))
        elif self._mode not in ('none', 'energy'):
            mapper = {}
            for dim, hv in self._high_value.items():
                h_val = max(1e-4, hv)
                mapper[dim] = lambda value, high_val=h_val: (
                    float(interp1d([0.0, high_val], [0.0, 1.0], bounds_error=False,
                                   fill_value=(0.0, 1.0))(abs(value))))
        else:
            raise NotImplemented(f'Cannot make a opacity handler with mode {self._mode}')
        return mapper


def compute_zigzag_line(start, end, nb_nodes, width) -> tuple[np.ndarray, np.ndarray]:
    nb_nodes = max(int(nb_nodes), 1)
    start, end = np.array(start).reshape((2,)), np.array(end).reshape((2,))
    length = np.linalg.norm(np.subtract(end, start))
    u_t = np.subtract(end, start) / length
    u_n = np.array([[0, -1], [1, 0]]).dot(u_t)
    spring_coords = np.zeros((2, nb_nodes + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end
    normal_dist = np.sqrt(max(0, width ** 2 - (length ** 2 / nb_nodes ** 2))) / 2
    # Compute the coordinates of each point (each node).
    for i in range(1, nb_nodes + 1):
        if i <= 4:
            spring_coords[:, i] = (start + ((length * (2 * i - 1) * u_t) / (2 * nb_nodes)))
        elif i >= nb_nodes - 3:
            spring_coords[:, i] = (start + ((length * (2 * i - 1) * u_t) / (2 * nb_nodes)))
        else:
            spring_coords[:, i] = (
                    start
                    + ((length * (2 * i - 1) * u_t) / (2 * nb_nodes))
                    + (normal_dist * (-1) ** i * u_n))

    return spring_coords[0, :], spring_coords[1, :]


def compute_arc_line(center, radius, start_angle, end_angle, nb_points=30) -> tuple[np.ndarray, np.ndarray]:
    # generate the silhouette points
    theta = np.linspace(start_angle, end_angle, nb_points)
    points = np.empty((nb_points, 2))
    points[:, 0] = center[0] + radius * np.cos(theta)
    points[:, 1] = center[1] + radius * np.sin(theta)
    return points[:, 0], points[:, 1]


def compute_requirements_for_animation(_result: Result, assembly_appearance):
    _assembly = _result.get_model().get_assembly()
    existing_shape_unit_dimensions = set()
    for _el in _assembly.get_elements():
        existing_shape_unit_dimensions.add(shape_unit_dimensions[type(_el.get_shape())])
    existing_shape_unit_dimensions = sorted(existing_shape_unit_dimensions)

    _initial_coordinates = _assembly.get_general_coordinates()
    u = _result.get_displacements(include_preloading=True)
    energy_means = []
    energy_derivative_means = {dim: [] for dim in existing_shape_unit_dimensions}
    energy_second_derivative_means = {dim: [] for dim in existing_shape_unit_dimensions}
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    characteristic_lengths = []
    for i in range(u.shape[0]):
        _assembly.set_general_coordinates(_initial_coordinates + u[i, :])
        match assembly_appearance['element_coloring_mode']:
            case 'energy':
                energy_means.append(np.mean(list(_assembly.compute_elemental_energies().values())))
            case 'generalized_force':
                energy_derivatives = _assembly.compute_elemental_energy_derivatives()
                for dim in existing_shape_unit_dimensions:
                    energy_derivative_means[dim].append(np.mean(np.abs([energy_derivatives[el]
                                                                        for el in _assembly.get_elements()
                                                                        if shape_unit_dimensions[
                                                                            type(el.get_shape())] == dim])))
            case 'generalized_stiffness':
                energy_second_derivatives = _assembly.compute_elemental_energy_second_derivatives()
                for dim in existing_shape_unit_dimensions:
                    energy_second_derivative_means[dim].append(np.mean(np.abs([energy_second_derivatives[el]
                                                                               for el in _assembly.get_elements()
                                                                               if shape_unit_dimensions[
                                                                                   type(el.get_shape())] == dim])))
        bounds = _assembly.get_dimensional_bounds()
        xmin = min(xmin, bounds[0])
        ymin = min(ymin, bounds[1])
        xmax = max(xmax, bounds[2])
        ymax = max(ymax, bounds[3])
        characteristic_lengths.append(_assembly.compute_characteristic_length())

    match assembly_appearance['element_coloring_mode']:
        case 'energy':
            high_value = np.max(np.abs(np.percentile(energy_means, [10, 90])))
        case 'generalized_force':
            high_value = {dim: np.max(np.abs(np.percentile(energy_derivative_means[dim], [10, 90])))
                          for dim in existing_shape_unit_dimensions}
        case 'generalized_stiffness':
            high_value = {dim: np.max(np.abs(np.percentile(energy_second_derivative_means[dim], [10, 90])))
                          for dim in existing_shape_unit_dimensions}
        case _:
            high_value = None

    color_handler = ColorHandler(high_value, mode=assembly_appearance['element_coloring_mode']) if assembly_appearance[
                                                                                                       'element_coloring_mode'] != 'none' else None
    opacity_handler = OpacityHandler(high_value,
                                     mode=assembly_appearance['element_coloring_mode']) if assembly_appearance[
                                                                                               'element_coloring_mode'] != 'none' else None
    characteristic_length = np.mean(characteristic_lengths)
    _assembly.set_general_coordinates(_initial_coordinates)

    return (xmin, ymin, xmax, ymax), characteristic_length, color_handler, opacity_handler


def print_progress(frame_index, nb_frames):
    progress = float(frame_index + 1) / nb_frames
    bar_length = 10
    if progress < 0.0:
        progress = 0.0
    if progress >= 1:
        progress = 1.0
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] ({1}/{2})".format("#" * block + "-" * (bar_length - block), frame_index + 1, nb_frames)
    sys.stdout.write(text)
    sys.stdout.flush()
