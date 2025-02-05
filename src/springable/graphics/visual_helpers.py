from ..mechanics.static_solver import Result
from ..mechanics.element import Element
from ..mechanics import shape
from .default_graphics_settings import AssemblyAppearanceOptions
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, trapezoid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.path import Path
import sys

shape_unit_dimensions: dict[type[shape], int] = {shape.Segment: 1,
                                                 shape.Area: 2,
                                                 shape.HoleyArea: 2,
                                                 shape.Angle: 0,
                                                 shape.Path: 1,
                                                 shape.DistancePointLine: -1,
                                                 shape.SignedDistancePointLine: -1,
                                                 shape.SquaredDistancePointSegment: -2}


def is_dark(hex_color):
    color = hex_color[1:]
    hex_red = int(color[0:2], base=16)
    hex_green = int(color[2:4], base=16)
    hex_blue = int(color[4:6], base=16)
    return (hex_red * 0.2126 + hex_green * 0.7152 + hex_blue * 0.0722) < 140


def get_bg_color():
    return (plt.rcParams['savefig.facecolor'] if not plt.rcParams['savefig.facecolor'] == 'auto'
            else plt.rcParams['figure.facecolor'])


class PropertyHandler:
    def __init__(self, high_value, mode):
        self._high_value = high_value
        self._mode = mode
        self._mapper = self._make_mapper()

    def _make_mapper(self):
        raise NotImplemented

    def determine_property_value(self, quantity: Element | float):
        if isinstance(quantity, Element):
            _element = quantity
            match self._mode:
                case 'energy':
                    return self._mapper(_element.compute_energy())
                case 'generalized_force':
                    return self._mapper[shape_unit_dimensions[type(_element.get_shape())]](
                        _element.compute_generalized_force())
                case 'generalized_stiffness':
                    return self._mapper[shape_unit_dimensions[type(_element.get_shape())]](
                        _element.compute_generalized_stiffness())
                case _:
                    raise ValueError(f"Unknown mode {self._mode}")
        elif isinstance(quantity, float):
            match self._mode:
                case 'energy':
                    force_work = quantity
                    return self._mapper(force_work)
                case 'generalized_force':
                    force_signed_magnitude = quantity
                    return self._mapper[1](force_signed_magnitude)
                case 'generalized_stiffness':
                    stiffness = quantity
                    return self._mapper[1](stiffness)
                case _:
                    raise ValueError(f"Unknown mode {self._mode}")


class ColorHandler(PropertyHandler):

    def __init__(self, high_value, mode, cmap):
        super().__init__(high_value, mode)
        self._cmap = cmap

    def _make_mapper(self):
        if self._mode == 'energy':
            h_val = max(1e-5, self._high_value)
            cn = plt.Normalize(vmin=-h_val, vmax=h_val, clip=True)

            def mapper(value, norm=cn):
                return mcm.ScalarMappable(norm=norm, cmap=self._cmap).to_rgba(value)

        elif self._mode in ('generalized_force', 'generalized_stiffness'):
            mapper = {}
            for dim, hv in self._high_value.items():
                h_val = max(1e-5, hv)
                cn = plt.Normalize(vmin=-h_val, vmax=h_val, clip=True)
                mapper[dim] = lambda value, norm=cn: mcm.ScalarMappable(norm=norm, cmap=self._cmap).to_rgba(value)
        else:
            raise ValueError(f'Cannot make a color handler with mode {self._mode}')
        return mapper


class OpacityHandler(PropertyHandler):

    def _make_mapper(self):
        if self._mode == 'energy':
            h_val = max(1e-5, self._high_value)

            def mapper(value, high_val=h_val):
                return float(
                    interp1d([0.0, high_val], [0.0, 1.0], bounds_error=False, fill_value=(0.0, 1.0))(abs(value)))

        elif self._mode in ('generalized_force', 'generalized_stiffness'):
            mapper = {}
            for dim, hv in self._high_value.items():
                h_val = max(1e-5, hv)
                mapper[dim] = lambda value, high_val=h_val: (
                    float(interp1d([0.0, high_val], [0.0, 1.0], bounds_error=False,
                                   fill_value=(0.0, 1.0))(abs(value))))
        else:
            raise ValueError(f'Cannot make a opacity handler with mode {self._mode}')
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


def compute_coil_line(start, end, nb_coils, diameter, straight_ratio=0.4, aspect=0.4):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    angle = np.arctan2(dy, dx)
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    radius = diameter / 2.

    x1_ = x1 + straight_ratio / 2 * length * np.cos(angle)
    y1_ = y1 + straight_ratio / 2 * length * np.sin(angle)
    x2_ = x2 - straight_ratio / 2 * length * np.cos(angle)
    y2_ = y2 - straight_ratio / 2 * length * np.sin(angle)

    omega = (2 * nb_coils - 1) * np.pi
    length_ = (1 - straight_ratio) * length
    coil_length = omega * radius
    min_radius = 0.4 * radius

    r_squared = max((coil_length ** 2 - length_ ** 2) / omega ** 2, min_radius ** 2)
    r = np.sqrt(r_squared)

    # Generate points along the path
    t = np.linspace(0, 1, nb_coils * 50)
    path_x = np.linspace(x1_, x2_, t.size)
    path_y = np.linspace(y1_, y2_, t.size)

    # Generate coil coordinates
    theta = np.linspace(np.pi, 2 * np.pi * nb_coils, t.size)
    coil_x = r * np.cos(theta) * aspect
    coil_y = r * np.sin(theta)

    # Rotate the coil to align with the start-end direction

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    coil = np.dot(rotation_matrix, np.array([coil_x, coil_y]))

    # Combine the coil with the path
    x_ = path_x + coil[0]
    y_ = path_y + coil[1]

    x = np.insert(x_, 0, x1)
    x = np.append(x, [x2])

    y = np.insert(y_, 0, y1)
    y = np.append(y, [y2])

    return x, y


def compute_coil_arc(center, radius, start_angle, end_angle, nb_coils, radii_ratio=2, aspect=0.4, nb_points_per_coil=50):
    nb_points = nb_coils * nb_points_per_coil
    arc_x, arc_y = compute_arc_line(center, radius, start_angle, end_angle, nb_points=nb_points)
    angle = np.linspace(start_angle, end_angle, nb_points) + np.pi/2

    # Generate coil coordinates
    theta = np.linspace(np.pi, 2 * np.pi * nb_coils, nb_points)
    coil_x = radius / radii_ratio * np.cos(theta) * aspect
    coil_y = radius / radii_ratio * np.sin(theta)

    curved_coil_x = np.cos(angle) * coil_x - np.sin(angle) * coil_y
    curved_coil_y = np.sin(angle) * coil_x + np.cos(angle) * coil_y

    x = arc_x + curved_coil_x
    y = arc_y + curved_coil_y
    return x, y


def compute_arc_line(center, radius, start_angle, end_angle, nb_points=30) -> tuple[np.ndarray, np.ndarray]:
    # generate the silhouette points
    theta = np.linspace(start_angle, end_angle, nb_points)
    points = np.empty((nb_points, 2))
    points[:, 0] = center[0] + radius * np.cos(theta)
    points[:, 1] = center[1] + radius * np.sin(theta)
    return points[:, 0], points[:, 1]


# def compute_coil_line(start, end, nb_coils, radius, straight_ratio=0.4, aspect=0.5):
#     x1, y1 = start
#     x2, y2 = end
#     dx, dy = x2 - x1, y2 - y1
#     angle = np.arctan2(dy, dx)
#     l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#     x1_ = x1 + straight_ratio/2 * l * np.cos(angle)
#     y1_ = y1 + straight_ratio/2 * l * np.sin(angle)
#     x2_ = x2 - straight_ratio/2 * l * np.cos(angle)
#     y2_ = y2 - straight_ratio/2 * l * np.sin(angle)
#
#
#     # Number of coils (segments)
#     num_coils = nb_coils
#
#     # Generate points along the path
#     t = np.linspace(0, 1, num_coils * 50)
#     path_x = np.linspace(x1_, x2_, len(t))
#     path_y = np.linspace(y1_, y2_, len(t))
#
#     # Generate the sine wave with a softened start and end
#     sine_wave = radius * np.sin(2 * np.pi * num_coils * t)
#     # softening = 0.5 * (1 - np.cos(2 * np.pi * t))  # Scales the amplitude smoothly from 0 to 1 to 0
#     snake_wave = sine_wave * 1.0
#
#     # Rotate the sine wave to align with the path
#     snake_x = snake_wave * np.cos(angle + np.pi / 2)
#     snake_y = snake_wave * np.sin(angle + np.pi / 2)
#
#     # Combine the snake wave with the path
#     x_ = path_x + snake_x
#     y_ = path_y + snake_y
#
#     x = np.insert(x_, 0, x1)
#     x = np.append(x, [x2])
#
#     y = np.insert(y_, 0, y1)
#     y = np.append(y, [y2])
#
#     return x, y

def _reorder(poly, cw=True):
    """Reorders the polygon to run clockwise or counter-clockwise
    according to the value of cw. It calculates whether a polygon is
    cw or ccw by summing (x2-x1)*(y2+y1) for all edges of the polygon,
    see https://stackoverflow.com/a/1165943/898213.
    """
    # Close polygon if not closed
    if not np.allclose(poly[:, 0], poly[:, -1]):
        poly = np.c_[poly, poly[:, 0]]
    direction = ((poly[0] - np.roll(poly[0], 1)) *
                 (poly[1] + np.roll(poly[1], 1))).sum() < 0
    if direction == cw:
        return poly
    else:
        return poly[:, ::-1]


def _ring_coding(n):
    """Returns a list of len(n) of this format:
    [MOVETO, LINETO, LINETO, ..., LINETO, LINETO CLOSEPOLY]
    """
    codes = [Path.LINETO] * n
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return codes


def compute_pathpatch_vertices(polys, compute_code=True):
    """Returns a matplotlib patch representing the polygon with holes.

    polys is an iterable (i.e list) of polygons, each polygon is a numpy array
    of shape (2, N), where N is the number of points in each polygon. The first
    polygon is assumed to be the exterior polygon and the rest are holes. The
    first and last points of each polygon may or may not be the same.

    This is inspired by
    https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html

    Example usage:
    ext = np.array([[-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4]])
    t = -np.linspace(0, 2 * np.pi)
    hole1 = np.array([2 + 0.4 * np.cos(t), 2 + np.sin(t)])
    hole2 = np.array([np.cos(t) * (1 + 0.2 * np.cos(4 * t + 1)),
                      np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
    hole2 = np.array([-2 + np.cos(t) * (1 + 0.2 * np.cos(4 * t)),
                      1 + np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
    hole3 = np.array([np.cos(t) * (1 + 0.5 * np.cos(4 * t)),
                      -2 + np.sin(t)])
    holes = [ext, hole1, hole2, hole3]
    vertices, codes = patchify([ext, hole1, hole2, hole3])
    ax = plt.gca()
    ax.add_patch(PathPatch(Path(vertices, codes)))
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    """
    ccw = [True] + ([False] * (len(polys) - 1))
    polys = [_reorder(poly, c) for poly, c in zip(polys, ccw)]
    vertices = np.concatenate(polys, axis=1).T
    if compute_code:
        codes = np.concatenate([_ring_coding(p.shape[1]) for p in polys])
    else:
        codes = None
    return vertices, codes


def compute_requirements_for_animation(_result: Result, assembly_appearance: AssemblyAppearanceOptions):
    aa = assembly_appearance

    _assembly = _result.get_model().get_assembly()
    u = _result.get_displacements(include_preloading=True)
    f = _result.get_forces(include_preloading=True)

    _initial_coordinates = _assembly.get_coordinates()
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    element_color_handler = None
    element_opacity_handler = None
    force_color_handler = None
    force_amounts = {}
    preforce_amounts = {}
    assembly_scanned = False
    start = _result.get_starting_index()
    n = u.shape[0] - start
    characteristic_lengths = []
    if aa.coloring_mode != 'none':
        if aa.show_forces and aa.color_forces:
            loaded_nodes = _result.get_model().get_loaded_nodes()
            preloaded_nodes = _result.get_model().get_preloaded_nodes()
            node_nb_to_dof_indices = _result.get_model().get_assembly().get_nodes_dof_indices()
            final_force_vector = _result.get_model().get_force_vector()
            for _node in loaded_nodes:
                dof_indices = node_nb_to_dof_indices[_node.get_node_nb()]
                final_force = final_force_vector[dof_indices]
                direction = final_force / np.linalg.norm(final_force)
                if not np.isnan(direction).any():
                    match aa.coloring_mode:
                        case 'energy':
                            forces = np.inner(f[start:, dof_indices] - f[start, dof_indices], direction)
                            displacements = np.inner(u[start:, dof_indices], direction)
                            force_amounts[_node] = cumulative_trapezoid(forces, displacements, initial=0)
                        case 'generalized_force':
                            force_amounts[_node] = np.inner(f[start:, dof_indices] - f[start, dof_indices], direction)
                        case 'generalized_stiffness':
                            forces = np.inner(f[start:, dof_indices], direction)
                            displacements = np.inner(u[start:, dof_indices], direction)
                            stiffnesses = np.diff(forces) / np.diff(displacements)
                            force_amounts[_node] = np.append(stiffnesses, stiffnesses[-1])

            for _node in preloaded_nodes:
                dof_indices = node_nb_to_dof_indices[_node.get_node_nb()]
                preforce_vector = f[start, dof_indices]
                direction = preforce_vector / np.linalg.norm(preforce_vector)
                if not np.isnan(direction).any():
                    match aa.coloring_mode:
                        case 'energy':
                            preforces = np.inner(f[:start, dof_indices], direction)
                            predisplacements = np.inner(u[:start, dof_indices], direction)
                            work_after_preloading = trapezoid(preforces, predisplacements)
                            pre_displacements = u[start, dof_indices]
                            preforce_amounts[_node] = (work_after_preloading
                                                       + np.inner(preforce_vector,
                                                                  u[start:, dof_indices] - pre_displacements))
                        case 'generalized_force':
                            preforce_amounts[_node] = np.inner(preforce_vector, direction) * np.ones(n)
                        case 'generalized_stiffness':
                            preforce_amounts[_node] = np.zeros(n)
        if aa.color_elements or (aa.color_forces and aa.show_forces):
            unit_dimensions = set()
            if aa.color_elements:
                existing_shape_unit_dimensions = set()
                for _el in _assembly.get_elements():
                    existing_shape_unit_dimensions.add(shape_unit_dimensions[type(_el.get_shape())])
                unit_dimensions |= existing_shape_unit_dimensions
            if aa.color_forces and aa.show_forces:
                unit_dimensions |= {1}

            energy_maxs = []
            generalized_force_highs = {dim: [] for dim in unit_dimensions}
            generalized_stiffness_highs = {dim: [] for dim in unit_dimensions}

            for i in range(n):
                _assembly.set_coordinates(_initial_coordinates + u[start + i, :])
                match aa.coloring_mode:
                    case 'energy':
                        energies = []
                        if aa.color_elements:
                            element_energies = list(_assembly.compute_elemental_energies().values())
                            energies += element_energies
                        if aa.color_forces and aa.show_forces:
                            force_works = [force_amount[i] for force_amount in force_amounts.values()]
                            preforce_works = [preforce_amount[i] for preforce_amount in preforce_amounts.values()]
                            energies += force_works + preforce_works
                        energy_maxs.append(np.max(np.abs(energies)))
                    case 'generalized_force':
                        generalized_forces = {dim: [] for dim in unit_dimensions}
                        if aa.color_elements:
                            element_to_generalized_forces = _assembly.compute_elemental_generalized_forces()
                            for dim in unit_dimensions:
                                generalized_forces[dim] += [element_to_generalized_forces[el]
                                                            for el in _assembly.get_elements()
                                                            if shape_unit_dimensions[type(el.get_shape())] == dim]
                        if aa.color_forces and aa.show_forces:
                            generalized_forces[1] += [force_amount[i] for force_amount in force_amounts.values()]
                            generalized_forces[1] += [preforce_amount[i]
                                                      for preforce_amount in preforce_amounts.values()]
                        for dim in unit_dimensions:
                            generalized_force_highs[dim].append(np.quantile(np.abs(generalized_forces[dim]), .7))

                    case 'generalized_stiffness':
                        generalized_stiffnesses = {dim: [] for dim in unit_dimensions}
                        if aa.color_elements:
                            element_to_generalized_stiffnesses = _assembly.compute_elemental_generalized_stiffnesses()
                            for dim in unit_dimensions:
                                generalized_stiffnesses[dim] += [element_to_generalized_stiffnesses[el]
                                                                 for el in _assembly.get_elements()
                                                                 if shape_unit_dimensions[type(el.get_shape())] == dim]
                        if aa.color_forces and aa.show_forces:
                            generalized_stiffnesses[1] += [force_amount[i] for force_amount in force_amounts.values()]
                            generalized_stiffnesses[1] += [preforce_amount[i]
                                                           for preforce_amount in preforce_amounts.values()]
                        for dim in unit_dimensions:
                            generalized_stiffness_highs[dim].append(
                                np.quantile(np.abs(generalized_stiffnesses[dim]), .7))

                bounds = _assembly.get_dimensional_bounds()
                xmin = min(xmin, bounds[0])
                ymin = min(ymin, bounds[1])
                xmax = max(xmax, bounds[2])
                ymax = max(ymax, bounds[3])
                characteristic_lengths.append(_assembly.compute_characteristic_length())
            _assembly.set_coordinates(_initial_coordinates)
            assembly_scanned = True

            match aa.coloring_mode:
                case 'energy':
                    high_value = np.quantile(energy_maxs, .9)
                case 'generalized_force':
                    high_value = {dim: np.quantile(generalized_force_highs[dim], .9)
                                  for dim in unit_dimensions}
                case 'generalized_stiffness':
                    high_value = {dim: np.quantile(generalized_stiffness_highs[dim], .9)
                                  for dim in unit_dimensions}
                case _:
                    high_value = None

            element_color_handler = (ColorHandler(high_value, mode=aa.coloring_mode, cmap=aa.colormap)
                                     if aa.color_elements else None)

            element_opacity_handler = (OpacityHandler(high_value, mode=aa.coloring_mode)
                                       if aa.color_elements else None)

            force_color_handler = (ColorHandler(high_value, mode=aa.coloring_mode, cmap=aa.colormap)
                                   if aa.color_forces and aa.show_forces else None)

    if not assembly_scanned:
        for i in range(n):
            _assembly.set_coordinates(_initial_coordinates + u[i + start, :])
            bounds = _assembly.get_dimensional_bounds()
            xmin = min(xmin, bounds[0])
            ymin = min(ymin, bounds[1])
            xmax = max(xmax, bounds[2])
            ymax = max(ymax, bounds[3])
            characteristic_lengths.append(_assembly.compute_characteristic_length())
        _assembly.set_coordinates(_initial_coordinates)

    characteristic_length = np.quantile(characteristic_lengths, 0.75)
    force_amounts = force_amounts if force_amounts else None
    preforce_amounts = preforce_amounts if preforce_amounts else None
    return ((xmin, ymin, xmax, ymax), characteristic_length, element_color_handler, element_opacity_handler,
            force_color_handler, force_amounts, preforce_amounts)


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
