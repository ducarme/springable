import numpy as np
from ..dynamic import dynamics
from graphics.graphical_utils import visual_helpers
from ..graphics.graphic_settings import AssemblyAppearance as AA


def compute_requirements_for_animation(mdl: dynamics.DynamicModel, t, q, dqdt, f):
    _assembly = mdl.get_assembly()
    existing_shape_unit_dimensions = set()
    for _el in _assembly.get_elements():
        existing_shape_unit_dimensions.add(visual_helpers.shape_unit_dimensions[type(_el.get_shape())])
    existing_shape_unit_dimensions = sorted(existing_shape_unit_dimensions)

    energy_means = []
    energy_derivative_means = {dim: [] for dim in existing_shape_unit_dimensions}
    energy_second_derivative_means = {dim: [] for dim in existing_shape_unit_dimensions}
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    characteristic_lengths = []
    for i in range(q.shape[0]):
        _assembly.set_coordinates(q[i, :])
        _assembly.set_velocities(dqdt[i, :])
        match AA.element_coloring_mode:
            case 1:
                energy_means.append(np.mean(list(_assembly.compute_elemental_energies().values())))
            case 2:
                energy_derivatives = _assembly.compute_elemental_energy_derivatives()
                for dim in existing_shape_unit_dimensions:
                    energy_derivative_means[dim].append(np.mean(np.abs([energy_derivatives[el]
                                                                        for el in _assembly.get_elements()
                                                                        if visual_helpers.shape_unit_dimensions[
                                                                            type(el.get_shape())] == dim])))
            case 3:
                energy_second_derivatives = _assembly.compute_elemental_energy_second_derivatives()
                for dim in existing_shape_unit_dimensions:
                    energy_second_derivative_means[dim].append(np.mean(np.abs([energy_second_derivatives[el]
                                                                               for el in _assembly.get_elements()
                                                                               if visual_helpers.shape_unit_dimensions[
                                                                                   type(el.get_shape())] == dim])))
        bounds = _assembly.get_dimensional_bounds()
        xmin = min(xmin, bounds[0])
        ymin = min(ymin, bounds[1])
        xmax = max(xmax, bounds[2])
        ymax = max(ymax, bounds[3])
        characteristic_lengths.append(_assembly.compute_characteristic_length())

    match AA.element_coloring_mode:
        case 1:
            high_value = np.max(np.abs(np.percentile(energy_means, [10, 90])))
        case 2:
            high_value = {dim: np.max(np.abs(np.percentile(energy_derivative_means[dim], [10, 90])))
                          for dim in existing_shape_unit_dimensions}
        case 3:
            high_value = {dim: np.max(np.abs(np.percentile(energy_second_derivative_means[dim], [10, 90])))
                          for dim in existing_shape_unit_dimensions}
        case _:
            high_value = None

    color_handler = visual_helpers.ColorHandler(high_value, mode=AA.element_coloring_mode) if AA.element_coloring_mode >= 1 else None
    opacity_handler = visual_helpers.OpacityHandler(high_value,
                                                    mode=AA.element_coloring_mode) if AA.element_coloring_mode >= 1 else None
    characteristic_length = np.mean(characteristic_lengths)
    _assembly.set_coordinates(q[0, :])
    _assembly.set_velocities(dqdt[0, :])

    return (xmin, ymin, xmax, ymax), characteristic_length, color_handler, opacity_handler


