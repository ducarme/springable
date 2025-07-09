from ..mechanics.element import *
from ..mechanics.mechanical_behavior import *
from ..mechanics.shape import *
from ..mechanics.node import Node
from ..mechanics.assembly import Assembly
from ..mechanics.load import NodalLoad, LoadStep
from ..mechanics.model import Model
from . import simpleeval as se
from .keywords import usable_shapes, usable_behaviors
import os


def smart_split(string, separators) -> list[str]:
    """ separate the input string when separators are encountered,
    except if those are within brackets (), {}, or []"""
    parts = []
    current = []
    bracket_level = 0
    for char in string:
        if char in '[({':
            bracket_level += 1
        elif char in '])}':
            bracket_level -= 1
        elif char in separators and bracket_level == 0:
            parts.append(''.join(current))
            current = []
            continue
        current.append(char)
    parts.append(''.join(current))
    return [part.strip() for part in parts]


def parse_simple_arithmetic(string, operators='+-') -> tuple[list[str], list[str]]:
    operator_sequence = ['+']

    parts = []
    current = []
    bracket_level = 0
    for char in string:
        if char in '[({':
            bracket_level += 1
        elif char in '])}':
            bracket_level -= 1
        elif char in operators and bracket_level == 0:
            parts.append(''.join(current))
            operator_sequence.append(char)
            current = []
            continue
        current.append(char)
    parts.append(''.join(current))
    return [part.strip(' ()') for part in parts], operator_sequence


def basic_split(string, separator) -> list[str]:
    return [part.strip() for part in string.split(separator)]


def node_to_text(_node: Node) -> str:
    return (f"{_node.get_node_nb()}, "
            f"{_node.get_x()}, "
            f"{_node.get_y()}, "
            f"{'1' if _node.is_fixed_horizontally() else '0'}, "
            f"{'1' if _node.is_fixed_vertically() else '0'}"
            )


def text_to_node(text, evaluator: se.SimpleEval = None) -> Node:
    if evaluator is None:
        evaluator = se.SimpleEval()
    parsed_data = smart_split(text, ',')
    node_nb = int(parsed_data[0])
    x = evaluator.eval(parsed_data[1])
    y = evaluator.eval(parsed_data[2])
    fixed_horizontally = int(parsed_data[3]) == 1
    fixed_vertically = int(parsed_data[4]) == 1
    return Node(x, y, fixed_horizontally, fixed_vertically, node_nb)


def behavior_to_text(_behavior: MechanicalBehavior, fmt='',
                     full_name=False, specify_natural_measure=True) -> str:
    int_fmt = '.0f'
    if isinstance(_behavior, LinearBehavior) and not full_name:
        text = f'k={_behavior.get_spring_constant():{fmt}}'
    else:
        text = usable_behaviors.type_to_name[type(_behavior)]
        text += '('
        text += '; '.join([f'{par_name}={par_val:{fmt if par_name != "mode" else int_fmt}}' if not isinstance(par_val, list)
                           else f'{par_name}=[{"; ".join([f"{par_val_i:{fmt}}" for par_val_i in par_val])}]'
                           for par_name, par_val in _behavior.get_parameters().items()
                           if not (par_name == "mode" and par_val == 0)])
        text += ')'

    if specify_natural_measure:
        natural_measure = _behavior.get_natural_measure()
        if natural_measure is not None:
            text += f', {natural_measure:{fmt}}'
    return text


def text_to_behavior(behavior_text: str, evaluator: se.SimpleEval = None,
                     natural_measure: float | None = None) -> MechanicalBehavior:
    """ if no natural measure is specified in the behavior_text, the natural_measure argument will be used
    (except if this happens when the natural_measure argument is None,
    in which case an UnknownNaturalMeasure exception is raised) """
    if evaluator is None:
        evaluator = se.SimpleEval()

    parsed_behavior_txt = smart_split(behavior_text, ',')
    if len(parsed_behavior_txt) > 1:
        core_behavior_txt, natural_measure_txt = parsed_behavior_txt
        natural_measure = evaluator.eval(natural_measure_txt)
    else:
        core_behavior_txt = parsed_behavior_txt[0]

    if core_behavior_txt.startswith('FROMFILE(') and core_behavior_txt.endswith(')'):
        parameters_txt = core_behavior_txt.removeprefix('FROMFILE(')
        parameters_txt = parameters_txt.removesuffix(')')
        path = smart_split(parameters_txt, ';')
        path = [evaluator.eval(p) for p in path]
        behavior_text_path = os.path.join('', *path)
        with open(behavior_text_path) as fr:
            line = fr.readline()
        return text_to_behavior(line, evaluator, natural_measure)

    if natural_measure is None:
        raise UnknownNaturalMeasure

    for behavior_type, behavior_name in usable_behaviors.type_to_name.items():
        if core_behavior_txt.startswith(behavior_name + '(') and core_behavior_txt.endswith(')'):
            parameters_txt = core_behavior_txt.removeprefix(behavior_name + '(')
            parameters_txt = parameters_txt.removesuffix(')')
            parameters = {}
            for parameter_txt in smart_split(parameters_txt, ';'):
                par_name = smart_split(parameter_txt, '=')[0]
                par_val_txt = smart_split(parameter_txt, '=')[1]
                if par_val_txt.startswith('[') and par_val_txt.endswith(']'):
                    par_val = []
                    par_val_txt = par_val_txt.removeprefix('[')
                    par_val_txt = par_val_txt.removesuffix(']')
                    for par_comp_txt in smart_split(par_val_txt, ';'):
                        par_val.append(evaluator.eval(par_comp_txt.strip()))
                else:
                    par_val = evaluator.eval(par_val_txt)
                parameters[par_name] = par_val
            return behavior_type(natural_measure, **parameters)
    else:  # the behavior does not match any name --> linear behavior
        if '=' in core_behavior_txt:
            par_name, par_val_text = core_behavior_txt.split('=')
        else:
            par_name = 'k'
            par_val_text = core_behavior_txt
        par_val = evaluator.eval(par_val_text)
        return LinearBehavior(natural_measure, **{par_name: par_val})


def _determine_usable_shape_type_name(_shape):
    if type(_shape) in usable_shapes.keys():
        return usable_shapes.type_to_name[type(_shape)]
    elif isinstance(_shape, HoleyArea):
        return 'AREA'
    else:
        raise NotImplementedError(f'Cannot associate a name to shape type {type(_shape)}')


def _determine_shape_description(_shape: Shape):
    if isinstance(_shape, HoleyArea):
        bulk = _shape.get_bulk_area()
        holes = _shape.get_holes()
        areas = (bulk,) + holes
        area_shape_descriptions = [f'({_determine_shape_description(area)})' for area in areas]
        return '-'.join(area_shape_descriptions)
    if type(_shape) in usable_shapes.keys():
        return '-'.join([str(_node.get_node_nb()) for _node in _shape.get_nodes()])
    else:
        raise NotImplementedError(f'Cannot associate a description to shape type {type(_shape)}')


def shape_to_text(_shape: Shape) -> tuple[str, str]:
    usable_shape_type_name = _determine_usable_shape_type_name(_shape)
    shape_description = _determine_shape_description(_shape)
    return usable_shape_type_name, shape_description


def text_to_shape(shape_text: tuple[str, str], nodes: set[Node]) -> Shape:
    shape_type_name, shape_description = shape_text
    # if shape_type_name.startswith('COMPOUND '):
    #     core_shape_type_name = shape_type_name.removeprefix('COMPOUND').strip()
    #     core_shape_descriptions, operators = parse_simple_arithmetic(shape_description, operators='+-')
    #     core_shapes = [text_to_shape((core_shape_type_name, core_shape_description), nodes)
    #                    for core_shape_description in core_shape_descriptions]
    #     signed_shapes = [core_shapes[i] if operators[i] == '+' else -core_shapes[i]
    #                      for i in range(len(core_shapes))]
    #     return Sum(*signed_shapes)
    if shape_type_name == 'AREA':
        if shape_description.startswith('('):
            core_area_descriptions, _ = parse_simple_arithmetic(shape_description, operators='-')
            core_areas = [text_to_shape(('AREA', core_area_description), nodes)
                          for core_area_description in core_area_descriptions]
            return HoleyArea(*core_areas)
    try:
        shape_type = usable_shapes.name_to_type[shape_type_name]
    except KeyError:
        raise NotImplementedError(f'Shape type {shape_type_name} is unknown.')
    node_nbs = [int(node_nb) for node_nb in smart_split(shape_description, '-')]
    shape_nodes = []
    for node_nb in node_nbs:
        shape_nodes.append(Assembly.get_node_from_set(nodes, node_nb))
    return shape_type(*shape_nodes)


def element_to_text(_element: Element) -> str:
    shape_type_name, shape_description = shape_to_text(_element.get_shape())
    text = shape_type_name + ' SPRING\n'
    text = text.lstrip()
    text += shape_description
    text += ', '
    text += behavior_to_text(_element.get_behavior())
    return text


def text_to_element(element_txt: str,
                    nodes: set[Node], evaluator: se.SimpleEval = None) -> Element:
    if evaluator is None:
        evaluator = se.SimpleEval()

    # shape
    shape_type_text, to_parse = basic_split(element_txt, '\n')
    element_description = smart_split(to_parse, ',')
    shape_description = element_description[0]
    _shape = text_to_shape((shape_type_text.removesuffix('SPRING').rstrip(), shape_description), nodes)

    # behavior
    behavior_txt = ', '.join(element_description[1:])
    try:
        behavior = text_to_behavior(behavior_txt, evaluator)
    except UnknownNaturalMeasure:
        natural_measure = _shape.compute(output_mode=Shape.MEASURE)
        behavior = text_to_behavior(behavior_txt, evaluator, natural_measure)
    return Element(_shape, behavior)


def assembly_to_text(assembly: Assembly) -> str:
    # nodes
    text = 'NODES'
    node_nb = 0
    while True:
        try:
            _node = Assembly.get_node_from_set(assembly.get_nodes(), node_nb)
        except ValueError:
            break
        text += '\n' + node_to_text(_node)
        node_nb += 1
    # elements
    element_sections = {}
    for _element in assembly.get_elements():
        el_txt = element_to_text(_element)
        header, element_description = basic_split(el_txt, '\n')
        if header + 'S' in element_sections:
            element_sections[header + 'S'] += '\n' + element_description
        else:
            element_sections[header + 'S'] = '\n' + element_description
    for header, element_section in element_sections.items():
        text += '\n' + header + element_section
    return text


def text_to_assembly(assembly_text: str, evaluator: se.SimpleEval = None) -> Assembly:
    if evaluator is None:
        evaluator = se.SimpleEval()
    all_lines = basic_split(assembly_text, '\n')
    reading_nodes = False
    reading_elements = False
    nodes: set[Node] = set()
    elements: list[Element] = []
    current_shape_type = None
    for line in all_lines:
        if line == 'NODES':
            reading_nodes = True
            reading_elements = False
            continue
        if line.endswith('SPRINGS'):
            reading_nodes = False
            reading_elements = True
            current_shape_type = line.rstrip('S')
            continue
        if reading_nodes:
            new_node = text_to_node(line, evaluator)
            nodes.add(new_node)
            node_nb = new_node.get_node_nb()
            evaluator.names.update({f'X{node_nb}': new_node.get_x(), f'Y{node_nb}': new_node.get_y()})
        if reading_elements:
            element_text = current_shape_type + '\n' + line
            elements.append(text_to_element(element_text, nodes, evaluator))
    return Assembly(nodes, elements, auto_node_numbering=False)


def nodal_load_to_text(nodal_load: NodalLoad) -> str:
    text = str(nodal_load.get_node().get_node_nb())
    text += ', ' + nodal_load.get_direction()
    text += ', ' + str(nodal_load.get_force())
    if nodal_load.get_max_displacement() is not None:
        text += ', ' + str(nodal_load.get_max_displacement())
    return text


def text_to_nodal_load(nodal_load_text: str, nodes: set[Node], evaluator: se.SimpleEval = None) -> NodalLoad:
    if evaluator is None:
        evaluator = se.SimpleEval()
    parsed_data = smart_split(nodal_load_text, ',')
    _node = Assembly.get_node_from_set(nodes, int(parsed_data[0]))
    direction = parsed_data[1]
    force = evaluator.eval(parsed_data[2])
    if len(parsed_data) >= 4:
        max_displacement = evaluator.eval(parsed_data[3])
    else:
        max_displacement = None
    return NodalLoad(_node, direction, force, max_displacement)


def loading_to_text(loadsteps: list[LoadStep]) -> str:
    load_strs = []
    for _load_step in loadsteps:
        load_str = ''
        blocked_nodes_directions = _load_step.get_blocked_nodes_directions()
        nodes, directions = blocked_nodes_directions
        if nodes:
            load_str = 'block'
            for node, direction in zip(nodes, directions):
                load_str += f'\n{node.get_node_nb()}, {direction}'

        for nodal_load in _load_step.get_nodal_loads():
            load_str += '\n' + nodal_load_to_text(nodal_load)
        load_strs.append(load_str.lstrip())
    return 'LOADING\n' + '\nthen\n'.join(load_strs)


def text_to_loading(loading_text: str, nodes: set[Node], evaluator: se.SimpleEval = None) -> list[LoadStep]:
    if evaluator is None:
        evaluator = se.SimpleEval()
    _loading: list[LoadStep] = []

    node_list: list[Node] = []
    directions: list[str] = []
    forces: list[float] = []
    max_displacements: list[float | None] = []
    blocked_nodes_directions: tuple[list[Node], list[str]] = ([], [])
    lines = basic_split(loading_text, '\n')
    reading_blocked_nodes = False
    if lines[0] != 'LOADING':
        raise ValueError('Loading section does not start with "LOADING".')
    for line in lines[1:]:
        if line == 'then':
            _loading.append(LoadStep(node_list, directions, forces, max_displacements, blocked_nodes_directions))
            node_list: list[Node] = []
            directions: list[str] = []
            forces: list[float] = []
            max_displacements: list[float | None] = []
            blocked_nodes_directions: tuple[list[Node], list[str]] = ([], [])
            continue
        if line == 'block':
            reading_blocked_nodes = True
            continue

        if reading_blocked_nodes:
            parsed_data = smart_split(line, ',')
            if len(parsed_data) == 2:
                blocked_nodes_directions[0].append(Assembly.get_node_from_set(nodes, int(parsed_data[0])))
                blocked_nodes_directions[1].append(parsed_data[1])
            else:
                reading_blocked_nodes = False
        if not reading_blocked_nodes:
            nodal_load = text_to_nodal_load(line, nodes, evaluator)
            node_list.append(nodal_load.get_node())
            directions.append(nodal_load.get_direction())
            forces.append(nodal_load.get_force())
            max_displacements.append(nodal_load.get_max_displacement())
    _loading.append(LoadStep(node_list, directions, forces, max_displacements, blocked_nodes_directions))
    return _loading


def model_to_text(_model: Model) -> str:
    text = assembly_to_text(_model.get_assembly())
    text += '\n' + loading_to_text(_model.get_loading())
    return text


def text_to_model(model_text: str, evaluator: se.SimpleEval = None) -> Model:
    if evaluator is None:
        evaluator = se.SimpleEval()
    parts = basic_split(model_text, 'LOADING')
    if len(parts) < 2:
        assembly_text = parts[0]
        loading_text = ''
    else:
        assembly_text, loading_text = parts[0], parts[1]
    assem = text_to_assembly(assembly_text, evaluator)
    ldg = text_to_loading(('LOADING\n' + loading_text).strip(), assem.get_nodes(), evaluator)
    return Model(assem, ldg)


def parameters_to_text(parameters: dict[str, str | float]) -> str:
    text = 'PARAMETERS'
    for par_name, par_val in parameters.items():
        text += '\n' + par_name
        try:
            par_val = float(par_val)
        except ValueError:
            par_val = "'" + par_val + "'"
        text += ', ' + str(par_val)
    return text


def text_to_parameters(parameters_text: str) -> tuple[dict[str, float | str], dict[str, dict]]:
    parameters = {}
    static_parameters = {}
    design_parameter_data = {}
    reading_parameters = False
    for line in basic_split(parameters_text, '\n'):
        if line == 'PARAMETERS':
            reading_parameters = True
            continue
        if reading_parameters:
            data = smart_split(line, ',')
            parameter_name = data[0]
            try:
                parameter_value = float(data[1])
            except ValueError:
                parameter_value = data[1].strip().strip("'")
            parameters[parameter_name] = parameter_value
            if len(data) == 2:
                static_parameters[parameter_name] = parameter_value
            else:
                bound_evaluator = se.SimpleEval(names=static_parameters)
                if data[2].startswith('[') and data[2].endswith(']'):
                    parameter_data = basic_split(data[2].strip('[]'), ';')
                    if len(parameter_data) == 2:
                        lb = bound_evaluator.eval(parameter_data[0])
                        ub = bound_evaluator.eval(parameter_data[1])
                        nb_samples = None
                    elif len(parameter_data) == 3:
                        lb = bound_evaluator.eval(parameter_data[0])
                        ub = bound_evaluator.eval(parameter_data[1])
                        nb_samples = int(round(float(parameter_data[2])))
                    else:
                        raise ValueError('Cannot read parameters')
                    design_parameter_data[parameter_name] = {'default value': parameter_value,
                                                             'lower bound': lb,
                                                             'upper bound': ub,
                                                             'nb samples': nb_samples if nb_samples is not None else 2,
                                                             'is numeric parameter': True,
                                                             'is range parameter': True}
                if data[2].startswith('{') and data[2].endswith('}'):
                    all_possible_values = np.array(basic_split(data[2].strip('{}'), ';'))
                    try:
                        all_possible_values = all_possible_values.astype(float)
                        is_numeric_parameter = True
                    except ValueError:
                        all_possible_values = [possible_value.strip().strip("'") for possible_value in
                                               all_possible_values]
                        is_numeric_parameter = False
                    design_parameter_data[parameter_name] = {'default value': parameter_value,
                                                             'all possible values': all_possible_values,
                                                             'nb samples': len(all_possible_values),
                                                             'is numeric parameter': is_numeric_parameter,
                                                             'is range parameter': False}
    return parameters, design_parameter_data


class UnknownNaturalMeasure(Exception):
    """ raise this when the natural measure is unknown """
