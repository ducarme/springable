from ..simulation.node import Node
from ..simulation.loadmap import Load
from ..readwrite import interpreting
from ..readwrite import simpleeval as se
from . import dynamics
import csv


def text_to_node(text, evaluator: se.SimpleEval = None) -> dynamics.DynamicNode:
    if evaluator is None:
        evaluator = se.SimpleEval()
    _node = interpreting.text_to_node(text, evaluator)
    node_nb = _node.get_node_nb()
    x = _node.get_x()
    y = _node.get_y()
    fixed_horizontally = _node.is_fixed_horizontally()
    fixed_vertically = _node.is_fixed_vertically()
    dynamic_info = interpreting.split(text, ',')[5:]
    mass = evaluator.eval(dynamic_info[0])
    if len(dynamic_info) > 1:
        vx = dynamic_info[1]
        vy = dynamic_info[2]
    else:
        vx = vy = 0.0
    return dynamics.DynamicNode(x, y, fixed_horizontally, fixed_vertically, mass, vx, vy, node_nb)


def text_to_element(element_txt: str,
                    nodes: set[Node], damping, evaluator: se.SimpleEval = None) -> dynamics.DynamicElement:
    if evaluator is None:
        evaluator = se.SimpleEval()
    _element = interpreting.text_to_element(element_txt, nodes, evaluator)
    return dynamics.DynamicElement(_element.get_shape(), _element.get_natural_measure(), _element.get_behavior(),
                                   damping)


def text_to_assembly(assembly_text: str, damping, evaluator: se.SimpleEval = None) -> dynamics.DynamicAssembly:
    if evaluator is None:
        evaluator = se.SimpleEval()
    all_lines = interpreting.split(assembly_text, '\n')
    reading_nodes = False
    reading_elements = False
    nodes: set[dynamics.DynamicNode] = set()
    elements: set[dynamics.DynamicElement] = set()
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
            nodes.add(text_to_node(line, evaluator))
        if reading_elements:
            element_text = current_shape_type + '\n' + line
            elements.add(text_to_element(element_text, nodes, damping, evaluator))
    return dynamics.DynamicAssembly(nodes, elements, auto_node_numbering=False)


def text_to_loadmap(loadmap_text: str, nodes: set[Node], evaluator: se.SimpleEval = None) -> Load:
    if evaluator is None:
        evaluator = se.SimpleEval()
    node_list: list[Node] = []
    directions: list[str] = []
    forces: list[float] = []
    max_displacements: list[float | None] = []
    reading_loadmap = False
    for line in interpreting.split(loadmap_text, '\n'):
        if line == 'LOADMAP':
            reading_loadmap = True
            continue
        if reading_loadmap:
            nodal_load = interpreting.text_to_nodal_load(line, nodes, evaluator)
            node_list.append(nodal_load.get_node())
            directions.append(nodal_load.get_direction())
            forces.append(nodal_load.get_force())
            max_displacements.append(nodal_load.get_max_displacement())
    return Load(node_list, directions, forces, max_displacements)


def text_to_model(model_text: str, damping, evaluator: se.SimpleEval = None) -> dynamics.DynamicModel:
    if evaluator is None:
        evaluator = se.SimpleEval()
    assembly_text, loadmap_text = interpreting.split(model_text, 'LOADMAP\n')
    assem = text_to_assembly(assembly_text, damping, evaluator)
    ldm = text_to_loadmap('LOADMAP\n' + loadmap_text, assem.get_nodes(), evaluator)
    return dynamics.DynamicModel(assem, ldm)


def read_model(model_path, damping, parameters: dict[str, float | str] = None) -> dynamics.DynamicModel:
    _parameters, _ = interpreting.read_parameters_from_model_file(model_path)
    if parameters is not None:
        _parameters.update(parameters)
    evaluator = se.SimpleEval(names=_parameters)
    model_text = ''
    reading_model = False
    with open(model_path, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if len(row) == 0 or row[0].lstrip().startswith('#'):
                continue
            if len(row) == 1 and row[0].strip() == 'NODES':
                reading_model = True
            if reading_model:
                model_text += ', '.join([row_item.strip() for row_item in row]) + '\n'
    return text_to_model(model_text.strip(), damping, evaluator)
