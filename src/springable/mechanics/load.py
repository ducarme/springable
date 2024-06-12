from .node import Node
import numpy as np


class NodalLoad:
    def __init__(self, node: Node, direction: str, force: float, max_displacement: float | None):
        self._node = node
        self._direction = direction
        self._force = force
        self._max_displacement = max_displacement

    def get_node(self) -> Node:
        return self._node

    def get_direction(self) -> str:
        return self._direction

    def get_force(self) -> float:
        return self._force

    def get_max_displacement(self) -> float | None:
        return self._max_displacement


class LoadStep:

    def __init__(self, nodes: list[Node], directions: list[str], forces: list[float], max_displacements: list[float | None]):
        self._load = []
        ld = {}
        for _node, _direction, _force, _max_displacement in zip(nodes, directions, forces, max_displacements):
            if (_node, _direction) not in ld:
                ld[(_node, _direction)] = [0.0, None]
            ld[_node, _direction][0] += _force
            if _max_displacement is not None:
                ld[_node, _direction][1] = _max_displacement
        for node_and_dir, force_and_maxu in ld.items():
            _node, _direction = node_and_dir
            _force, _max_u = force_and_maxu
            self._load.append(NodalLoad(_node, _direction, _force, _max_u))

    def get_nodal_loads(self) -> list[NodalLoad]:
        return self._load




