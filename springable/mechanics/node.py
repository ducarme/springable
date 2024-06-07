class Node:
    """ Class describing a node """

    def __init__(self, x, y, fixed_horizontally, fixed_vertically, node_nb=None):
        self._x = x
        self._y = y
        self._fixed_horizontally = fixed_horizontally
        self._fixed_vertically = fixed_vertically
        self._node_nb = node_nb

    def set_position(self, position):
        self._x = position[0]
        self._y = position[1]

    def displace(self, u):
        self._x += u[0]
        self._y += u[1]

    def get_node_nb(self):
        return self._node_nb

    def set_node_nb(self, node_number: int):
        self._node_nb = node_number

    def is_fixed_horizontally(self):
        return self._fixed_horizontally

    def is_fixed_vertically(self):
        return self._fixed_vertically

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

