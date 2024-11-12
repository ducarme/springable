class Node:
    """ Class describing a node """

    def __init__(self, x, y, fixed_horizontally, fixed_vertically, node_nb=None):
        self._x = x
        self._y = y
        self._fixed_horizontally = fixed_horizontally
        self._fixed_vertically = fixed_vertically
        self._node_nb = node_nb

        # only used for dynamic simulation
        self._vx = None
        self._vy = None

    def block_horizontally(self):
        self._fixed_horizontally = True

    def block_vertically(self):
        self._fixed_vertically = True

    def release_horizontally(self):
        self._fixed_horizontally = False

    def release_vertically(self):
        self._fixed_vertically = False

    def set_position(self, position):
        self._x = position[0]
        self._y = position[1]

    def displace(self, u):
        self._x += u[0]
        self._y += u[1]

    def get_node_nb(self) -> int:
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

    # HERE IN BELOW IS ONLY USED FOR DYNAMIC SIMULATIONS
    def set_velocity(self, velocity):
        self._vx = velocity[0]
        self._vy = velocity[1]

    def get_vx(self):
        return self._vx

    def get_vy(self):
        return self._vy
