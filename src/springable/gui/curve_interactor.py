import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from ..mechanics import mechanical_behavior as mb
from ..readwrite import interpreting
from ..readwrite.keywords import usable_behaviors


class CurveInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

    """
    show_vertices = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly: Polygon,
                 _behavior: mb.MechanicalBehavior, gui=None):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.poly.set_visible(False)
        x, y = zip(*self.poly.xy)

        # self._behavior = _behavior
        # self._behavior_type = type(_behavior)
        # self._natural_measure = _behavior.get_natural_measure()
        # self._behavior_parameters = _behavior.get_parameters()
        # self._behavior_parameters_valid = True
        #
        #
        # update_behavior_parameters_from_control_points(self._behavior_type, self._behavior_parameters, x, y)
        # try:
        #     self._behavior = self._behavior_type(self._natural_measure, **self._behavior_parameters)
        # except mb.InvalidBehaviorParameters:
        #     self._behavior_parameters_valid = False

        # draw line that connect control points
        self.line = Line2D(x, y, linestyle='--', color='#CECECE', lw=3,
                           marker='o', markersize=10, markerfacecolor='tab:green',
                           animated=True)
        self.ax.add_line(self.line)
        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        # draw curve defined by control points
        if self._behavior_parameters_valid:
            if isinstance(self._behavior, mb.UnivariateBehavior):
                span = np.max(self.poly.xy[:, 0]) - np.min(self.poly.xy[:, 0])
                t = np.linspace(np.min(self.poly.xy[:, 0]), np.max(self.poly.xy[:, 0]) + 0.2 * span, 250)
                self.curve = Line2D(t, self._behavior.gradient_energy(t)[0], animated=True)
                self.curve.set_color('tab:blue')
            elif isinstance(self._behavior, mb.BivariateBehavior):
                t = np.linspace(0, 1.1, 3000)
                self.curve = Line2D(self._behavior._a(t), self._behavior._b(t), animated=True)
                self.curve.set_color('tab:blue')
            else:
                self.curve = Line2D([], [], animated=True)
                self.curve.set_color('salmon')

        self.ax.add_line(self.curve)

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

        self.behavior_creator_gui = gui
        if self._behavior_parameters_valid:
            self.behavior_creator_gui.update_behavior_txt(interpreting.behavior_to_text(self._behavior, fmt='.2E'))
            self.behavior_creator_gui.enable_copy_button()
        else:
            self.behavior_creator_gui.update_behavior_txt(f'PARAMETERS DO NOT DEFINE A VALID '
                                                          f'{usable_behaviors.type_to_name[self._behavior_type]} BEHAVIOR')
            self.behavior_creator_gui.disable_copy_button()

    def get_behavior(self):
        return self._behavior

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.curve)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.show_vertices:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.show_vertices:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.show_vertices = not self.show_vertices
            self.line.set_visible(self.show_vertices)
            if not self.show_vertices:
                self._ind = None
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.show_vertices:
            return
        if self._ind is None:
            return
        if self._ind == 0:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y

        # draw line
        self.line.set_data(zip(*self.poly.xy))

        # update behavior parameters
        update_behavior_parameters_from_control_points(self._behavior_type, self._behavior_parameters,
                                                       self.poly.xy[:, 0], self.poly.xy[:, 1])
        try:
            self._behavior = self._behavior_type(1.0, **self._behavior_parameters)
            self._behavior_parameters_valid = True
        except mb.InvalidBehaviorParameters:
            self._behavior_parameters_valid = False

        if self._behavior_parameters_valid:
            self.behavior_creator_gui.update_behavior_txt(interpreting.behavior_to_text(self._behavior, fmt='.2E'))
            # self.behavior_creator_gui.update_energy_landscape(self._behavior)
            self.behavior_creator_gui.enable_copy_button()
            self.curve.set_color('tab:blue')
            if isinstance(self._behavior, mb.UnivariateBehavior):
                span = np.max(self.poly.xy[:, 0]) - np.min(self.poly.xy[:, 0])
                t = np.linspace(np.min(self.poly.xy[:, 0]), np.max(self.poly.xy[:, 0]) + 0.2 * span, 250)
                self.curve.set_data(t, self._behavior.gradient_energy(t)[0])
            elif isinstance(self._behavior, mb.BivariateBehavior):
                t = np.linspace(0, 1.1, 1000)
                self.curve.set_data(self._behavior._a(t), self._behavior._b(t))
                # self.curve2.set_data(t * np.max(self._behavior._a(t)), self._behavior._dbda(t))
                # self.curve3.set_data(t * np.max(self._behavior._a(t)), self._behavior._k(t))
                # self.curve4.set_data(t * np.max(self._behavior._a(t)), k_star * np.ones_like(t))
                # self.curve5.set_data(self._behavior._a(t) - t, self._behavior._b(t))

        else:
            self.behavior_creator_gui.update_behavior_txt(f'PARAMETERS DO NOT DEFINE A VALID '
                                                          f'{usable_behaviors.type_to_name[self._behavior_type]} BEHAVIOR')
            self.behavior_creator_gui.disable_copy_button()
            self.curve.set_color('salmon')

        # update canvas
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.curve)
        # self.ax.draw_artist(self.curve2)
        # self.ax.draw_artist(self.curve3)
        # self.ax.draw_artist(self.curve4)
        # self.ax.draw_artist(self.curve5)
        self.canvas.blit(self.ax.bbox)