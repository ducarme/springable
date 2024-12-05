import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from .gui_event_handler import GUIEventHandler
from ..mechanics.mechanical_behavior import MechanicalBehavior
from ..readwrite import interpreting
from ..readwrite.keywords import usable_behaviors
import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class DrawingSpace:
    def __init__(self, drawing_frame: ttk.Frame, handler: GUIEventHandler):
        self.handler = handler
        fig = Figure(figsize=(5, 4))
        self._bg = None
        self.ax = fig.add_subplot()
        self.ax.spines['top'].set_position('center')
        self.ax.spines['right'].set_position('center')
        self.ax.set_xlim((-5, 5))
        self.ax.set_ylim((-5, 5))
        # self.ax.spines['right'].set_color('none')
        # self.ax.spines['top'].set_color('none')
        # self.ax.xaxis.set_ticks_position('bottom')
        # self.ax.yaxis.set_ticks_position('left')

        self.ax.set_xlabel("$\\Delta \\alpha$")
        self.ax.set_ylabel("$\\nabla{\\alpha} U$")
        self.canvas = FigureCanvasTkAgg(fig, master=drawing_frame)  # A tk.DrawingArea.
        toolbar = NavigationToolbar2Tk(self.canvas, drawing_frame, pack_toolbar=False)
        toolbar.update()

        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.curves: dict[str, Line2D] = {}
        self.curve_interactors: dict[str, CurveInteractor] = {}
        self._active_curve_interactor: CurveInteractor | None = None
        self.cid = self.canvas.mpl_connect("draw_event", self.on_draw)

    def add_curve(self, name: str, u, f, is_controllable: bool, cp_x=None, cp_y=None):
        if not is_controllable:
            self.curves[name], = self.ax.plot(u, f, animated=True)
        else:
            self.curves[name], = self.ax.plot(u, f, animated=True)
            self.curve_interactors[name] = CurveInteractor(self.ax, cp_x, cp_y, name, self, self.handler)
            self._active_curve_interactor = self.curve_interactors[name]
        self.update()

    def remove_curve(self, name):
        self.curves[name].remove()
        try:
            self.curves.pop(name)
        except KeyError:
            pass
        self.update()

    def update_curve(self, name: str, u, f):
        self.curves[name].set_data(u, f)
        self.update()

    def set_focus(self, name):
        pass

    def _send_cp_update_event(self, name):
        pass

    def on_draw(self, event):
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        fig = self.canvas.figure
        for line in self.curves.values():
            fig.draw_artist(line)
        if self._active_curve_interactor is not None:
            fig.draw_artist(self._active_curve_interactor.get_poly())
            fig.draw_artist(self._active_curve_interactor.get_line())

    def update(self):
        fig = self.canvas.figure
        if self._bg is None:
            self.on_draw(None)
        else:
            self.canvas.restore_region(self._bg)
            self._draw_animated()
            self.canvas.blit(fig.bbox)
        self.canvas.flush_events()

    def get_control_points(self, name):
        return self.curve_interactors[name].get_control_points()


class CurveInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

    """
    show_vertices = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, cp_x, cp_y, name, drawing_space: DrawingSpace, handler: GUIEventHandler):
        self.ax = ax
        self.poly = Polygon(np.column_stack([cp_x, cp_y]), animated=True, closed=False)
        self.ax.add_patch(self.poly)
        self.name = name
        self.ds = drawing_space
        self.handler = handler
        self.canvas = self.poly.figure.canvas
        self.poly.set_visible(False)
        x, y = zip(*self.poly.xy)

        # draw line that connect control points
        self.line = Line2D(x, y, linestyle='--', color='#CECECE', lw=3,
                           marker='o', markersize=10, markerfacecolor='tab:green',
                           animated=True)
        self.ax.add_line(self.line)
        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def get_line(self):
        return self.line

    def get_poly(self):
        return self.poly

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
        print(f'update {self.name} because of mouse move')
        self.poly.xy[self._ind] = x, y
        self.line.set_data(zip(*self.poly.xy))
        self.handler.update_behavior_parameter_from_control_points(self.name)


        # draw line
        # self.ds.update()

    def get_control_points(self):
        cp_x, cp_y = zip(*self.poly.xy)
        cp_x = np.array(cp_x)
        cp_y = np.array(cp_y)
        return cp_x, cp_y

