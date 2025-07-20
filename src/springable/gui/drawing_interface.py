import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from .gui_utils import SimpleToolbar, get_current_recursion_depth, get_recursion_limit
from .gui_event_handler import GUIEventHandler
from .gui_settings import XLIM, YLIM
import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator


class DrawingSpace:
    def __init__(self, drawing_frame: ttk.Frame, handler: GUIEventHandler, window: tk.Tk):
        self.win = window
        self.handler = handler
        fig = Figure(figsize=(6, 4.5))
        self._bg = None
        self.ax = fig.add_subplot()

        # Axis control panel
        xmin = XLIM[0]
        xmax = XLIM[1]
        ymin = YLIM[0]
        ymax = YLIM[1]
        x_pnl = ttk.Frame(drawing_frame)
        y_pnl = ttk.Frame(drawing_frame)
        self.axis_status_lbl = tk.Label(drawing_frame, text="", fg='green')
        tk.Label(x_pnl, text="min x").grid(row=0, column=0)
        self.xmin_entry = tk.Entry(x_pnl, width=5)
        self.xmin_entry.grid(row=0, column=1)
        self.xmin_entry.insert(0, f"{xmin}")
        tk.Label(x_pnl, text="max x").grid(row=0, column=2)
        self.xmax_entry = tk.Entry(x_pnl, width=5)
        self.xmax_entry.grid(row=0, column=3)
        self.xmax_entry.insert(0, f"{xmax}")

        tk.Label(y_pnl, text="min y").grid(row=0, column=0)
        self.ymin_entry = tk.Entry(y_pnl, width=5)
        self.ymin_entry.grid(row=0, column=1)
        self.ymin_entry.insert(0, f"{ymin}")
        tk.Label(y_pnl, text="max y").grid(row=0, column=2)
        self.ymax_entry = tk.Entry(y_pnl, width=5)
        self.ymax_entry.grid(row=0, column=3)
        self.ymax_entry.insert(0, f"{ymax}")

        self.xmin_entry.bind("<KeyRelease>", self._update_xaxis_limits)
        self.xmax_entry.bind("<KeyRelease>", self._update_xaxis_limits)
        self.ymin_entry.bind("<KeyRelease>", self._update_yaxis_limits)
        self.ymax_entry.bind("<KeyRelease>", self._update_yaxis_limits)

        self.ax.set_xlim((xmin, xmax))
        self.ax.set_ylim((ymin, ymax))
        self._xaxis_line, = self.ax.plot([xmin, xmax], [0, 0], 'k-', lw=1, animated=True)
        self._yaxis_line, = self.ax.plot([0, 0], [ymin, ymax], 'k-', lw=1, animated=True)

        fig_frame = ttk.Frame(drawing_frame)
        self.ax.set_xlabel("generalized displacement $U=\\alpha-\\alpha_0$ ")
        self.ax.set_ylabel("generalized force $F=\\partial_{\\alpha} E$")
        self.canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        toolbar = SimpleToolbar(self.canvas, fig_frame, self._reset_axis)
        toolbar.update()

        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.curves: dict[str, Line2D] = {}
        # self._extra_curves: dict[str, Line2D] = {}
        self.response_curve: Line2D | None = None

        self.curve_interactors: dict[str, CurveInteractor] = {}
        self.curve_control_points: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._exp_curves = []
        self._active_curve_interactor: CurveInteractor | None = None
        self.cid = self.canvas.mpl_connect("draw_event", self.on_draw)

        bottom_frame = ttk.Frame(drawing_frame)


        self.add_remove_cp_btn_frame = ttk.Frame(bottom_frame)
        self.add_cp_btn = ttk.Button(self.add_remove_cp_btn_frame, text='Add CP', command=self.add_control_point)
        self.remove_cp_btn = ttk.Button(self.add_remove_cp_btn_frame, text='Remove CP',
                                        command=self.remove_control_point)
        self.add_cp_btn.grid(column=0, row=0)
        self.remove_cp_btn.grid(column=1, row=0)

        exp_frame = ttk.Frame(bottom_frame)
        self.show_exp_btn = ttk.Button(exp_frame, text='Load an experimental curve',
                                       command=self.handler.load_experimental_curve)
        self.remove_exp_btn = ttk.Button(exp_frame, text='Remove curve',
                                         command=self.handler.remove_experimental_curve)
        self.show_exp_btn.grid(column=0, row=0)
        self.remove_exp_btn.grid(column=1, row=0)

        x_pnl.grid(row=0, column=0, sticky='W')
        y_pnl.grid(row=0, column=2, sticky='E')
        self.axis_status_lbl.grid(row=0, column=1)
        bottom_frame.grid(column=0, row=2, columnspan=3, sticky='W')
        fig_frame.grid(row=1, column=0, columnspan=3)
        self.add_remove_cp_btn_frame.grid(row=0, column=1, sticky='E')
        exp_frame.grid(row=0, column=0, sticky='W')
        self.add_remove_cp_btn_frame.grid_remove()

        self._current_curve_name = None

    def _reset_axis(self):
        self.xmin_entry.delete(0, tk.END)
        self.xmax_entry.delete(0, tk.END)
        self.ymin_entry.delete(0, tk.END)
        self.ymax_entry.delete(0, tk.END)
        self.xmin_entry.insert(0, f"{XLIM[0]}")
        self.xmax_entry.insert(0, f"{XLIM[1]}")
        self.ymin_entry.insert(0, f"{YLIM[0]}")
        self.ymax_entry.insert(0, f"{YLIM[1]}")
        self._update_xaxis_limits()
        self._update_yaxis_limits()

    def _update_xaxis_limits(self, *args):
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
        except ValueError:
            self.axis_status_lbl.config(text="Invalid axis limits (invalid numbers)", fg="red")
        else:
            if xmin >= xmax:
                self.axis_status_lbl.config(text="Invalid axis limits (min >= max)", fg="red")
            else:
                self.axis_status_lbl.config(text="", fg="green")
                self.ax.set_xlim((xmin, xmax))
                self.ax.xaxis.set_major_locator(AutoLocator())
                self._xaxis_line.set_data([xmin, xmax], [0, 0])
                self.handler.update_xlimits(xmin, xmax)
                # update() is called through the handler
                self.canvas.draw()

    def _update_yaxis_limits(self, *args):
        try:
            ymin = float(self.ymin_entry.get())
            ymax = float(self.ymax_entry.get())
        except ValueError:
            self.axis_status_lbl.config(text="Invalid y-axis limits (invalid numbers)", fg="red")
        else:
            if ymin >= ymax:
                self.axis_status_lbl.config(text="Invalid y-axis limits (min >= max)", fg="red")
            else:
                self.axis_status_lbl.config(text="", fg="green")
                self.ax.set_ylim((ymin, ymax))
                self.ax.yaxis.set_major_locator(AutoLocator())
                self._yaxis_line.set_data([0, 0], [ymin, ymax])
                self.update()  # manual update() is needed because the handler is not notified
                self.canvas.draw()

    def print_curves(self):
        print('Curves:')
        print(self.curves.keys())
        print('Curves interactors:')
        print(self.curve_interactors.keys())
        print("Curve control points:")
        print(self.curve_control_points.keys())
        print(self.curve_control_points.values())
        print("Active curve interactor:")
        print(self._active_curve_interactor.name if self._active_curve_interactor is not None else 'None')

    def draw_exp_curve(self, u, f):
        exp_curve, = self.ax.plot(u, f, animated=True, lw=3, alpha=0.5)
        self._exp_curves.append(exp_curve)
        self.update()
        self.handler.show_popup('Experimental curve added!', 750)

    def remove_exp_curve(self):
        if self._exp_curves:
            self._exp_curves.pop()
            self.update()
            self.handler.show_popup('Experimental curve removed!', 750)
        else:
            self.handler.show_popup('No experimental curve to remove!', 750)

    def load_new_curve(self, name: str, u, f, is_controllable: bool, cp_x=None, cp_y=None):
        self.curves[name].set_data(u, f)
        self.curves[name].set_linestyle('-')

        if name in self.curve_interactors.keys():
            curve_interactor = self.curve_interactors.pop(name)
            curve_interactor.disconnect()
            curve_interactor.get_poly().remove()
            curve_interactor.get_line().remove()
            self._active_curve_interactor = None
            del curve_interactor

        if is_controllable:
            self.curve_control_points[name] = cp_x, cp_y
            self.curve_interactors[name] = CurveInteractor(self.ax, cp_x, cp_y, name, self, self.handler)
            self._active_curve_interactor = self.curve_interactors[name]
            self.add_remove_cp_btn_frame.grid()
        else:
            self.add_remove_cp_btn_frame.grid_remove()
        self.update()

    def add_curve(self, name: str, u, f, is_controllable: bool, cp_x=None, cp_y=None):
        if self._active_curve_interactor is not None:
            self._active_curve_interactor.disconnect()
        if u is not None and f is not None:
            self.curves[name], = self.ax.plot(u, f, animated=True, lw=1.5)
            # energy = [np.trapezoid(f[u > 0][:i], u[u > 0][:i]) for i in range(u[u > 0].shape[0])]
            # self._extra_curves[name], = self.ax.plot(u[u>0], energy, animated=True, lw=1.5)
        else:
            self.curves[name], = self.ax.plot([], [], animated=True, lw=1.5)
            # self._extra_curves[name], = self.ax.plot([], [], animated=True, lw=1.5)

        if not is_controllable:
            self.curve_control_points[name] = None
        else:
            self.curve_interactors[name] = CurveInteractor(self.ax, cp_x, cp_y, name, self, self.handler)
            self.curve_control_points[name] = cp_x, cp_y
            self._active_curve_interactor = self.curve_interactors[name]
        self.update()

    def add_response_curve(self, u, f):
        if u is not None and f is not None:
            self.response_curve, = self.ax.plot(u, f, 'o', animated=True, color='#000000', markersize=1.0)
        else:
            self.response_curve, = self.ax.plot([], [], 'o', animated=True, color='#000000', markersize=1.0)
        self.update()

    def update_response_curve(self, u, f):
        if u is not None and f is not None:
            self.response_curve.set_xdata(u)
            self.response_curve.set_ydata(f)
        else:
            self.response_curve.set_xdata([])
            self.response_curve.set_ydata([])
        self.update()

    def change_curve_type(self, name, u, f, is_controllable, cp_x=None, cp_y=None):
        if u is not None and f is not None:
            self.curves[name].set_data(u, f)
            self.curves[name].set_linestyle('-')
        else:
            self.curves[name].set_data([], [])

        if is_controllable:
            self.curve_control_points[name] = cp_x, cp_y
            if name not in self.curve_interactors.keys():
                self.curve_interactors[name] = CurveInteractor(self.ax, cp_x, cp_y, name, self, self.handler)
            self._active_curve_interactor = self.curve_interactors[name]
            self.add_remove_cp_btn_frame.grid()
        else:
            if name in self.curve_interactors.keys():
                curve_interactor = self.curve_interactors.pop(name)
                curve_interactor.disconnect()
                curve_interactor.get_poly().remove()
                curve_interactor.get_line().remove()
                self._active_curve_interactor = None
                del curve_interactor
            self.add_remove_cp_btn_frame.grid_remove()
        self.update()

    def switch_to_curve(self, name: str):
        if self._active_curve_interactor is not None:
            self._active_curve_interactor.disconnect()
            self._active_curve_interactor = None
        if name in self.curve_interactors.keys():
            self._active_curve_interactor = self.curve_interactors[name]
            self._active_curve_interactor.reconnect()
            self._active_curve_interactor.ensure_interactor_is_visible()
            self.add_remove_cp_btn_frame.grid()
        else:
            self.add_remove_cp_btn_frame.grid_remove()
        for curve in self.curves.values():
            curve.set_linewidth(1.5)
        self.curves[name].set_linewidth(2.5)
        self._current_curve_name = name
        self.update()

    def remove_curve(self, name):
        self._current_curve_name = None
        self.curves[name].remove()
        try:
            self.curves.pop(name)
        except KeyError:
            pass
        try:
            self.curve_control_points.pop(name)
        except KeyError:
            pass
        try:
            curve_interactor = self.curve_interactors.pop(name)
            curve_interactor.disconnect()
            curve_interactor.get_poly().remove()
            curve_interactor.get_line().remove()
            self._active_curve_interactor = None
            del curve_interactor
        except KeyError:
            pass
        self.update()

    def update_curve(self, name: str, u, f):
        if u is not None and f is not None:
            self.curves[name].set_data(u, f)
            self.curves[name].set_linestyle('-')
            # energy = [np.trapezoid(f[u > 0][:i], u[u > 0][:i]) for i in range(u[u > 0].shape[0])]
            # self._extra_curves[name].set_data(u[u>0], energy)
        else:
            # self._extra_curves[name], = self.ax.plot([], [], animated=True, lw=1.5)
            self.curves[name].set_linestyle('--')
        self.update()

    def update_all_curves(self, uf_data: dict[str, tuple[np.ndarray, np.ndarray]]):
        for name, data in uf_data.items():
            u, f = data
            if u is not None and f is not None:
                self.curves[name].set_data(u, f)
                self.curves[name].set_linestyle('-')
                # energy = [np.trapezoid(f[u > 0][:i], u[u > 0][:i]) for i in range(u[u > 0].shape[0])]
                # self._extra_curves[name].set_data(u[u>0], energy)
            else:
                self.curves[name].set_linestyle('--')
                # self._extra_curves[name], = self.ax.plot([], [], animated=True, lw=1.5)
        self.update()

    def on_draw(self, event):
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        fig = self.canvas.figure
        fig.draw_artist(self._xaxis_line)
        fig.draw_artist(self._yaxis_line)

        for exp_curve in self._exp_curves:
            fig.draw_artist(exp_curve)

        if self.response_curve is not None:
            fig.draw_artist(self.response_curve)

        for name, line in self.curves.items():
            if name != self._current_curve_name:
                fig.draw_artist(line)
        # for name, line in self._extra_curves.items():
        #     fig.draw_artist(line)
        if self._current_curve_name is not None:
            fig.draw_artist(self.curves[self._current_curve_name])
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

    def get_control_points(self, name) -> tuple[np.ndarray, np.ndarray] | None:
        if name in self.curve_interactors.keys():
            return self.curve_interactors[name].get_control_points()
        else:
            return None

    def get_previous_control_points(self, name) -> tuple[np.ndarray, np.ndarray] | None:
        return self.curve_control_points[name]

    def add_control_point(self):
        if self._active_curve_interactor is not None:
            name = self._active_curve_interactor.name
            new_cp_x, new_cp_y = self.handler.get_elevated_control_points(name)
            if new_cp_x.shape[0] > 12:
                self.handler.show_popup('Max 12 control points!', 750)
                return

            self.curve_control_points[name] = new_cp_x, new_cp_y

            # Delete
            curve_interactor = self.curve_interactors.pop(name)
            curve_interactor.disconnect()
            curve_interactor.get_poly().remove()
            curve_interactor.get_line().remove()
            self._active_curve_interactor = None
            del curve_interactor

            # Re-create
            self.curve_interactors[name] = CurveInteractor(self.ax, new_cp_x, new_cp_y, name, self, self.handler)
            self._active_curve_interactor = self.curve_interactors[name]
            self.handler.update_behavior_parameter_from_control_points(name)
            self.update()

    def remove_control_point(self):
        if self._active_curve_interactor is not None:
            name = self._active_curve_interactor.name
            cp_x, cp_y = self.curve_control_points[name]
            if cp_x.shape[0] <= 3:
                self.handler.show_popup('Min 3 control points!', 750)
                return
            new_cp_x = cp_x[:-1].copy()
            new_cp_y = cp_y[:-1].copy()
            self.curve_control_points[name] = new_cp_x, new_cp_y

            # Delete
            curve_interactor = self.curve_interactors.pop(name)
            curve_interactor.disconnect()
            curve_interactor.get_poly().remove()
            curve_interactor.get_line().remove()
            self._active_curve_interactor = None
            del curve_interactor

            # Re-create
            self.curve_interactors[name] = CurveInteractor(self.ax, new_cp_x, new_cp_y, name, self, self.handler)
            self._active_curve_interactor = self.curve_interactors[name]
            self.handler.update_behavior_parameter_from_control_points(name)
            self.update()


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
        self.canvas = drawing_space.canvas
        self.poly.set_visible(False)
        x, y = zip(*self.poly.xy)

        # draw line that connect control points
        self.line = Line2D(x, y, linestyle=(0, (1, 0.75)), color='k', lw=2, alpha=0.75,
                           marker='o', markersize=10, markerfacecolor='tab:green', markeredgewidth=2,
                           animated=True)
        self.ax.add_line(self.line)
        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.cid_btn_pressed = self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_key_pressed = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_btn_released = self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.cid_mouse_moved = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid_btn_pressed)
        self.canvas.mpl_disconnect(self.cid_key_pressed)
        self.canvas.mpl_disconnect(self.cid_btn_released)
        self.canvas.mpl_disconnect(self.cid_mouse_moved)

    def reconnect(self):
        self.cid_btn_pressed = self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_key_pressed = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_btn_released = self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.cid_mouse_moved = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

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
        if self._ind is not None and self._ind != 0:
            self.canvas.get_tk_widget().config(cursor="dot")

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.show_vertices:
            return
        if event.button != 1:
            return
        self._ind = None
        self.canvas.get_tk_widget().config(cursor="arrow")

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

    def ensure_interactor_is_visible(self):
        self.show_vertices = True
        self.line.set_visible(self.show_vertices)
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if self._ind is None:
            ind = self.get_ind_under_point(event)
            if ind is not None and ind != 0:
                self.canvas.get_tk_widget().config(cursor="hand1")
            elif ind == 0:
                self.canvas.get_tk_widget().config(cursor="X_cursor")
            else:
                self.canvas.get_tk_widget().config(cursor="arrow")

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
        cp_x, cp_y = self.get_control_points()
        self.line.set_data(cp_x, cp_y)
        self.ds.curve_control_points[self.name] = cp_x, cp_y

        if get_current_recursion_depth() > 0.8 * get_recursion_limit():
            self.canvas.mpl_disconnect(self.cid_mouse_moved)
            self.handler.update_behavior_parameter_from_control_points(self.name)
            self.ds.win.after_idle(self._reconnect_mouse_move)
        else:
            self.handler.update_behavior_parameter_from_control_points(self.name)


    def get_control_points(self):
        cp_x, cp_y = zip(*self.poly.xy)
        cp_x = np.array(cp_x)
        cp_y = np.array(cp_y)
        return cp_x, cp_y
    
    def _reconnect_mouse_move(self):
        self.cid_mouse_moved = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
