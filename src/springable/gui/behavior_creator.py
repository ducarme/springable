import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import TextBox, Button
import pyperclip
from ..mechanics import mechanical_behavior as mb
from ..mechanics.math_utils import smooth_zigzag_curve as zigzag_utils
from ..readwrite import interpreting
from ..readwrite.keywords import usable_behaviors


def beta_plus(s, delta_inf=0.5, kappa=1.0):
    return 0.5 * (np.sqrt((s + delta_inf) ** 2 + kappa ** 2) + s + delta_inf)


def beta_minus(s, delta_inf=0.5, kappa=1.0):
    return -beta_plus(-s, delta_inf, kappa)


def gamma_plus(s, delta_s, delta_inf, kappa):
    return beta_plus(s - delta_s, delta_inf, kappa) + delta_s


def gamma_minus(s, delta_s, delta_inf, kappa):
    return beta_minus(s - delta_s, delta_inf, kappa) + delta_s


def mu_plus(s, k_star, delta, epsilon):
    s_star_plus = k_star - delta
    return ((s <= s_star_plus - epsilon) * k_star
            + (s_star_plus - epsilon < s) * (s <= s_star_plus + epsilon) * (
                    0.5 / epsilon * (0.5 * s ** 2 - (s_star_plus - epsilon) * s) + k_star + 0.25 / epsilon * (
                    s_star_plus - epsilon) ** 2)
            + (s > s_star_plus + epsilon) * (s + delta))


def mu_minus(s, k_star, delta, epsilon):
    s_star_minus = k_star + delta
    return ((s <= s_star_minus - epsilon) * (s - delta)
            + (s_star_minus - epsilon < s) * (s <= s_star_minus + epsilon) * (
                    0.5 / epsilon * ((s_star_minus + epsilon) * s - 0.5 * s ** 2) + k_star - 0.25 / epsilon * (
                    s_star_minus + epsilon) ** 2)
            + (s > s_star_minus + epsilon) * k_star)


def k_star_fun(delta_k, epsilon_k, k_max, delta, epsilon):
    c = k_max + delta + epsilon - 0.25 / epsilon_k * (k_max + delta + epsilon) / (delta + epsilon) * (
            delta + epsilon + epsilon_k) ** 2
    return ((delta_k < delta + epsilon - epsilon_k) * delta_k * (k_max + delta + epsilon) / (delta + epsilon)
            + (np.abs(delta_k - (delta + epsilon)) < epsilon_k) *
            (0.5 / epsilon_k * (k_max + delta + epsilon) / (delta + epsilon)
             * ((delta + epsilon + epsilon_k) * delta_k - 0.5 * delta_k ** 2) + c)
            + (delta_k > delta + epsilon + epsilon_k) * (k_max + delta + epsilon)
            )


def update_behavior_parameters_from_control_points(behavior_type: type[mb.MechanicalBehavior],
                                                   behavior_parameters: dict[str, ...],
                                                   x, y):
    if behavior_type == mb.ZigZagBehavior:
        a, x = zigzag_utils.compute_zizag_slopes_and_transitions_from_control_points(x, y)
        behavior_parameters['a'] = a
        behavior_parameters['x'] = x
    elif behavior_type in (mb.BezierBehavior, mb.Bezier2Behavior, mb.ZigZag2Behavior):
        behavior_parameters['u_i'] = list(x[1:])
        behavior_parameters['f_i'] = list(y[1:])


def get_control_points_from_behavior(_behavior: mb.MechanicalBehavior) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(_behavior, mb.ZigZagBehavior):
        a = _behavior.get_parameters()['a']
        x = _behavior.get_parameters()['x']
        cp_x, cp_y = zigzag_utils.compute_zigzag_control_points(a, x)
        return cp_x, cp_y
    elif isinstance(_behavior, (mb.BezierBehavior, mb.Bezier2Behavior, mb.ZigZag2Behavior)):
        cp_x = np.array([0.0] + _behavior.get_parameters()['u_i'])
        cp_y = np.array([0.0] + _behavior.get_parameters()['f_i'])
        return cp_x, cp_y
    else:
        raise NotImplementedError("Cannot define control points for that behavior")


class CurveInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

    """
    show_vertices = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly,
                 _behavior: mb.MechanicalBehavior, gui=None):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.poly.set_visible(False)
        self._behavior = _behavior
        self._behavior_type = type(_behavior)
        self._behavior_parameters = _behavior.get_parameters()
        self._behavior_parameters_valid = True

        x, y = zip(*self.poly.xy)

        update_behavior_parameters_from_control_points(self._behavior_type, self._behavior_parameters, x, y)
        try:
            self._behavior = self._behavior_type(1.0, **self._behavior_parameters)
        except mb.InvalidBehaviorParameters:
            self._behavior_parameters_valid = False

        # draw line that connect control points
        self.line = Line2D(x, y, linestyle='--', color='#CECECE', lw=3,
                           marker='o', markersize=10, markerfacecolor='tab:green',
                           animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        # draw curve defined by control points
        # self.curve2 = None
        # self.curve3 = None
        # self.curve4 = None
        # self.curve5 = None
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
                # self.curve5 = Line2D(self._behavior._a(t) - t, self._behavior._b(t), animated=True)
                # self.curve5.set_color('tab:orange')
                # self.curve2 = Line2D(t * np.max(self._behavior._a(t)), self._behavior._dbda(t),
                #                      linestyle='', color='#a0a0a0', marker='o', markersize=2,
                #                      markerfacecolor='#a0a0a0', animated=True)
                # self.curve3 = Line2D(t * np.max(self._behavior._a(t)), self._behavior._k(t),
                #                      linestyle='', color='r', marker='o', markersize=2,
                #                      markerfacecolor='r', animated=True)
                # self.curve4 = Line2D(t * np.max(self._behavior._a(t)), k_star * np.ones_like(t),
                # linestyle='', color='g', marker='o', markersize=2,
                # markerfacecolor='g', animated=True)
            else:
                self.curve = Line2D([], [], animated=True)
                self.curve.set_color('salmon')

        self.ax.add_line(self.curve)
        # self.ax.add_line(self.curve2)
        # self.ax.add_line(self.curve3)
        # self.ax.add_line(self.curve4)
        # self.ax.add_line(self.curve5)

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

        self.behavior_creator_gui = gui
        # self.behavior_creator_gui.update_energy_landscape(self._behavior)
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
        # self.ax.draw_artist(self.curve5)

        # self.ax.draw_artist(self.curve2)
        # self.ax.draw_artist(self.curve3)
        # self.ax.draw_artist(self.curve4)
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


class BehaviorCreatorGUI:
    DEFAULT_COLOR = '#CCCCCC'
    DEFAULT_HOVER_COLOR = '#EEEEEE'

    def __init__(self, initial_behavior: mb.MechanicalBehavior):
        self._fig, ax0 = plt.subplots(figsize=(8, 5))
        # self._fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))
        # ax1.remove()
        # ax1 = self._fig.add_subplot(1, 2, 2, projection='3d')
        shift_x = +0.0
        shift_y = +0.05

        self.ax_main = ax0
        self.ax_main.set_xlabel('$\\alpha$ (displacement, angle, ...)')
        self.ax_main.set_ylabel('$\\nabla_{\\alpha} U $ (force, torque, ...)')

        behavior_txt_ax = self._fig.add_axes((0.15, 0.9, 0.75, 0.075))
        self.behavior_txt = TextBox(behavior_txt_ax, "")
        self.behavior_txt.set_active(False)

        copy_behavior_ax = self._fig.add_axes((0.9, 0.9, 0.075, 0.075))
        self._copy_button = Button(copy_behavior_ax, 'Copy', hovercolor='0.975')
        self._copy_button_cid = self._copy_button.on_clicked(self.copy_behavior_txt)
        self._is_copy_button_enabled = True

        # self.el_ax = ax1
        # self._el_surface = None

        cp_x, cp_y = get_control_points_from_behavior(initial_behavior)

        xspan = np.max(cp_x) - np.min(cp_x)
        yspan = np.max(cp_y) - np.min(cp_y)
        self.poly = Polygon(np.column_stack([cp_x, cp_y]), animated=True, closed=False)
        self.ax_main.add_patch(self.poly)
        self.curve_interactor = CurveInteractor(self.ax_main, self.poly, initial_behavior, gui=self)

        self.ax_main.set_xlim((np.min(cp_x) - 0.1 * xspan, np.max(cp_x) + 0.1 * xspan))
        self.ax_main.set_ylim((np.min(cp_y) - 0.1 * yspan, np.max(cp_y) + 0.1 * yspan))

        ax_alpha_min = self._fig.add_axes((0.20 + shift_x, 0.025 + shift_y, 0.1, 0.04))
        self.alpha_min = TextBox(ax_alpha_min, '$\\alpha_\\text{min}$',
                                 color=BehaviorCreatorGUI.DEFAULT_COLOR, initial=f'{self.ax_main.get_xlim()[0]:.2f}',
                                 hovercolor=BehaviorCreatorGUI.DEFAULT_HOVER_COLOR)
        self.alpha_min.on_submit(self.set_alpha_min)
        self.alpha_min.on_text_change(lambda txt: BehaviorCreatorGUI.color_according_to_validity(self.alpha_min, txt))

        ax_alpha_max = self._fig.add_axes((0.40 + shift_x, 0.025 + shift_y, 0.1, 0.04))
        self.alpha_max = TextBox(ax_alpha_max, '$\\alpha_\\text{max}$', initial=f'{self.ax_main.get_xlim()[1]:.2f}',
                                 color=BehaviorCreatorGUI.DEFAULT_COLOR,
                                 hovercolor=BehaviorCreatorGUI.DEFAULT_HOVER_COLOR)
        self.alpha_max.on_submit(self.set_alpha_max)
        self.alpha_max.on_text_change(lambda txt: BehaviorCreatorGUI.color_according_to_validity(self.alpha_max, txt))

        ax_nabla_min = self._fig.add_axes((0.6 + shift_x, 0.025 + shift_y, 0.1, 0.04))
        self.nabla_min = TextBox(ax_nabla_min, '$\\nabla U_\\text{min}$', initial=f'{self.ax_main.get_ylim()[0]:.2f}',
                                 color=BehaviorCreatorGUI.DEFAULT_COLOR,
                                 hovercolor=BehaviorCreatorGUI.DEFAULT_HOVER_COLOR)
        self.nabla_min.on_submit(self.set_nabla_min)
        self.nabla_min.on_text_change(lambda txt: BehaviorCreatorGUI.color_according_to_validity(self.nabla_min, txt))

        ax_nabla_max = self._fig.add_axes((0.8 + shift_x, 0.025 + shift_y, 0.1, 0.04))
        self.nabla_max = TextBox(ax_nabla_max, '$\\nabla U_\\text{max}$', initial=f'{self.ax_main.get_ylim()[1]:.2f}',
                                 color=BehaviorCreatorGUI.DEFAULT_COLOR,
                                 hovercolor=BehaviorCreatorGUI.DEFAULT_HOVER_COLOR)
        self.nabla_max.on_submit(self.set_nabla_max)
        self.nabla_max.on_text_change(lambda txt: BehaviorCreatorGUI.color_according_to_validity(self.nabla_max, txt))

        plt.subplots_adjust(bottom=0.25, left=0.25)
        plt.show()

    def update_behavior_txt(self, txt):
        self.behavior_txt.set_val(txt)

    def update_energy_landscape(self, _behavior: mb.Bezier2Behavior):
        self.el_ax.clear()
        self.el_surface = _behavior.plot_energy_landscape(self.el_ax)
        self.el_ax.set_zlim((0, 45))
        self._fig.canvas.draw_idle()

    def enable_copy_button(self):
        if not self._is_copy_button_enabled:
            # print('enable')
            self._copy_button_cid = self._copy_button.on_clicked(self.copy_behavior_txt)
            self._copy_button.ax.set_facecolor('white')  # Reset the color to indicate it's enabled
            self._copy_button.label.set_color('black')
            self._is_copy_button_enabled = True

    def disable_copy_button(self):
        if self._is_copy_button_enabled:
            # print('disable')
            self._copy_button.disconnect(self._copy_button_cid)
            self._copy_button.ax.set_facecolor('lightgray')  # Change the color to indicate it's disabled
            self._copy_button.label.set_color('gray')
            self._copy_button_cid = None  # Clear the connection ID
            self._is_copy_button_enabled = False

    def copy_behavior_txt(self, event):
        # print('copy click')
        pyperclip.copy(interpreting.behavior_to_text(self.curve_interactor.get_behavior()))

    def set_alpha_min(self, txt):
        try:
            xlim = self.ax_main.get_xlim()
            new_xlim = float(txt), xlim[1]
            self.ax_main.set_xlim(new_xlim)
            self._fig.canvas.draw_idle()
        except ValueError:
            pass

    def set_alpha_max(self, txt):
        try:
            xlim = self.ax_main.get_xlim()
            new_xlim = xlim[0], float(txt)
            self.ax_main.set_xlim(new_xlim)
            self._fig.canvas.draw_idle()
        except ValueError:
            pass

    def set_nabla_min(self, txt):
        try:
            ylim = self.ax_main.get_ylim()
            new_ylim = float(txt), ylim[1]
            self.ax_main.set_ylim(new_ylim)
            self._fig.canvas.draw_idle()
        except ValueError:
            pass

    def set_nabla_max(self, txt):
        try:
            ylim = self.ax_main.get_ylim()
            new_ylim = ylim[0], float(txt)
            self.ax_main.set_ylim(new_ylim)
            self._fig.canvas.draw_idle()
        except ValueError:
            pass

    @staticmethod
    def color_according_to_validity(txt_box, txt):
        try:
            txt_box.label.set_color('k')
        except ValueError:
            txt_box.label.set_color('red')

    def b(self, event):
        print('B')

    def c(self, event):
        print('C')

    def d(self, event):
        print('D')

    def e(self, event):
        print('E')
