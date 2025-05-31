from dataclasses import dataclass
from tkinter import filedialog, messagebox
from .gui_utils import show_popup
from ..mechanics.mechanical_behavior import *
from ..utils import bezier_curve
from ..readwrite.interpreting import behavior_to_text
from ..readwrite import fileio
from .gui_settings import DEFAULT_BEHAVIORS, DEFAULT_NATURAL_MEASURE, XLIM, NB_SAMPLES, FORCE_COLUMN_INDEX, \
    DISPLACEMENT_COLUMN_INDEX, DELIMITER, OOB_TOL, FMAX, SAMPLING

import numpy as np


@dataclass
class BehaviorInfo:
    behavior_type: str
    behavior_parameters: dict
    behavior_natural_measure: float


print_messages = False


class GUIEventHandler:
    def __init__(self):
        self._behaviors: dict[str, MechanicalBehavior] = {}
        self._behavior_errors: dict[str, str] = {}
        self._umin = XLIM[0]
        self._umax = XLIM[1]
        self._nb_samples = NB_SAMPLES
        
        self._behavior_notebook = None
        self._drawing_space = None

    def update_xlimits(self, xmin, xmax):
        self._umin = xmin
        self._umax = xmax
        uf_data = {}
        for name, behavior in self._behaviors.items():
            if not self._behavior_errors[name]:
                if isinstance(behavior, UnivariateBehavior):
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = behavior.gradient_energy(behavior.get_natural_measure() + u)[0]
                elif isinstance(behavior, BivariateBehavior):
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = behavior.a(t)
                    f = behavior.b(t)
                else:
                    raise ValueError('Unknown behavior family')
            else:
                u = None
                f = None
            uf_data[name] = u, f
        self._drawing_space.update_all_curves(uf_data)

    def print_behaviors(self):
        if print_messages:
            for name, behavior in self._behaviors.items():
                print(f'{name}: {behavior_to_text(behavior, fmt='.2E', full_name=True, specify_natural_measure=True)}')

    def connect_to_notebook(self, behavior_notebook):
        self._behavior_notebook = behavior_notebook

    def connect_to_drawing_space(self, drawing_space):
        self._drawing_space = drawing_space

    def remove_behavior(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle the removal of the behavior named {tab_name}')
        try:
            self._behaviors.pop(tab_name)
        except KeyError:
            pass
        try:
            self._behavior_errors.pop(tab_name)
        except KeyError:
            pass
        self._drawing_space.remove_curve(tab_name)
        self.print_behaviors()

    def get_elevated_control_points(self, name):
        b = self._behaviors[name]
        if isinstance(b, (BezierBehavior, Bezier2Behavior)):
            cp_x, cp_y = b.get_control_points()
            new_cp_x = bezier_curve.elevate_order(cp_x)
            new_cp_y = bezier_curve.elevate_order(cp_y)
            return new_cp_x, new_cp_y
        elif isinstance(b, PiecewiseBehavior):
            cp_x, cp_y = b.get_control_points()
            new_cp_x = np.append(cp_x, cp_x[-1] + 4 * b.get_parameters()['us'])
            new_cp_y = np.append(cp_y, cp_y[-1] + b.get_parameters()['k'][-1] * 4 * b.get_parameters()['us'])
            return new_cp_x, new_cp_y
        elif isinstance(b, (ZigzagBehavior, Zigzag2Behavior)):
            cp_x, cp_y = b.get_control_points()
            s = 0.5
            new_cp_x = np.append(cp_x, cp_x[-1] + (cp_x[-1] - cp_x[-2]) * s)
            new_cp_y = np.append(cp_y, cp_y[-1] + (cp_y[-1] - cp_y[-2]) * s)
            return new_cp_x, new_cp_y
        else:
            raise ValueError

    def switch_focus(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle a focus switch to behavior named {tab_name}')
        self._drawing_space.switch_to_curve(tab_name)

    def add_behavior(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle the addition of a new behavior named {tab_name}')

        behavior_type_name = self._behavior_notebook.get_behavior_type(tab_name)
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        notebook_parameters = self._behavior_notebook.get_behavior_parameters(tab_name)
        behavior = DEFAULT_BEHAVIORS[behavior_type_name].copy()
        self._behaviors[tab_name] = behavior
        error = ''
        try:
            behavior.update(natural_measure, **notebook_parameters)
        except InvalidBehaviorParameters as e:
            error = e.get_message()

        if isinstance(behavior, (BezierBehavior, Bezier2Behavior, PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior)):
            cp_x, cp_y = behavior.get_control_points()
            if isinstance(behavior, UnivariateBehavior):
                u = np.linspace(self._umin, self._umax, self._nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
            elif isinstance(behavior, BivariateBehavior):
                t = np.linspace(-1.25, 1.25, self._nb_samples)
                u = behavior.a(t)
                f = behavior.b(t)
            else:
                raise ValueError('Unknown behavior family')
            self._drawing_space.add_curve(tab_name, u, f, True, cp_x, cp_y)
        else:
            u = np.linspace(self._umin, self._umax, self._nb_samples)
            f = behavior.gradient_energy(natural_measure + u)[0]
            self._drawing_space.add_curve(tab_name, u, f, False)

        self._behavior_errors[tab_name] = error
        self.update_behavior_text(tab_name)

        if (all(err == '' for err in self._behavior_errors.values())
                and all(name in self._behaviors.keys() for name in ('B0', 'B1', 'B2'))
                and not isinstance(self._behaviors['B0'], BivariateBehavior)
                and not isinstance(self._behaviors['B1'], BivariateBehavior)
                and isinstance(self._behaviors['B2'], BivariateBehavior)):
            b0 = self._behaviors['B0']
            b1 = self._behaviors['B1']
            b2: BivariateBehavior = self._behaviors['B2']

            tt = np.linspace(0, 1, SAMPLING)
            u0 = np.linspace(0, self._umax, SAMPLING)
            f  = np.linspace(0, FMAX, SAMPLING)
            T, U0, F = np.meshgrid(tt, u0, f)

            F0 = b0.gradient_energy(U0 + b0.get_natural_measure())[0]

            U2 = b2.a(T)
            F2 = b2.b(T)

            U1 = U0 + U2
            F1 = b1.gradient_energy(U1 + b1.get_natural_measure())[0]

            OOB = (F0 - F1 - F2) ** 2 + (F0 + F1 - F) ** 2

            balance_indices = OOB < OOB_TOL
            balance_f = F[balance_indices].flatten()
            balance_u = (U0[balance_indices] + U1[balance_indices]).flatten()
            self._drawing_space.add_response_curve(balance_u, balance_f)
        self.print_behaviors()

    def change_behavior_type(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle '
                  f'the change of behavior type of the behavior named {tab_name}')

        behavior_type_name = self._behavior_notebook.get_behavior_type(tab_name)
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)
        notebook_parameters = self._behavior_notebook.get_behavior_parameters(tab_name)
        previous_control_points = self._drawing_space.get_previous_control_points(tab_name)
        behavior = DEFAULT_BEHAVIORS[behavior_type_name].copy()
        self._behaviors[tab_name] = behavior

        error = ''
        if isinstance(behavior, (BezierBehavior, Bezier2Behavior, PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior, Spline2Behavior)):
            if previous_control_points is not None:
                try:
                    behavior.update(natural_measure, **notebook_parameters)
                except InvalidBehaviorParameters:
                    pass
                cp_x, cp_y = previous_control_points
                try:
                    behavior.update_from_control_points(cp_x, cp_y)
                except InvalidBehaviorParameters as e:
                    error = e.get_message()
            else:
                try:
                    behavior.update(natural_measure, **notebook_parameters)
                except InvalidBehaviorParameters as e:
                    error = e.get_message()
                cp_x, cp_y = behavior.get_control_points()

            if isinstance(behavior, UnivariateBehavior):
                if not error:
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = behavior.gradient_energy(natural_measure + u)[0]
                else:
                    u = None
                    f = None
            elif isinstance(behavior, BivariateBehavior):
                if not error:
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = behavior.a(t)
                    f = behavior.b(t)
                else:
                    u = None
                    f = None
            else:
                raise ValueError('Unknown behavior family (not univariate nor bivariate)')
            self._drawing_space.change_curve_type(tab_name, u, f, True, cp_x, cp_y)
        else:
            try:
                behavior.update(natural_measure, **notebook_parameters)
            except InvalidBehaviorParameters as e:
                error = e.get_message()
            if not error:
                u = np.linspace(self._umin, self._umax, self._nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
            else:
                u = None
                f = None
            self._drawing_space.change_curve_type(tab_name, u, f, False)
        self._behavior_errors[tab_name] = error

        if (all(err == '' for err in self._behavior_errors.values())
                and all(name in self._behaviors.keys() for name in ('B0', 'B1', 'B2'))
                and not isinstance(self._behaviors['B0'], BivariateBehavior)
                and not isinstance(self._behaviors['B1'], BivariateBehavior)
                and isinstance(self._behaviors['B2'], BivariateBehavior)):
            b0 = self._behaviors['B0']
            b1 = self._behaviors['B1']
            b2: BivariateBehavior = self._behaviors['B2']

            tt = np.linspace(0, 1, SAMPLING)
            u0 = np.linspace(0, self._umax, SAMPLING)
            f  = np.linspace(0, FMAX, SAMPLING)
            T, U0, F = np.meshgrid(tt, u0, f)

            F0 = b0.gradient_energy(U0 + b0.get_natural_measure())[0]

            U2 = b2.a(T)
            F2 = b2.b(T)

            U1 = U0 + U2
            F1 = b1.gradient_energy(U1 + b1.get_natural_measure())[0]

            OOB = (F0 - F1 - F2) ** 2 + (F0 + F1 - F) ** 2

            balance_indices = OOB < OOB_TOL
            balance_f = F[balance_indices].flatten()
            balance_u = (U0[balance_indices] + U1[balance_indices]).flatten()
            self._drawing_space.add_response_curve(balance_u, balance_f)

        self.update_behavior_text(tab_name)
        self.print_behaviors()

    def update_behavior_parameter(self, tab_name, parameter_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle the update of '
                  f'the parameter {parameter_name} of the behavior named {tab_name}')
        behavior = self._behaviors[tab_name]
        par_val = self._behavior_notebook.get_behavior_parameter(tab_name, parameter_name)

        error = ''
        try:
            behavior.update(**{parameter_name: par_val})
        except InvalidBehaviorParameters as e:
            error = e.get_message()

        natural_measure = behavior.get_natural_measure()

        if isinstance(behavior, (BezierBehavior, Bezier2Behavior,
                                 PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior, Spline2Behavior)):
            if isinstance(behavior, UnivariateBehavior):
                if not error:
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = behavior.gradient_energy(natural_measure + u)[0]
                else:
                    u = None
                    f = None
            elif isinstance(behavior, BivariateBehavior):
                if not error:
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = behavior.a(t)
                    f = behavior.b(t)
                else:
                    u = None
                    f = None
            else:
                raise ValueError
        else:
            if not error:
                u = np.linspace(self._umin, self._umax, self._nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
            else:
                u = None
                f = None

        self._drawing_space.update_curve(tab_name, u, f)
        self._behavior_errors[tab_name] = error

        if (all(err == '' for err in self._behavior_errors.values())
                and all(name in self._behaviors.keys() for name in ('B0', 'B1', 'B2'))
                and not isinstance(self._behaviors['B0'], BivariateBehavior)
                and not isinstance(self._behaviors['B1'], BivariateBehavior)
                and isinstance(self._behaviors['B2'], BivariateBehavior)):
            b0 = self._behaviors['B0']
            b1 = self._behaviors['B1']
            b2: BivariateBehavior = self._behaviors['B2']

            tt = np.linspace(0, 1, SAMPLING)
            u0 = np.linspace(0, self._umax, SAMPLING)
            f  = np.linspace(0, FMAX, SAMPLING)
            T, U0, F = np.meshgrid(tt, u0, f)

            F0 = b0.gradient_energy(U0 + b0.get_natural_measure())[0]

            U2 = b2.a(T)
            F2 = b2.b(T)

            U1 = U0 + U2
            F1 = b1.gradient_energy(U1 + b1.get_natural_measure())[0]

            OOB = (F0 - F1 - F2) ** 2 + (F0 + F1 - F) ** 2

            balance_indices = OOB < OOB_TOL
            balance_f = F[balance_indices].flatten()
            balance_u = (U0[balance_indices] + U1[balance_indices]).flatten()
            self._drawing_space.update_response_curve(balance_u, balance_f)

        self.update_behavior_text(tab_name)
        self.print_behaviors()
        # to be extended

    def load_experimental_curve(self):
        if print_messages:
            print(f'Drawing space GUI sent event to handler to handle the addition of an experimental curve')

        try:
            file_path = filedialog.askopenfilename(
                title="Select experimental curve",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
            )
            if file_path:
                u, f = fileio.read_experimental_force_displacement_data(file_path,
                                                                        displacement_column_index=DISPLACEMENT_COLUMN_INDEX,
                                                                        force_column_index=FORCE_COLUMN_INDEX,
                                                                        delimiter=DELIMITER)
            else:
                return False
            if u.shape[0] == 0 or f.shape[0] == 0:
                raise ValueError('Could not read any valid numbers in the displacement or force column.')
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid experimental data file. {str(e)}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Error when reading behavior file. {e}")
            return False
        else:
            self._drawing_space.draw_exp_curve(u, f)
            return True


    def remove_experimental_curve(self):
        if print_messages:
            print(f'Drawing space GUI sent event to handler to handle the removal of the last added experimental curve')
        self._drawing_space.remove_exp_curve()

    def update_behavior_parameter_from_control_points(self, name):
        if print_messages:
            print(f'Drawing space GUI sent event to handler to handle the update of the behavior named {name}')
        behavior = self._behaviors[name]
        natural_measure = self._behaviors[name].get_natural_measure()

        if isinstance(behavior, (BezierBehavior, Bezier2Behavior,
                                 PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior, Spline2Behavior)):
            cp_x, cp_y = self._drawing_space.get_control_points(name)
            error = ''
            try:
                behavior.update_from_control_points(cp_x, cp_y)
            except InvalidBehaviorParameters as e:
                error = e.get_message()
            if error:
                u = None
                f = None
            else:
                if isinstance(behavior, UnivariateBehavior):
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = behavior.gradient_energy(natural_measure + u)[0]
                elif isinstance(behavior, BivariateBehavior):
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = behavior.a(t)
                    f = behavior.b(t)
                else:
                    raise ValueError
            # first updating the message, then drawing the curve.
            # Otherwise, curve depiction and error message are sometimes not consistent, somehow.
            self._behavior_errors[name] = error
            self.update_behavior_text(name)
            self._drawing_space.update_curve(name, u, f)

            if (all(err == '' for err in self._behavior_errors.values())
                    and all(name in self._behaviors.keys() for name in ('B0', 'B1', 'B2'))
                    and not isinstance(self._behaviors['B0'], BivariateBehavior)
                    and not isinstance(self._behaviors['B1'], BivariateBehavior)
                    and isinstance(self._behaviors['B2'], BivariateBehavior)):
                b0 = self._behaviors['B0']
                b1 = self._behaviors['B1']
                b2: BivariateBehavior = self._behaviors['B2']

                tt = np.linspace(0, 1, SAMPLING)
                u0 = np.linspace(0, self._umax, SAMPLING)
                f = np.linspace(0, FMAX, SAMPLING)
                T, U0, F = np.meshgrid(tt, u0, f)

                F0 = b0.gradient_energy(U0 + b0.get_natural_measure())[0]

                U2 = b2.a(T)
                F2 = b2.b(T)

                U1 = U0 + U2
                F1 = b1.gradient_energy(U1 + b1.get_natural_measure())[0]

                OOB = (F0 - F1 - F2) ** 2 + (F0 + F1 - F) ** 2

                balance_indices = OOB < OOB_TOL
                balance_f = F[balance_indices].flatten()
                balance_u = (U0[balance_indices] + U1[balance_indices]).flatten()
                self._drawing_space.update_response_curve(balance_u, balance_f)
        self.print_behaviors()

    def update_behavior_natural_measure(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to handle the update of '
                  f'the natural measure of the behavior named {tab_name}')
        behavior = self._behaviors[tab_name]
        natural_measure = self._behavior_notebook.get_natural_measure(tab_name)

        error = ''
        try:
            behavior.update(natural_measure)
        except InvalidBehaviorParameters as e:
            error = e.get_message()

        if error:
            u = None
            f = None
        else:
            if isinstance(behavior, (BezierBehavior, Bezier2Behavior,
                                     PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior, Spline2Behavior)):
                if isinstance(behavior, UnivariateBehavior):
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = behavior.gradient_energy(natural_measure + u)[0]
                elif isinstance(behavior, BivariateBehavior):
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = behavior.a(t)
                    f = behavior.b(t)
                else:
                    raise ValueError
            else:
                u = np.linspace(self._umin, self._umax, self._nb_samples)
                f = behavior.gradient_energy(natural_measure + u)[0]
        self._drawing_space.update_curve(tab_name, u, f)
        self._behavior_errors[tab_name] = error
        self.update_behavior_text(tab_name)
        self.print_behaviors()

    def update_behavior_text(self, tab_name):
        error = self._behavior_errors[tab_name]
        if not error:
            specify_natural_measure: bool = self._behavior_notebook.get_specify_natural_measure_state(tab_name)
            self._behavior_notebook.set_behavior_text(tab_name,
                                                      self._get_behavior_text(tab_name, specify_natural_measure))
        else:
            self._behavior_notebook.set_behavior_text(tab_name, self._behavior_errors[tab_name])
        self._behavior_notebook.set_behavior_validity(tab_name, not error)

    def _get_behavior_text(self, tab_name: str, specify_natural_measure: bool, fmt='.2E') -> str:
        return behavior_to_text(self._behaviors[tab_name],
                                fmt=fmt, full_name=True, specify_natural_measure=specify_natural_measure)

    def copy_behavior_to_clipboard(self, tab_name):
        specify_natural_measure = self._behavior_notebook.get_specify_natural_measure_state(tab_name)
        text = self._get_behavior_text(tab_name, specify_natural_measure, fmt='.3E')
        self._behavior_notebook.win.clipboard_clear()
        self._behavior_notebook.win.clipboard_append(text)
        self._behavior_notebook.win.update()
        self.show_popup('Copied to clipboard!', duration=400)

    def show_popup(self, message: str, duration: int):
        show_popup(self._behavior_notebook.win, message, duration)

    def write_behavior(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to write the behavior named {tab_name} into a CSV file')
        file_path = filedialog.asksaveasfilename(
            title="Save behavior as",
            defaultextension=".csv",
            initialfile=tab_name + '.csv',
            filetypes=[("CSV Files", "*.csv")],
        )
        b = self._behaviors[tab_name]
        if file_path:
            try:
                specify_natural_measure = self._behavior_notebook.get_specify_natural_measure_state(tab_name)
                fileio.write_behavior(b, file_path, '.3E', specify_natural_measure)
            except Exception as e:
                messagebox.showerror("Error", f"Error while saving behavior. {e}")
            else:
                self.show_popup('Saved!', duration=400)

    def load_from_file(self, tab_name):
        if print_messages:
            print(f'Notebook GUI sent event to handler to read and load a new  behavior in place of the behavior named'
                  f'{tab_name}')
        try:
            file_path = filedialog.askopenfilename(
                title="Select behavior file",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
            )
            if file_path:
                self._behaviors[tab_name] = fileio.read_behavior(file_path, natural_measure=DEFAULT_NATURAL_MEASURE)
            else:
                return False
        except InvalidBehaviorParameters as e:
            messagebox.showerror("Error", f"Invalid behavior parameters. {e.get_message()}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Error when reading behavior file. {e}")
            return False
        else:
            # else is probably not needed in this case,
            # because all except clauses end with 'return'
            b = self._behaviors[tab_name]
            self._behavior_errors[tab_name] = ''
            if isinstance(b, (BezierBehavior, Bezier2Behavior,
                              PiecewiseBehavior, ZigzagBehavior, Zigzag2Behavior, Spline2Behavior)):
                cp_x, cp_y = b.get_control_points()
                if isinstance(b, UnivariateBehavior):
                    u = np.linspace(self._umin, self._umax, self._nb_samples)
                    f = b.gradient_energy(b.get_natural_measure() + u)[0]
                elif isinstance(b, BivariateBehavior):
                    t = np.linspace(-1.25, 1.25, self._nb_samples)
                    u = b.a(t)
                    f = b.b(t)
                else:
                    raise ValueError('Unknown behavior family')
                self._drawing_space.load_new_curve(tab_name, u, f, True, cp_x, cp_y)
            else:
                u = np.linspace(self._umin, self._umax, self._nb_samples)
                f = b.gradient_energy(b.get_natural_measure() + u)[0]
                self._drawing_space.load_new_curve(tab_name, u, f, False)
            return True

    def get_behavior_type_name(self, name) -> str:
        for j, bt_name in enumerate(list(DEFAULT_BEHAVIORS.keys())):
            if type(DEFAULT_BEHAVIORS[bt_name]) is type(self._behaviors[name]):
                return bt_name
        else:
            raise ValueError(f'unknown behavior type "{type(self._behaviors[name])}"')

    def get_behavior_parameters(self, name) -> dict:
        return self._behaviors[name].copy().get_parameters()

    def get_behavior_natural_measure(self, name) -> float:
        return self._behaviors[name].get_natural_measure()
