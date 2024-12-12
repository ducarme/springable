import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector


def value_to_text(val, fmt='.3E', parameter_name='Value'):
    return f"{parameter_name}={val:{fmt}}"


def make_validate_and_update_function(low_lim_var, high_lim_var, slider, err_lbl):
    def validate_and_update_limits(*args):
        try:
            new_low = float(low_lim_var.get())
            new_high = float(high_lim_var.get())
            if new_low < new_high:
                slider.config(from_=new_low, to=new_high)
                err_lbl.config(text="")
            else:
                err_lbl.config(text="Low limit must be less than high limit")
        except ValueError:
            err_lbl.config(text="Please enter valid numbers")

    return validate_and_update_limits


def make_update_value_text_function(parameter_name, slider_value_lbl):
    def update_value_text(val):
        slider_value_lbl.config(text=value_to_text(float(val), parameter_name=parameter_name))

    return update_value_text


def slider_panel(root, parameter_name, initial_val, low, high, command, row):
    low_limit_var = tk.StringVar(value=str(low))
    high_limit_var = tk.StringVar(value=str(high))

    low_limit_entry = ttk.Entry(root, textvariable=low_limit_var, width=6)
    high_limit_entry = ttk.Entry(root, textvariable=high_limit_var, width=6)
    slider_value_lbl = ttk.Label(root, text=value_to_text(initial_val, parameter_name=parameter_name))
    slider = ttk.Scale(root, orient=tk.HORIZONTAL, length=120, from_=low, to=high, value=initial_val,
                       command=lambda val: [make_update_value_text_function(parameter_name, slider_value_lbl)(val),
                                            command()])
    error_lbl = ttk.Label(root, text="", foreground="red")
    validate_and_update_limits = make_validate_and_update_function(low_limit_var, high_limit_var, slider, error_lbl)
    low_limit_entry.bind("<KeyRelease>", validate_and_update_limits)
    high_limit_entry.bind("<KeyRelease>", validate_and_update_limits)

    # placement
    slider_value_lbl.grid(column=0, row=row, sticky='W')
    low_limit_entry.grid(column=1, row=row)
    high_limit_entry.grid(column=3, row=row)
    slider.grid(column=2, row=row)
    error_lbl.grid(column=1, row=row + 1, columnspan=3)
    return slider


class SimpleToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, home: callable):
        NavigationToolbar2Tk.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),)
        super().__init__(canvas, parent, pack_toolbar=False)
        self._home = home

    def home(self, *args):
        self._home(*args)


class Tooltip:
    def __init__(self, widget, text_var, movable=False):
        self.widget = widget
        self.text_var = text_var
        self.tooltip = None

        # Bind events to the widget
        self.widget.bind("<Enter>", self.show_tooltip)
        if movable:
            self.widget.bind("<Motion>", self.move_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if not self.tooltip:
            # Create a Toplevel window for the tooltip
            self.tooltip = tk.Toplevel(self.widget)
            self.tooltip.wm_overrideredirect(True)  # Remove window decorations
            self.tooltip.attributes("-topmost", True)  # Always on top

            # Create a Label inside the Toplevel for the text
            label = tk.Label(
                self.tooltip,
                text=self.text_var.get(),
                bg="lightyellow",
                fg="black",
                relief="solid",
                borderwidth=1,
                padx=5,
                pady=2
            )
            label.pack()

        # Position the tooltip
        self.move_tooltip(event)

    def move_tooltip(self, event):
        if self.tooltip:
            # Update the tooltip text dynamically
            self.tooltip.children["!label"].config(text=self.text_var.get())
            # Position the Toplevel near the cursor
            x = event.x_root + 5
            y = event.y_root + 5
            self.tooltip.geometry(f"+{x}+{y}")

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def show_popup(root: tk.Tk, message, duration):
    popup = tk.Toplevel(root)
    popup.wm_overrideredirect(True)  # Remove window decorations
    popup.attributes("-topmost", True)  # Ensure it stays on top

    popup.attributes("-alpha", 0.5)
    label = tk.Label(popup, text=message, bg="white", fg="black",
                     font=("Helvetica", 30, "bold"), relief="flat", padx=10, pady=5)
    label.pack()

    # Position the popup in the center of the main window
    root.update_idletasks()  # Ensure geometry updates
    x = root.winfo_x() + (root.winfo_width() // 2) - (popup.winfo_reqwidth() // 2)
    y = root.winfo_y() + (root.winfo_height() // 2) - (popup.winfo_reqheight() // 2)
    popup.geometry(f"+{x}+{y}")

    # Destroy the popup after `duration` milliseconds
    popup.after(duration, popup.destroy)
