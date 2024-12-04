import tkinter as tk
import tkinter.ttk as ttk

_PADDING = (3, 3, 12, 12)
_BORDERWIDTH = 2
_RELIEF = 'groove'


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
    # panel = ttk.Frame(window, padding=_PADDING)
    # panel['borderwidth'] = _BORDERWIDTH
    # panel['relief'] = _RELIEF
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
    error_lbl.grid(column=1, row=row+1, columnspan=3)
    return slider


def natural_panel(root, low, high, slider_command):
    panel = ttk.Frame(root, padding=_PADDING)
    panel['borderwidth'] = _BORDERWIDTH
    panel['relief'] = _RELIEF
    slider = ttk.Scale(panel, orient=tk.HORIZONTAL, length=100, from_=low, to=high, command=slider_command)
    slider.grid(column=0, row=0)
    return panel
