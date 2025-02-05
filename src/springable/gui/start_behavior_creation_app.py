import tkinter as tk
import tkinter.ttk as ttk
from .control_panel_interface import BehaviorNotebook
from .drawing_interface import DrawingSpace
from .gui_event_handler import GUIEventHandler
import warnings
import sys
import os


def suppress_output():
    sys.stderr = open(os.devnull, 'w')


# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def start_behavior_creation():
    window = tk.Tk()
    suppress_output()
    window.wm_title('Behavior creation')
    handler = GUIEventHandler()
    main_frame = ttk.Frame(window, padding=(3, 3, 12, 12))
    main_frame.grid(column=0, row=0)
    drawing_frame = ttk.Frame(main_frame, borderwidth=5, relief="ridge", width=200, height=100)
    drawing_frame.grid(column=0, row=0, columnspan=3, rowspan=2)
    notebook_frame = ttk.Frame(main_frame)
    notebook_frame.grid(column=3, row=0, sticky='N')
    ds = DrawingSpace(drawing_frame, handler)
    handler.connect_to_drawing_space(ds)
    bn = BehaviorNotebook(notebook_frame, handler, window)
    handler.connect_to_notebook(bn)
    window.mainloop()
