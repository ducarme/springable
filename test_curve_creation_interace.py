import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
from src.springable.gui.behaviors_tabs import BehaviorNotebook
from src.springable.gui.curve_interactor import DrawingSpace
from src.springable.gui.gui_event_handler import GUIEventHandler


def handle_keypress(event):
    print(event.char)


window = tk.Tk()
window.wm_title('Test tkinter gui')
handler = GUIEventHandler()

main_frame = ttk.Frame(window, padding=(3, 3, 12, 12))
main_frame.grid(column=0, row=0)

drawing_frame = ttk.Frame(main_frame, borderwidth=5, relief="ridge", width=200, height=100)
drawing_frame.grid(column=0, row=0, columnspan=3, rowspan=2)

notebook_frame = ttk.Frame(main_frame)
notebook_frame.grid(column=3, row=0, sticky='N')

ds = DrawingSpace(drawing_frame, handler)
handler.connect_to_drawing_space(ds)

bn = BehaviorNotebook(notebook_frame, handler)
handler.connect_to_notebook(bn)

# window.bind("<Key>", handle_keypress)
window.mainloop()
