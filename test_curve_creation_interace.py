import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
from src.springable.gui.behaviors_tabs import BehaviorNotebook
from src.springable.gui.gui_event_handler import GUIEventHandler


def handle_keypress(event):
    print(event.char)


root = tk.Tk()
root.wm_title('Test tkinter gui')
content = ttk.Frame(root, padding=(3, 3, 12, 12))
drawing_frame = ttk.Frame(content, borderwidth=5, relief="ridge", width=200, height=100)

fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
ax = fig.add_subplot()
line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
ax.set_xlabel("$\\Delta \\alpha$")
ax.set_ylabel("$\\nabla{\\alpha} U$")

canvas = FigureCanvasTkAgg(fig, master=drawing_frame)  # A tk.DrawingArea.
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, drawing_frame, pack_toolbar=False)
toolbar.update()
content.grid(column=0, row=0)
drawing_frame.grid(column=0, row=0, columnspan=3, rowspan=2)
toolbar.pack(side=tk.BOTTOM, fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

handler = GUIEventHandler()
bn = BehaviorNotebook(content, handler)
handler.connect_to_notebook(bn)
bn.get_tab_menu().grid(column=3, row=0, sticky='N')

# root.bind("<Key>", handle_keypress)
root.mainloop()
