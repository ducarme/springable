import tkinter as tk
import tkinter.ttk as ttk
import inspect
from .gui_utils import slider_panel
from .gui_event_handler import GUIEventHandler
from ..readwrite.keywords import usable_behaviors
from .default_behaviors import DEFAULT_BEHAVIORS


class BehaviorNotebook:

    def __init__(self, parent, handler):
        tab_menu = ttk.Notebook(parent)
        tab_plus = ttk.Frame(tab_menu)
        tab_menu.add(tab_plus, text='+')
        self._tab_menu = tab_menu
        self._tab_plus = tab_plus
        self._tabs = dict()
        self._tab_menu.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self.handler: GUIEventHandler = handler

    def on_tab_selected(self, event):
        selected_tab = self._tab_menu.select()
        if self._tab_menu.tab(selected_tab)["text"] == "+":
            self.add_behavior_tab()

    def on_button_clicked(self):
        self.remove_selected_behavior_tab()

    def add_behavior_tab(self):
        i = 0
        while True:
            tab_name = f'B{i}'
            if tab_name not in self._tabs.values():
                new_behavior_tab = BehaviorTab(self, tab_name)
                self._tabs[new_behavior_tab] = tab_name
                break
            else:
                i += 1

        # send event to the handler
        self.handler.add_behavior(new_behavior_tab.get_name(),
                                  new_behavior_tab.get_behavior_type(),
                                  new_behavior_tab.get_parameters())

    def remove_selected_behavior_tab(self):
        selected_tab_id = self._tab_menu.select()
        selected_tab = self._tab_menu.nametowidget(selected_tab_id)

        # destroy the children widget of the selected tab
        for widget in selected_tab.winfo_children():
            widget.destroy()

        # remove the selected tab for the dictionary of tabs (and gets its name)
        tab_name = None
        for tab in self._tabs.keys():
            if tab.tab is selected_tab:
                tab_name = self._tabs.pop(tab)
                break

        # sets the focus/selection to another tab when the last tab was selected to be removed
        selected_tab_index = self._tab_menu.index(selected_tab_id)
        if selected_tab_index == len(self._tab_menu.tabs()) - 2:
            self._tab_menu.select(self._tab_menu.tabs()[len(self._tab_menu.tabs()) - 3])

        # remove the tab from the notebook/tab menu
        self._tab_menu.forget(selected_tab_id)

        # send event to the handler
        self.handler.remove_behavior(tab_name)

    def get_tab_menu(self):
        return self._tab_menu


class BehaviorTab:

    def __init__(self, behavior_notebook: BehaviorNotebook, name):
        self._currently_displayed_par_pnl: ttk.Frame | None = None
        self._behavior_notebook = behavior_notebook
        tab_menu = self._behavior_notebook.get_tab_menu()
        self.tab = ttk.Frame(tab_menu)
        self._name = name
        self._parameter_panels = dict()

        self._behavior_type_var = tk.StringVar()
        behavior_type_menu = ttk.Combobox(self.tab, textvariable=self._behavior_type_var)
        behavior_type_menu['values'] = list(DEFAULT_BEHAVIORS.keys())
        behavior_type_menu.current(0)

        behavior_type_menu.state(["readonly"])
        behavior_type_menu.bind('<<ComboboxSelected>>', self.on_behavior_type_menu_change)

        chk = ttk.Checkbutton(self.tab, text='specify alpha0', onvalue='ON', offvalue='OFF')
        self.add_parameter_panel()
        self._remove_btn = ttk.Button(self.tab, text='Remove behavior',
                                      command=self._behavior_notebook.on_button_clicked)

        behavior_type_menu.grid(column=0, row=0)
        chk.grid(column=0, row=1)
        self._remove_btn.grid(column=0, row=3)

        # Add new tab to the notebook
        tab_menu.insert(len(tab_menu.tabs()) - 1, self.tab, text=name)
        tab_menu.select(self.tab)  # Focus the newly created tab

    def update_parameters(self, val, par_name):
        self._behavior_notebook.handler.update_behavior_parameters(self.get_name(), {par_name: val})

    def get_parameters(self):
        current_parameter_panel, sliders = self._parameter_panels[self.get_behavior_type()]
        return {k: v.get() for k, v in sliders.items()}

    def add_parameter_panel(self):
        behavior_type_name = self._behavior_type_var.get()
        behavior_parameters = DEFAULT_BEHAVIORS[behavior_type_name].get_parameters()
        pnl = ttk.Frame(self.tab)
        self._parameter_panels[behavior_type_name] = (pnl, {})
        j = 0
        for par_name, par_val in behavior_parameters.items():
            if isinstance(par_val, (int, float)):
                span = max(abs(float(par_val)), 1e-5)
                sp, slider = slider_panel(pnl,
                                          par_name, par_val, par_val-span/2, par_val+span/2,
                                          lambda val, par_name_=par_name: self.update_parameters(val, par_name_))
                self._parameter_panels[behavior_type_name][1][par_name] = slider
                sp.grid(row=j, column=0)
                j += 1
        pnl.grid(row=2, column=0)

        self._currently_displayed_par_pnl = pnl

    def on_behavior_type_menu_change(self, event):
        new_behavior_type = self._behavior_type_var.get()
        self._currently_displayed_par_pnl.grid_remove()
        if new_behavior_type in self._parameter_panels:
            print('that behavior panel already exists, lets make it visible')
            self._currently_displayed_par_pnl = self._parameter_panels[new_behavior_type][0]
            self._currently_displayed_par_pnl.grid()
        else:
            print('that behavior panel did not already existed, a fresh one is made')
            self.add_parameter_panel()
        self._behavior_notebook.handler.change_behavior_type(self._name, new_behavior_type)

    def get_remove_button(self):
        return self._remove_btn

    def get_name(self):
        return self._name

    def get_behavior_type(self):
        return self._behavior_type_var.get()
