import tkinter as tk
import tkinter.ttk as ttk
from .gui_utils import slider_panel
from .gui_event_handler import GUIEventHandler
from .default_behaviors import DEFAULT_BEHAVIORS


class BehaviorNotebook:

    def __init__(self, parent, handler):
        self.handler: GUIEventHandler = handler
        tab_menu = ttk.Notebook(parent)
        tab_plus = ttk.Frame(tab_menu)
        tab_menu.add(tab_plus, text='+')
        self._tab_menu = tab_menu
        self._tab_plus = tab_plus
        self._tabs: dict[BehaviorTab, str] = dict()
        self._tab_menu.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self._tab_menu.grid(column=0, row=0)

    def on_tab_selected(self, event):
        selected_tab_id = self._tab_menu.select()
        if self._tab_menu.tab(selected_tab_id)["text"] == "+":
            self.add_behavior_tab()
        else:
            selected_tab = self._tab_menu.nametowidget(selected_tab_id)
            for tab in self._tabs.keys():
                if tab.tab is selected_tab:
                    self.handler.switch_focus(self._tabs[tab])
                    break

    def on_remove_button_clicked(self):
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
        self.handler.add_behavior(tab_name)

        # update widgets based on info handler has updated
        displayed_text = self.handler.get_behavior_text(tab_name, new_behavior_tab.get_specify_natural_measure_state())
        new_behavior_tab.set_behavior_text(displayed_text)

    def remove_selected_behavior_tab(self):
        selected_tab_id = self._tab_menu.select()
        selected_tab = self._tab_menu.nametowidget(selected_tab_id)

        # remove the selected tab for the dictionary of tabs (and gets its name)
        tab_name = None
        for tab in self._tabs.keys():
            if tab.tab is selected_tab:
                tab_name = self._tabs.pop(tab)
                break


        # send event to the handler before any start at removing the tab,
        # which might start the creation of a new tab and lead to errors
        self.handler.remove_behavior(tab_name)


        # destroy the children widget of the selected tab
        for widget in selected_tab.winfo_children():
            widget.destroy()

        # sets the focus/selection to another tab when the last tab was selected to be removed
        selected_tab_index = self._tab_menu.index(selected_tab_id)
        if selected_tab_index == len(self._tab_menu.tabs()) - 2:
            self._tab_menu.select(self._tab_menu.tabs()[len(self._tab_menu.tabs()) - 3])

        # remove the tab from the notebook/tab menu
        self._tab_menu.forget(selected_tab_id)



    def get_tab_menu(self):
        return self._tab_menu

    def get_behavior_type(self, tab_name: str) -> str:
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.get_behavior_type()
        raise ValueError("Unknown name")

    def get_natural_measure(self, tab_name: str) -> float:
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.get_natural_measure()
        raise ValueError("Unknown name")

    def get_behavior_parameters(self, tab_name: str) -> dict:
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.get_parameters()
        raise ValueError("Unknown name")

    def get_behavior_parameter(self, tab_name: str, par_name: str) -> float:
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.get_parameter(par_name)
        raise ValueError("Unknown name")

    def get_specify_natural_measure_state(self, tab_name) -> bool:
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.get_specify_natural_measure_state()
        raise ValueError("Unknown name")

    def set_behavior_text(self, tab_name: str, text: str):
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.set_behavior_text(text)
        raise ValueError("Unknown name")



class BehaviorTab:

    def __init__(self, behavior_notebook: BehaviorNotebook, name):
        self._currently_displayed_alpha_par_pnl: ttk.Frame | None = None
        self._handler = behavior_notebook.handler
        tab_menu = behavior_notebook.get_tab_menu()
        self.tab = ttk.Frame(tab_menu)
        self._name = name
        self.alpha_and_parameter_panels: dict[str, tuple[ttk.Frame, dict[str, ttk.Scale], ttk.Scale]] = dict()

        self._behavior_type_var = tk.StringVar()
        behavior_type_menu = ttk.Combobox(self.tab, textvariable=self._behavior_type_var)
        behavior_type_menu['values'] = list(DEFAULT_BEHAVIORS.keys())
        behavior_type_menu.current(5)

        behavior_type_menu.state(["readonly"])
        behavior_type_menu.bind('<<ComboboxSelected>>', self.on_behavior_type_menu_change)

        self._behavior_text_var = tk.StringVar(value="...")
        behavior_text_entry = ttk.Entry(self.tab, textvariable=self._behavior_text_var, width=60)
        behavior_text_entry.state(['readonly'])

        alpha_and_parameters_panel = ttk.Frame(self.tab)
        start_row = 0
        parameter_sliders = self._add_parameter_sliders_to_panel(alpha_and_parameters_panel, start_row)
        start_row = len(parameter_sliders) * 2
        alpha0 = DEFAULT_BEHAVIORS[self._behavior_type_var.get()].copy().get_natural_measure()
        alpha0_slider = slider_panel(alpha_and_parameters_panel, 'alpha0', alpha0,
                                     min(alpha0 / 2, alpha0 * 3 / 2),
                                     max(alpha0 * 3 / 2, alpha0 / 2),
                                     self._update_natural_measure, row=start_row)
        self.alpha_and_parameter_panels[self._behavior_type_var.get()] = (alpha_and_parameters_panel,
                                                                          parameter_sliders, alpha0_slider)
        self._currently_displayed_alpha_par_pnl = alpha_and_parameters_panel

        self._remove_btn = ttk.Button(self.tab, text='Remove behavior',
                                      command=behavior_notebook.on_remove_button_clicked)

        save_pnl = ttk.Frame(self.tab)
        self._save_btn = ttk.Button(save_pnl, text='Save behavior',
                                    command=self._on_save_button_clicked)

        self._specify_natural_measure_var = tk.BooleanVar(value=True)
        self._specify_natural_measure_btn = ttk.Checkbutton(save_pnl,
                                                            text='specify alpha 0',
                                                            variable=self._specify_natural_measure_var,
                                                            onvalue=True, offvalue=False,
                                                            command=self.on_specify_natural_measure_clicked)
        self._save_btn.grid(column=0, row=0, sticky='W')
        self._specify_natural_measure_btn.grid(column=1, row=0, sticky='W')

        behavior_type_menu.grid(column=0, row=0, sticky='NW')
        self._remove_btn.grid(column=0, row=1, sticky='NW')
        alpha_and_parameters_panel.grid(column=0, row=2)
        behavior_text_entry.grid(column=0, row=3)
        save_pnl.grid(column=0, row=4, sticky='W', pady=5)

        # Add new tab to the notebook
        tab_menu.insert(len(tab_menu.tabs()) - 1, self.tab, text=name)
        tab_menu.select(self.tab)  # Focus the newly created tab

    def _update_parameter(self, parameter_name):
        # send event to notify handler that a parameter has been changed
        self._handler.update_behavior_parameter(self._name, parameter_name)

        # update widgets based on info updated by handler
        displayed_text = self._handler.get_behavior_text(self._name, self._specify_natural_measure_var.get())
        self.set_behavior_text(displayed_text)

    def _update_natural_measure(self):
        # send event to handler
        self._handler.update_behavior_natural_measure(self._name)

        # update widgets based on info updated by handler
        displayed_text = self._handler.get_behavior_text(self._name, self._specify_natural_measure_var.get())
        self.set_behavior_text(displayed_text)

    def get_parameters(self) -> dict:
        _, parameter_sliders, _ = self.alpha_and_parameter_panels[self.get_behavior_type()]
        return {k: v.get() for k, v in parameter_sliders.items()}

    def get_parameter(self, par_name: str) -> float:
        _, parameter_sliders, _ = self.alpha_and_parameter_panels[self.get_behavior_type()]
        return parameter_sliders[par_name].get()

    def get_natural_measure(self) -> float | None:
        _, _, alpha0_slider = self.alpha_and_parameter_panels[self.get_behavior_type()]
        return alpha0_slider.get()

    def get_behavior_type(self) -> str:
        return self._behavior_type_var.get()

    def _add_parameter_sliders_to_panel(self, pnl: ttk.Frame, start_row=0) -> dict[str, ttk.Scale]:
        behavior_type_name = self._behavior_type_var.get()
        behavior_parameters = DEFAULT_BEHAVIORS[behavior_type_name].copy().get_parameters()
        j = start_row
        sliders: dict[str, ttk.Scale] = {}
        for par_name, par_val in behavior_parameters.items():
            if isinstance(par_val, (int, float)):
                span = max(abs(float(par_val)), 1e-5)
                slider = slider_panel(pnl,
                                      par_name, par_val, par_val - span / 2, par_val + span / 2,
                                      lambda par_name_=par_name: self._update_parameter(par_name_), row=j)
                sliders[par_name] = slider
                j += 2
        return sliders

    def on_behavior_type_menu_change(self, event):
        new_behavior_type = self._behavior_type_var.get()
        self._currently_displayed_alpha_par_pnl.grid_remove()
        if new_behavior_type in self.alpha_and_parameter_panels:
            print('that behavior panel already exists, lets make it visible')
            self._currently_displayed_alpha_par_pnl = self.alpha_and_parameter_panels[new_behavior_type][0]
            self._currently_displayed_alpha_par_pnl.grid()
        else:
            print('that behavior panel did not already existed, a fresh one is made')
            alpha_and_parameters_panel = ttk.Frame(self.tab)
            start_row = 0
            parameter_sliders = self._add_parameter_sliders_to_panel(alpha_and_parameters_panel, start_row)
            start_row = len(parameter_sliders) * 2
            alpha0 = DEFAULT_BEHAVIORS[self._behavior_type_var.get()].copy().get_natural_measure()
            alpha0_slider = slider_panel(alpha_and_parameters_panel, 'alpha0', alpha0,
                                         alpha0 / 2,
                                         max(alpha0 * 3 / 2, 1.0),
                                         self._update_natural_measure, row=start_row)
            self.alpha_and_parameter_panels[self._behavior_type_var.get()] = (alpha_and_parameters_panel,
                                                                              parameter_sliders, alpha0_slider)
            alpha_and_parameters_panel.grid(column=0, row=2)
            self._currently_displayed_alpha_par_pnl = alpha_and_parameters_panel

        # send event to handler
        self._handler.change_behavior_type(self._name)

        # update widgets based on updated info updated by the handler
        displayed_text = self._handler.get_behavior_text(self._name, self._specify_natural_measure_var.get())
        self.set_behavior_text(displayed_text)

    def on_specify_natural_measure_clicked(self):
        text = self._handler.get_behavior_text(self._name, self._specify_natural_measure_var.get())
        self.set_behavior_text(text)

    def get_specify_natural_measure_state(self) -> bool:
        return self._specify_natural_measure_var.get()

    def _on_save_button_clicked(self):
        x = self
        print('save button')

    def set_behavior_text(self, behavior_text: str):
        return self._behavior_text_var.set(behavior_text)
