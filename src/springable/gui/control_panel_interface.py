import tkinter as tk
import tkinter.ttk as ttk
from .gui_utils import slider_panel, Tooltip, get_recursion_limit, get_current_recursion_depth
from .gui_event_handler import GUIEventHandler
from .gui_settings import DEFAULT_BEHAVIORS, INITIALLY_SELECTED_BEHAVIOR_INDEX


class BehaviorNotebook:

    def __init__(self, parent, handler, window: tk.Tk):
        self.win = window
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

    def on_remove_button(self):
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

    def set_behavior_validity(self, tab_name, valid: bool):
        for tab, name in self._tabs.items():
            if name == tab_name:
                return tab.set_behavior_validity(valid)
        raise ValueError("Unknown name")
    

    def open_edit_behavior_win(self, name):
        popup = tk.Toplevel(self.win)
        popup.title("Edit behavior")
        popup.transient(self.win)
        popup.grab_set()
        popup.resizable(False, False)  # prevent resizing
        result = {'behavior_text': ''}
        
        initial_behavior_string_w_nm = self.handler.request_behavior_string_even_if_invalid(name, True)
        initial_behavior_string, natural_measure_str = initial_behavior_string_w_nm.split(',')
        open_idx = initial_behavior_string.find('(')
        close_idx = initial_behavior_string.rfind(')')
        behavior_type_str = initial_behavior_string[:open_idx]
        behavior_content_str = initial_behavior_string[open_idx+1:close_idx]

        lbl = ttk.Label(popup, text="Enter the behavior parameters"+ "\n/" + "-"*90 + "\\", anchor="center", justify="center")
        lbl.grid(row=0, column=1, columnspan=2)
        behavior_type_label_start = ttk.Label(popup, text=f"{behavior_type_str}(")
        behavior_type_label_end = ttk.Label(popup, text=")" + "," + natural_measure_str)
        text_var = tk.StringVar(value=behavior_content_str)
        entry = ttk.Entry(popup, textvariable=text_var, width=80)

        behavior_type_label_start.grid(row=1, column=0)
        entry.grid(row=1, column=1, columnspan=2)
        behavior_type_label_end.grid(row=1, column=3) 

        error_label = ttk.Label(popup, text="", foreground="red")
        error_text_var = tk.StringVar(popup, value="")
        Tooltip(error_label, error_text_var)
        error_label.grid(row=2, column=1, columnspan=2)

        ok_button = ttk.Button(popup, text="OK", state="disabled")
        popup.bind("<Return>", lambda event: ok_button.invoke())
        ok_button.grid(row=3, column=1)
        cancel_button = ttk.Button(popup, text="Cancel", command=popup.destroy)
        cancel_button.grid(row=3, column=2)


        def validate(*_):
            behavior_str = behavior_type_str + '(' + entry.get() + ')'
            error = self.handler.is_behavior_string_invalid(behavior_str, float(natural_measure_str))
            error_text_var.set(error)

            if error:
                error_msg = (error if error in ('Invalid syntax', 'Ill-defined_behavior')
                             else 'Invalid parameters')
                error_label.config(text=error_msg)
                ok_button.config(state="disabled")
                return False
            else:
                error_label.config(text="")
                ok_button.config(state="normal")
                return True

        def submit():
            result['behavior_text'] = behavior_type_str + '(' + entry.get() + '), ' + natural_measure_str
            popup.destroy()

        ok_button.config(command=submit)
        entry.bind("<KeyRelease>", validate)
        entry.focus()
        validate()


        popup.update_idletasks()
        width = max(400, popup.winfo_width())
        height = popup.winfo_height()
        parent_x = self.win.winfo_rootx()
        parent_y = self.win.winfo_rooty()
        parent_w = self.win.winfo_width()
        parent_h = self.win.winfo_height()
        x = parent_x + (parent_w // 2) - (width // 2)
        y = parent_y + (parent_h // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        popup.wait_window()
        return result['behavior_text']


class BehaviorTab:

    def __init__(self, behavior_notebook: BehaviorNotebook, name: str):
        self._currently_displayed_alpha_par_pnl: ttk.Frame | None = None
        self._bn = behavior_notebook
        self._handler = behavior_notebook.handler
        tab_menu = behavior_notebook.get_tab_menu()
        self.tab = ttk.Frame(tab_menu)
        self._name = name
        self.alpha_and_parameter_panels: dict[str, tuple[ttk.Frame, dict[str, ttk.Scale], ttk.Scale]] = dict()

        self._behavior_type_var = tk.StringVar()
        behavior_type_menu = ttk.Combobox(self.tab, textvariable=self._behavior_type_var)
        behavior_type_menu['values'] = list(DEFAULT_BEHAVIORS.keys())
        behavior_type_menu.current(INITIALLY_SELECTED_BEHAVIOR_INDEX)

        behavior_type_menu.state(["readonly"])
        behavior_type_menu.bind('<<ComboboxSelected>>', self.on_behavior_type_menu_change)

        self._behavior_text_var = tk.StringVar(value="...")
        self.behavior_text_entry = ttk.Entry(self.tab, textvariable=self._behavior_text_var,
                                             width=60, foreground="green")

        self.behavior_text_entry.bind("<Button-1>", lambda *args: self._start_editing_behavior())
           
        
        # Create a right-click (context) menu
        # context_menu = tk.Menu(self._bn.win, tearoff=0)
        # context_menu.add_command(label="Edit", command=self._on_edit_button_clicked)
        # def show_menu(event):
        #     context_menu.tk_popup(event.x_root, event.y_root)
        # self.behavior_text_entry.bind("<Button-3>", show_menu)
        
        
        self.behavior_text_entry.state(['readonly'])
        self.behavior_text_tooltip = Tooltip(self.behavior_text_entry, self._behavior_text_var)

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

        general_btn_frame = ttk.Frame(self.tab)

        self._remove_btn = ttk.Button(general_btn_frame, text='Remove',
                                      command=behavior_notebook.on_remove_button)
        self._load_btn = ttk.Button(general_btn_frame, text='Load from file...',
                                    command=self._on_load_from_file_button)
        self._remove_btn.grid(column=0, row=1, sticky='NW')
        self._load_btn.grid(column=1, row=1, sticky='NW')

        save_pnl = ttk.Frame(self.tab)
        self._copy_btn = ttk.Button(save_pnl, text='Copy',
                                    command=self._on_copy_button_clicked)
        self._save_btn = ttk.Button(save_pnl, text='Save...',
                                    command=self._on_save_button_clicked)
        self._edit_btn = ttk.Button(save_pnl, text='Edit...',
                                    command=self._start_editing_behavior)

        self._add_png_var = tk.BooleanVar(value=False)
        self._add_png_btn = ttk.Checkbutton(save_pnl, text='+ PNG', variable=self._add_png_var, onvalue=True, offvalue=False)
        self._add_pdf_var = tk.BooleanVar(value=False)
        self._add_pdf_btn = ttk.Checkbutton(save_pnl, text='+ PDF', variable=self._add_pdf_var, onvalue=True, offvalue=False)
        self._specify_natural_measure_var = tk.BooleanVar(value=False)
        self._specify_natural_measure_btn = ttk.Checkbutton(save_pnl,
                                                            text='specify alpha 0',
                                                            variable=self._specify_natural_measure_var,
                                                            onvalue=True, offvalue=False,
                                                            command=self.on_specify_natural_measure_clicked)
        general_btn_frame.grid(column=0, row=0, sticky='NW')
        self._copy_btn.grid(column=0, row=0, sticky='W')
        self._save_btn.grid(column=1, row=0, sticky='W')
        self._specify_natural_measure_btn.grid(column=2, row=0, sticky='W')
        self._add_pdf_btn.grid(column=3, row=0, sticky='W')
        self._add_png_btn.grid(column=4, row=0, sticky='W')
        

        behavior_type_menu.grid(column=0, row=1, sticky='NW')
        self._remove_btn.grid(column=0, row=1, sticky='NW')

        alpha_and_parameters_panel.grid(column=0, row=2)
        self.behavior_text_entry.grid(column=0, row=3)
        save_pnl.grid(column=0, row=4, sticky='W', pady=5)

        # Add new tab to the notebook
        tab_menu.insert(len(tab_menu.tabs()) - 1, self.tab, text=name)
        tab_menu.select(self.tab)  # Focus the newly created tab

    def _update_parameter(self, parameter_name):
        # send event to notify handler that a parameter has been changed
        relative_recursion_depth = get_current_recursion_depth() / get_recursion_limit()
        if relative_recursion_depth > .8:
            cmd = self.disconnect_parameter_slider(parameter_name)
            self._handler.update_behavior_parameter(self._name, parameter_name)
            self._bn.win.after_idle(self.reconnect_parameter_slider, parameter_name, cmd)
        else:
            self._handler.update_behavior_parameter(self._name, parameter_name)

    def disconnect_parameter_slider(self, par_name):
        _, parameter_sliders, _ = self.alpha_and_parameter_panels[self.get_behavior_type()]
        cmd = parameter_sliders[par_name].cget("command")
        parameter_sliders[par_name].config(command="")
        return cmd

    def reconnect_parameter_slider(self, par_name, cmd):
        _, parameter_sliders, _ = self.alpha_and_parameter_panels[self.get_behavior_type()]
        parameter_sliders[par_name].config(command=cmd)

    def disconnect_alpha0_slider(self):
        _, _, alpha0_slider = self.alpha_and_parameter_panels[self.get_behavior_type()]
        cmd = alpha0_slider.cget("command")
        alpha0_slider.config(command="")
        return cmd

    def reconnect_alpha0_slider(self, cmd):
        _, _, alpha0_slider = self.alpha_and_parameter_panels[self.get_behavior_type()]
        alpha0_slider.config(command=cmd)

    def _update_natural_measure(self):
        # send event to handler
        relative_recursion_depth = get_current_recursion_depth() / get_recursion_limit()
        if relative_recursion_depth > .8:
            cmd = self.disconnect_alpha0_slider()
            self._handler.update_behavior_natural_measure(self._name)
            self._bn.win.after_idle(self.reconnect_alpha0_slider, cmd)
        else:
            self._handler.update_behavior_natural_measure(self._name)

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

    def _add_parameter_sliders_to_panel(self, pnl: ttk.Frame, start_row=0,
                                        behavior_parameters: dict | None = None) -> dict[str, ttk.Scale]:
        if behavior_parameters is None:
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
            self._currently_displayed_alpha_par_pnl = self.alpha_and_parameter_panels[new_behavior_type][0]
            self._currently_displayed_alpha_par_pnl.grid()
        else:
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

    def on_specify_natural_measure_clicked(self):
        self._handler.update_behavior_text(self._name)

    def get_specify_natural_measure_state(self) -> bool:
        return self._specify_natural_measure_var.get()

    def _on_save_button_clicked(self):
        self._handler.save_behavior(self._name, add_png=self._add_png_var.get(), add_pdf=self._add_pdf_var.get())

    def _on_load_from_file_button(self):
        # SEND EVENT to handler
        success = self._handler.load_from_file(self._name)
        if success:
            # UPDATE WIDGETS based on updated information fetched from handler
            # fetch updated info from handler
            behavior_type_name = self._handler.get_behavior_type_name(self._name)
            parameters = self._handler.get_behavior_parameters(self._name)
            natural_measure = self._handler.get_behavior_natural_measure(self._name)

            # set behavior type value in the dropdown menu
            self._behavior_type_var.set(behavior_type_name)

            # remove current parameter-slider panel
            self._currently_displayed_alpha_par_pnl.grid_remove()

            # make new parameter-alpha0-slider frame
            alpha_and_parameters_panel = ttk.Frame(self.tab)
            start_row = 0
            parameter_sliders = self._add_parameter_sliders_to_panel(alpha_and_parameters_panel, start_row, parameters)
            start_row = len(parameter_sliders) * 2
            alpha0 = natural_measure
            alpha0_slider = slider_panel(alpha_and_parameters_panel, 'alpha0', alpha0,
                                         alpha0 / 2,
                                         max(alpha0 * 3 / 2, 1.0),
                                         self._update_natural_measure, row=start_row)
            self.alpha_and_parameter_panels[behavior_type_name] = (alpha_and_parameters_panel,
                                                                   parameter_sliders, alpha0_slider)
            alpha_and_parameters_panel.grid(column=0, row=2)
            self._currently_displayed_alpha_par_pnl = alpha_and_parameters_panel
            self._handler.update_behavior_text(self._name)
            self._handler.show_popup('Loaded successfully!', 500)

    def _on_copy_button_clicked(self):
        self._handler.copy_behavior_to_clipboard(self._name)

    
    def set_behavior_text(self, behavior_text: str):
        return self._behavior_text_var.set(behavior_text)

    def set_behavior_validity(self, valid):
        if valid:
            self.behavior_text_entry.config(foreground='green')
            self._save_btn.state(['!disabled'])
            self._copy_btn.state(['!disabled'])
        else:
            self.behavior_text_entry.config(foreground='red')
            self._save_btn.state(['disabled'])
            self._copy_btn.state(['disabled'])

    def _start_editing_behavior(self):
        success = self._handler.update_behavior_from_text(self._name)
        if success:
            # UPDATE WIDGETS based on updated information fetched from handler
            # fetch updated info from handler
            behavior_type_name = self._handler.get_behavior_type_name(self._name)
            parameters = self._handler.get_behavior_parameters(self._name)
            natural_measure = self._handler.get_behavior_natural_measure(self._name)

            # set behavior type value in the dropdown menu
            self._behavior_type_var.set(behavior_type_name)

            # remove current parameter-slider panel
            self._currently_displayed_alpha_par_pnl.grid_remove()

            # make new parameter-alpha0-slider frame
            alpha_and_parameters_panel = ttk.Frame(self.tab)
            start_row = 0
            parameter_sliders = self._add_parameter_sliders_to_panel(alpha_and_parameters_panel, start_row, parameters)
            start_row = len(parameter_sliders) * 2
            alpha0 = natural_measure
            alpha0_slider = slider_panel(alpha_and_parameters_panel, 'alpha0', alpha0,
                                         alpha0 / 2,
                                         max(alpha0 * 3 / 2, 1.0),
                                         self._update_natural_measure, row=start_row)
            self.alpha_and_parameter_panels[behavior_type_name] = (alpha_and_parameters_panel,
                                                                   parameter_sliders, alpha0_slider)
            alpha_and_parameters_panel.grid(column=0, row=2)
            self._currently_displayed_alpha_par_pnl = alpha_and_parameters_panel
            self._handler.update_behavior_text(self._name)
            self._handler.show_popup('Edited successfully!', 500)
        
