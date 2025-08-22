from .interpreting import *
from ..simulation import static_solver
import os
import platform
import subprocess
import numpy as np
import csv
import shutil
import json
import sys
from collections.abc import Iterator

if sys.version_info >= (3, 11):
    from tomllib import load as load_toml_file
else:
    from tomli import load as load_toml_file


def print_model_file(model_path, print_title=True):
    title = ''
    if print_title:
        title = f'MODEL: {os.path.basename(model_path)}'
        print(''.join(['-']*(len(title) + 4)))
        print(f'| {title} |')
        print(''.join(['-']*(len(title) + 4)))
    with open(model_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(", ".join(row))
    if print_title:
        print(''.join(['-'] * (len(title) + 4)))



def read_model(model_path, parameters: dict[str, float | str] = None) -> Model:
    _parameters, _ = read_parameters_from_model_file(model_path)
    if parameters is not None:
        _parameters.update(parameters)
    evaluator = se.SimpleEval(names=_parameters)
    model_text = ''
    reading_model = False
    with open(model_path, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if len(row) == 0 or (len(row) == 1 and not row[0].strip('')) or row[0].lstrip().startswith('#'):
                continue
            if len(row) == 1 and row[0].strip() == 'NODES':
                reading_model = True
            if reading_model:
                model_text += ', '.join([row_item.strip() for row_item in row]) + '\n'
    return text_to_model(model_text.strip(), evaluator)


def write_model(_model: Model, save_dir, save_name='model'):
    model_txt = model_to_text(_model)
    lines = basic_split(model_txt, '\n')
    rows = []
    for line in lines:
        if line.lstrip().startswith('#') or not line.strip():
            continue
        rows.append(smart_split(line, ','))

    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def write_behavior(_behavior: MechanicalBehavior | None, filepath, fmt, specify_natural_measure, writing_mode='w'):
    with open(filepath, writing_mode, newline='') as file:
        writer = csv.writer(file)
        if _behavior is None:
            b_txt = 'NONE'
        else:
            b_txt = behavior_to_text(_behavior, fmt, True, specify_natural_measure)
        row = smart_split(b_txt, ',')
        writer.writerow(row)


def read_behavior(behavior_path: str, natural_measure=1.0) -> MechanicalBehavior:
    """ Reads behavior file.
    Only if an UnknownNaturalMeasure is triggered, the natural_measure argument will be used"""
    with open(behavior_path, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if len(row) == 0 or (len(row) == 1 and not row[0].strip('')) or row[0].lstrip().startswith('#'):
                continue
            try:
                return text_to_behavior(', '.join(row))
            except UnknownNaturalMeasure:
                return text_to_behavior(', '.join(row), natural_measure=natural_measure)
        raise InvalidBehaviorParameters("No text to read as behavior")


def stream_behaviors(behaviors_path: str, natural_measure=1.0) -> Iterator[MechanicalBehavior | None]:
    """ Read and yield behaviors (one by one) listed in csv file"""
    with open(behaviors_path, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if len(row) == 0 or (len(row) == 1 and not row[0].strip('')) or row[0].lstrip().startswith('#'):
                continue
            if row[0] == 'NONE':
                _behavior = None
            else:
                try:
                    try:
                        _behavior = text_to_behavior(', '.join(row))
                    except UnknownNaturalMeasure:
                        _behavior = text_to_behavior(', '.join(row), natural_measure=natural_measure)
                except InvalidBehaviorParameters:
                    _behavior = None
            yield _behavior


def read_parameters_from_model_file(model_path) -> tuple[dict[str, float | str], dict[str, dict]]:
    parameters_text = ''
    reading_parameters = False
    with open(model_path, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if len(row) == 0 or (len(row) == 1 and not row[0].strip('')) or row[0].lstrip().startswith('#'):
                continue
            if len(row) == 1 and row[0].strip() == 'PARAMETERS':
                reading_parameters = True
            elif len(row) == 1 and row[0].strip() == 'NODES':
                break
            if reading_parameters:
                parameters_text += ', '.join([row_item.strip() for row_item in row]) + '\n'
    parameters, design_parameter_data = text_to_parameters(parameters_text.strip())
    return parameters, design_parameter_data


def write_solver_parameters(solver_parameters, save_dir, save_name='solver_parameters.csv'):
    # Save the values of the solver parameters used for this simulation
    with open(os.path.join(save_dir, save_name), 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        for k, v in solver_parameters.items():
            writer.writerow([k, v])


def _read_settings_file(file_path):
    with open(file_path, "rb") as f:
        settings = load_toml_file(f)
    return settings


def read_solver_settings_file(file_path):
    return _read_settings_file(file_path)


def read_graphics_settings_file(file_path):
    settings = _read_settings_file(file_path)
    return (settings.get('general_options', {}),
            settings.get('plot_options', {}),
            settings.get('animation_options', {}),
            settings.get('assembly_appearance', {}))

def read_experimental_force_displacement_data(filepath,
                                              displacement_column_index=0, force_column_index=1, delimiter=','):
    """ read a force displacement data from a csv file. Force and dispalcement values should be in separate columns.
    Skips header automatically. """
    exp_displacements = []
    exp_forces = []
    with open(filepath, newline='') as file_object:
        reader = csv.reader(file_object, delimiter=delimiter)
        reading_data = False
        for row in reader:
            if not reading_data:
                if row:
                    try:
                        _ = float(row[displacement_column_index].replace(',', '.'))
                        _ = float(row[force_column_index].replace(',', '.'))
                    except ValueError:
                        continue
                    else:
                        reading_data = True
            if reading_data:
                exp_displacements.append(float(row[displacement_column_index].replace(',', '.')))
                exp_forces.append(float(row[force_column_index].replace(',', '.')))
    return np.array(exp_displacements), np.array(exp_forces)



def write_model_parameters(parameters, save_dir, save_name='parameters.csv'):
    # save the values of the parameters used for a simulation (if any)
    if parameters is not None:
        with open(os.path.join(save_dir, save_name), 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for k, v in parameters.items():
                writer.writerow([k, v])


def read_model_parameters(save_dir, save_name='parameters.csv'):
    path = os.path.join(save_dir, save_name)
    parameters = {}
    try:
        with open(path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    parameters[row[0]] = float(row[1])
                except ValueError:
                    parameters[row[0]] = row[1]
    except FileNotFoundError:
        return None
    return parameters


def write_design_parameters(design_parameters, save_dir, save_name='design_parameters.csv'):
    # save the values of the parameters used for a simulation (if any)
    if design_parameters is not None:
        with open(os.path.join(save_dir, save_name), 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for k, v in design_parameters.items():
                writer.writerow([k, v])


def read_design_parameters(save_dir, save_name='design_parameters.csv'):
    # Get the dictionary describing the design parameters used for the simulation
    parameters = {}
    with open(os.path.join(save_dir, save_name), mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                parameters[row[0]] = float(row[1])
            except ValueError:
                parameters[row[0]] = row[1]
    return parameters


def write_solving_process_info(info: dict | None, save_dir, save_name='solving_process_info.csv'):
    # save solving process information
    if info is not None:
        with open(os.path.join(save_dir, save_name), 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for k, v in info.items():
                writer.writerow([k, v])


def read_solving_process_info(save_dir, save_name='solving_process_info.csv'):
    # get the solving process information
    info = {}
    try:
        with open(os.path.join(save_dir, save_name), mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == 'duration (s)':
                    try:
                        info[row[0]] = float(row[1])
                    except ValueError:
                        info[row[0]] = None
                else:
                    try:
                        info[row[0]] = int(row[1])
                    except ValueError:
                        info[row[0]] = None
    except FileNotFoundError:
        pass
    return info


def write_results(result: static_solver.Result, save_dir: str):
    np.savetxt(os.path.join(save_dir, 'displacements.csv'),
               result.get_displacements(include_preloading=True, check_usability=False), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'forces.csv'),
               result.get_forces(include_preloading=True, check_usability=False), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'stability.csv'),
               result.get_stability(include_preloading=True, check_usability=False), delimiter=',', fmt="%s")
    np.savetxt(os.path.join(save_dir, 'eigval_stats.csv'),
               result.get_eigval_stats(include_preloading=True, check_usability=False), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'step_indices.csv'),
               result.get_step_indices(), delimiter=',', fmt='%d')
    write_solving_process_info(result.get_solving_process_info(), save_dir)
    write_model(result.get_model(), save_dir)


def read_results(save_dir):
    _model = read_model(os.path.join(save_dir, 'model.csv'))
    displacements = np.loadtxt(os.path.join(save_dir, 'displacements.csv'), delimiter=',')
    forces = np.loadtxt(os.path.join(save_dir, 'forces.csv'), delimiter=',')
    stability = np.loadtxt(os.path.join(save_dir, 'stability.csv'), delimiter=',', dtype=str)
    eigval_stats = np.loadtxt(os.path.join(save_dir, "eigval_stats.csv"), delimiter=',')
    step_indices = np.loadtxt(os.path.join(save_dir, "step_indices.csv"), delimiter=',', dtype=int)
    solving_process_info = read_solving_process_info(save_dir)
    return static_solver.Result(_model, displacements, forces, stability, eigval_stats, step_indices,
                                solving_process_info)


def write_scanning_general_info(scanning_general_info: dict, save_dir, save_name='general_info.json'):
    with open(os.path.join(save_dir, save_name), 'w') as fp:
        json.dump(scanning_general_info, fp)


def read_scanning_general_info(scan_results_dir, name='general_info.json'):
    with open(os.path.join(scan_results_dir, name), 'r') as fp:
        return json.load(fp)


def copy_model_file(save_dir, model_path):
    shutil.copy(model_path, save_dir)


def copy_solver_settings_file(save_dir, solver_settings_path):
    shutil.copy(solver_settings_path, save_dir)


def copy_graphics_settings_file(save_dir, graphics_settings_path):
    shutil.copy(graphics_settings_path, save_dir)


def mkdir(dir_path):
    original_dir_path = dir_path
    i = 1
    while True:
        try:
            os.makedirs(dir_path)
            break
        except FileExistsError:
            dir_path = original_dir_path + '-' + str(i)
        i += 1
    return dir_path


def open_file_with_default_os_app(file_path):
    current_os = platform.system()
    if current_os == "Windows":
        os.startfile(file_path)
    else:
        if current_os == "Linux":
            commands = ["xdg-open"]
        elif current_os.startswith("CYGWIN"):
            commands = ["cygstart"]
        elif current_os == "Darwin":
            commands = ["open"]
        else:
            raise OSError("Cannot open file, because unable to identify your operating system")
        commands.append(file_path)
        subprocess.run(commands)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter
    except ModuleNotFoundError:
        return False


def play_media_in_notebook_if_possible(file_path, format_type):
    try:
        from IPython.display import display, Video, Image
        if format_type == 'video':
            display(Video(file_path, width=800, html_attributes="muted loop autoplay"))
        elif format_type == 'image':
            display(Image(data=open(file_path, 'rb').read(), format='png'))
    except ModuleNotFoundError:
        pass
