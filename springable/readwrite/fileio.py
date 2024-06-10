from .interpreting import *
from ..simulation import static_solver
import os
import numpy as np
import csv
import tomllib
import shutil


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
    rows = [smart_split(line, ',') for line in lines]
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


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
    return text_to_parameters(parameters_text.strip())


def write_solver_parameters(solver_parameters, save_dir, save_name='solver_parameters.csv'):
    # Save the values of the solver parameters used for this simulation
    with open(os.path.join(save_dir, save_name), 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        for k, v in solver_parameters.items():
            writer.writerow([k, v])


def _read_settings_file(file_path):
    with open(file_path, "rb") as f:
        settings = tomllib.load(f)
    return settings


def read_solver_settings_file(file_path):
    return _read_settings_file(file_path)


def read_graphics_settings_file(file_path):
    settings = _read_settings_file(file_path)
    return (settings.get('general_options', {}),
            settings.get('plot_options', {}),
            settings.get('animation_options', {}),
            settings.get('assembly_appearance', {}))


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


def write_results(result: static_solver.Result, save_dir: str):
    np.savetxt(os.path.join(save_dir, 'displacements.csv'), result.get_displacements(include_preloading=True),
               delimiter=',')
    np.savetxt(os.path.join(save_dir, 'forces.csv'), result.get_forces(include_preloading=True), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'stability.csv'), result.get_stability(include_preloading=True), delimiter=',',
               fmt="%s")
    np.savetxt(os.path.join(save_dir, 'eigval_stats.csv'), result.get_eigval_stats(include_preloading=True),
               delimiter=',')
    np.savetxt(os.path.join(save_dir, 'step_indices.csv'), result.get_step_indices(), delimiter=',', fmt='%d')
    write_model(result.get_model(), save_dir)


def read_results(save_dir):
    _model = read_model(os.path.join(save_dir, 'model.csv'))
    displacements = np.loadtxt(os.path.join(save_dir, 'displacements.csv'), delimiter=',')
    forces = np.loadtxt(os.path.join(save_dir, 'forces.csv'), delimiter=',')
    stability = np.loadtxt(os.path.join(save_dir, 'stability.csv'), delimiter=',', dtype=str)
    eigval_stats = np.loadtxt(os.path.join(save_dir, "eigval_stats.csv"), delimiter=',')
    step_indices = np.loadtxt(os.path.join(save_dir, "step_indices.csv"), delimiter=',', dtype=int)
    return static_solver.Result(_model, displacements, forces, stability, eigval_stats, step_indices)


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
