from .static_solver import Result
from .stability_states import StabilityStates
import numpy as np


def extract_branches(result: Result) -> dict[str, list[list[int]]]:
    stability: np.ndarray = result.get_stability()
    branch_dict = {StabilityStates.STABLE: [],
                   StabilityStates.STABILIZABLE: [],
                   StabilityStates.UNSTABLE: []}
    current_branch = [0]
    for i in range(1, stability.shape[0]):
        if stability[i] == stability[i - 1]:
            current_branch.append(i)
        else:
            branch_dict[stability[i - 1]].append(current_branch)
            current_branch = [i]
    branch_dict[stability[-1]].append(current_branch)
    return branch_dict


def extract_loading_path(result: Result, drive_mode: str, starting_index: int = 0):
    u_load, f_load = result.get_equilibrium_path()
    stability = result.get_stability()

    if starting_index < 0:
        starting_index = u_load.shape[0] + starting_index

    match drive_mode:
        case 'force':
            path_indices = []
            current_load = f_load[starting_index]
            for index in range(starting_index, f_load.shape[0]):
                load = f_load[index]
                stable = stability[index] == StabilityStates.STABLE
                if load >= current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case 'displacement':
            path_indices = []
            current_load = u_load[starting_index]
            for index in range(starting_index, u_load.shape[0]):
                load = u_load[index]
                stable = stability[index] != StabilityStates.UNSTABLE
                if load >= current_load and stable:
                    current_load = u_load[index]
                    path_indices.append(index)
        case _:
            raise ValueError(f'invalid drive mode {drive_mode}')
    if not path_indices:
        raise LoadingPathEmpty
    is_restabilization = np.diff(path_indices, prepend=[0]) > 1
    restabilization_indices = np.array(path_indices)[is_restabilization]
    tmp = [is_restabilization[index + 1] == True for index in range(len(is_restabilization) - 1)]
    tmp.append(False)
    critical_indices = np.array(path_indices)[tmp]
    return path_indices, critical_indices.tolist(), restabilization_indices.tolist()


def extract_unloading_path(result: Result, drive_mode: str, starting_index: int = -1):
    u_load, f_load = result.get_equilibrium_path()
    stability = result.get_stability()
    if starting_index < 0:
        starting_index = u_load.shape[0] + starting_index

    match drive_mode:
        case 'force':
            path_indices = []
            current_load = f_load[starting_index]
            for index in range(starting_index, -1, -1):
                load = f_load[index]
                stable = stability[index] == StabilityStates.STABLE
                if load <= current_load and stable:
                    current_load = f_load[index]
                    path_indices.append(index)
        case 'displacement':
            path_indices = []
            current_load = u_load[starting_index]
            for index in range(starting_index, -1, -1):
                load = u_load[index]
                stable = stability[index] != StabilityStates.UNSTABLE
                if load <= current_load and stable:
                    current_load = u_load[index]
                    path_indices.append(index)
        case _:
            raise ValueError(f'invalid drive mode {drive_mode}')
    if not path_indices:
        raise LoadingPathEmpty
    is_restabilization = np.diff(path_indices, prepend=path_indices[0]) < -1
    restabilization_indices = np.array(path_indices)[is_restabilization]
    tmp = [is_restabilization[index + 1] for index in range(len(is_restabilization) - 1)]
    tmp.append(False)
    critical_indices = np.array(path_indices)[tmp]
    return path_indices, critical_indices.tolist(), restabilization_indices.tolist()


class LoadingPathEmpty(Exception):
    """ raise this when the loading (or unloading) path is empty"""
