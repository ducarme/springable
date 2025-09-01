from .static_solver import Result
from .stability_states import StabilityStates
from scipy.interpolate import interp1d
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

def extract_branches_in_order(result: Result, drive_mode: str):
    """ get a list of branches that are stable under the specific drive mode """

    if drive_mode not in ("force", "displacement"):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')
    
    stable_status = (StabilityStates.STABLE, ) if drive_mode == 'force' else (StabilityStates.STABLE, StabilityStates.STABILIZABLE)
    stability = result.get_stability()
    branches = []
    in_stable_branch = False
    for i in range(stability.shape[0]):
        if not in_stable_branch:
            if stability[i] in stable_status:
                start = i
                in_stable_branch = True
        else:
            if stability[i] not in stable_status:
                branches.append((start,  i - 1))
                in_stable_branch = False
    if in_stable_branch:
        branches.append((start, stability.shape[0] - 1))
    return branches

def extract_transition_graph(result: Result, drive_mode: str):
    if drive_mode not in ('force', 'displacement'):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')
    
    u_load, f_load = result.get_equilibrium_path()
    branches = extract_branches_in_order(result, drive_mode)

    load = f_load if drive_mode == 'force' else u_load

    # check whether the load increases monotonically on each branch,
    # otherwise it is a sign that the solution path doubled back or connects
    # equilibria that should not be directly connected
    for branch in branches:
        start, end = branch
        if (np.diff(load[start:end+1]) < 0.0).any():
            raise DiscontinuityInTheSolutionPath

    transition_graph = []    
    for i, branch_i in enumerate(branches):
        start_i, end_i = branch_i
        critical_plus = load[end_i] if i != len(branches) - 1 else None
        critical_minus = load[start_i] if i != 0 else None
        transition_graph.append([None, None])
        if critical_plus is not None:
            for j, branch_j in enumerate(branches):
                if j <= i: continue
                start, end = branch_j
                # look for plus-transition
                if load[start] <= critical_plus <= load[end]:
                    transition_graph[-1][1] = j
                    break
        if critical_minus is not None:
            for jj, branch_j in enumerate(branches[::-1]):
                j = len(branches) - 1 - jj
                if j >= i: continue
                start, end = branch_j
                # look for plus-transition
                if load[start] <= critical_minus <= load[end]:
                    transition_graph[-1][0] = j
                    break
    return transition_graph

def extract_loading_path(result: Result, drive_mode: str, starting_index: int = 0):
    if drive_mode not in ('force', 'displacement'):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')
    
    u_load, f_load = result.get_equilibrium_path()
    branches = extract_branches_in_order(result, drive_mode)

    load = f_load if drive_mode == 'force' else u_load

    # check whether the load increases monotonically on each branch,
    # otherwise it is a sign that the solution path doubled back or connects
    # equilibria that should not be directly connected
    for branch in branches:
        start, end = branch
        if (np.diff(load[start:end+1]) < 0.0).any():
            raise DiscontinuityInTheSolutionPath


    if starting_index < 0:
            starting_index = load.shape[0] + starting_index

    # find starting branch
    for branch_ix, branch in enumerate(branches):
        start, end = branch
        if starting_index < start:
            current_index = start
            current_branch_index = branch_ix
            break
        if start <= starting_index <= end:
            current_index = starting_index
            current_branch_index = branch_ix
            break
    else:  # the starting index is beyond the last stable point
        raise LoadingPathEmpty

    path_indices = []
    critical_indices = []
    restabilization_indices = []
    while True:
        # look for branch
        for i in range(current_branch_index, len(branches)):
            start, end = branches[i]
            if load[start] <= load[current_index] <= load[end]:  # should always be true for the first iteration of the while loop 
                landing_index = int(interp1d(load[start:end + 1], 
                                             list(range(start, end + 1)),
                                             kind='next')(load[current_index]))
                path_indices.extend(list(range(landing_index, end + 1)))
                current_index = end
                current_branch_index = i + 1
                restabilization_indices.append(landing_index)
                critical_indices.append(end)
                break
        else:  # could not find a branch
            break

    if not path_indices:
        raise LoadingPathEmpty
    
    restabilization_indices = restabilization_indices[1:]
    critical_indices = critical_indices[:-1]
    return path_indices, critical_indices, restabilization_indices


def extract_unloading_path(result: Result, drive_mode: str, starting_index: int = -1):
    if drive_mode not in ("force", "displacement"):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')

    u_load, f_load = result.get_equilibrium_path()
    branches = extract_branches_in_order(result, drive_mode)

    load = f_load if drive_mode == "force" else u_load

    # check whether the load increases monotonically on each branch,
    # otherwise it is a sign that the solution path doubled back or connects
    # equilibria that should not be directly connected
    for branch in branches:
        start, end = branch
        if (np.diff(load[start : end + 1]) < 0.0).any():
            raise DiscontinuityInTheSolutionPath

    if starting_index < 0:
        starting_index = load.shape[0] + starting_index

    # find starting branch
    for branch_ix, branch in enumerate(branches[::-1]):
        start, end = branch
        if starting_index > end:
            current_index = end
            current_branch_index = len(branches) - 1 - branch_ix
            break
        if start <= starting_index <= end:
            current_index = starting_index
            current_branch_index = len(branches) - 1 - branch_ix
            break
    else:  # the starting index is beyond the last stable point
        raise LoadingPathEmpty
    

    path_indices = []
    critical_indices = []
    restabilization_indices = []
    while True:
        # look for branch
        for i in range(current_branch_index, -1, -1):
            start, end = branches[i]
            # when one looks for the first branch from the starting index,
            # one should ensure that we don't land on a branch post starting index
            if current_index < start:
                continue
            if load[start] <= load[current_index] <= load[end]:
                landing_index = int(interp1d(load[start : end + 1],
                                             list(range(start, end + 1)), kind="previous")(load[current_index]))
                
                path_indices.extend(list(range(landing_index, start - 1, -1)))
                current_index = start
                current_branch_index = i - 1
                restabilization_indices.append(landing_index)
                critical_indices.append(start)
                break
        else:  # could not find a branch
            break

    if not path_indices:
        raise LoadingPathEmpty
    restabilization_indices = restabilization_indices[1:]
    critical_indices = critical_indices[:-1]
    return path_indices, critical_indices, restabilization_indices


class LoadingPathEmpty(Exception):
    """ raise this when the loading (or unloading) path is empty"""

class DiscontinuityInTheSolutionPath(Exception):
    """raise this when you detect discontinuities in the solutoin path """