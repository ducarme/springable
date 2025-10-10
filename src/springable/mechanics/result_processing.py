from .static_solver import Result
from .stability_states import StabilityStates
from scipy.interpolate import interp1d
import numpy as np


def determine_hysteron_branch_id_from_stable_branch(res: Result, stable_branch):
    start, _ = stable_branch
    hysteron_branch_id = ''.join(res.compute_elemental_hysteron_ids_at_state_index(start))
    return hysteron_branch_id

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

def extract_stable_branches_in_order(result: Result, drive_mode: str):
    """ get a list of branches that are stable under the specific drive mode """
    if drive_mode not in ("force", "displacement"):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')
    if drive_mode == 'force' and not result.solution_describes_a_force_driven_loading():
        raise LoadingPathIsNotDescribedBySolution
    elif drive_mode == 'displacement' and not result.solution_describes_a_displacement_driven_loading():
        raise LoadingPathIsNotDescribedBySolution
    
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

def extract_all_transitions(result: Result, drive_mode: str, check_energy_release=True):
    if drive_mode not in ('force', 'displacement'):
        raise ValueError(f'Invalid drive mode "{drive_mode}"')
    if drive_mode == 'force' and not result.solution_describes_a_force_driven_loading():
        raise LoadingPathIsNotDescribedBySolution
    elif drive_mode == 'displacement' and not result.solution_describes_a_displacement_driven_loading():
        raise LoadingPathIsNotDescribedBySolution
    
    u_load, f_load = result.get_equilibrium_path()  # wrt to preload
    f0 = result.get_forces()[0, :]  # preload forces
    u = result.get_displacements() - result.get_displacements()[0, :]  # displacement wrt to the preloaded configuration

    branches = extract_stable_branches_in_order(result, drive_mode)

    load = f_load if drive_mode == 'force' else u_load

    # check whether the load increases monotonically on each branch,
    # otherwise it is a sign that the solution path doubled back or connects
    # equilibria that should not be directly connected
    for branch in branches:
        start, end = branch
        if (np.diff(load[start:end+1]) < 0.0).any():
            raise DiscontinuityInTheSolutionPath
        

    transitions = []
    for i, branch_i in enumerate(branches):
        transitions.append({'previous_branch_index': None, 'next_branch_index': None,
                            'previous_critical_index': None, 'next_critical_index': None,
                            'previous_restabilization_index': None, 'next_restabilization_index': None,
                            'previous_elastic_energy_jump': None, 'next_elastic_energy_jump': None,
                            'previous_work_in': None, 'next_work_in': None})
        start_i, end_i = branch_i
        critical_plus = load[end_i] if i != len(branches) - 1 else None
        critical_minus = load[start_i] if i != 0 else None
        if critical_plus is not None:
            for j, branch_j in enumerate(branches):
                if j <= i: continue
                start, end = branch_j
                # look for plus-transition
                if load[start] <= critical_plus <= load[end]:
                    landing_index = int(interp1d(load[start:end + 1], 
                                        list(range(start, end + 1)),
                                        kind='next')(critical_plus))
                    elastic_energy_jump = result.compute_energy_at_state_index(landing_index)- result.compute_energy_at_state_index(end_i)

                    # work done by the additional forces (relative to preload) during snapping
                    work_in_additional = (f_load[end_i] * (u_load[landing_index] - u_load[end_i])
                                            if drive_mode == 'force' else
                                            0.0)
                    # work done by the preload forces (absolute) during snapping
                    work_in_preload = np.inner(f0, u[landing_index, :] - u[end_i, :])

                    # total work in during snapping
                    work_in = work_in_preload + work_in_additional

                    # print(f"{work_in_preload=:.3E}")
                    # print(f"{work_in_additional=:.3E}")
                    # print(f"{work_in=:.3E}")
                    # print(f"{elastic_energy_jump=:.3E}")
                    if check_energy_release and elastic_energy_jump > work_in:
                        # the structure gained more elastic energy than was provided by the external forces,
                        # the transition is not valid
                        continue
                    
                    transitions[-1]['next_branch_index'] = j
                    transitions[-1]['next_critical_index'] = end_i
                    transitions[-1]['next_restabilization_index'] = landing_index
                    transitions[-1]['next_elastic_energy_jump'] = elastic_energy_jump
                    transitions[-1]['next_work_in'] = work_in
                    break
        if critical_minus is not None:
            for jj, branch_j in enumerate(branches[::-1]):
                j = len(branches) - 1 - jj
                if j >= i: continue
                start, end = branch_j
                # look for plus-transition
                if load[start] <= critical_minus <= load[end]:
                    landing_index = int(interp1d(load[start:end + 1], 
                                        list(range(start, end + 1)),
                                        kind='previous')(critical_minus))
                    elastic_energy_jump = elastic_energy_jump = result.compute_energy_at_state_index(landing_index)- result.compute_energy_at_state_index(start_i)
                    
                    # work done by the additional forces (relative to preload) during snapping
                    work_in_additional = (f_load[start_i] * (u_load[landing_index] - u_load[start_i])
                                          if drive_mode == 'force' else
                                          0.0)
                    # work done by the preload forces (absolute) during snapping
                    work_in_preload = np.inner(f0, u[landing_index, :] - u[start_i, :])

                    # total work in during snapping
                    work_in = work_in_preload + work_in_additional

                    # print(f"{work_in_preload=:.3E}")
                    # print(f"{work_in_additional=:.3E}")
                    # print(f"{work_in=:.3E}")
                    # print(f"{elastic_energy_jump=:.3E}")
                    if check_energy_release and elastic_energy_jump > work_in:
                        # the structure gained more elastic energy than was provided by the external forces,
                        # the transition is not valid
                        continue
                
                    transitions[-1]['previous_branch_index'] = j
                    transitions[-1]['previous_critical_index'] = start_i
                    transitions[-1]['previous_restabilization_index'] = landing_index
                    transitions[-1]['previous_elastic_energy_jump'] = elastic_energy_jump
                    transitions[-1]['previous_work_in'] = work_in
                    break
    return branches, transitions



def extract_loading_path(result: Result, drive_mode: str, starting_index: int = 0, check_energy_release=True):
    branches, transitions = extract_all_transitions(result, drive_mode, check_energy_release)

    if starting_index < 0:
        starting_index = result.get_displacements().shape[0] + starting_index

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
        path_indices.extend(list(range(current_index, branches[current_branch_index][1] + 1)))
        transition = transitions[current_branch_index]
        if transition['next_branch_index'] is not None:
            current_index = transition['next_restabilization_index']
            current_branch_index = transition['next_branch_index']
            critical_indices.append(transition['next_critical_index'])
            restabilization_indices.append(transition['next_restabilization_index'])
        else:
            break
    return path_indices, critical_indices, restabilization_indices


def extract_unloading_path(result: Result, drive_mode: str, starting_index: int = -1, check_energy_release=True):
    branches, transitions = extract_all_transitions(result, drive_mode, check_energy_release)

    if starting_index < 0:
        starting_index = result.get_displacements().shape[0] + starting_index

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
        path_indices.extend(list(range(current_index, branches[current_branch_index][0] - 1, -1)))
        transition = transitions[current_branch_index]
        if transition['previous_branch_index'] is not None:
            current_index = transition['previous_restabilization_index']
            current_branch_index = transition['previous_branch_index']
            critical_indices.append(transition['previous_critical_index'])
            restabilization_indices.append(transition['previous_restabilization_index'])
        else:
            break
    return path_indices, critical_indices, restabilization_indices


class LoadingPathEmpty(Exception):
    """ raise this when the loading (or unloading) path is empty"""

class DiscontinuityInTheSolutionPath(Exception):
    """raise this when you detect discontinuities in the solution path """

class LoadingPathIsNotDescribedBySolution(Exception):
    """raise this when you the solution path does not describe a loading path under a given condition (force or displacement) """