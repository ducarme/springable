import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from ..simulation.static_solver import update_progress
from . import dynamics
from . import dynamic_animation
from ..io_handling.graphic_settings import GeneralOptions as GO
from . import text_utils as dti
from ..easy_springable import text_utils as ti
from ..io_handling import visuals
from . import dynamic_solver
from ..io_handling import plot
import matplotlib.pyplot as plt
import os
import shutil


def _simulate(mdl: dynamics.DynamicModel, **solver_parameters):
    slv = dynamic_solver.DynamicSolver(mdl)
    return slv.solve(**solver_parameters)


def make_static_load(loaded_node_nb, direction, force):
    return {'node_nb': loaded_node_nb,
            'direction': direction,
            'force': force}


def make_oscillating_load(loaded_node_nb, direction, amplitude, frequency):
    return {'node_nb': loaded_node_nb,
            'direction': direction,
            'amplitude': amplitude,
            'frequency': frequency}


def make_impulse_load(loaded_node_nb, direction, delta_t, amplitude):
    def pulse(t):
        if 0 <= t <= delta_t:
            return amplitude * np.sin(np.pi * t / delta_t )
        else:
            return 0.0

    return {'node_nb': loaded_node_nb,
            'direction': direction,
            'f(t)': pulse}


def _make_oscillating_force_vector_function(mdl: dynamics.DynamicModel, oscillating_load):
    loaded_dof_index = 2 * oscillating_load['node_nb'] + (0 if oscillating_load['direction'] == 'X' else 1)

    def oscillating_load_function(t):
        f = np.zeros(mdl.get_assembly().get_nb_dofs())
        f[loaded_dof_index] = oscillating_load['amplitude'] * np.sin(2 * np.pi * oscillating_load['frequency'] * t)
        return f

    return oscillating_load_function


def _make_static_force_vector_function(mdl: dynamics.DynamicModel, static_load):
    loaded_dof_index = 2 * static_load['node_nb'] + (0 if static_load['direction'] == 'X' else 1)

    def static_load_function(t):
        f = np.zeros(mdl.get_assembly().get_nb_dofs())
        f[loaded_dof_index] = static_load['force']
        return f

    return static_load_function


def _make_custom_force_vector_function(mdl: dynamics.DynamicModel, custom_load):
    loaded_dof_index = 2 * custom_load['node_nb'] + (0 if custom_load['direction'] == 'X' else 1)

    def custom_load_function(t):
        f = np.zeros(mdl.get_assembly().get_nb_dofs())
        f[loaded_dof_index] = custom_load['f(t)'](t)
        return f

    return custom_load_function


def extract_amplitude(s, n_last):
    peak_indices, _ = signal.find_peaks(s)
    valley_indices, _ = signal.find_peaks(-s)
    if len(peak_indices) < n_last or len(valley_indices) < n_last:
        raise ValueError(f'There are {len(peak_indices)} peaks and {len(valley_indices)} valleys,'
                         f'while {n_last} peaks and valleys are required.')
    peak_values = s[peak_indices[-n_last:]]
    valley_values = s[valley_indices[-n_last:]]
    amplitudes = (peak_values - valley_values) / 2
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(s.shape[0]),s[:], '-')
    # ax.plot(np.arange(s.shape[0])[peak_indices],s[peak_indices], 'ro')
    # ax.plot(np.arange(s.shape[0])[valley_indices],s[valley_indices], 'go')
    # plt.show()
    return np.mean(amplitudes), np.std(amplitudes)


def extract_phase_shift(t, s1, s2, n_last):
    s1_peak_indices, _ = signal.find_peaks(s1)
    s2_peak_indices, _ = signal.find_peaks(s2)
    if len(s1_peak_indices) < n_last or len(s2_peak_indices) < n_last:
        raise ValueError(f'There are {len(s1_peak_indices)} peaks in signal 1 and {len(s2_peak_indices)} in signal 2,'
                         f'while {n_last} peaks in both signals are required.')
    s1_peak_times = t[s1_peak_indices[-n_last:]]
    s2_peak_times = t[s2_peak_indices[-n_last:]]
    phase_shifts = s1_peak_times - s2_peak_times
    return np.mean(phase_shifts), np.std(phase_shifts)


def _make_total_force_vector_function(mdl: dynamics.DynamicModel, static_load_functions, dynamic_load_functions,
                                      custom_load_functions):
    def force_vector_function(t):
        f = np.zeros(mdl.get_assembly().get_nb_dofs())
        for static_load_function in static_load_functions:
            f += static_load_function(t)
        for dynamic_load_function in dynamic_load_functions:
            f += dynamic_load_function(t)
        for custom_load_function in custom_load_functions:
            f += custom_load_function(t)
        return f

    return force_vector_function


def simulate_model(model_path, damping, static_loads, oscillating_loads, custom_loads, data_dir, node_nbs=None,
                   directions=None,
                   show=True):
    data_dir = ti.mkdir(data_dir)
    mdl = dti.read_model(model_path, damping)
    print('Start dynamic simulation')

    static_force_vector_functions = [_make_static_force_vector_function(mdl, static_load) for static_load in
                                     static_loads]
    dynamic_force_vector_functions = [_make_oscillating_force_vector_function(mdl, oscillating_load) for
                                      oscillating_load in oscillating_loads]
    custom_load_functions = [_make_custom_force_vector_function(mdl, custom_load) for
                             custom_load in custom_loads]
    mdl.set_force_vector_function(
        _make_total_force_vector_function(mdl, static_force_vector_functions, dynamic_force_vector_functions,
                                          custom_load_functions))

    if GO.generate_model_drawing and GO.show_model_drawing:
        visuals.draw_model(mdl, show=True)

    time, q, dqdt, f = _simulate(mdl)
    print('Dynamic simulation finished successfully')
    dynamic_animation.animate(mdl, time, q, dqdt, f, data_dir, save_name='animation', show=True)
    load_dof_indices = [2 * node_nb + (0 if direction == 'X' else 1) for node_nb, direction in
                        zip(node_nbs, directions)]
    # mean_q, std_q = extract_amplitude(q[:, load_dof_indices[0]], 10)
    # mean_f, std_f = extract_amplitude(f[:, load_dof_indices[0]], 10)
    # mean_phi, std_phi = extract_phase_shift(time, q[:, load_dof_indices[0]], f[:, load_dof_indices[0]], 10)
    # print(f'amplification: {mean_q}/{mean_f} = {mean_q / mean_f}')
    # print(f'std q, f [%]: {std_q / mean_q * 100} %, {std_f / mean_f * 100} %')
    # frequency = oscillating_loads[0]['frequency']
    # print(f'phase shift = {mean_phi} [s] = {2 * np.pi * frequency * mean_phi} [rad]')
    # print(f'std phi [s]: {std_phi}')

    if node_nbs is not None and directions is not None:
        fig, axs = plt.subplots(3, 1, sharex='all')
        load_dof_indices = [2 * node_nb + (0 if direction == 'X' else 1) for node_nb, direction in
                            zip(node_nbs, directions)]
        # peak_indices, _ = signal.find_peaks(q[:, load_dof_indices[0]])
        # valley_indices, _ = signal.find_peaks(-q[:, load_dof_indices[0]])
        axs[0].plot(time, q[:, load_dof_indices])
        axs[1].plot(time, dqdt[:, load_dof_indices])
        axs[2].plot(time, f[:, load_dof_indices])
        if show:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        load_dof_indices = [2 * node_nb + (0 if direction == 'X' else 1) for node_nb, direction in
                            zip(node_nbs, directions)]
        ax.plot(q[:, load_dof_indices], dqdt[:, load_dof_indices], 'k-', lw=0.5, alpha=0.5)
        ax.scatter(q[:, load_dof_indices], dqdt[:, load_dof_indices], c=time)
        if show:
            plt.show()
        else:
            plt.close()

    # fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    # t_u = np.linspace(time[0], time[-1], 5*time.shape[0])
    # q_u = interp1d(time, q, axis=0)(t_u)
    # dqdt_u = interp1d(time, dqdt, axis=0)(t_u)
    # f_u = interp1d(time, f, axis=0)(t_u)
    # load_dof_index = [2 * node_nb + (0 if direction == 'X' else 1) for node_nb, direction in
    #                     zip(node_nbs, directions)][0]
    # Q = np.fft.fft(q_u[:, load_dof_index])
    # F = np.fft.fft(f_u[:, load_dof_index])
    # freq = np.fft.fftfreq(q_u.shape[0], t_u[1] - t_u[0])
    # good_indices = freq > 0
    #
    # axs[0].plot(freq[good_indices], np.abs(Q[good_indices]/F[good_indices]), 'k-', lw=0.5, alpha=0.5)
    # axs[0].scatter(freq[good_indices], np.abs(Q[good_indices]/F[good_indices]))
    # axs[1].plot(freq[good_indices], np.angle(Q[good_indices]/F[good_indices]), 'k-', lw=0.5, alpha=0.5)
    # axs[1].scatter(freq[good_indices], np.angle(Q[good_indices]/F[good_indices]))
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    indices = [2 * node_nbs[0], 2 * node_nbs[0] + 1]
    ax.plot(q[:, indices[0]], q[:, indices[1]], 'k-', lw=0.5, alpha=0.5)
    ax.scatter(q[:, indices[0]], q[:, indices[1]], c=time)
    ax.set_aspect('equal')
    if show:
        plt.show()
    else:
        plt.close()


def compute_frequency_response(model_path, damping, loaded_node_nb, direction, displacement_input_amplitude,
                               frequencies,
                               data_dir):
    data_dir = ti.mkdir(data_dir)
    shutil.copy(model_path, data_dir)
    mdl = dti.read_model(model_path, damping)
    loaded_node = mdl.get_assembly().get_node_from_set(mdl.get_assembly().get_nodes(), loaded_node_nb)
    mass = loaded_node.get_mass()
    static_loads = [make_static_load(loaded_node_nb, 'X', mass * 9.81)]
    static_force_vector_functions = [_make_static_force_vector_function(mdl, static_load) for static_load in
                                     static_loads]
    # if GO.generate_model_drawing and GO.show_model_drawing:
    #     visuals.draw_model(mdl, show=True)

    nb_frequencies = frequencies.shape[0]
    update_progress(0, 0, nb_frequencies, 'in progress')
    for i, frequency in enumerate(frequencies):
        subdata_dir = ti.mkdir(os.path.join(data_dir, f'sim{i}'))

        oscillating_load = make_oscillating_load(loaded_node_nb, direction,
                                                 mass * displacement_input_amplitude * (2 * np.pi * frequency) ** 2,
                                                 frequency)
        dynamic_force_vector_functions = [_make_oscillating_force_vector_function(mdl, oscillating_load)]
        mdl.set_force_vector_function(
            _make_total_force_vector_function(mdl, static_force_vector_functions, dynamic_force_vector_functions))
        time, q, dqdt, f = _simulate(mdl, t_max=30 / frequency)
        np.savetxt(os.path.join(subdata_dir, 'time.csv'), time, delimiter=',')
        np.savetxt(os.path.join(subdata_dir, 'q.csv'), q, delimiter=',')
        np.savetxt(os.path.join(subdata_dir, 'dqdt.csv'), dqdt, delimiter=',')
        np.savetxt(os.path.join(subdata_dir, 'f.csv'), f, delimiter=',')
        ti.write_design_parameters({'frequency': frequency,
                                    'mass': mass,
                                    'force_input_amplitude': displacement_input_amplitude * mass * (
                                            2 * np.pi * frequency) ** 2,
                                    'displacement_input_amplitude': displacement_input_amplitude,
                                    'damping': damping,
                                    'loaded_node_nb': loaded_node_nb,
                                    'direction': direction},
                                   subdata_dir, 'sim_data.csv')

        update_progress((i + 1) / nb_frequencies, (i + 1), nb_frequencies,
                        'in progress' if i + 1 < nb_frequencies else 'completed')


def read_data_and_plot_frequency_response(data_dir, save_dir, save_name, show=True):
    output_amplitudes = []
    std_output_amplitudes = []
    phase_shifts = []
    std_phase_shifts = []
    frequencies = []
    input_amplitudes = []
    for folder in os.listdir(data_dir):
        subdata_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(subdata_dir):
            continue
        state = 'state0' if 'STATE0' in data_dir else 'state1'
        t = np.loadtxt(os.path.join(subdata_dir, 'time.csv'), delimiter=',')
        q = np.loadtxt(os.path.join(subdata_dir, 'q.csv'), delimiter=',')
        dqdt = np.loadtxt(os.path.join(subdata_dir, 'dqdt.csv'), delimiter=',')
        f = np.loadtxt(os.path.join(subdata_dir, 'f.csv'), delimiter=',')
        sim_data = ti.read_design_parameters(subdata_dir, 'sim_data.csv')
        load_dof_index = 2 * int(sim_data['loaded_node_nb']) + (0 if sim_data['direction'] == 'X' else 1)
        y = - f / (2 * np.pi * sim_data['frequency']) ** 2
        mean_q, std_q = extract_amplitude(q[:, load_dof_index], 10)
        mean_phi, std_phi = extract_phase_shift(t, q[:, load_dof_index], y[:, load_dof_index], 10)
        frequencies.append(sim_data['frequency'])
        output_amplitudes.append(mean_q)
        std_output_amplitudes.append(std_q)
        phase_shifts.append(mean_phi)
        std_phase_shifts.append(std_phi)
        input_amplitudes.append(sim_data['displacement_input_amplitude'])
    output_amplitudes = np.array(output_amplitudes)
    std_output_amplitudes = np.array(std_output_amplitudes)
    phase_shifts = np.array(phase_shifts)
    frequencies = np.array(frequencies)
    std_phase_shifts = np.array(std_phase_shifts)
    input_amplitudes = np.array(input_amplitudes)
    phase_shifts_in_rad = 2 * np.pi * frequencies * phase_shifts

    fig, axs = plt.subplots(2, 1)
    fig.suptitle(
        f"state = {state}, damping [Ns/m] = {sim_data['damping']}, input amplitude [mm] = {sim_data['displacement_input_amplitude'] * 1000}",
        fontsize=8)
    axs[0].plot(frequencies, output_amplitudes / input_amplitudes, 'o')
    axs[0].errorbar(frequencies, output_amplitudes / input_amplitudes,
                    yerr=std_output_amplitudes / input_amplitudes, ls='none')
    axs[1].plot(frequencies, phase_shifts_in_rad, 'o')
    axs[1].errorbar(frequencies, phase_shifts_in_rad, yerr=2 * np.pi * frequencies * std_phase_shifts, ls='none')
    plot.save_fig(fig, save_dir, save_name, ['png'])
    if show:
        plt.show()
    else:
        plt.close()
