import numpy as np
from numba import njit
from measure_sync_library import array_validation


@njit
def get_shifted(signal: np.array, shifts: np.array, signal_size, n):
    shifted_signal = np.zeros((n, signal_size))
    for i in range(n):
        shifted_signal[i, :] = np.roll(signal, shifts[i])
    return shifted_signal


def get_noisy_samples_from_signal(signal: np.array, n, sigma):
    """
    :param signal: the signal we want to hide with noise
    :param n: amount of noisy samples
    :param sigma:
    """
    array_validation(signal)
    signal_size = signal.shape[0]

    # Adding shifts
    # TODO: improve efficiency
    shifts = np.random.randint(0, signal_size, n)
    shifted_signal = get_shifted(signal, shifts, signal_size, n)

    # Adding noise
    noise = np.random.normal(0, scale=sigma, size=(n, signal_size))



    shifted_noisy = shifted_signal + noise
    pos_shifted_noisy = shifted_noisy + np.abs(np.min(shifted_noisy, axis=1))[:, None]
    print(shifted_noisy[0])
    pos_shifted_noisy_normalized = pos_shifted_noisy / np.linalg.norm(pos_shifted_noisy, axis=1)[:, None]
    return pos_shifted_noisy_normalized, noise, shifts
