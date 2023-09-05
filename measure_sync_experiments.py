import numpy as np
from numba import njit
from measure_sync_library import array_validation


def get_shifted(signal: np.array, shifts: np.array, signal_size, n):
    shifted_signal = np.zeros((n, signal_size))
    for i in range(n):
        shifted_signal[i, :] = np.roll(signal, shifts[i])
    return shifted_signal


def create_outliers_in_matrix(good_matrix : np.array, outlier_proportion):
    """
    # TODO: add noisy outliers as well.
    """
    n, d = good_matrix.shape
    outlier_count = int(n * outlier_proportion)

    random_samples = np.random.choice(np.arange(n), size=outlier_count)

    garbage_values = np.zeros((outlier_count, d))
    garbage_indices = np.random.randint(0, d, size=outlier_count)

    garbage_values[np.arange(outlier_count), garbage_indices] = 1.

    good_matrix[random_samples, :] = garbage_values


def get_noisy_samples_from_signal(signal: np.array, n, sigma, outliers=0.):
    """
    :param signal: the signal we want to hide with noise
    :param n: amount of noisy samples
    :param sigma:
    :param outliers:
    """
    array_validation(signal)
    signal_size = signal.shape[0]

    # Adding shifts
    # TODO: improve efficiency
    shifts = np.random.randint(0, signal_size, n)
    shifted_signal = get_shifted(signal, shifts, signal_size, n)

    noise = np.random.normal(0, scale=sigma, size=(n, signal_size))

    shifted_noisy = shifted_signal + noise

    if outliers != 0:
        create_outliers_in_matrix(shifted_noisy, outlier_proportion=outliers)

    return shifted_noisy, noise, shifts


def get_noisy_normalized(noisy_samples):
    pos_shifted_noisy = noisy_samples + np.abs(np.min(noisy_samples, axis=1))[:, None]
    print(noisy_samples[0])
    return pos_shifted_noisy / np.linalg.norm(pos_shifted_noisy, axis=1)[:, None]
