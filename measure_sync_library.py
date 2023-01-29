import numpy as np
import numpy.typing as npt
import scipy
from numba import njit

from sklearn import linear_model


@njit
def array_validation(arr):
    assert len(arr.shape) == 1


@njit
def dimension_validation(arr1, arr2):
    assert arr1.shape == arr2.shape


@njit
def discrete_convolution(arr1: np.array, arr2: np.array):
    # array_validation(arr1)
    # array_validation(arr2)
    # dimension_validation(arr1, arr2)
    n = arr1.shape[0]

    conv_arr = np.zeros_like(arr1)
    for k in range(n):
        for i in range(n):
            conv_arr[k] += arr1[i] * arr2[(i - k + n) % n]
    return conv_arr


@njit
def add_noise(signal: np.array, snr=1.):
    array_validation(signal)
    noise = np.random.normal(0, scale=1 / snr, size=signal.shape)
    return noise + signal


@njit
def solve_best_shift(arr1: np.array, arr2: np.array):
    array_validation(arr1)
    array_validation(arr2)
    dimension_validation(arr1, arr2)
    n = arr1.shape[0]

    minimum = np.inf
    argmin = -1
    for i in range(n):
        error = np.linalg.norm(np.roll(arr1, i) - arr2)
        if error < minimum:
            argmin = i
            minimum = error

    return argmin


@njit
def cost_function(samples, predictions):
    assert len(samples.shape) == 2
    assert len(predictions.shape) == 2
    assert samples.shape == predictions.shape
    n, d = samples.shape
    # n = distributions number
    # d = distributions dimension
    cost = 0
    for j in range(n):
        for i in range(j):
            cost += np.linalg.norm(samples[i, j], discrete_convolution[predictions[i, :], predictions[j, :]])
    return cost


@njit
def scipy_get_cost(x0: np.ndarray, distributions, samples_size, dimension_size):
    input = x0.reshape((samples_size, dimension_size))

    cost = 0
    for j in range(samples_size):
        for i in range(j):
            term = distributions[i, j] - discrete_convolution(input[i], input[j])
            cost += np.linalg.norm(term) ** 2

    return cost


def scipy_constraints(sample_size, dimensions):
    constraints = []

    for i in range(sample_size):
        constraints.append({'type': 'eq', 'fun': lambda x0, i=i, dimensions=dimensions: np.sum(
            x0[dimensions * i: dimensions * i + dimensions]) - 1})

    constraints.append({'type': 'ineq', 'fun': lambda x0: x0})

    return constraints


@njit
def get_distributions_from_noisy_samples(noisy_samples, samples, dimension):
    distributions = np.zeros((samples, samples, dimension))
    for j in range(samples):
        for i in range(j):
            # Cross correlation between signal and noisy shifted copies are stores as our samples.
            distributions[i, j, :] = discrete_convolution(noisy_samples[i], noisy_samples[j])

    return distributions

def solve_measure_sync_scipy(noisy_samples):
    assert len(noisy_samples.shape) == 2

    samples = noisy_samples.shape[0]
    dimension = noisy_samples.shape[1]

    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)

    # initial guess
    row_max_indices = np.argmax(noisy_samples, axis=1)
    wrong_guesses = np.zeros((samples, dimension))
    wrong_guesses[np.arange(samples), row_max_indices] = 1

    x0 = wrong_guesses.reshape((samples * dimension))

    constraints = scipy_constraints(samples, dimension)
    return scipy.optimize.minimize(scipy_get_cost, x0, args=(distributions, samples, dimension),
                                   constraints=constraints)


def solve_measure_sync(noisy_samples):
    """
    TODO: implement properly
    """
    assert len(noisy_samples.shape) == 2
    samples = noisy_samples.shape[0]
    signal_size = noisy_samples.shape[1]

    distributions = np.zeros((samples, samples, signal_size))
    for j in range(samples):
        for i in range(j):
            # Cross correlation between signal and noisy shifted copies are stores as our samples.
            distributions[i, j, :] = discrete_convolution(noisy_samples[i], noisy_samples[j])

    return distributions
