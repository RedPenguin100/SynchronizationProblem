import numpy as np
import scipy
from numba import njit


@njit
def array_validation(arr):
    assert len(arr.shape) == 1


@njit
def dimension_validation(arr1, arr2):
    assert arr1.shape == arr2.shape


@njit
def discrete_convolution(arr1: np.array, arr2: np.array):
    """
    Convolution
    | ... , ... , ... |
    -->
    | ... , ... , ... |
    <--
       <--
          <--
    """
    array_validation(arr1)
    array_validation(arr2)
    # dimension_validation(arr1, arr2)
    n = arr1.shape[0]

    conv_arr = np.zeros_like(arr1, dtype=np.float64)
    for k in range(n):
        for i in range(n):
            # conv_arr[k] += arr1[i] * arr2[(i - k + n) % n]
            conv_arr[k] += arr1[i] * arr2[k - i]

    return conv_arr


@njit
def discrete_cross_correlation(arr1: np.array, arr2: np.array):
    """
    Cross correlation
    | ... , ... , ... |
    -->
    | ... , ... , ... |
    -->
       -->
           -->
    """
    # array_validation(arr1)
    # array_validation(arr2)
    # dimension_validation(arr1, arr2)
    n = arr1.shape[0]

    conv_arr = np.zeros_like(arr1, dtype=np.float64)
    for k in range(n):
        for i in range(n):
            conv_arr[k] += arr1[i] * arr2[(i + k) % n]

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


# @njit
# def scipy_get_cost(x0: np.ndarray, distributions, samples_size, dimension_size):
#     variables = x0.reshape((samples_size, dimension_size))
#
#     cost = 0.
#     for j in range(samples_size):
#         for i in range(j):
#             term = distributions[i, j] - discrete_cross_correlation(variables[i], variables[j])
#             cost += np.linalg.norm(term)

# # Internal consistency condition
# for x in range(samples_size):
#     for y in range(x):
#         for z in range(y):
#             v1 = discrete_cross_correlation(variables[z], variables[y])
#             v2 = discrete_cross_correlation(variables[x], variables[y])
#             v3 = discrete_cross_correlation(variables[y], variables[x])
#             v4 = discrete_cross_correlation(variables[y], variables[z])
#             term1 = discrete_cross_correlation(v1, v2)
#             term2 = discrete_cross_correlation(v3, v4)
#             cost += np.linalg.norm(term1 - term2) ** 2

@njit
def scipy_get_cost(x0: np.ndarray, distributions, samples_size, dimension_size):
    variables = x0.reshape((samples_size, dimension_size))

    cost = 0.
    for j in range(samples_size):
        for i in range(j):
            term = distributions[i, j] - discrete_cross_correlation(variables[i], variables[j])
            cost += np.linalg.norm(term)
    return cost


def scipy_constraints(sample_size, dimensions):
    constraints = []

    for i in range(sample_size):
        # sum of x_i = 1
        var_start = i * dimensions
        var_end = i * dimensions + dimensions
        constraints.append({'type': 'eq', 'fun': lambda x0, var_end=var_end, var_start=var_start: np.sum(
            x0[var_start: var_end]) - 1})

    # for i in range(sample_size):
    #     # sum of x_i = 1
    #     constraints.append({'type': 'eq', 'fun': lambda x0, i=i, dimensions=dimensions: np.linalg.norm(
    #         x0[dimensions * i: dimensions * i + dimensions]) - 1})

    # x_i >= -1
    # constraints.append({'type': 'ineq', 'fun': lambda x0: x0 + 1})
    constraints.append({'type': 'ineq', 'fun': lambda x0: x0})

    return constraints


@njit
def get_distributions_from_noisy_samples(noisy_samples, samples, dimension):
    distributions = np.empty((samples, samples, dimension))
    for j in range(samples):
        for i in range(j):
            # Cross correlation between signal and noisy shifted copies are stores as our samples.
            distributions[i, j, :] = discrete_cross_correlation(noisy_samples[i], noisy_samples[j])
    return distributions


def stupid_solution(noisy_samples):
    assert len(noisy_samples.shape) == 2

    samples = noisy_samples.shape[0]
    dimension = noisy_samples.shape[1]

    # initial guess
    row_max_indices = np.argmax(noisy_samples, axis=1)
    wrong_guesses = np.zeros((samples, dimension))
    wrong_guesses[np.arange(samples), row_max_indices] = 1

    return wrong_guesses


def stupid_solution_distributions(distributions):
    assert len(distributions.shape) == 3

    samples = distributions.shape[0]
    samples2 = distributions.shape[1]
    assert samples == samples2
    dimension = distributions.shape[2]

    wrong_guesses = np.zeros((samples, dimension))
    wrong_guesses[0, 0] = 1
    for j in range(1, samples):
        index = np.argmax(distributions[0, j])
        wrong_guesses[j, dimension - index - 1] = 1

    return wrong_guesses


def solve_measure_sync_scipy(distributions, guesses=None):
    assert len(distributions.shape) == 3

    samples = distributions.shape[0]
    samples2 = distributions.shape[1]
    assert samples == samples2
    dimension = distributions.shape[2]

    constraints = scipy_constraints(samples, dimension)
    return scipy.optimize.minimize(scipy_get_cost, guesses, args=(distributions, samples, dimension),
                                   constraints=constraints,
                                   options={'maxiter': 1000, 'ftol': 1e-2})
