import os
import time
import json

import pytest

from directories import PROJECT_ROOT
from measure_sync_experiments import *
from measure_sync_library import *
from sync_library import solve_sync_with_spectral, Problem, get_n_roll_matrix, get_shift_vec_from_matrix
from test_utils import OptAlgorithm, Result, Setting, Experiment


def test_convolution():
    a = np.array([0.9, 0.1])
    b = np.array([0.7, 0.3])
    conv = discrete_convolution(a, b)
    assert sum(conv) == pytest.approx(1)


def test_best_shift_sanity():
    a = np.array([1, 0, 0], dtype=np.float64)
    b = np.array([0, 0, 1], dtype=np.float64)

    index = solve_best_shift(a, b)
    assert index == 2


def test_best_shift_with_slight_errors():
    a = np.array([1.1, 0, 0], dtype=np.float64)
    b = np.array([0, 0, 1], dtype=np.float64)

    index = solve_best_shift(a, b)
    assert index == 2


def test_best_shift_distribution_sanity():
    a = np.array([1, 0, 0], dtype=np.float64)
    b = np.array([0, 1, 0], dtype=np.float64)

    index = discrete_convolution(a, b)
    assert np.sum(index) == pytest.approx(1)

    print(index)


@pytest.mark.skip
def test_noisy_samples():
    # Create signal [1, 0, 0, ... ,0]
    dimension = 15
    samples = 45
    x = np.zeros(dimension)
    x[0] = 1

    # Add noise:
    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=0.1)

    assert pytest.approx(np.linalg.norm(noisy_samples, axis=1)) == np.ones(samples)


def align_samples(sample1, sample2):
    assert sample1.shape == sample2.shape
    return np.argmax(discrete_cross_correlation(sample1, sample2))


def test_shift():
    np.random.seed(42)
    # Create signal [1, -1, 0, ... ,0]
    dimension = 15
    samples = 45
    x = np.zeros(dimension)
    x[0] = -1
    x[1] = 1
    noise = np.random.normal(0, 1, dimension)

    assert align_samples(x, np.roll(x, 1)) == 1
    assert align_samples(x, noise + np.roll(x, 1)) == 1


@njit
def shifts_to_matrix(shifts: np.array, samples, dimension):
    # assert len(shifts.shape) == 1
    # assert samples == shifts.shape[0]
    matrix = np.zeros((samples, dimension))
    for i in range(samples):
        matrix[i, shifts[i]] = 1
    return matrix


def get_best_apriori_guess(noisy_samples, dimension, samples):
    row_max_indices = np.argmax(noisy_samples, axis=1)
    wrong_guesses = np.zeros((samples, dimension), dtype=np.float64)
    wrong_guesses[np.arange(samples), row_max_indices] = 1
    return wrong_guesses.reshape((samples * dimension))


def reconstruct_signal_from_solution(noisy_samples, solution):
    assert len(solution.shape) == 2
    assert len(noisy_samples.shape) == 2
    samples, dimension = solution.shape
    noiseless = attempt_samples_to_noiseless(solution)
    signal = np.zeros(dimension)
    for sample in range(samples):
        signal += np.roll(noisy_samples[sample, :], - np.argmax(noiseless[sample, :]))
    return signal / samples


def get_distance_mra(signal, expected_signal):
    array_validation(signal)
    dimension = signal.shape[0]
    distances = []
    for i in range(dimension):
        distance = np.linalg.norm(signal - np.roll(expected_signal, i))
        distances.append(distance)
    return np.min(distances)


@pytest.mark.parametrize('roll1', [1, 2])
@pytest.mark.parametrize('roll2', [1, 2])
def test_convolution_associativity(roll1, roll2):
    np.set_printoptions(precision=2)

    dimension = 5
    x = np.zeros(dimension)
    x[0] = 1
    x[1] = 1
    y = np.roll(x, roll1)
    z = np.roll(x, roll2)
    print("x,y : ", discrete_convolution(x, y))
    print("y,x : ", discrete_convolution(y, x))
    print("y,z : ", discrete_convolution(y, z))
    print("z,y : ", discrete_convolution(z, y))
    print("z,x : ", discrete_convolution(z, x))
    print("x,z : ", discrete_convolution(x, z))

    first_side = discrete_convolution(x, discrete_convolution(y, z))
    second_side = discrete_convolution(discrete_convolution(x, y), z)
    print(first_side)
    print(second_side)
    assert np.all(first_side == second_side)


@pytest.mark.parametrize('roll1', [1, 2])
@pytest.mark.parametrize('roll2', [1, 2])
def test_cross_correlation_triplet(roll1, roll2):
    np.set_printoptions(precision=2)

    dimension = 5
    x = np.zeros(dimension)
    x[0] = 1
    x[1] = 1
    y = np.roll(x, roll1)
    z = np.roll(x, roll2)
    print("x,y : ", discrete_cross_correlation(x, y))
    print("y,x : ", discrete_cross_correlation(y, x))
    print("y,z : ", discrete_cross_correlation(y, z))
    print("z,y : ", discrete_cross_correlation(z, y))
    print("z,x : ", discrete_cross_correlation(z, x))
    print("x,z : ", discrete_cross_correlation(x, z))

    first_side = discrete_cross_correlation(discrete_cross_correlation(z, y), discrete_cross_correlation(x, y))
    second_side = discrete_cross_correlation(discrete_cross_correlation(y, x), discrete_cross_correlation(y, z))
    print(first_side)
    print(second_side)
    assert np.all(first_side == second_side)


@pytest.mark.parametrize('roll1', [1, 2])
@pytest.mark.parametrize('roll2', [1, 2])
def test_cross_correlation_triplet2(roll1, roll2):
    np.set_printoptions(precision=2)

    dimension = 5
    x = np.zeros(dimension)
    x[0] = 1
    x[1] = 1
    y = np.roll(x, roll1)
    z = np.roll(x, roll2)
    print("x,x : ", discrete_cross_correlation(x, x))
    print("x,y : ", discrete_cross_correlation(x, y))
    print("y,x : ", discrete_cross_correlation(y, x))
    print("y,z : ", discrete_cross_correlation(y, z))
    print("z,y : ", discrete_cross_correlation(z, y))
    print("z,x : ", discrete_cross_correlation(z, x))
    print("x,z : ", discrete_cross_correlation(x, z))

    first_side = discrete_cross_correlation(discrete_cross_correlation(x, y), discrete_cross_correlation(x, z))
    second_side = discrete_cross_correlation(discrete_cross_correlation(y, y), discrete_cross_correlation(y, z))
    # second_side = discrete_cross_correlation(y, z)
    print(first_side)
    print(second_side)
    assert np.all(first_side == second_side)


@pytest.mark.parametrize('roll1', [1, 2])
@pytest.mark.parametrize('roll2', [1, 2])
def test_cross_correlation_commutativity(roll1, roll2):
    np.set_printoptions(precision=2)
    dimension = 5
    x, y = np.random.normal(loc=0., scale=0.4, size=(2, dimension))
    print(x)
    first_side = discrete_cross_correlation(x, y)
    second_side = discrete_cross_correlation(y, x)
    assert np.allclose(np.roll(np.flip(first_side), 1), second_side)
    assert np.allclose(np.flip(np.roll(first_side, 4)), second_side)


def solve_distributions(algorithm, distributions, samples, dimension, verbose=False, solutions=None):
    if algorithm == OptAlgorithm.measure_best_apriori:
        # initial guess
        best_apriori_guess = stupid_solution_distributions(distributions)
        res2 = solve_measure_sync_scipy(distributions, best_apriori_guess)
        if verbose:
            print("stupid_sol cost: ",
                  scipy_get_cost(best_apriori_guess.reshape(dimension * samples), distributions, samples, dimension))
            print(f"nit: {res2.nit},")
            print(f"njev: {res2.njev},")
            print(f"fun: {res2.fun}, ")
        assert res2.success, res2.message
        return np.reshape(res2.x, (samples, dimension))
    elif algorithm == OptAlgorithm.measure_best_apriori_fourier:
        # initial guess
        best_apriori_guess = solve_distributions(OptAlgorithm.sync_mra, distributions, samples, dimension)
        best_apriori_guess_fft = np.fft.fft(best_apriori_guess)
        print('best_apriori_guess_fft: ', best_apriori_guess_fft)
        distributions_fft = np.fft.fft(distributions)
        print('distributions fft: ', distributions_fft)
        res = solve_measure_fourier_sync(distributions_fft, best_apriori_guess_fft)
        if verbose:
            # print("stupid_sol cost: ",
            #       scipy_get_cost_fourier(
            #           best_apriori_guess_fft,
            #           distributions, samples, dimension))
            print(f"nit: {res.nit},")
            print(f"njev: {res.njev},")
            print(f"fun: {res.fun}, ")
        assert res.success, res.message

        complex_result = res.x.reshape(-1, 2).view(np.complex128)
        res_reshaped = np.reshape(complex_result, (samples, dimension))
        res_real = np.fft.ifft(res_reshaped)
        return res_real.astype(np.float64)
    elif algorithm == OptAlgorithm.stupid_solution:
        # initial guess
        return stupid_solution_distributions(distributions)
    elif algorithm == OptAlgorithm.pure_random:
        wrong_guesses = np.zeros((samples, dimension))
        wrong_guesses[np.arange(samples), np.random.randint(0, dimension, samples)] = 1
        return wrong_guesses
    elif algorithm == OptAlgorithm.sync_mra:
        B = np.empty((samples * dimension, samples * dimension))

        for i in range(samples):
            for j in range(samples):
                d = dimension
                if i == j:
                    B[d * i: d * (i + 1), d * j: d * (j + 1)] = np.eye(d)
                else:
                    n_roll = np.argmax(distributions[i, j])
                    # print("Samples: {}, {}".format(noisy_samples[i], noisy_samples[j]))
                    # print("Roll matrix: ", dimension - n_roll)
                    B[d * i: d * (i + 1), d * j: d * (j + 1)] = get_n_roll_matrix(dimension, dimension - n_roll)

        R_hat = solve_sync_with_spectral(B, dimension, problem=Problem.mra)
        solution = np.zeros((samples, dimension))
        for i in range(samples):
            solution[i] = get_shift_vec_from_matrix(R_hat[i])
        return solution
    elif algorithm == OptAlgorithm.best_possible:
        return solutions
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def util_test_noisy_measure(setting, algorithm=None, seed=None, verbose=False, debug=False,
                            distribution_cleaner=None):
    # printing
    np.set_printoptions(precision=2)
    if verbose:
        print()
    test_start = time.time()

    # Setup
    sigma = setting.sigma
    dimension = setting.dimension
    samples = setting.samples
    signal = setting.signal
    if seed is not None:
        np.random.seed(seed)

    if signal is not None:
        x = signal
    else:
        x = np.zeros(dimension)
        x[0] = 1
        x[1] = 0.5
        x = x / np.linalg.norm(x)

    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma)

    # --  Problem setup --
    truth = get_shifted(x, shifts, dimension, samples)
    shift_matrix = shifts_to_matrix(shifts, samples, dimension)
    if verbose:
        print("Shift matrix: ", shift_matrix)
    flat_shift_matrix = shift_matrix.reshape((samples * dimension))
    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)

    # -- Problem starts here --
    if distribution_cleaner is not None:
        distributions = distribution_cleaner(distributions)
    if verbose:
        # print("shift_matrix: ", shift_matrix)
        print("shift_matrix cost", scipy_get_cost(flat_shift_matrix, distributions, samples, dimension))
        print("truth cost", scipy_get_cost(truth.reshape((samples * dimension)), distributions, samples, dimension))

    solver_solution = solve_distributions(algorithm, distributions, samples, dimension, verbose=verbose,
                                          solutions=truth)
    if verbose:
        print("Solver solution: \n", solver_solution.shape)
        print("Shift matrix shape: \n", shift_matrix.shape)
    apriori_best = compare_samples_up_to_shift(shift_matrix, solver_solution, debug=debug)
    signal_1 = reconstruct_signal_from_solution(noisy_samples, solver_solution)
    reconstruction_error = get_distance_mra(x, signal_1)

    perfect = reconstruct_signal_from_solution(noisy_samples, shift_matrix)
    # if debug:
    print("best_apriori_solutions ", solver_solution)
    if verbose:
        print("Perfect ", perfect)
        print("Total noise perfect alignment: ", np.linalg.norm(perfect - x))

    test_end = time.time()

    return Result(apriori_best[1], reconstruction_error, test_end - test_start, seed)


@pytest.mark.parametrize('algorithm',
                         [
                             # OptAlgorithm.sync_mra,
                             # OptAlgorithm.measure_best_apriori
                             OptAlgorithm.measure_best_apriori_fourier
                             # OptAlgorithm.best_possible
                             # OptAlgorithm.pure_random,
                             # OptAlgorithm.stupid_solution
                         ])
@pytest.mark.parametrize('sigma', [
    0.,
    0.1,
    0.2,
    # 0.25,
    0.3,
    0.4,
    0.5,
    # 0.8,
    # 1.
])
@pytest.mark.parametrize('samples', [
    # 15,
    25,
    # 45,
    # 70,
    # 100
])
@pytest.mark.parametrize('dimension', [
    5,
    # 10,
    # 15
])
def test_measure_uniform_noise(sigma, samples, dimension, algorithm):
    np.random.seed(0)
    # Create signal
    signal = np.zeros(dimension)
    signal[0] = 1
    signal[1] = 0.5
    signal[4] = 0.2
    signal /= np.linalg.norm(signal)

    # Create setting
    setting = Setting(sigma, samples, dimension, signal, algorithm)
    errors = []
    results = Experiment(setting)

    FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", algorithm)
    print(f"Data directory: {FULL_DATA_DIR}")
    result_path = os.path.join(FULL_DATA_DIR, str(setting)) + ".json"
    if os.path.exists(result_path):
        pytest.skip("Experiment already executed")

    experiment_count = 20
    # Run tests
    for i in range(experiment_count):
        try:
            result = util_test_noisy_measure(setting, algorithm, seed=i, verbose=True)
            results.add_result(result)
        except Exception as e:
            errors.append(e)
            print("Encountered error:", e)
    print("Done execution")
    print(f"Total errors: {len(errors)}")
    results.print()
    setting.print_summary()
    results.setting.signal = list(results.setting.signal)
    json_string = json.dumps(results, default=lambda x: x.__dict__, sort_keys=True, indent=4)
    with open(result_path, 'w+') as f:
        f.write(json_string)


def util_test_noiseless_outliers(setting, algorithm=None, seed=None, verbose=False, debug=False,
                                 distribution_cleaner=None):
    # printing
    np.set_printoptions(precision=2)
    if verbose:
        print()
    test_start = time.time()

    # Setup

    sigma = setting.sigma
    dimension = setting.dimension
    samples = setting.samples
    signal = setting.signal
    outliers = setting.outliers

    if seed is not None:
        np.random.seed(seed)

    if signal is not None:
        x = signal
    else:
        x = np.zeros(dimension)
        x[0] = 1
        x[1] = 0.5
        x = x / np.linalg.norm(x)

    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma, outliers=outliers)

    # --  Problem setup --
    truth = get_shifted(x, shifts, dimension, samples)
    shift_matrix = shifts_to_matrix(shifts, samples, dimension)
    if verbose:
        print("Shift matrix: ", shift_matrix)
    flat_shift_matrix = shift_matrix.reshape((samples * dimension))
    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)

    # -- Problem starts here --
    if distribution_cleaner is not None:
        distributions = distribution_cleaner(distributions)
    if verbose:
        # print("shift_matrix: ", shift_matrix)
        print("shift_matrix cost", scipy_get_cost(flat_shift_matrix, distributions, samples, dimension))
        print("truth cost", scipy_get_cost(truth.reshape((samples * dimension)), distributions, samples, dimension))

    solver_solution = solve_distributions(algorithm, distributions, samples, dimension, verbose=verbose)
    if verbose:
        print("Solver solution: \n", solver_solution.shape)
        print("Shift matrix shape: \n", shift_matrix.shape)
    apriori_best = compare_samples_up_to_shift(shift_matrix, solver_solution, debug=debug)
    signal_1 = reconstruct_signal_from_solution(noisy_samples, solver_solution)
    reconstruction_error = get_distance_mra(x, signal_1)

    perfect = reconstruct_signal_from_solution(noisy_samples, shift_matrix)
    # if debug:
    print("best_apriori_solutions ", solver_solution)
    if verbose:
        print("Perfect ", perfect)
        print("Total noise perfect alignment: ", np.linalg.norm(perfect - x))

    test_end = time.time()

    return Result(apriori_best[1], reconstruction_error, test_end - test_start, seed)


@pytest.mark.parametrize('algorithm',
                         [
                             OptAlgorithm.sync_mra,
                             OptAlgorithm.measure_best_apriori,
                             OptAlgorithm.pure_random,
                             OptAlgorithm.stupid_solution
                         ])
@pytest.mark.parametrize('outliers', [
    0.,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
])
@pytest.mark.parametrize('sigma', [0,
                                   0.1,
                                   0.2,
                                   0.3,
                                   0.4,
                                   0.5
                                   ])
@pytest.mark.parametrize('samples', [
    15,
    25,
    45,
    70,
    100
])
@pytest.mark.parametrize('dimension', [
    5,
    10,
    15
])
def test_measure_noiseless_outliers(outliers, sigma, samples, dimension, algorithm):
    np.random.seed(0)
    # Create signal
    signal = np.zeros(dimension)
    signal[0] = 1
    signal[1] = 0.5
    signal[4] = 0.2
    signal /= np.linalg.norm(signal)

    # Create setting
    setting = Setting(sigma, samples, dimension, signal, algorithm, outliers=outliers)
    errors = []
    results = Experiment(setting)

    FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", algorithm + "_outliers")
    print(f"Data directory: {FULL_DATA_DIR}")
    result_path = os.path.join(FULL_DATA_DIR, str(setting)) + ".json"
    if os.path.exists(result_path):
        pytest.skip("Experiment already executed")
    if not os.path.exists(FULL_DATA_DIR):
        os.makedirs(FULL_DATA_DIR)

    experiment_count = 20
    # Run tests
    for i in range(experiment_count):
        try:
            result = util_test_noiseless_outliers(setting, algorithm, seed=i, verbose=False)
            results.add_result(result)
        except Exception as e:
            errors.append(e)
            print("Encountered error:", e)
    print("Done execution")
    print(f"Total errors: {len(errors)}")
    results.print()
    setting.print_summary()
    results.setting.signal = list(results.setting.signal)
    json_string = json.dumps(results, default=lambda x: x.__dict__, sort_keys=True, indent=4)
    with open(result_path, 'w+') as f:
        f.write(json_string)
