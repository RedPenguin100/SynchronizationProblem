import os
import time
import json

import pytest
import numpy as np
import scipy.optimize

from math_utils import average

from directories import DATA_DIR
from measure_sync_experiments import *
from measure_sync_library import *
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


def compare_samples_up_to_shift(truth, result, should_shift=True, debug=False):
    if len(truth.shape) != 2:
        raise ValueError("samples.shape is not len 2")
    if truth.shape != result.shape:
        raise ValueError("Bad shapes")
    sample_size, dimension = truth.shape

    noiseless = attempt_samples_to_noiseless(result)
    total_cost = np.inf
    total_wrongs_in_best_shift = 0

    wrong_samples = []
    correct_was = []
    winning_shift = 0
    shifts = sample_size
    if not should_shift:
        shifts = 1

    for shift in range(shifts):
        cost = 0.
        wrongs = 0
        wrong_samples_temp = []
        correct_was_temp = []
        for sample in range(sample_size):
            rolled_sample = np.roll(noiseless[sample], shift)
            temp_cost = np.linalg.norm(truth[sample] - rolled_sample)
            cost += temp_cost
            if temp_cost > 0:
                wrongs += 1
                wrong_samples_temp.append(np.roll(result[sample], shift))
                correct_was_temp.append(truth[sample])
        if cost < total_cost:
            total_wrongs_in_best_shift = wrongs
            total_cost = cost
            wrong_samples = wrong_samples_temp
            correct_was = correct_was_temp
            winning_shift = shift
    if debug:
        print("wrong_samples: ", wrong_samples)
        # print("correct_was: ", correct_was)
        print("winning_shift: ", winning_shift)
    tol = 0.7
    total_eliminated = 0
    correctly_eliminated = 0
    for sample in range(sample_size):
        sort = np.sort(result[sample])
        if (sort[-1] - sort[-2]) < 0.1:
            # print("Too close to second peak ", result[sample])
            pass
        if sort[-1] < tol:
            total_eliminated += 1
            if debug:
                print("Too small value ", np.roll(result[sample], shift=winning_shift))
            for wrong_sample in wrong_samples:
                if np.array_equal(np.roll(result[sample], shift=winning_shift), wrong_sample):
                    correctly_eliminated += 1
    if total_eliminated != 0:
        remaining_errors = total_wrongs_in_best_shift - correctly_eliminated
        if debug:
            print(
                f"Eliminated {correctly_eliminated} / {total_eliminated} correctly out of {total_wrongs_in_best_shift}")
            print(sample_size)
            print(
                f"New error: {100.0 * remaining_errors / (sample_size - total_eliminated)}%, Old error:{100.0 * total_wrongs_in_best_shift / sample_size}% of samples removed")
    return total_cost, total_wrongs_in_best_shift


def attempt_samples_to_noiseless(samples):
    assert len(samples.shape) == 2
    samples_copy = np.copy(samples)
    sample_size, dimension = samples_copy.shape
    for sample in range(sample_size):
        # argmax = np.argmax(np.abs(samples_copy[sample]))
        argmax = np.argmax(samples_copy[sample])
        samples_copy[sample] = np.zeros(dimension)
        samples_copy[sample][argmax] = 1.
    return samples_copy


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
    wrong_guesses = np.zeros((samples, dimension))
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
def test_convolution_accociativity(roll1, roll2):
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


@pytest.mark.skip("Re-evalutate this test")
@pytest.mark.parametrize('times', list(range(10)))
@pytest.mark.parametrize('sigma', [
    # 0.,
    # 0.1,
    # 0.2,
    # 0.25,
    # 0.3,
    0.4,
    # 0.5,
    # 0.8,
    # 1.
])
@pytest.mark.parametrize('samples', [15,
                                     # 25, 35, 45
                                     ])
@pytest.mark.parametrize('dimension', [5,
                                       # 7, 10, 12, 15
                                       ])
def test_measure_sync(sigma, samples, dimension, times):
    np.random.seed(times)
    np.set_printoptions(precision=2)

    # Create signal [1, 0, 0, ... ,0]
    x = np.zeros(dimension)
    x[0] = 1
    # x[1] = -1
    x = x / np.linalg.norm(x)

    print("Signal: ", x)
    if sigma == 0.:
        print("Noiseless setting")
    else:
        print(f"\nSNR={1 / sigma}")
    # Add noise:
    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma)
    truth = get_shifted(x, shifts, dimension, samples)
    shift_matrix = shifts_to_matrix(shifts, samples, dimension)
    print("shift_matrix: ", shift_matrix)
    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)
    print("shift_matrix cost", scipy_get_cost(shift_matrix, distributions, samples, dimension))
    print("truth cost", scipy_get_cost(truth, distributions, samples, dimension))

    # initial guess

    best_apriori_guess = stupid_solution(noisy_samples)
    best_possible_guess = shift_matrix.reshape((samples * dimension))

    res2 = solve_measure_sync_scipy(distributions, best_apriori_guess)
    res3 = solve_measure_sync_scipy(distributions, best_possible_guess)
    stupid_sol = stupid_solution(noisy_samples)

    print("stupid_sol cost: ",
          scipy_get_cost(stupid_sol.reshape(dimension * samples), distributions, samples, dimension))
    print(f"nit: {res2.nit},"
          f" {res3.nit},"
          # f" {res4.nit}"
          )
    print(f"njev: {res2.njev},"
          f" {res3.njev},"
          # f" {res4.njev}"
          )
    print(f"fun: {res2.fun}, "
          f"{res3.fun},"
          # f" {res4.fun}"
          )
    assert res3.success, res3.message

    best_apriori_solutions = np.reshape(res2.x, (samples, dimension))
    best_possible_solutions = np.reshape(res3.x, (samples, dimension))
    print("best_apriori_solutions ", best_apriori_solutions)
    apriori_best = compare_samples_up_to_shift(shift_matrix, best_apriori_solutions)
    print("Apriori best guess: ", apriori_best)
    print("Best possible guess: ", compare_samples_up_to_shift(shift_matrix, best_possible_solutions))
    stupid_sol_best = compare_samples_up_to_shift(shift_matrix, stupid_sol)
    print("Stupid solution: ", stupid_sol_best)
    signal_1 = reconstruct_signal_from_solution(noisy_samples, best_apriori_solutions)
    signal_2 = reconstruct_signal_from_solution(noisy_samples, best_possible_solutions)
    signal_3 = reconstruct_signal_from_solution(noisy_samples, stupid_sol)
    perfect = reconstruct_signal_from_solution(noisy_samples, shift_matrix)
    print("Signal 1", signal_1)
    print("Signal 2", signal_2)
    print("Signal 3", signal_3)
    print("Perfect ", perfect)
    print("Total noise 1: ", np.linalg.norm(signal_1 - x))
    print("Total noise 2: ", np.linalg.norm(signal_2 - x))
    print("Total noise 3: ", np.linalg.norm(signal_3 - x))
    print("Total noise perfect alignment: ", np.linalg.norm(perfect - x))

    assert apriori_best[1] <= stupid_sol_best[1]


def test_stupid_solution(setting, algorithm=None, seed=None, verbose=False, debug=False, distribution_cleaner=None):
    if verbose:
        print()
    test_start = time.time()

    sigma = setting.sigma
    dimension = setting.dimension
    samples = setting.samples
    signal = setting.signal
    if seed is not None:
        np.random.seed(seed)
    if signal is None:
        raise ValueError("Signal is None")

    x = signal

    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma)
    truth = get_shifted(signal, shifts, dimension, samples)
    shift_matrix = shifts_to_matrix(shifts, samples, dimension)
    flat_shift_matrix = shift_matrix.reshape((samples * dimension))
    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)
    # -- Problem starts here --
    if distribution_cleaner is not None:
        distributions = distribution_cleaner(distributions)

    if verbose:
        # print("shift_matrix: ", shift_matrix)
        print("shift_matrix cost", scipy_get_cost(flat_shift_matrix, distributions, samples, dimension))
        print("truth cost", scipy_get_cost(truth.reshape((samples * dimension)), distributions, samples, dimension))

    # initial guess
    good_apriori_guess = stupid_solution_distributions(distributions)
    best_guess_result = compare_samples_up_to_shift(shift_matrix, good_apriori_guess, debug=debug)
    signal_1 = reconstruct_signal_from_solution(noisy_samples, good_apriori_guess)
    print("Signal 1", signal_1)
    reconstruction_error = get_distance_mra(x, signal_1)

    test_end = time.time()

    print(f"Test took: {test_end - test_start}")
    return Result(best_guess_result[1], reconstruction_error, test_end - test_start, seed)


def test_measure_generic(setting, algorithm=None, seed=None, verbose=False, debug=False,
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

    # --  Problem setup --
    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma)
    truth = get_shifted(x, shifts, dimension, samples)
    shift_matrix = shifts_to_matrix(shifts, samples, dimension)
    flat_shift_matrix = shift_matrix.reshape((samples * dimension))
    distributions = get_distributions_from_noisy_samples(noisy_samples, samples, dimension)

    # -- Problem starts here --
    if distribution_cleaner is not None:
        distributions = distribution_cleaner(distributions)

    if verbose:
        # print("shift_matrix: ", shift_matrix)
        print("shift_matrix cost", scipy_get_cost(flat_shift_matrix, distributions, samples, dimension))
        print("truth cost", scipy_get_cost(truth.reshape((samples * dimension)), distributions, samples, dimension))

    # initial guess
    best_apriori_guess = stupid_solution_distributions(distributions)
    solve_start = time.time()
    res2 = solve_measure_sync_scipy(distributions, best_apriori_guess)
    solve_end = time.time()
    print(f"Solve apriori took: {solve_end - solve_start}s")
    best_apriori_solutions = np.reshape(res2.x, (samples, dimension))
    apriori_best = compare_samples_up_to_shift(shift_matrix, best_apriori_solutions, debug=debug)
    signal_1 = reconstruct_signal_from_solution(noisy_samples, best_apriori_solutions)
    print("Signal 1", signal_1)
    reconstruction_error = get_distance_mra(x, signal_1)
    print("Signal1: Get distance MRA: ", reconstruction_error)

    best_possible_guess = flat_shift_matrix
    solve_start = time.time()
    res3 = solve_measure_sync_scipy(distributions, best_possible_guess)
    solve_end = time.time()
    print(f"Solve best took: {solve_end - solve_start}s")

    stupid_sol = stupid_solution_distributions(distributions)
    if verbose:
        print("stupid_sol cost: ",
              scipy_get_cost(stupid_sol.reshape(dimension * samples), distributions, samples, dimension))
        print(f"nit: {res2.nit},"
              f" {res3.nit},"
              # f" {res4.nit}"
              )
        print(f"njev: {res2.njev},"
              f" {res3.njev},"
              # f" {res4.njev}"
              )
        print(f"fun: {res2.fun}, "
              f"{res3.fun},"
              # f" {res4.fun}"
              )
    assert res3.success, res3.message

    best_possible_solutions = np.reshape(res3.x, (samples, dimension))
    best_possible = compare_samples_up_to_shift(shift_matrix, best_possible_solutions, debug=debug)
    stupid_sol_best = compare_samples_up_to_shift(shift_matrix, stupid_sol, debug=debug)

    signal_2 = reconstruct_signal_from_solution(noisy_samples, best_possible_solutions)
    signal_3 = reconstruct_signal_from_solution(noisy_samples, stupid_sol)
    perfect = reconstruct_signal_from_solution(noisy_samples, shift_matrix)
    if debug:
        print("best_possible ", best_possible)
        print("best_apriori_solutions ", best_apriori_solutions)
        print("Stupid solution: ", stupid_sol_best)
    print("Signal 2", signal_2)
    print("Signal 3", signal_3)
    print("Perfect ", perfect)
    print("Signal2: Get distance MRA: ", get_distance_mra(x, signal_2))
    print("Signal3: Get distance MRA: ", get_distance_mra(x, signal_3))
    print("Total noise perfect alignment: ", np.linalg.norm(perfect - x))

    test_end = time.time()

    return Result(apriori_best[1], reconstruction_error, test_end - test_start, seed)


@pytest.mark.parametrize('sigma', [
    0.,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    # 0.8,
    # 1.
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
def test_measure_sync2(sigma, samples, dimension):
    np.random.seed(0)
    # Create signal
    signal = np.zeros(dimension)
    signal[0] = 1
    signal[1] = 0.5
    signal[4] = 0.2
    signal /= np.linalg.norm(signal)

    # Create setting
    algorithm = OptAlgorithm.best_apriori
    setting = Setting(sigma, samples, dimension, signal, algorithm)
    errors = []
    results = Experiment(setting)
    result_path = os.path.join(DATA_DIR, str(setting)) + ".json"
    if os.path.exists(result_path):
        pytest.skip("Experiment already executed")

    experiment_count = 20
    # Run tests
    for i in range(experiment_count):
        try:
            result = test_stupid_solution(setting, seed=i, verbose=True)
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
