import pytest
import numpy as np
import cvxpy as cp
import scipy.optimize

from measure_sync_experiments import get_noisy_samples_from_signal
from measure_sync_library import *


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


def cvxpy_discrete_convolution(arr1: cp.Expression, arr2: cp.Expression):
    n = arr1.shape[0]

    res = [arr1 @ arr2]
    for k in range(1, n):
        stack = cp.hstack([arr2[((n - k) % n):], arr2[:((n - k) % n)]])
        res.append(arr1 @ stack)

    return res


def compare_samples_up_to_shift(samples, result):
    assert len(samples.shape) == 2
    assert samples.shape == result.shape
    sample_size, dimension = samples.shape

    total_cost = np.inf
    for shift in range(sample_size):
        cost = 0.
        for sample in range(sample_size):
            cost += np.linalg.norm(samples[sample] - np.roll(result[sample], shift))
        total_cost = np.min([total_cost, cost])

    return total_cost


def attempt_samples_to_noiseless(samples):
    assert len(samples.shape) == 2
    sample_size, dimension = samples.shape
    for sample in range(sample_size):
        argmax = np.argmax(samples[sample])
        samples[sample] = np.zeros(dimension)
        samples[sample][argmax] = 1.


def test_least_squares_solver_scipy():
    samples = 4
    dimension = 3

    noiseless = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

    a = np.array([1.1, 0.1, 0.2], dtype=np.float64)
    b = np.array([0, 1, 0.2], dtype=np.float64)
    c = np.array([0, 0.1, 1], dtype=np.float64)
    d = np.array([0, 1.1, 0], dtype=np.float64)

    sample_data = np.array([a, b, c, d])
    res = solve_measure_sync_scipy(sample_data)  # rho hat

    assert res.success, res
    assert res.fun < 0.5

    np.set_printoptions(precision=3)
    solutions = np.reshape(res.x, (samples, dimension))
    print("\nx=", solutions)

    attempt_samples_to_noiseless(solutions)

    print(compare_samples_up_to_shift(noiseless, solutions))


def test_least_squares_solver():
    samples = 4
    dimension = 3

    a = np.array([1.1, 0.1, 0.2], dtype=np.float64)
    b = np.array([0, 1, 0.2], dtype=np.float64)
    c = np.array([0, 0.1, 1], dtype=np.float64)
    d = np.array([0, 1, 0], dtype=np.float64)

    sample_data = np.array([a, b, c, d])
    distributions = solve_measure_sync(sample_data)

    vars = list()
    for i in range(samples):
        vars.append(cp.Variable(dimension, nonneg=True))

    # constraints = [1 == np.inner(np.ones(samples), p[i]) for i in range(samples)]
    constraints = [cp.sum(var) == 1 for var in vars]
    constraints.extend([var >= 0 for var in vars])

    print("")

    cost = 0
    for j in range(samples):
        for i in range(j):
            cost_inner = 0
            variable_conv = cvxpy_discrete_convolution(vars[i], vars[j])
            for l in range(dimension):
                term = cp.power((cp.Constant(distributions[i, j, l]) - variable_conv[l]), 2)
                print("Is term something ", term.is_dqcp())
                print("Is term something ", term.is_quasiconcave())
                print("Is term something ", term.is_quasiconvex())
                cost_inner += term

            cost += cost_inner
    for constraint in constraints:
        print("Is constraint dqcp: ", constraint.is_dqcp())
    problem = cp.Problem(cp.Minimize(cost), constraints)
    print("Is problem dqcp: ", problem.is_dqcp())
    problem.solve(qcp=True)


def test_measure_sync_sanity():
    # Create signal [1, 0, 0, ... ,0]
    x = np.zeros(15)
    x[0] = 1

    # Add noise:
    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=15, sigma=0.1)
    print(noisy_samples[0])
