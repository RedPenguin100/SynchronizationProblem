import pytest
import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from measure_sync_experiments import get_noisy_samples_from_signal
from measure_sync_library import get_distributions_from_noisy_samples, discrete_convolution, discrete_cross_correlation
from sync_library import get_so_projection, get_error, solve_sync_with_spectral, truly_random_so_matrix, \
    block_assignment, add_noise_to_matrix, add_holes_to_matrix, create_d_matrix, Problem, get_mra_projection, \
    get_n_roll_matrix, truly_random_mra_matrix


def shift_permutation_by_n(X, n):
    """
    Shift the X-roll matrix from rolling x to (x+n) % d
    """
    return np.roll(X, -n, axis=1)


def get_best_shift(arr1, arr2):
    err = np.inf
    assert len(arr1.shape) == 1
    assert len(arr2.shape) == 1
    d = arr1.shape[0]
    assert arr2.shape[0] == d

    best_shift = np.nan
    for i in range(d):
        new_err = np.linalg.norm(arr1, np.roll(arr2, d))
        if new_err < err:
            best_shift = d
            err = new_err
    return best_shift


@pytest.mark.parametrize('n', [0, 1, 2])
def test_permutation_projection(n):
    mat1 = get_n_roll_matrix(3, n)
    a = np.array([1, 2, 3])
    print(mat1 @ a)
    assert np.array_equal(mat1 @ a, np.roll(a, n))


def test_shift_permutation():
    mat1 = get_n_roll_matrix(d=3, n=1)
    mat2 = get_n_roll_matrix(d=3, n=2)
    assert np.array_equal(mat2, shift_permutation_by_n(mat1, 1))


# def test_shift_permutation_nontrivial_array():
# array = np.array([1, 2, 3])
# mat1 = get_n_roll_matrix(d=3, n=1, arr=array)
# mat2 = get_n_roll_matrix(d=3, n=2, arr=array)
# assert np.array_equal(mat2, shift_permutation_by_n(mat1, 1))
#
# mat_eye = get_n_roll_matrix(d=3, n=0, arr=array)
# assert np.array_equal(mat_eye, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))


@pytest.mark.parametrize('n', [1, 2, 4])
@pytest.mark.parametrize('d', [3, 5, 10])
def test_mra_projection(d, n):
    original = get_n_roll_matrix(d, n)
    noisy = np.copy(original)
    noisy[0, 0] = 0.5  # adding error
    projection = get_mra_projection(noisy)
    assert np.array_equal(projection, original)


@pytest.mark.parametrize('times', list(range(10)))
@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('d', [3])
def test_sanity_sync_mra(n, d, times):
    stack = [truly_random_mra_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    print("B", B)
    assert n * V == pytest.approx(B @ V)

    R_hat = solve_sync_with_spectral(B, d, problem=Problem.mra)
    V = V.reshape((n, d, d))
    print(V)
    print(R_hat)
    assert 0.0 == pytest.approx(get_error(R_hat, V, d, problem=Problem.mra))


@pytest.mark.parametrize('n', [10, 30, 50])
@pytest.mark.parametrize('d', [3, 5, 7])
def test_sync_mra_shift_invariant(n, d):
    stack = [truly_random_mra_matrix(d) for _ in range(n)]
    stack2 = [shift_permutation_by_n(mat, 1) for mat in stack]
    V = np.vstack(stack)
    V2 = np.vstack(stack2)
    B = V @ V.T.conj()

    assert n * V == pytest.approx(B @ V)

    R_hat = solve_sync_with_spectral(B, d, problem=Problem.mra)
    V = V.reshape((n, d, d))
    V2 = V2.reshape((n, d, d))
    print(V)
    print(R_hat)
    assert 0 == pytest.approx(get_error(R_hat, V, d, problem=Problem.mra), )
    assert 0 == pytest.approx(get_error(R_hat, V2, d, problem=Problem.mra), )


@pytest.mark.parametrize('times', list(range(10)))
@pytest.mark.parametrize('sigma', [
    # 0.,
    0.1,
    0.2,
    0.3,
    # 0.4
])
def test_sync_mra_with_measure_setting(sigma, times):
    dimension = 3
    samples = 15

    x = np.zeros(dimension)
    x[0] = 1
    # x[1] = -1A

    x = x / np.linalg.norm(x)

    noisy_samples, noise, shifts = get_noisy_samples_from_signal(x, n=samples, sigma=sigma)

    B = np.empty((samples * dimension, samples * dimension))

    for i in range(samples):
        for j in range(samples):
            d = dimension
            if i == j:
                B[d * i: d * (i + 1), d * j: d * (j + 1)] = np.eye(d)
            else:
                corr = discrete_cross_correlation(noisy_samples[i], noisy_samples[j])
                n_roll = np.argmax(corr)
                print("Samples: {}, {}".format(noisy_samples[i], noisy_samples[j]))
                print("Roll matrix: ", dimension - n_roll)
                B[d * i: d * (i + 1), d * j: d * (j + 1)] = get_n_roll_matrix(dimension, dimension - n_roll)


    V = np.vstack([get_n_roll_matrix(dimension,  shift) for shift in shifts])
    print(B.shape)
    R_hat = solve_sync_with_spectral(B, dimension, problem=Problem.mra)
    V = V.reshape((samples, dimension, dimension))
    print("B", B)
    print("V", V)

    print(R_hat)
    assert 0 == pytest.approx(get_error(R_hat, V, dimension, problem=Problem.mra), )
    # noisy samples is [g_1, g_2, g_3, ..., g_n]
