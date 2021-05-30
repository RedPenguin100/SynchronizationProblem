import pytest
import numpy as np
from matplotlib import pyplot as plt

from exer1 import get_so_projection, get_error, solve_sync_with_spectral, truly_random_so_matrix, \
    block_assignment, add_noise_to_matrix


def test_rotation_matrix_sanity():
    X = truly_random_so_matrix(3)
    print(X.conj().T @ X)
    assert np.allclose(X.conj().T @ X, np.eye(3), atol=1.e-8)
    assert 1 == pytest.approx(np.linalg.det(X))


def test_projection_matrix_for_rotation():
    X = truly_random_so_matrix(3)
    print(X)
    projection = get_so_projection(X)
    print(projection)
    assert projection == pytest.approx(X)

    assert np.eye(3) == pytest.approx(projection.T.conj() @ projection)
    assert 1 == pytest.approx(np.linalg.det(projection))


def test_projection_matrix_not_rotation():
    X = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    projection = get_so_projection(X)
    assert projection == pytest.approx(np.eye(3))


def test_error_sanity():
    rot1 = truly_random_so_matrix(3)
    rot2 = truly_random_so_matrix(3)

    rot1_disturbed = rot1
    rot2_disturbed = rot2
    rot = np.array((rot1, rot2))
    rot_dis = np.array((rot1_disturbed, rot2_disturbed))

    assert 0 == pytest.approx(get_error(expected=rot, actual=rot_dis))


def test_error_noisy():
    rot1 = truly_random_so_matrix(3)
    rot2 = truly_random_so_matrix(3)

    rot1_disturbed = rot1 + np.random.normal(0, 0.1, (3, 3))
    rot2_disturbed = rot2 + np.random.normal(0, 0.1, (3, 3))
    rot = np.array((rot1, rot2))
    rot_dis = np.array((rot1_disturbed, rot2_disturbed))

    assert get_error(expected=rot, actual=rot_dis) < 0.4


@pytest.mark.parametrize('times', range(100))
def test_eigenvalue(times):
    n = 5
    d = 2
    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()

    assert n * V == pytest.approx(B @ V)

    R_hat = solve_sync_with_spectral(B, d)
    V = V.reshape((n, d, d))
    assert 0 == pytest.approx(get_error(R_hat, V, d))


@pytest.mark.skip
def test_pure_half_circle():
    n = 600
    d = 3
    p = 0.2  # Probability of getting good matrix

    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    B_noisy = np.copy(B)
    add_noise_to_matrix(B_noisy, d, p)
    W = B_noisy - p * (V @ V.T.conj())
    v = np.linalg.eigvals(W)

    plt.figure(2)

    plt.hist(v, bins=30, label='imaginary')

    plt.title('Histograms of the spectrum matrix W.  n={n}, p={p}'.format(n=n, p=p))
    plt.show()


@pytest.mark.skip
def test_histogram_B_noisy():
    n = 600
    d = 3
    p = 0.04  # Probability of getting good matrix

    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    B_noisy = np.copy(B)
    add_noise_to_matrix(B_noisy, d, p)
    v = np.linalg.eigvals(B_noisy)  # Note: B_noisy = W + \lambda u ut

    plt.figure(2)

    plt.hist(np.real(v), bins=30, label='real')
    plt.hist(np.imag(v), bins=30, label='imaginary')

    plt.title('Histograms of the spectrum matrix W.  n={n}, p={p}'.format(n=n, p=p))
    plt.show()


def test_block_assignment():
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    z = np.zeros((2, 2))
    block_assignment(a, z, 1, 1)
    print(a)


def test_create_noisy_matrix():
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype='complex128')
    add_noise_to_matrix(a, 2, 0.5)
    print(a)
