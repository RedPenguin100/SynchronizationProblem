import pytest
import numpy as np
from matplotlib import pyplot as plt

from sync_library import get_so_projection, get_error, solve_sync_with_spectral, truly_random_so_matrix, \
    block_assignment, add_noise_to_matrix, add_holes_to_matrix, create_d_matrix, Problem, truly_random_matrix


def haar_measure_sampling_evidence(n):
    """
    Choose a high dimension(~1000+) so that it is more visible.
    :note: computation might take a while.
    """
    M = truly_random_so_matrix(n)
    w, v = np.linalg.eig(M)
    count, bins, ignored = plt.hist(np.angle(w), 15, density=True)
    density = np.ones_like(bins) / (2 * np.pi)
    plt.plot(bins, density, linewidth=2, color='r')
    plt.show()


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
    # Projection of x should be same as x in SO(3)
    assert projection == pytest.approx(X)

    assert np.eye(3) == pytest.approx(projection.T.conj() @ projection)
    assert 1 == pytest.approx(np.linalg.det(projection))


def test_projection_matrix_not_rotation():
    X = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    projection = get_so_projection(X)
    assert projection == pytest.approx(np.eye(3))


@pytest.mark.parametrize('problem', [Problem.mra, Problem.rotation])
def test_error_sanity(problem):
    d = 3

    rot1 = truly_random_matrix(d, problem)
    rot2 = truly_random_matrix(d, problem)

    rot = np.array((rot1, rot2))
    rot_dis = np.array((rot1, rot2))

    assert 0 == pytest.approx(get_error(expected=rot, actual=rot_dis, dim=d, problem=problem))


def test_error_noisy():
    dimension = 3

    rot1 = truly_random_so_matrix(dimension)
    rot2 = truly_random_so_matrix(dimension)

    rot1_disturbed = rot1 + np.random.normal(0, 0.1, (dimension, dimension))
    rot2_disturbed = rot2 + np.random.normal(0, 0.1, (dimension, dimension))
    rot = np.array((rot1, rot2))
    rot_dis = np.array((rot1_disturbed, rot2_disturbed))

    assert get_error(expected=rot, actual=rot_dis, dim=dimension) < 0.4


@pytest.mark.parametrize('times', range(10))
@pytest.mark.parametrize('n', [10, 50, 100])
@pytest.mark.parametrize('d', [2, 3])
def test_eigenvalue(times, n, d):
    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()

    assert n * V == pytest.approx(B @ V)

    R_hat = solve_sync_with_spectral(B, d, problem=Problem.rotation)
    V = V.reshape((n, d, d))
    assert 0 == pytest.approx(get_error(R_hat, V, d))


@pytest.mark.parametrize('n', [100])
@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('p', [0.2])
def test_partial_graph_eigenvalue(n, d, p):
    # Setup
    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    B_partial = np.copy(B)
    hole_indexes = add_holes_to_matrix(B_partial, d, p)
    # Weights
    ones = np.ones((n, n))
    for index in hole_indexes:
        ones[index] = 0
    # logic
    R_hat = solve_sync_with_spectral(B_partial, d, ones, problem=Problem.rotation)
    V = V.reshape((n, d, d))
    assert 0 == pytest.approx(get_error(R_hat, V, d))


@pytest.mark.parametrize('n', [100])
@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('p', [0.8])
def test_graph_with_false_measurements_eigenvalue(n, d, p):
    # Setup
    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    B_partial = np.copy(B)
    add_noise_to_matrix(B_partial, d, p)

    # logic
    R_hat = solve_sync_with_spectral(B_partial, d, problem=Problem.rotation)
    V = V.reshape((n, d, d))
    total_error = get_error(R_hat, V, d)
    average_error = total_error / n
    assert 0 == pytest.approx(average_error, abs=0.02)


def test_D_matrix():
    weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    D = create_d_matrix(3, weights)
    assert (2 * np.eye(3 * 3) == D).all()


def test_spectral_validity_dimension():
    n = 5
    d = 3

    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()

    ones = np.ones((n, n))

    # This should not throw
    solve_sync_with_spectral(B, d, ones, problem=Problem.rotation)

    with pytest.raises(AssertionError):
        solve_sync_with_spectral(B, d, ones[1:], problem=Problem.rotation)


@pytest.mark.skip
def test_pure_half_circle():
    n = 100
    d = 3
    p = 0.2  # Probability of getting good matrix

    stack = [truly_random_so_matrix(d) for _ in range(n)]
    V = np.vstack(stack)
    B = V @ V.T.conj()
    B_noisy = np.copy(B)
    add_noise_to_matrix(B_noisy, d, p)
    W = B_noisy - p * (V @ V.T.conj())
    v = np.linalg.eigvals(W)
    b_noisy_eigenvalues = np.linalg.eigvals(B_noisy)

    plt.figure(2)

    plt.hist(b_noisy_eigenvalues, bins=30, label='imaginary')

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
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.float64)
    z = np.zeros((2, 2))
    block_assignment(a, z, 1, 1)
    print(a)


def test_create_noisy_matrix():
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.complex128)
    add_noise_to_matrix(a, 2, 0.5)
    print(a)
