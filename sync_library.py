import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def get_so_projection(X, d=3):
    """
    :param X: matrix to get projection of
    :param d: dimension of matrix
    """
    U, sigma, Vt = np.linalg.svd(X)
    # The dtype conversion is absolutely necessary because
    # the numerical error is too large otherwise.
    S = np.zeros((d, d), dtype=X.dtype)
    np.fill_diagonal(S, 1)
    S[-1][-1] = np.conj(np.linalg.det(U @ Vt))
    # The conj() might not be necessary for the real matrices
    # but is absolutely-must for the complex matrices.
    return (U @ S) @ Vt


def _get_minimizer(expected, actual):
    n, d, d2 = expected.shape
    expected = expected.reshape((n * d, d))
    actual = actual.reshape((n * d, d))

    return get_so_projection(actual.T @ expected, d)


def get_error(expected, actual, dim=3):
    # Validation logic
    assert expected.shape == actual.shape, "Dimension mismatch!"
    assert len(expected.shape) == 3
    n, d, d2 = expected.shape
    assert d == dim
    assert d2 == dim

    # Error retrieving logic
    Q = _get_minimizer(expected.conj(), actual)
    error = 0
    for i in range(n):
        # The addition of `conj` to expected[i] is crucial for the complex case.
        error += np.linalg.norm(expected[i] - actual[i] @ Q.conj()) ** 2

    return error


@njit
def create_d_matrix(d: int, weights: np.ndarray):
    assert len(weights.shape) == 2
    assert weights.shape[0] == weights.shape[1]

    n = weights.shape[0]
    i_matrix = np.eye(d)

    D = np.zeros((d * n, d * n))
    for i in range(n):
        D[d * i:d * i + d, d * i:d * i + d] = i_matrix * np.sum(weights[i, :])

    return D


def solve_sync_with_spectral(data, d, weights=None):
    """
    :param data: Graph matrix containing description of all nodes
    :param d: Dimension of rotation
    :param weights: if weights exist, display confidence in data
    """
    # Assertions
    assert len(data.shape) == 2
    nd, nd2 = data.shape
    assert nd == nd2
    n = nd // d
    assert n * d == nd
    if weights is not None:
        assert weights.shape == (n, n)

    for i in range(n):
        assert np.isclose(data[i,i], 1)

    # Actual derivation
    if weights is not None:
        D = create_d_matrix(d, weights)
        data = np.linalg.inv(D) @ data

    w, v = np.linalg.eig(data)
    v_args = np.argsort(w)[-d:]
    V_hat = v[:, v_args].reshape((n, d, d))

    R_hat = np.copy(V_hat)
    for i in range(n):
        R_hat[i] = get_so_projection(V_hat[i], d)
    return R_hat


@njit
def truly_random_so_matrix(d):
    """
    :param d: dimension
    """
    z = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(z)
    D = np.empty(d, dtype=np.complex128)
    for i in range(d):
        D[i] = R[i, i]
    normalizing_matrix = D / np.absolute(D)
    Q_tag = np.multiply(Q, normalizing_matrix)
    Q_tag = Q_tag.astype('complex_')
    return get_so_projection(Q_tag, d)


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


@njit
def block_assignment(wall, block, i, j):
    """
    Insert a `block` matrix to `wall` in position `i,j` with respect
    to the blocks, i.e if `wall` has 4 block matrices of 2x2 then i,j runs through 0,1 and NOT
    0,1,2,3.
    """
    # Validations
    assert len(wall.shape) == 2
    nd, nd2 = wall.shape
    assert nd == nd2

    assert len(block.shape) == 2
    d, d2 = block.shape
    assert d == d2

    n = nd // d
    assert n * d == nd

    # Actual logic
    wall[d * i: d * (i + 1), d * j:d * (j + 1)] = block


@njit
def add_noise_to_matrix(H, d, p):
    """
    H - the matrix we want to add noise to

    """
    # Validations
    assert len(H.shape) == 2
    nd, nd2 = H.shape
    assert nd == nd2
    n = nd // d
    assert n * d == nd

    # Actual logic
    index_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if 1 == np.random.binomial(1, 1 - p):
                index_list.append((i, j))

    for i, j in index_list:
        block_assignment(H, truly_random_so_matrix(d), i, j)


@njit
def add_holes_to_matrix(H, d, p):
    """
    H - the matrix we want to add noise to
    p - concentration of non-holes
    d - dimension of rotation
    :return: indexes of newly added holes
    """
    # Validations
    assert len(H.shape) == 2
    nd, nd2 = H.shape
    assert nd == nd2
    n = nd // d
    assert n * d == nd

    # Actual logic
    index_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if 1 == np.random.binomial(1, 1 - p):
                index_list.append((i, j))

    zeros = np.zeros((d, d))
    for i, j in index_list:
        block_assignment(H, zeros, i, j)

    return index_list
