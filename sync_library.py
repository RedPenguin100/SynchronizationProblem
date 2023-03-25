import numpy as np
from numba import njit


class Problem:
    mra = "mra"
    rotation2d = "rotation2d"


def get_projection(X, d=3, problem=Problem.mra):
    if problem == Problem.mra:
        return get_mra_projection(X)
    if problem == Problem.rotation2d:
        return get_so_projection(X, d)
    raise NotImplemented()


@njit
def get_n_roll_matrix(d, n):
    mat = np.zeros((d, d))
    for i in range(d):
        mat[i, (i - n) % d] = 1  # Compatible with modulo numbers
    return mat


@njit
def get_mra_projection(X):
    shape = np.shape(X)
    if len(shape) != 2:
        raise ValueError("Shape should be 2-->matrix!")
    d1, d = shape
    if d1 != d:
        raise ValueError("Matrix should be square!")

    min_error = np.inf
    min_matrix = None

    for i in range(d):
        mat = get_n_roll_matrix(d, i)
        error = np.linalg.norm(X - mat)
        if error < min_error:
            min_error = error
            min_matrix = mat

    return min_matrix


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
    S[-1, -1] = np.conj(np.linalg.det(U @ Vt))
    # The conj() might not be necessary for the real matrices
    # but is absolutely-must for the complex matrices.
    return (U @ S) @ Vt


def _get_minimizer_mra(expected, actual):
    n, d, d2 = expected.shape
    expected = expected.reshape((n * d, d))
    actual = actual.reshape((n * d, d))

    return get_mra_projection(actual.T @ expected)


def get_minimizer(expected, actual, problem):
    if problem == Problem.mra:
        return _get_minimizer_mra(expected, actual)
    if problem == Problem.rotation2d:
        return _get_minimizer_so(expected, actual)
    raise ValueError("Unknown problem ", problem)


def _get_minimizer_so(expected, actual):
    n, d, d2 = expected.shape
    expected = expected.reshape((n * d, d))
    actual = actual.reshape((n * d, d))

    return get_so_projection(actual.T @ expected, d)


def get_error(expected, actual, dim, problem=Problem.rotation2d):
    # Validation logic
    assert expected.shape == actual.shape, "Dimension mismatch!"
    assert len(expected.shape) == 3
    n, d, d2 = expected.shape
    assert d == dim
    assert d2 == dim

    # Error retrieving logic
    Q = get_minimizer(expected.conj(), actual, problem=Problem.rotation2d)
    print("Q", Q)
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


def solve_sync_with_spectral(data, d, weights=None, problem=Problem.mra):
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
        assert np.abs(data[i, i] - 1) < 1e-8

    # Actual derivation
    if weights is not None:
        D = create_d_matrix(d, weights)
        data = np.linalg.inv(D) @ data

    w, v = np.linalg.eig(data)
    v_args = np.argsort(w)[-d:]
    print("Before reshape: ", v[:, v_args])
    V_hat = v[:, v_args].reshape((n, d, d))

    R_hat = np.empty_like(V_hat)

    if problem == Problem.rotation2d:
        for i in range(n):
            R_hat[i] = get_projection(V_hat[i], d, problem=Problem.rotation2d)
    elif problem == Problem.mra:
        baseline = get_projection(V_hat[0], d, problem=Problem.rotation2d)
        base_inv = np.linalg.inv(baseline)  # This is so we can get to the MRA matrices.
        for i in range(n):
            R_hat[i] = get_projection(get_projection(V_hat[i], d, problem=Problem.rotation2d) @ base_inv, d=d, problem=Problem.mra)
    else:
        raise ValueError(f"Unknown problem {problem}")

    return R_hat


def truly_random_matrix(d, problem):
    if problem == Problem.mra:
        return truly_random_mra_matrix(d)
    if problem == Problem.rotation2d:
        return truly_random_so_matrix(d)
    return None


@njit
def truly_random_mra_matrix(d):
    roll = np.random.randint(d)
    print("roll=", roll)
    return get_n_roll_matrix(d, roll)


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
