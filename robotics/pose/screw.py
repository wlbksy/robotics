import numpy as np

# skew-symmetric matrix is a square matrix whose transpose equals its negative.
# A.T = -A


def skew2D(v: float) -> np.array:
    """The skew symmetric representation of vector v
    """
    return np.array([[0, -v], [v, 0]])


def skew3D(v: np.array) -> np.array:
    """The skew symmetric representation of vector v
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vex2D(S: np.array) -> float:
    """Convert skew symmetric representation to vector v
    """
    return S[1, 0]


def vex3D(S: np.array) -> np.array:
    """Convert skew symmetric representation to vector v
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])
