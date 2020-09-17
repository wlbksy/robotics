from typing import Tuple

import numpy as np


def wrap_2_pi(angle: float) -> float:
    """Wrap given angle to [-π, +π)
    """
    result = (angle + np.pi) % (2 * np.pi) - np.pi
    return result


def rotation_inv(R: np.array) -> np.array:
    """Returns the inverse of the rotation.
    """
    return R.T


def rotation2D(angle: float) -> np.array:
    """Generate a SO(2) matrix R.

    np.linalg.det(R) == 1
    np.linalg.inv(R) == R.T
    rotation2D(-angle) == rotation2D(angle).T
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation2D_to_angle(R: np.array) -> float:
    """Return the angle for the rotation matrix R
    """
    return np.arctan2(R[1, 0], R[0, 0])


def transform2D(x: float, y: float, angle: float) -> np.array:
    """Generate a SE(2) matrix T.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def rotation3D_x(angle: float) -> np.array:
    """Generate the SO(3) rotation matrix about the x axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation3D_y(angle: float) -> np.array:
    """Generate the SO(3) rotation matrix about the y axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation3D_z(angle: float) -> np.array:
    """Generate the SO(3) rotation matrix about the z axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation3D_rpy(roll: float, pitch: float, yaw: float) -> np.array:
    """Generate the ZYX rotation matrix for roll, pitch, yaw
    return rotation3D_z(yaw) @ rotation3D_y(pitch) @ rotation3D_x(roll)
    """

    sr = np.sin(roll)
    cr = np.cos(roll)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def transform3D(x: float, y: float, z: float, R: np.array) -> np.array:
    """Generate a SE(3) matrix T for x, y, z and rotation matrix R.
    """
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:, 3] = [x, y, z, 1.0]

    return T


def transform3D_rpy(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.array:
    """Generate the ZYX transform matrix for x, y, z, roll, pitch, yaw
    """
    sr = np.sin(roll)
    cr = np.cos(roll)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0, 0, 0, 1.0],
        ]
    )


def rotation3D_to_rpy(R: np.array) -> Tuple[float, float, float]:
    """Return the roll, pitch, yaw for the ZYX rotation matrix R
    """
    if abs(1 - R[2, 0]) < 1e-6:
        roll = 0.0
        pitch = -np.arcsin(R[2, 0])
        if R[2, 0] < 0:
            yaw = np.arctan2(-R[0, 1], R[0, 2])
        else:
            yaw = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = -np.arctan(R[2, 0] * np.cos(roll) / R[2, 2])
    return roll, pitch, yaw
