from typing import Tuple

import numpy as np

from .quaternion import Quaternion
from .screw import skew3D, vex3D


def rotation_to_quaternion(m: np.array):
    """Converts a homogeneous rotation matrix to a Quaternion object
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    """
    w = m.trace() + 1

    if w > 1:
        x = m[2, 1] - m[1, 2]
        y = m[0, 2] - m[2, 0]
        z = m[1, 0] - m[0, 1]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        w = m[2, 1] - m[1, 2]
        x = 1 + m[0, 0] - m[1, 1] - m[2, 2]
        y = m[0, 1] + m[1, 0]
        z = m[0, 2] + m[2, 0]
    elif m[1, 1] > m[2, 2]:
        w = m[0, 2] - m[2, 0]
        x = m[0, 1] + m[1, 0]
        y = 1 + m[1, 1] - m[0, 0] - m[2, 2]
        z = m[1, 2] + m[2, 1]
    else:
        w = m[1, 0] - m[0, 1]
        x = m[0, 2] + m[2, 0]
        y = m[1, 2] + m[2, 1]
        z = 1 + m[2, 2] - m[0, 0] - m[1, 1]

    return Quaternion(w, x, y, z).to_unit()


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    r = 0.5 * roll
    p = 0.5 * pitch
    y = 0.5 * yaw

    sr = np.sin(r)
    cr = np.cos(r)
    sp = np.sin(p)
    cp = np.cos(p)
    sy = np.sin(y)
    cy = np.cos(y)

    return Quaternion(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        sr * cp * sy + cr * sp * cy,
        cr * cp * sy - sr * sp * cy,
    )


def axis_angle_to_quaternion(axis: np.array) -> Quaternion:
    angle = np.linalg.norm(axis)
    w = np.cos(angle / 2)
    sin_half_angle = np.sin(angle / 2)
    new_axis = axis / angle * sin_half_angle
    return Quaternion(w, new_axis[0], new_axis[1], new_axis[2])


def quaternion_to_axis_angle(q: Quaternion) -> np.array:
    if not q.is_unit():
        q = q.to_unit()
    angle = 2 * np.arccos(q.w)
    s = np.sqrt(1 - q.w ** 2)
    axis = q.get_vector()
    if s >= 1e-3:
        axis /= s
    return axis * angle


def rotation3D_axis_angle(axis: np.array) -> np.array:
    """Generate the rotation matrix about the axis for angle
    """
    angle = np.linalg.norm(axis)
    if angle == 0:
        return np.eye(3)
    normalized_aixs = axis / angle
    sk = skew3D(normalized_aixs)
    R = np.eye(3) + sk * np.sin(angle) + sk @ sk * (1 - np.cos(angle))
    return R


def rotation3D_to_axis_angle(R: np.array) -> np.array:
    """Return the rotation the axis and angle for rotation matrix
    """
    trace_R = np.trace(R)
    if abs(trace_R - 3) < 1e-6:
        return np.zeros(3)

    if abs(trace_R + 1) < 1e-6:
        diag_R = np.diag(R)
        max_idx = np.argmax(diag_R)
        max_v = diag_R[max_idx]
        I = np.eye(3)
        col = R[:, max_idx] + I[:, max_idx]
        aixs = col / np.sqrt(2 * (1 + max_v))
        return aixs * np.pi

    angle = np.arccos((trace_R - 1) / 2)
    skw = (R - R.T) / 2 / np.sin(angle)
    axis = vex3D(skw)
    return axis * angle


def rpy_to_axis_angle(roll: float, pitch: float, yaw: float) -> np.array:
    q = rpy_to_quaternion(roll, pitch, yaw)

    angle = 2 * np.arccos(q.w)
    vec = q.get_vector()
    norm_square = vec @ vec

    if norm_square < 1e-3:
        # when all euler angles are zero angle =0 so
        # we can set axis to anything to avoid divide by zero
        return np.array([angle, 0, 0])

    norm = np.sqrt(norm_square)
    return vec * angle / norm


def axis_angle_to_rpy(axis: np.array) -> np.array:
    q = axis_angle_to_quaternion(axis)

    return q.to_rpy()
