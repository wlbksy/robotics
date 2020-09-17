from typing import List

import numpy as np

from .conversions import rotation_to_quaternion
from .quaternion import Quaternion, vector_to_quaternion
from .transform import (
    rotation2D,
    rotation2D_to_angle,
    transform2D,
    transform3D,
    wrap_2_pi,
)


def angle_interpolation(angle0: float, angle1: float, ratio: float) -> float:
    return wrap_2_pi(angle0 + wrap_2_pi(angle1 - angle0) * ratio)


def quaternion_slerp(q0: Quaternion, q1: Quaternion, ratio: float) -> Quaternion:
    """Slerp is shorthand for spherical linear interpolation
    https://en.wikipedia.org/wiki/Slerp
    """
    v0 = q0.to_vector()
    v1 = q1.to_vector()
    dot_v0_v1 = np.dot(v0, v1)

    if dot_v0_v1 < 0.0:
        v1 *= -1
        dot_v0_v1 *= -1

    v = None
    DOT_THRESHOLD = 0.9995
    if dot_v0_v1 > DOT_THRESHOLD:
        v = v0 + ratio * (v1 - v0)
    else:
        theta_0 = np.arccos(dot_v0_v1)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * ratio
        sin_theta = np.sin(theta)

        s1 = sin_theta / sin_theta_0
        s0 = np.cos(theta) - dot_v0_v1 * s1
        v = s0 * v0 + s1 * v1

    return vector_to_quaternion(v).to_unit()


def quaternion_slerp_array(
    q0: Quaternion, q1: Quaternion, ratio_list
) -> List[Quaternion]:
    """Slerp is shorthand for spherical linear interpolation
    https://en.wikipedia.org/wiki/Slerp
    """
    ratio_array = np.array(ratio_list).reshape(-1, 1)
    v0 = q0.to_vector()
    v1 = q1.to_vector()
    dot_v0_v1 = np.dot(v0, v1)

    if dot_v0_v1 < 0.0:
        v1 *= -1
        dot_v0_v1 *= -1

    v = None
    DOT_THRESHOLD = 0.9995
    if dot_v0_v1 > DOT_THRESHOLD:
        v = v0 + np.outer(ratio_array, v1 - v0)
    else:
        theta_0 = np.arccos(dot_v0_v1)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * ratio_array
        sin_theta = np.sin(theta)

        s1 = sin_theta / sin_theta_0
        s0 = np.cos(theta) - dot_v0_v1 * s1
        v = np.outer(s0, v0) + np.outer(s1, v1)

    return [vector_to_quaternion(r).to_unit() for r in v]


def transform3D_slerp(T0: np.array, T1: np.array, ratio: float) -> np.array:
    # rotation
    r0 = T0[:3, :3]
    r1 = T1[:3, :3]
    # translation
    l0 = T0[:3, 3]
    l1 = T1[:3, 3]

    q0 = rotation_to_quaternion(r0)
    q1 = rotation_to_quaternion(r1)
    q = quaternion_slerp(q0, q1, ratio)
    r = q.to_rotation()
    l = l0 * (1 - ratio) + l1 * ratio

    return transform3D(l[0], l[1], l[2], r)


def transform3D_slerp_array(T0: np.array, T1: np.array, ratio_list) -> List[np.array]:
    # rotation
    r0 = T0[:3, :3]
    r1 = T1[:3, :3]
    # translation
    l0 = T0[:3, 3]
    l1 = T1[:3, 3]

    q0 = rotation_to_quaternion(r0)
    q1 = rotation_to_quaternion(r1)
    q = quaternion_slerp_array(q0, q1, ratio_list)
    ratio_array = np.array(ratio_list).reshape(-1, 1)
    l = l0 * (1 - ratio_array) + l1 * ratio_array
    result = [
        transform3D(l_i[0], l_i[1], l_i[2], q_i.to_rotation()) for l_i, q_i in zip(l, q)
    ]
    return result


def transform2D_slerp(T0: np.array, T1: np.array, ratio: float) -> np.array:
    # rotation
    r0 = T0[:2, :2]
    r1 = T1[:2, :2]
    # translation
    l0 = T0[:2, 2]
    l1 = T1[:2, 2]

    angle0 = rotation2D_to_angle(r0)
    angle1 = rotation2D_to_angle(r1)
    a = angle_interpolation(angle0, angle1, ratio)
    l = l0 * (1 - ratio) + l1 * ratio

    return transform2D(l[0], l[1], a)


def transform2D_slerp_array(T0: np.array, T1: np.array, ratio_list) -> List[np.array]:
    # rotation
    r0 = T0[:2, :2]
    r1 = T1[:2, :2]
    # translation
    l0 = T0[:2, 2]
    l1 = T1[:2, 2]

    angle0 = wrap_2_pi(rotation2D_to_angle(r0))
    angle1 = wrap_2_pi(rotation2D_to_angle(r1))
    ratio_array = np.array(ratio_list).reshape(-1, 1)
    l = l0 * (1 - ratio_array) + l1 * ratio_array
    result = [
        transform2D(l_i[0], l_i[1], angle_interpolation(angle0, angle1, ratio))
        for l_i, ratio in zip(l, ratio_array[:, 0])
    ]
    return result
