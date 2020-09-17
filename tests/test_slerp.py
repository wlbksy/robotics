import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import robotics as rbt


class TestSlerp:
    def test_angle_interpolation(self):
        assert_almost_equal(
            rbt.angle_interpolation(0.75 * np.pi, -0.75 * np.pi, 0.5), -np.pi
        )

    def test_quaternion_slerp(self):
        q0 = rbt.Quaternion(1, 0, 0, 0)
        q1 = rbt.Quaternion(0, 1, 0, 0)
        assert rbt.quaternion_slerp(q0, q1, 0.5) == rbt.Quaternion(
            np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0
        )

        q_list = rbt.quaternion_slerp_array(q0, q1, [0.5, 0.5, 0.5])
        assert q_list[0] == rbt.Quaternion(np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
        assert q_list[1] == rbt.Quaternion(np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
        assert q_list[2] == rbt.Quaternion(np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)

    def test_transform2D_slerp(self):
        T0 = rbt.transform2D(0, 0, 0.75 * np.pi)
        T1 = rbt.transform2D(0, 0, -0.75 * np.pi)

        assert_array_almost_equal(
            rbt.transform2D_slerp(T0, T1, 0.5),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        )

    def test_transform2D_slerp_array(self):
        T0 = rbt.transform2D(0, 0, np.pi * 0.75)
        T1 = rbt.transform2D(0, 0, -np.pi * 0.75)

        T_list = rbt.transform2D_slerp_array(T0, T1, [0.3, 0.5])
        assert_array_almost_equal(
            T_list[0],
            np.array([[-0.951057, -0.309017, 0], [0.309017, -0.951057, 0], [0, 0, 1]]),
        )
        assert_array_almost_equal(
            T_list[1], np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )

    def test_transform3D_slerp(self):
        T0 = rbt.transform3D(0, 0, 0, np.eye(3))
        T1 = rbt.transform3D(10, -10, 10, np.eye(3))

        assert_array_almost_equal(
            rbt.transform3D_slerp(T0, T1, 0.3),
            np.array([[1, 0, 0, 3], [0, 1, 0, -3], [0, 0, 1, 3], [0, 0, 0, 1]]),
        )

    def test_transform3D_slerp_array(self):
        T0 = rbt.transform3D(0, 0, 0, np.eye(3))
        T1 = rbt.transform3D(10, -10, 10, np.eye(3))

        T_list = rbt.transform3D_slerp_array(T0, T1, [0.1, 0.3, 0.5])
        assert_array_almost_equal(
            T_list[0],
            np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]]),
        )
        assert_array_almost_equal(
            T_list[1],
            np.array([[1, 0, 0, 3], [0, 1, 0, -3], [0, 0, 1, 3], [0, 0, 0, 1]]),
        )
        assert_array_almost_equal(
            T_list[2],
            np.array([[1, 0, 0, 5], [0, 1, 0, -5], [0, 0, 1, 5], [0, 0, 0, 1]]),
        )
