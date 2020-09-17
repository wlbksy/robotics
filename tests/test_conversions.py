import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import robotics as rbt


class TestConversions:
    def test_rotation3D_axis_angle(self):
        axis_z = np.array([0, 0, 1])

        assert_array_almost_equal(np.eye(3), rbt.rotation3D_axis_angle(np.zeros(3)))

        assert_array_almost_equal(
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            rbt.rotation3D_axis_angle(axis_z * np.pi),
        )

        assert_array_almost_equal(
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            rbt.rotation3D_axis_angle(axis_z * np.pi / 2),
        )

    def test_rotation3D_to_axis_angle(self):
        axis_z = np.array([0, 0, 1])
        axis = rbt.rotation3D_to_axis_angle(np.eye(3))

        assert_array_almost_equal(axis, np.zeros(3))

        axis = rbt.rotation3D_to_axis_angle(
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        assert_array_almost_equal(axis, axis_z * np.pi)

        axis = rbt.rotation3D_to_axis_angle(
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )
        assert_array_almost_equal(axis, axis_z * np.pi / 2)
