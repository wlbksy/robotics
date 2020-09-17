import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import robotics as rbt


class TestTransform:
    def test_wrap_2_pi(self):
        assert rbt.wrap_2_pi(0) == pytest.approx(0, 1e-9)
        assert rbt.wrap_2_pi(0.5 * np.pi) == pytest.approx(0.5 * np.pi, 1e-9)
        assert rbt.wrap_2_pi(-0.5 * np.pi) == pytest.approx(-0.5 * np.pi, 1e-9)
        assert rbt.wrap_2_pi(np.pi) == pytest.approx(-np.pi, 1e-9)
        assert rbt.wrap_2_pi(-np.pi) == pytest.approx(-np.pi, 1e-9)
        assert rbt.wrap_2_pi(1.5 * np.pi) == pytest.approx(-0.5 * np.pi, 1e-9)
        assert rbt.wrap_2_pi(-1.5 * np.pi) == pytest.approx(0.5 * np.pi, 1e-9)
        assert rbt.wrap_2_pi(2 * np.pi) == pytest.approx(0, 1e-9)
        assert rbt.wrap_2_pi(-2 * np.pi) == pytest.approx(0, 1e-9)

    def test_rotation2D(self):
        R = rbt.rotation2D(0.1)
        assert_array_almost_equal(
            R, np.array([[0.99500417, -0.09983342], [0.09983342, 0.99500417]])
        )
        assert np.linalg.det(R) == pytest.approx(1.0, 1e-9)
        assert_array_almost_equal(np.linalg.inv(R), R.T)
        assert_array_almost_equal(rbt.rotation2D(-0.1), R.T)

    def test_rotation2D_to_angle(self):
        R = rbt.rotation2D(np.pi / 2)
        assert rbt.rotation2D_to_angle(R) == pytest.approx(np.pi / 2, 1e-9)

        R = rbt.rotation2D(-np.pi / 2)
        assert rbt.rotation2D_to_angle(R) == pytest.approx(-np.pi / 2, 1e-9)

        R = rbt.rotation2D(np.pi)
        assert rbt.rotation2D_to_angle(R) == pytest.approx(np.pi, 1e-9)

    def test_transform2D(self):
        assert_array_almost_equal(
            rbt.transform2D(1, 2, 0.1),
            np.array(
                [
                    [0.99500417, -0.09983342, 1.0],
                    [0.09983342, 0.99500417, 2.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_rotation3D_x(self):
        R = rbt.rotation3D_x(0.1)
        assert_array_almost_equal(
            R,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.99500417, -0.09983342],
                    [0.0, 0.09983342, 0.99500417],
                ]
            ),
        )
        assert_array_almost_equal(np.linalg.inv(R), R.T)

    def test_rotation3D_y(self):
        R = rbt.rotation3D_y(0.1)
        assert_array_almost_equal(
            R,
            np.array(
                [
                    [0.99500417, 0.0, 0.09983342],
                    [0.0, 1.0, 0.0],
                    [-0.09983342, 0.0, 0.99500417],
                ]
            ),
        )
        assert_array_almost_equal(np.linalg.inv(R), R.T)

    def test_rotation3D_z(self):
        R = rbt.rotation3D_z(0.1)
        assert_array_almost_equal(
            R,
            np.array(
                [
                    [0.99500417, -0.09983342, 0.0],
                    [0.09983342, 0.99500417, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        assert_array_almost_equal(np.linalg.inv(R), R.T)

    def test_rotation3D_rpy(self):
        R = rbt.rotation3D_rpy(0.1, 0.2, 0.3)
        assert_array_almost_equal(
            R,
            np.array(
                [
                    [0.93629336, -0.27509585, 0.21835066],
                    [0.28962948, 0.95642509, -0.03695701],
                    [-0.19866933, 0.0978434, 0.97517033],
                ]
            ),
        )
        assert_array_almost_equal(np.linalg.inv(R), R.T)

    def test_transform3D_rpy(self):
        assert_array_almost_equal(
            rbt.transform3D_rpy(1, 2, 3, 0.1, 0.2, 0.3),
            np.array(
                [
                    [0.93629336, -0.27509585, 0.21835066, 1.0],
                    [0.28962948, 0.95642509, -0.03695701, 2.0],
                    [-0.19866933, 0.0978434, 0.97517033, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_rotation3D_to_rpy(self):
        R = rbt.rotation3D_rpy(0.1, 0.2, 0.3)
        roll, pitch, yaw = rbt.rotation3D_to_rpy(R)
        assert roll == pytest.approx(0.1, 1e-9)
        assert pitch == pytest.approx(0.2, 1e-9)
        assert yaw == pytest.approx(0.3, 1e-9)
