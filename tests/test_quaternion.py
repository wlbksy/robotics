import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import robotics as rbt


class TestQuaternion:
    def test_vanilla(self):
        q1 = rbt.Quaternion(1, 2, 3, 4)
        assert q1 == q1
        q2 = rbt.Quaternion(2, 3, 4, 5)
        assert q1 + q2 == rbt.Quaternion(3, 5, 7, 9)
        assert q2 - q1 == rbt.Quaternion(1, 1, 1, 1)
        assert 2 * q1 == rbt.Quaternion(2, 4, 6, 8)
        assert q1 * 2 == rbt.Quaternion(2, 4, 6, 8)
        assert q1 - 1 == rbt.Quaternion(0, 2, 3, 4)
        assert q1 + 1 == rbt.Quaternion(2, 2, 3, 4)
        assert 1 + q1 == rbt.Quaternion(2, 2, 3, 4)
        assert 1 - q1 == rbt.Quaternion(0, -2, -3, -4)

        assert_array_almost_equal(
            q1.to_left_matrix() @ q2.to_right_matrix(),
            q2.to_right_matrix() @ q1.to_left_matrix(),
        )

    def test_non_unit_quaternion(self):
        q = rbt.Quaternion(1, 2, 3, 4)
        q_c = q.conj()
        assert q_c.w == q.w
        assert q_c.x == -q.x
        assert q_c.y == -q.y
        assert q_c.z == -q.z
        assert q.norm_square() == 30
        assert q.to_unit().is_unit()
        assert rbt.vector_to_quaternion(q.to_vector()) == q
        assert_array_almost_equal(
            q.to_rotation(),
            np.array(
                [
                    [-0.666667, 0.133333, 0.733333],
                    [0.666667, -0.333333, 0.666667],
                    [0.333333, 0.933333, 0.133333],
                ]
            ),
        )
        r, p, y = rbt.rpy_to_quaternion(0.1, 0.2, 0.3).to_rpy()
        assert_almost_equal(r, 0.1)
        assert_almost_equal(p, 0.2)
        assert_almost_equal(y, 0.3)

        R = rbt.rotation3D_rpy(0.1, 0.2, 0.3)
        r, p, y = rbt.rotation_to_quaternion(R).to_rpy()
        assert_almost_equal(r, 0.1)
        assert_almost_equal(p, 0.2)
        assert_almost_equal(y, 0.3)

        assert_array_almost_equal(
            q.rotate_vector(np.array([1, 2, 3])), np.array([1.8, 2.0, 2.6])
        )

    def test_unit_quaternion(self):
        q = rbt.Quaternion(np.sqrt(2) / 4, np.sqrt(2) / 4, 0.5, np.sqrt(2) / 2)
        assert q.is_unit()
        assert q.conj().w == q.w
        assert q.conj().x == -q.x
        assert q.conj().y == -q.y
        assert q.conj().z == -q.z
