import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import robotics as rbt


class TestScrew:
    def test_screw(self):
        assert_array_almost_equal(rbt.skew2D(1), np.array([[0, -1], [1, 0]]))

        assert_array_almost_equal(
            rbt.skew3D(np.array([1, 2, 3])),
            np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]),
        )

        assert_array_almost_equal(
            rbt.skew3D(np.array([1, 2, 3])).transpose(),
            -rbt.skew3D(np.array([1, 2, 3])),
        )

    def test_vex(self):
        assert rbt.vex2D(np.array([[0, -1], [1, 0]])) == 1

        assert_array_almost_equal(
            rbt.vex3D(np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])),
            np.array([1, 2, 3]),
        )
