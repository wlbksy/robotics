import numpy as np
from matplotlib import pyplot as plt

from ..pose import rotation2D, transform2D


class Triangle:
    def __init__(self, p0=(0.5, 0), p1=(-0.4, 0.3), p2=(-0.4, -0.3), color="-k"):
        self.shape = np.array(
            [[p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [1.0, 1.0, 1.0]]
        )
        self.color = color

    def transform_plot(self, x, y, heading):
        P = transform2D(x, y, heading) @ self.shape

        plt.plot([P[0, 0], P[0, 1]], [P[1, 0], P[1, 1]], self.color)
        plt.plot([P[0, 1], P[0, 2]], [P[1, 1], P[1, 2]], self.color)
        plt.plot([P[0, 2], P[0, 0]], [P[1, 2], P[1, 0]], self.color)
        plt.plot(x, y, "*")


class Car:
    def __init__(
        self,
        color="-k",
        car_length=4.5,
        car_width=2.0,
        rear_axis_to_tail=1.0,
        wheel_base=2.5,
        wheel_length=0.6,
        wheel_width=0.4,
        axis_length=1.4,
    ):
        half_car_width = car_width / 2
        half_wheel_length = wheel_length / 2
        half_wheel_width = wheel_width / 2
        rear_axis_to_front = car_length - rear_axis_to_tail

        self.outline = np.array(
            [
                [
                    rear_axis_to_front,
                    car_length - 1.5 * rear_axis_to_tail,
                    rear_axis_to_front,
                    rear_axis_to_front,
                    -rear_axis_to_tail,
                    -rear_axis_to_tail,
                    rear_axis_to_front,
                ],
                [
                    half_car_width,
                    0,
                    -half_car_width,
                    half_car_width,
                    half_car_width,
                    -half_car_width,
                    -half_car_width,
                ],
            ]
        )

        self.origin_wheel = np.array(
            [
                [
                    half_wheel_length,
                    half_wheel_length,
                    -half_wheel_length,
                    -half_wheel_length,
                    half_wheel_length,
                ],
                [
                    half_wheel_width,
                    -half_wheel_width,
                    -half_wheel_width,
                    half_wheel_width,
                    half_wheel_width,
                ],
            ]
        )
        self.color = color
        self.half_axis = axis_length / 2
        self.wheel_base = wheel_base

    def transform_plot(self, x, y, heading=0.0, steering=0.0):
        rl_wheel = np.copy(self.origin_wheel) + np.array([[0], [self.half_axis]])
        rr_wheel = np.copy(self.origin_wheel) + np.array([[0], [-self.half_axis]])

        fl_wheel = np.copy(self.origin_wheel)

        R_steering = rotation2D(steering)
        fl_wheel = R_steering @ fl_wheel
        fr_wheel = np.copy(fl_wheel)

        fl_wheel += np.array([[self.wheel_base], [self.half_axis]])
        fr_wheel += np.array([[self.wheel_base], [-self.half_axis]])

        R_heading = rotation2D(heading)
        shift_vector = np.array([[x], [y]])
        for shape in [fr_wheel, fl_wheel, self.outline, rr_wheel, rl_wheel]:
            shape = R_heading @ shape
            shape += shift_vector
            plt.plot(shape[0, :], shape[1, :], self.color)

        plt.plot(x, y, "*")
