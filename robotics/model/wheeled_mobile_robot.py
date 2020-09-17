import numpy as np


class UnicycleModel:
    """Generate a unicycle model.

    x: x position in global coordinate frame.
    y: y position in global coordinate frame.
    theta: heading angle in global coordinate frame.
    omega: steering rate.
    v: speed.

    When calling update_xxx functions, dt should be provided,
    which stands for the sampling period for this update.
    """

    __slots__ = (
        "x",
        "y",
        "theta",
        "omega",
        "v",
        "time",
        "x_history",
        "y_history",
        "theta_history",
        "omega_history",
        "v_history",
        "time_history",
    )

    def __init__(self, x=0.0, y=0.0, theta=0.0, omega=0.0, v=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.omega = omega
        self.v = v

        self.time = 0
        self.time_history = [0]
        self.x_history = [x]
        self.y_history = [y]
        self.theta_history = [theta]
        self.omega_history = [omega]
        self.v_history = [v]

    def __update_history(self):
        self.time_history.append(self.time)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.theta_history.append(self.theta)
        self.omega_history.append(self.omega)
        self.v_history.append(self.v)

    def __update_observation(self, theta, v, dt):
        """
        dt: the sampling period.
        """
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = theta
        self.v = v
        self.time += dt

    def update_Euler_by_omega_and_v(self, omega, v, dt):
        """
        omega: steering rate.
        dt: the sampling period.
        """
        self.__update_observation(self.theta, v, dt)
        self.theta += omega * dt
        self.__update_history()

    def update_Euler_by_omega_and_accel(self, omega, accel, dt):
        """
        omega: steering rate.
        accel: acceleration.
        dt: the sampling period.
        """
        self.__update_observation(self.theta, self.v, dt)
        self.theta += omega * dt
        self.v += accel * dt
        self.__update_history()

    def update_Euler_by_alpha_and_accel(self, alpha, accel, dt):
        """
        alpha: steering angular acceleration.
        accel: acceleration.
        dt: the sampling period.
        """
        self.__update_observation(self.theta, self.v, dt)
        self.theta += self.omega * dt
        self.omega += alpha * dt
        self.v += accel * dt
        self.__update_history()

    def __repr__(self):
        return "x: {:.2f}, y: {:.2f}, θ: {:.2f}, v: {:.2f}".format(
            self.x, self.y, self.theta, self.v
        )


class BicycleModel:
    """Generate a bicycle model.

    x: x position in global coordinate frame.
    y: y position in global coordinate frame.
    theta: heading angle in global coordinate frame.
    phi: steering angle in body frame.
    omega: steering rate.
    v: speed.
    L: wheelbase which is the distance between the front and rear wheels.
    max_steering_angle: the maximum absolute value of steering angle.

    When calling update_xxx functions, dt should be provided,
    which stands for the sampling period for this update.
    """

    __slots__ = (
        "x",
        "y",
        "theta",
        "phi",
        "omega",
        "v",
        "L",
        "max_steering_angle",
        "time",
        "x_history",
        "y_history",
        "theta_history",
        "omega_history",
        "v_history",
        "time_history",
        "phi_history",
    )

    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        L=1.0,
        phi=0.0,
        omega=0.0,
        v=0.0,
        max_steering_angle=np.pi / 2,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.phi = phi
        self.omega = omega
        self.v = v
        self.L = L
        self.max_steering_angle = max_steering_angle

        self.time = 0
        self.time_history = [0]
        self.x_history = [x]
        self.y_history = [y]
        self.theta_history = [theta]
        self.phi_history = [phi]
        self.omega_history = [omega]
        self.v_history = [v]

    def __update_observation(self, phi, v, dt):
        phi = np.clip(phi, -self.max_steering_angle, self.max_steering_angle)
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += v * np.tan(phi) / self.L * dt
        self.phi = phi
        self.v = v
        self.time += dt

    def __update_history(self):
        self.time_history.append(self.time)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.theta_history.append(self.theta)
        self.phi_history.append(self.phi)
        self.omega_history.append(self.omega)
        self.v_history.append(self.v)

    def update_Euler_by_phi_and_accel(self, phi, accel, dt):
        """
        phi: steering angle.
        accel: acceleration.
        dt: the sampling period.
        """
        self.__update_observation(phi, self.v, dt)
        self.v += accel * dt
        self.__update_history()

    def update_Euler_by_omega_and_accel(self, omega, accel, dt):
        """
        omega: steering rate.
        accel: acceleration.
        dt: the sampling period.
        """
        self.__update_observation(self.phi, self.v, dt)
        self.phi += omega * dt
        self.v += accel * dt
        self.__update_history()

    def update_Euler_by_alpha_and_accel(self, alpha, accel, dt):
        """
        alpha: steering angular acceleration.
        accel: acceleration.
        dt: the sampling period.
        """
        self.__update_observation(self.phi, self.v, dt)
        self.phi += self.omega * dt
        self.omega += alpha * dt
        self.v += accel * dt
        self.__update_history()

    def __repr__(self):
        return "x: {:.2f}, y: {:.2f}, θ: {:.2f}, φ: {:.2f}, v: {:.2f}".format(
            self.x, self.y, self.theta, self.phi, self.v
        )
