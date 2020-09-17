import numpy as np


class PIDClamping:
    __slots__ = (
        "P",
        "I",
        "D",
        "kP",
        "kI",
        "kD",
        "output",
        "last_error",
        "last_time",
        "last_I",
        "delta_I",
        "max_output",
        "min_output",
        "dt",
    )

    def __init__(self, kP=0.0, kI=0.0, kD=0.0, max_output=np.inf, min_output=None):
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0

        self.kP = kP
        self.kI = kI
        self.kD = kD

        self.last_I = 0.0
        self.delta_I = 0.0

        self.max_output = max_output
        if min_output is None:
            self.min_output = -max_output
        else:
            self.min_output = min_output

        self.output = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.dt = 0

    def __update(self, current_error, t):
        """
        current_error: the difference between a desired setpoint SP and
            a measured process variable PV.
        t: the sampling timestamp in seconds.
        """
        self.P = current_error
        self.last_error = current_error
        if self.last_time is None:
            self.last_time = t
            return

        self.dt = t - self.last_time
        if self.dt <= 0.0:
            return

        self.last_I = self.I
        self.delta_I = current_error * self.dt
        self.I += self.delta_I
        self.D = (current_error - self.last_error) / self.dt
        self.last_time = t

    def __compute_output(self):
        u = self.kP * self.P + self.kI * self.I + self.kD * self.D
        if u > self.max_output or u < self.min_output or np.sign(u) == np.sign(self.P):
            self.I = self.last_I
            u -= self.kI * self.delta_I
        self.output = u

    def get_output(self, current_error, t):
        self.__update(current_error, t)
        self.__compute_output()
        return self.output
