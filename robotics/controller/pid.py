class PID:
    __slots__ = ("P", "I", "D", "kP", "kI", "kD", "output", "last_error", "last_time")

    def __init__(self, kP=0.0, kI=0.0, kD=0.0):
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0

        self.kP = kP
        self.kI = kI
        self.kD = kD

        self.output = 0.0
        self.last_error = 0.0
        self.last_time = None

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

        dt = t - self.last_time
        if dt <= 0.0:
            return

        self.I += current_error * dt
        self.D = (current_error - self.last_error) / dt
        self.last_time = t

    def __compute_output(self):
        self.output = self.kP * self.P + self.kI * self.I + self.kD * self.D

    def get_output(self, current_error, t):
        self.__update(current_error, t)
        self.__compute_output()
        return self.output
