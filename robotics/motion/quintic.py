class QuinticPolynomial:
    """A quintic polynomial is defined as
    f(x) = c0 * x^5 + c1 * x^4 + c2 * x^3 + c3 * x^2 + c4 * x + c5,

    where coefficients is denoted as [c0, c1, c2, c3, c4, c5]
    """

    def __init__(self, x, coefficients):
        self.x = x
        self.c = coefficients

    def get_value(self) -> float:
        v = 0.0
        for i in range(6):
            v *= self.x
            v += self.c[i]
        return v

    def get_first_derivative(self) -> float:
        v = 0.0
        for i in range(5):
            v *= self.x
            v += (5 - i) * self.c[i]
        return v

    def get_second_derivative(self) -> float:
        v = 0.0
        for i in range(4):
            v *= self.x
            v += (5 - i) * (4 - i) * self.c[i]
        return v
