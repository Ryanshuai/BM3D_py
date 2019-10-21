import itertools


class Schedule:
    def __init__(self):
        self.sigma = [25]

        self.nHard = 16
        self.kHard = 8
        self.NHard = 16
        self.pHard = 3
        self.lambdaHard3D = 2.7
        self.tauMatchHard = 2500 if self.sigma < 35 else 5000
        self.useSD_h = False
        self.tau_2D_hard = 'BIOR'

        self.nWien = 16
        self.kWien = 8
        self.NWien = 32
        self.pWien = 3
        self.tauMatchWien = 400 if self.sigma < 35 else 3500
        self.useSD_w = True
        self.tau_2D_wien = 'DCT'

    def product(self):
        itertools.product(self.sigma, )


if __name__ == '__main__':
    s = Schedule()
    print(vars(s))
