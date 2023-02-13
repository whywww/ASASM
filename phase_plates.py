import numpy as np


class VortexPlate():
    def __init__(self, Nx, Ny, m=1) -> None:
        
        self._build_plate(Nx, Ny, m)
    

    def _build_plate(self, Nx, Ny, m):

        x = np.linspace(-Nx // 2, Nx // 2, Nx)
        y = np.linspace(-Ny // 2, Ny // 2, Ny)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        self.plate = np.remainder(np.arctan2(yy, xx) * m, 2 * np.pi)


    def forward(self, Uin):

        tf = np.exp(1j * self.plate)
        Uout = Uin * tf

        return Uout



class CubicPhasePlate():
    def __init__(self, xi, eta, k, s, m=1e+4) -> None:
        
        r = abs(xi).max()
        self._build_plate(xi, eta, k, m)
        self.fc = k / (2 * np.pi) * 3 / 2 * m * r**2
        self.fb = self.fc * 2 * s
    

    def _build_plate(self, xi, eta, k, m):

        xx, yy = np.meshgrid(xi, eta, indexing='xy')
        self.plate = k * (xx**3 + yy**3) * m


    def forward(self, Uin):

        tf = np.exp(1j * self.plate)
        Uout = Uin * tf

        return Uout

    
    def new_field(self):

        return np.exp(1j * self.plate)