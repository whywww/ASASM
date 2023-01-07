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