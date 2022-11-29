import numpy as np
from tqdm import tqdm


class RSDiffraction_INT():
    def __init__(self) -> None:
        pass

    def __call__(self, E0, z, xvec, yvec, svec, tvec, wavelengths):
        
        k = 2 * np.pi / wavelengths
        xx, yy = np.meshgrid(xvec, yvec)

        LS, LT = len(svec), len(tvec)
        Eout = np.empty((LS, LT), dtype=np.complex)
        for i in tqdm(range(LS), leave=False):
            for j in range(LT):
                r = np.sqrt((svec[i] - xx)**2 + (tvec[j] - yy)**2 + z**2)
                h = -1 / (2 * np.pi) * (1j * k - 1 / r) * np.exp(1j * k * r) * z / r**2
                Eout[i,j] = np.sum(E0 * h)

        return Eout