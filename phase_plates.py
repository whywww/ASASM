import numpy as np
from utils import effective_bandwidth
import cv2


# class VortexPlate():
#     def __init__(sedlf, Nx, Ny, m=1) -> None:
        
#         self._build_plate(Nx, Ny, m)
    

#     def _build_plate(self, Nx, Ny, m):

#         x = np.linspace(-Nx // 2, Nx // 2, Nx)
#         y = np.linspace(-Ny // 2, Ny // 2, Ny)
#         xx, yy = np.meshgrid(x, y, indexing='xy')
#         self.plate = np.remainder(np.arctan2(yy, xx) * m, 2 * np.pi)


#     def forward(self, Uin):

#         tf = np.exp(1j * self.plate)
#         Uout = Uin * tf

#         return Uout


class CubicPhasePlate():
    def __init__(self, k, r, m=1e+3) -> None:
        
        self.m = m
        self.k = k
        self.r = r
        self.fcX = self.fcY = k / (2 * np.pi) * 3 / 2 * m * r**2
    

    def forward(self, E0, xi_, eta_, compensate):

        plate = self.k * (xi_**3 + eta_**3) * self.m
        E = np.exp(1j * plate)

        if compensate:
            E *= np.exp(1j * 2 * np.pi * (- self.fcX * xi_ - self.fcY * eta_))

        return E0 * E

    
    def grad_symm(self, xi, eta):
        '''
        Computes gradient of phi_symm / pi given a value of xi
        phi_u = k*m*(xi**3 + eta**3)
        phi_comp = - 2*pi*fc*xi
        phi_symm = phi_u + phi_comp 
        '''

        grad_uX = 3 * self.k * self.m * xi**2
        grad_compX = - 2 * np.pi * self.fcX
        gradientX = grad_uX + grad_compX

        grad_uY = 3 * self.k * self.m * eta**2
        grad_compY = - 2 * np.pi * self.fcY
        gradientY = grad_uY + grad_compY

        return gradientX, gradientY

    
    def grad_nonsymm(self, xi, eta):
        '''
        Computes gradient of phi_u / pi given a value of xi
        '''

        grad_uX = 3 * self.k * self.m * xi**2
        grad_uY = 3 * self.k * self.m * eta**2

        return grad_uX, grad_uY

    
class SphericalWave():
    def __init__(self, k, x0, y0, z0, angles, zf) -> None:
        
        self.k = k
        thetaX, thetaY = angles
        self.fcX = - np.sin(thetaX / 180 * np.pi) * k / (2 * np.pi)
        self.fcY = - np.sin(thetaY / 180 * np.pi) * k / (2 * np.pi)
        self.x0, self.y0, self.z0 = x0, y0, z0 
        self.zf = zf


    def forward(self, E0, xi_, eta_, compensate):
        ''' 
        Get the phase shift of the spherical wave from a single point source 
        
        :param x0, y0: spatial coordinate of the source point
        :param to_xx, to_yy: coordinate grid at the destination plane
        :param distance: scalar tensor, travel distance
        :return: the spherical wave at destination
        '''

        radius = np.sqrt(self.z0**2 + (xi_ - self.x0)**2 + (eta_ - self.y0)**2)
        phase = self.k * radius
        amplitude = 1 / radius

        E = amplitude * np.exp(1j * phase)

        if compensate:
            E *= np.exp(1j * 2 * np.pi * (-self.fcX * xi_ - self.fcY * eta_))

        return E0 * E

    
    def grad_symm(self, xi, eta):

        denom = np.sqrt((xi - self.x0)**2 + (eta - self.y0)**2 + self.z0**2)
        grad_uX = self.k * (xi - self.x0) / denom
        grad_uY = self.k * (eta - self.y0) / denom

        grad_linearX = 2 * np.pi * self.fcX
        grad_linearY = 2 * np.pi * self.fcY

        gradientX = grad_uX - grad_linearX
        gradientY = grad_uY - grad_linearY

        return gradientX, gradientY

    
    def grad_nonsymm(self, xi, eta):

        denom = np.sqrt((xi - self.x0)**2 + (xi - self.y0)**2 + self.z0**2)
        grad_uX = self.k * (xi - self.x0) / denom
        grad_uY = self.k * (eta - self.y0) / denom

        return grad_uX, grad_uY

    
class PlaneWave():
    def __init__(self, k, r, x0, y0, z0) -> None:
        
        self.k = k
        self.r = r
        self.fcX = self.fcY = 0
        self.x0, self.y0, self.z0 = x0, y0, z0 


    def forward(self, E0, xi_, eta_, compensate):

        vec = np.array([-self.x0, -self.y0, self.z0])
        kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))
        phase = self.k * (kx * xi_ + ky * eta_ + kz)

        return E0 * np.exp(1j * phase)


    def grad_symm(self, xi, eta):

        return 0, 0


class ThinLens():
    def __init__(self, k, f) -> None:
        
        self.k = k
        self.f = f
        self.fcX = self.fcY = 0


    def forward(self, E0, xi_, eta_, compensate):

        phase = self.k / 2 * (-1 / self.f) * (xi_**2 + eta_**2)

        return E0 * np.exp(1j * phase)


    def grad_symm(self, xi, eta):

        grad_uX = -self.k / self.f * xi
        grad_uY = -self.k / self.f * eta

        return grad_uX, grad_uY


    def grad_nonsymm(self, xi, eta):

        grad_uX = -self.k / self.f * xi
        grad_uY = -self.k / self.f * eta
        
        return grad_uX, grad_uY
    

class Diffuser():
    def __init__(self, r, interpolation='nearest') -> None:
        '''
        Two types of diffusers: nearest or linear interpolated
        '''
        
        self.fcX = self.fcY = 0
        self.pitch = r / 10
        self.N = int(r * 2 / self.pitch)
        np.random.seed(0)
        self.plate = np.random.rand(self.N, self.N)
        self.interp = interpolation


    def forward(self, E0, xi_, eta_, compensate):

        if self.interp == 'nearest':
            plate_sample = cv2.resize(self.plate, xi_.shape, interpolation=cv2.INTER_NEAREST)
        else:
            plate_sample = cv2.resize(self.plate, xi_.shape, interpolation=cv2.INTER_LINEAR)

        phase = plate_sample * 4 * np.pi

        return E0 * np.exp(1j * phase)


    def grad_symm(self, xi, eta):

        grad_max = np.pi * effective_bandwidth(self.pitch, is_plane_wave = True)  # nearest interpolation

        if self.interp == 'linear':
            grad_max += 4 * np.pi / self.pitch  # linear interpolation

        return grad_max, grad_max

    
class TestPlate():
    def __init__(self, r) -> None:
        
        self.c = 41.2/(r*2)**2 * 2 * np.pi
        # self.c = 41.2 / 3 * 2**5 * np.pi / (r*2)**6
        self.fcX = self.fcY = 0
        self.r = r


    def forward(self, E0, xi_, eta_, compensate):

        phase = self.c * xi_ * eta_
        # phase = self.c * xi_**3 * eta_**3
        return E0 * np.exp(1j * phase)


    def grad_symm(self, xi, eta):

        # plane = np.pi * effective_bandwidth(self.r*2, is_plane_wave = True)

        return self.c * xi, self.c * eta
        # return 3 * self.c * xi**2 * eta**3 + plane, 3 * self.c * xi**3 * eta**2 + plane