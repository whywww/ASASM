"""
This script includes the modulations and components of the input field.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

    
Technical Paper:
Haoyu Wei, Xin Liu, Xiang Hao, Edmund Y. Lam, and Yifan Peng, "Modeling off-axis diffraction with the least-sampling angular spectrum method," Optica 10, 959-962 (2023)
"""


import numpy as np
from utils import effective_bandwidth
import cv2

    
class SphericalWave():
    def __init__(self, k, x0, y0, z0, angles, zf) -> None:
        
        self.k = k
        thetaX, thetaY = angles
        self.fcX = - np.sin(thetaX / 180 * np.pi) * k / (2 * np.pi)
        self.fcY = - np.sin(thetaY / 180 * np.pi) * k / (2 * np.pi)
        self.x0, self.y0, self.z0 = x0, y0, z0 
        self.zf = zf


    def forward(self, E0, xi_, eta_):
        ''' 
        Apply a spherical phase shift to E0 at coordinates xi_ and eta_
        '''

        radius = np.sqrt(self.z0**2 + (xi_ - self.x0)**2 + (eta_ - self.y0)**2)
        phase = self.k * radius
        amplitude = 1 / radius

        E = amplitude * np.exp(1j * phase)
        E *= np.exp(1j * 2 * np.pi * (-self.fcX * xi_ - self.fcY * eta_))  # LPC

        return E0 * E

    
    def phase_gradient(self, xi, eta):
        '''
        Compute phase gradients at point (xi, eta)
        '''

        denom = np.sqrt((xi - self.x0)**2 + (eta - self.y0)**2 + self.z0**2)
        grad_uX = self.k * (xi - self.x0) / denom
        grad_uY = self.k * (eta - self.y0) / denom

        grad_linearX = 2 * np.pi * self.fcX
        grad_linearY = 2 * np.pi * self.fcY

        gradientX = grad_uX - grad_linearX
        gradientY = grad_uY - grad_linearY

        return gradientX, gradientY

    
class PlaneWave():
    def __init__(self, k, r, x0, y0, z0) -> None:
        
        self.k = k
        self.r = r
        self.fcX = self.fcY = 0
        self.x0, self.y0, self.z0 = x0, y0, z0 


    def forward(self, E0, xi_, eta_):

        vec = np.array([-self.x0, -self.y0, self.z0])
        kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))
        phase = self.k * (kx * xi_ + ky * eta_ + kz)

        return E0 * np.exp(1j * phase)


    def phase_gradient(self, xi, eta):

        return 0, 0


class ThinLens():
    def __init__(self, k, f) -> None:
        
        self.k = k
        self.f = f
        self.fcX = self.fcY = 0


    def forward(self, E0, xi_, eta_):

        phase = self.k / 2 * (-1 / self.f) * (xi_**2 + eta_**2)

        return E0 * np.exp(1j * phase)


    def phase_gradient(self, xi, eta):

        grad_uX = -self.k / self.f * xi
        grad_uY = -self.k / self.f * eta

        return grad_uX, grad_uY
    

class Diffuser():
    def __init__(self, r, interpolation='nearest', rand_phase=True, rand_amp=False) -> None:
        '''
        Two types of diffusers: 'nearest' or 'linear' interpolated
        '''
        
        self.fcX = self.fcY = 0
        self.pitch = r / 10
        self.N = int(r * 2 / self.pitch)
        np.random.seed(0)
        self.plate = np.random.rand(self.N, self.N)
        self.interp = interpolation
        self.rand_phase = rand_phase
        self.rand_amp = rand_amp


    def forward(self, E0, xi_, eta_):

        if self.interp == 'nearest':
            plate_sample = cv2.resize(self.plate, xi_.shape, interpolation=cv2.INTER_NEAREST)
        elif self.interp == 'linear':
            plate_sample = cv2.resize(self.plate, xi_.shape, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError

        amp = np.ones_like(plate_sample)
        phase = np.zeros_like(plate_sample)
        if self.rand_phase:
            phase = plate_sample * 4 * np.pi  # random phase from 0 to 4pi
        if self.rand_amp:
            amp = plate_sample  # random amplitude from 0 to 1

        return E0 * amp * np.exp(1j * phase)


    def phase_gradient(self):
        '''
        :return: maximum phase gradient
        '''

        # Second term in Eq. 4
        grad_max = effective_bandwidth(self.pitch, is_plane_wave = True) 

        # nearest interpolation does not have phase gradient
        # but linear interpolation does
        if self.interp == 'linear':
            if self.rand_phase:
                grad_max += 4 / self.pitch
            if self.rand_amp:
                grad_max += 1 / self.pitch / np.pi

        return grad_max, grad_max