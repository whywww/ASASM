import numpy as np
from phase_plates import VortexPlate, CubicPhasePlate
from utils import effective_bandwidth


class InputField():
    def __init__(self, type:int, wvls:float, theta:float, r:float, z0=None, f=None, zf=None, s=2.5, compensate=True) -> None:

        self.wvls = wvls  # wavelength of light in vacuum
        self.k = 2 * np.pi / self.wvls  # wavenumebr
        thetaX = thetaY = theta

        # define incident wave
        r0 = z0 / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
        x0, y0 = r0 * np.sin(thetaX / 180 * np.pi), r0 * np.sin(thetaY / 180 * np.pi)
        print(f'aperture diameter = {2 * r}, offset = {x0:.4f}, theta = {thetaX}.')

        if type == 0: 
            print('Plane wave')

            Nx, Ny, fb = self.sampling_without_LPC(2 * r, s)
            self.set_input_plane(r, Nx, Ny)
            E0 = self.pupil * self.get_plane_wave(x0, y0, z0)
            # fc = 

        elif type == 1:
            print('Convex Lens + spherical wave')

            fc = -np.sin(thetaX / 180 * np.pi) / self.wvls
            if compensate:
                Nx, Ny, fb = self.sampling_with_LPC(2 * r, s, x0, y0, z0, thetaX, thetaY, f, zf, is_plane_wave=False)
                self.set_input_plane(r, Nx, Ny)
                E0 = self.pupil * self.get_spherical_wave(x0, y0, z0) * self.lens_transfer(f)
                E0 = E0 * np.exp(1j * 2 * np.pi * (-fc * self.xi_ - fc * self.eta_))
            else:
                Nx, Ny, fb = self.sampling_without_LPC(2 * r, s, x0, y0, z0, f)
                self.set_input_plane(r, Nx, Ny)
                E0 = self.pupil * self.get_spherical_wave(x0, y0, z0) * self.lens_transfer(f)

        # elif type == 3:
        #     print('Vortex plate')

        #     Nx, Ny, fb = self.sampling_without_LPC(2 * r, s)
        #     self.set_input_plane(r, Nx, Ny)
        #     phase_plate = VortexPlate(Nx, Ny, m=3)

        elif type == 2:
            print('Cubic phase plate')

            Nx = Ny = 1024  # TODO: calculate sampling requirement
            self.set_input_plane(r, Nx, Ny)
            phase_plate = CubicPhasePlate(self.xi, self.eta, self.k, s)
            fc = phase_plate.fc
            fb = phase_plate.fb
            E0 = phase_plate.new_field()
            if compensate:
                E0 = E0 * np.exp(1j * 2 * np.pi * (- fc * self.xi_ - fc * self.eta_))
        else:
            raise NotImplementedError

        self.fc = fc
        self.fb = fb
        self.E0 = E0
        self.s = s
        self.zf = zf

        
    def get_spherical_wave(self, x0, y0, distance):
        ''' 
        Get the phase shift of the spherical wave from a single point source 
        
        :param x0, y0: spatial coordinate of the source point
        :param to_xx, to_yy: coordinate grid at the destination plane
        :param distance: scalar tensor, travel distance
        :return: the spherical wave at destination
        '''

        radius = np.sqrt(distance**2 + (self.xi_ - x0)**2 + (self.eta_ - y0)**2)
        phase = self.k * radius
        amplitude = 1 / radius

        return amplitude * np.exp(1j * phase)


    def get_plane_wave(self, x0, y0, distance):

        vec = np.array([-x0, -y0, distance])
        kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))
        phase = self.k * (kx * self.xi_ + ky * self.eta_ + kz)

        return np.exp(1j * phase)


    def lens_transfer(self, f):

        phase = self.k / 2 * (-1 / f) * (self.xi_**2 + self.eta_**2)

        return np.exp(1j * phase)
            

    def sampling_with_LPC(self, D, s, x0, y0, z0, thetaX, thetaY, f, zf=None, is_plane_wave=False):
        '''
        Calculate the minimum number of aperture pixels 
        required to satisfy the sampling theorem with Linear Phase Separation
        using a thin lens modulation
        '''

        # using effective bandwidth (S7)
        fb1 = effective_bandwidth(D, self.wvls, is_plane_wave = is_plane_wave, zf = zf, s = s)
        
        # using phase gradient analysis
        ra_min = - D / 2
        denom = np.sqrt((ra_min - x0)**2 + (ra_min - y0)**2 + z0**2)
        tau_x1 = 2 * abs((ra_min - x0) / denom - ra_min / f + np.sin(thetaX / 180 * np.pi)) / self.wvls
        tau_y1 = 2 * abs((ra_min - y0) / denom - ra_min / f + np.sin(thetaY / 180 * np.pi)) / self.wvls

        ra_max = D / 2
        denom = np.sqrt((ra_max - x0)**2 + (ra_max - y0)**2 + z0**2)
        tau_x2 = 2 * abs((ra_max - x0) / denom - ra_max / f + np.sin(thetaX / 180 * np.pi)) / self.wvls
        tau_y2 = 2 * abs((ra_max - y0) / denom - ra_max / f + np.sin(thetaY / 180 * np.pi)) / self.wvls

        fb2 = max(max(tau_x1, tau_x2), max(tau_y1, tau_y2)) * s
        fb = max(fb1, fb2)

        Nx = Ny = int(fb * D * s)
        print(f'spatial sampling number = {Nx, Ny}.')
        return Nx, Ny, fb


    def sampling_without_LPC(self, D, s, x0, y0, z0, f):
        '''
        Calculate the minimum number of aperture pixels 
        required to satisfy the sampling theorem without Linear Phase Separation
        '''

        # using phase gradient analysis
        ra_min = - D / 2
        denom = np.sqrt((ra_min - x0)**2 + (ra_min - y0)**2 + z0**2)
        tau_x1 = 2 * abs((ra_min - x0) / denom - ra_min / f) / self.wvls
        tau_y1 = 2 * abs((ra_min - y0) / denom - ra_min / f) / self.wvls
        
        ra_max = D / 2
        denom = np.sqrt((ra_max - x0)**2 + (ra_max - y0)**2 + z0**2)
        tau_x2 = 2 * abs((ra_max - x0) / denom - ra_max / f) / self.wvls
        tau_y2 = 2 * abs((ra_max - y0) / denom - ra_max / f) / self.wvls

        fb = max(max(tau_x1, tau_x2), max(tau_y1, tau_y2))
        Nx = Ny = int(fb * D * s)
        print(f'spatial sampling number = {Nx, Ny}.')

        return Nx, Ny, fb

    
    def set_input_plane(self, r, Nx, Ny):

        # coordinates of aperture
        xi = np.linspace(-r, r, Nx, endpoint=False)
        eta = np.linspace(-r, r, Ny, endpoint=False)
        xi_, eta_ = np.meshgrid(xi, eta, indexing='xy')

        # circular aperture
        pupil = np.where(xi_**2 + eta_**2 <= r**2, 1, 0)

        self.pupil = pupil
        self.xi, self.eta = xi, eta
        self.xi_, self.eta_ = xi_, eta_