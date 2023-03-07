import numpy as np
from phase_plates import SphericalWave, PlaneWave, CubicPhasePlate, ThinLens
from operator import add 


class InputField():
    def __init__(self, type:str, wvls:float, angles, r:float, z0=None, f=None, zf=None, s=2.5, compensate=True) -> None:

        self.wvls = wvls  # wavelength of light in vacuum
        self.k = 2 * np.pi / self.wvls  # wavenumebr
        thetaX, thetaY = angles

        # define incident wave
        r0 = z0 / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
        x0, y0 = r0 * np.sin(thetaX / 180 * np.pi), r0 * np.sin(thetaY / 180 * np.pi)
        print(f'aperture diameter = {2 * r}, offset = {x0:.4f}, theta = {thetaX}.')

        fcX = 0
        fcY = 0
        typelist = [*type]
        wavelist = []

        print('Input field contains:')
        if "0" in typelist: 
            print('\t Plane wave')

            field = PlaneWave(self.k, r, x0, y0, z0)
            fcX += field.fcX
            fcY += field.fcY
            wavelist.append(field)

        if "1" in typelist:
            print('\t Spherical wave')

            field = SphericalWave(self.k, x0, y0, z0, angles, zf)
            fcX += field.fcX
            fcY += field.fcY
            wavelist.append(field)

        if "2" in typelist:
            print('\t Convex lens')
            lens = ThinLens(self.k, f)
            fcX += lens.fcX
            fcY += lens.fcY
            wavelist.append(lens)

        if "3" in typelist:
            print('\t Cubic phase plate')

            phase_plate = CubicPhasePlate(self.k, r, m=1e+3)
            fcX += phase_plate.fcX
            fcY += phase_plate.fcY
            wavelist.append(phase_plate)

        if compensate:
            Nx, Ny, fbX, fbY = self.sampling_with_LPC(r, s, wavelist)
        else:
            Nx, Ny, fbX, fbY = self.sampling_without_LPC(r, s, wavelist)
        # Nx, Ny, fbX, fbY = self.sampling_with_LPC(r, s, wavelist)
        # Nx, Ny, fbX, fbY = self.sampling_without_LPC(r, s, wavelist)
        
        self.set_input_plane(r, Nx, Ny)
        E0 = self.pupil
        for wave in wavelist:
            E0 = wave.forward(E0, self.xi_, self.eta_, compensate)

        self.fcX = fcX
        self.fcY = fcY
        self.fbX = fbX
        self.fbY = fbY
        self.E0 = E0
        self.s = s
        self.zf = zf
        self.D = 2 * r
        self.type = type


    def get_plane_wave(self, x0, y0, distance):

        vec = np.array([-x0, -y0, distance])
        kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))
        phase = self.k * (kx * self.xi_ + ky * self.eta_ + kz)

        return np.exp(1j * phase)


    def sampling_with_LPC(self, r, s, wavelist):

        grad1 = [0, 0]
        grad2 = [0, 0]
        for wave in wavelist:
            grad1 = list(map(add, grad1, wave.grad_symm(-r, -r))) 
            grad2 = list(map(add, grad2, wave.grad_symm(r, r))) 
        fbX = max(abs(grad1[0]), abs(grad2[0])) / np.pi * s
        fbY = max(abs(grad1[1]), abs(grad2[1])) / np.pi * s
        Nx = int(np.ceil(fbX * 2 * r))
        Ny = int(np.ceil(fbY * 2 * r))
        print(f'spatial sampling number = {Nx, Ny}.')

        return Nx, Ny, (Nx - 1) / (2 * r), (Ny - 1) / (2 * r)


    def sampling_without_LPC(self, r, s, wavelist):

        grad1 = [0, 0]
        grad2 = [0, 0]
        for wave in wavelist:
            grad1 = list(map(add, grad1, wave.grad_nonsymm(-r, -r))) 
            grad2 = list(map(add, grad2, wave.grad_nonsymm(r, r))) 
        fbX = max(abs(grad1[0]), abs(grad2[0])) / np.pi * s
        fbY = max(abs(grad1[1]), abs(grad2[1])) / np.pi * s
        Nx = int(np.ceil(fbX * 2 * r))
        Ny = int(np.ceil(fbY * 2 * r))
        print(f'spatial sampling number = {Nx, Ny}.')

        return Nx, Ny, (Nx - 1) / (2 * r), (Ny - 1) / (2 * r)

    
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


    # def sampling_with_LPC(self, D, s, x0=None, y0=None, z0=None, thetaX=None, thetaY=None, f=None, zf=None, is_plane_wave=False):
    #     '''
    #     Calculate the minimum number of aperture pixels 
    #     required to satisfy the sampling theorem with Linear Phase Separation
    #     using a thin lens modulation
    #     '''

    #     # using effective bandwidth (S7)
    #     fb = effective_bandwidth(D, self.wvls, is_plane_wave = is_plane_wave, zf = zf, s = s)
        
    #     # using phase gradient analysis
    #     if not is_plane_wave:
    #         ra_min = - D / 2
    #         denom = np.sqrt((ra_min - x0)**2 + (ra_min - y0)**2 + z0**2)
    #         tau_x1 = 2 * abs((ra_min - x0) / denom - ra_min / f + np.sin(thetaX / 180 * np.pi)) / self.wvls
    #         tau_y1 = 2 * abs((ra_min - y0) / denom - ra_min / f + np.sin(thetaY / 180 * np.pi)) / self.wvls

    #         ra_max = D / 2
    #         denom = np.sqrt((ra_max - x0)**2 + (ra_max - y0)**2 + z0**2)
    #         tau_x2 = 2 * abs((ra_max - x0) / denom - ra_max / f + np.sin(thetaX / 180 * np.pi)) / self.wvls
    #         tau_y2 = 2 * abs((ra_max - y0) / denom - ra_max / f + np.sin(thetaY / 180 * np.pi)) / self.wvls

    #         fb2 = max(max(tau_x1, tau_x2), max(tau_y1, tau_y2)) * s
    #         fb = max(fb, fb2)

    #     Nx = Ny = int(fb * D * s)
    #     print(f'spatial sampling number = {Nx, Ny}.')
    #     return Nx, Ny, fb


    # def sampling_without_LPC(self, D, s, x0, y0, z0, f):
    #     '''
    #     Calculate the minimum number of aperture pixels 
    #     required to satisfy the sampling theorem without Linear Phase Separation
    #     '''

    #     # using phase gradient analysis
    #     ra_min = - D / 2
    #     denom = np.sqrt((ra_min - x0)**2 + (ra_min - y0)**2 + z0**2)
    #     tau_x1 = 2 * abs((ra_min - x0) / denom - ra_min / f) / self.wvls
    #     tau_y1 = 2 * abs((ra_min - y0) / denom - ra_min / f) / self.wvls
        
    #     ra_max = D / 2
    #     denom = np.sqrt((ra_max - x0)**2 + (ra_max - y0)**2 + z0**2)
    #     tau_x2 = 2 * abs((ra_max - x0) / denom - ra_max / f) / self.wvls
    #     tau_y2 = 2 * abs((ra_max - y0) / denom - ra_max / f) / self.wvls

    #     fb = max(max(tau_x1, tau_x2), max(tau_y1, tau_y2))
    #     Nx = Ny = int(fb * D * s)
    #     print(f'spatial sampling number = {Nx, Ny}.')

    #     return Nx, Ny, fb

