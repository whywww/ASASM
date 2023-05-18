import numpy as np
from phase_plates import SphericalWave, PlaneWave, CubicPhasePlate, ThinLens, Diffuser
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

        if "4" in typelist:
            print('\t Random diffuser')

            phase_plate = Diffuser(r, interpolation='linear')
            fcX += phase_plate.fcX
            fcY += phase_plate.fcY
            wavelist.append(phase_plate)

        if compensate:
            Nx, Ny, fbX, fbY = self.sampling_with_LPC(r, s, wavelist)
        else:
            Nx, Ny, fbX, fbY = self.sampling_without_LPC(r, s, wavelist)
        # Nx, Ny, fbX, fbY = self.sampling_with_LPC(r, s, wavelist)
        
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
        xi = np.linspace(-r, r, Nx, endpoint=True)
        eta = np.linspace(-r, r, Ny, endpoint=True)
        xi_, eta_ = np.meshgrid(xi, eta, indexing='xy')

        # circular aperture
        pupil = np.where(xi_**2 + eta_**2 <= r**2, 1, 0)

        self.pupil = pupil
        self.xi, self.eta = xi, eta
        self.xi_, self.eta_ = xi_, eta_
