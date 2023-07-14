import numpy as np
from phase_plates import SphericalWave, PlaneWave, ThinLens, Diffuser
from operator import add 
from utils import effective_bandwidth


class InputField():
    '''
    Prepare compensated input field and spatial sampling
    '''
    def __init__(self, type:str, wvls:float, angles, r:float, 
                 z0=None, f=None, zf=None, s=1.5) -> None:

        self.wvls = wvls  # wavelength of light in vacuum
        self.k = 2 * np.pi / self.wvls  # wavenumebr
        thetaX, thetaY = angles

        # define incident wave
        r0 = z0 / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
        x0, y0 = r0 * np.sin(thetaX / 180 * np.pi), r0 * np.sin(thetaY / 180 * np.pi)
        print(f'aperture diameter = {2 * r}, offset = {x0:.4f}, theta = {thetaX}.')

        # prepare wave components
        typelist = [*type]
        wavelist = []
        fcX = 0  # frequency centers
        fcY = 0

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
            print('\t Random diffuser')

            phase_plate = Diffuser(r, interpolation='linear', rand_phase=True, rand_amp=True)
            fcX += phase_plate.fcX
            fcY += phase_plate.fcY
            wavelist.append(phase_plate)

        # Compute spatial sampling
        Nx, Ny, fbX, fbY = self.spatial_sampling(r, s, wavelist)
        
        # Prepare input field
        self.set_input_plane(r, Nx, Ny)
        E0 = self.pupil
        for wave in wavelist:
            E0 = wave.forward(E0, self.xi_, self.eta_)

        self.fcX = fcX
        self.fcY = fcY
        self.fbX = fbX
        self.fbY = fbY
        self.E0 = E0
        self.s = s
        self.zf = zf
        self.D = 2 * r
        self.type = type


    def spatial_sampling(self, r, s, wavelist):
        '''
        :param r: aperture radius
        :param s: oversampling factor
        :param wavelist: a list of input wave components
        :return: number of samples in both dimensions, bandwidths in both dimensions
        '''

        # Second term in Eq4, the size of Airy disk
        fplane = effective_bandwidth(r*2, is_plane_wave = True)

        # First term in Eq4, maximum phase gradient
        # as the phase terms are all monotonic here, 
        # we use the two boundaries of aperture (+-r) to find max
        grad1 = [0, 0]
        grad2 = [0, 0]
        diffuser = False
        for wave in wavelist:
            if isinstance(wave, Diffuser):
                diffuser = True
                grad = wave.phase_gradient()
                fbX_diffuser = grad[0] * s
                fbY_diffuser = grad[1] * s
            else:
                grad1 = list(map(add, grad1, wave.phase_gradient(-r, -r))) 
                grad2 = list(map(add, grad2, wave.phase_gradient(r, r))) 
        fbX = (max(abs(grad1[0]), abs(grad2[0])) / np.pi + fplane) * s
        fbY = (max(abs(grad1[1]), abs(grad2[1])) / np.pi + fplane) * s

        if diffuser:
            fbX = max(fbX, fbX_diffuser)
            fbY = max(fbY, fbY_diffuser)

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
