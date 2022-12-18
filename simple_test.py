'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client simple_test.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import Effective_Bandwidth


def get_exact_spherical_wave(src_point, dest_plane, distance):
    ''' 
    Get the phase shift of the spherical wave from a single point source 
    
    :param src: tuple (x,y), spatial coordinate of the source
    :param dest: tensor [uu,vv], coordinate grid at the destination
    :param distance: scalar tensor, travel distance
    :return: (DxCxUxV) amplitude and phase of the spherical wave
    '''

    x = src_point[0]
    y = src_point[1]
    radius = np.sqrt(distance**2 + (dest_plane[0]-x)**2 + (dest_plane[1]-y)**2)
    phase = k * radius

    # normalize the total energy of input light to 1
    amplitude = pupil * distance / lam / radius**2
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return amplitude * np.exp(1j * phase)


def get_plane_wave(src_point, dest_plane, distance):
    x = src_point[0]
    y = src_point[1]
    vec = np.array([-x, -y, distance])
    kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))

    # radius = np.sqrt((distance**2 + dest_plane[0]-x)**2 + (dest_plane[1]-y)**2)
    phase = k * (kx * dest_plane[0] + ky * dest_plane[1] + kz)

    # normalize the total energy of input light to 1
    amplitude = pupil * np.ones_like(phase)
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return np.exp(1j * phase)


def get_lens_phase_shift(f, ap_coords):
    phase = k/2 * (-1/f) * (ap_coords[0]**2 + ap_coords[1]**2)
    return np.exp(1j * phase)


def get_aperture_px(k, diam, z, f):
        '''
        Calculate the minimum number of aperture pixels required to satisfy the sampling theorem.
        '''

        # paraxial spherical wave sampling requirement
        # Nl1 = ap_diam**2/zo.min()/wvls.min()
        # Exact spherical wave sampling requirement
        Nl1 = k * diam**2 / (2 * np.pi) / np.sqrt(diam**2 / 4 + z**2)
        # Lens phase shift sampling requirement
        Nl2 = k * diam**2 / (2 * np.pi * f)
        
        s = 1.1  # oversampling factor for safety
        ap_px = int(max(Nl1, Nl2) * s) 
        print(f'The lens has {ap_px} pixels.')

        return ap_px


# parameters
lam = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / lam  # wavenumebr

z = 5e-3  # aperture-sensor distance
# z = 1/(1/25e-3 - 1/1.7)

f = 25e-3  # focal length of lens (if applicable)
r = 0.4e-3  # radius of aperture 0.4 mm
N = get_aperture_px(k, 2*r, z, f)  # number of source points
# N = 512  # number of source points
pitch = 2 * r / (N - 1)

# define incident wave
thetaX = 5  # incident angle
thetaY = 0  # incident angle
zo = 1.7  # source-aperture distance
x0, y0 = np.tan(thetaX / 180 * np.pi) * zo, np.tan(thetaY / 180 * np.pi) * zo  # object offset
s0, t0 = -np.tan(thetaX / 180 * np.pi) * z, -np.tan(thetaY / 180 * np.pi) * z  # image offset
print(f'aperture diameter = {2*r}, offset = {x0}')


# coordinates of aperture
x = np.linspace(-r, r, N)
y = np.linspace(-r, r, N)
xx, yy = np.meshgrid(x, y, indexing='xy')

# coordinates of observation plane
M = N
l = r * 2
s = np.linspace(-l / 2 + s0, l / 2 + s0, M)
t = np.linspace(-l / 2 + t0, l / 2 + t0, M)

# circular aperture
pupil = np.where(xx**2 + yy**2 <= r**2, 1, 0)

# input field
# E1 = pupil * get_plane_wave((x0, y0), np.stack((xx, yy), axis=0), zo)
E1 = pupil * get_exact_spherical_wave((x0, y0), np.stack((xx, yy), axis=0), zo) * get_lens_phase_shift(f, np.stack((xx, yy), axis=0))
# E0 = pupil * get_exact_spherical_wave((0, 0), np.stack((xx, yy), axis=0), zo)
effB = Effective_Bandwidth(False, 2*r, lam, 1/(1/f-1/zo), s=1.5)
# effB = Effective_Bandwidth(True, 2*r, lam, 1/(1/f-1/zo), s=5)

# print('-------------- Propagating with RS integral --------------')
# # from RS import RSDiffraction_INT
# # prop = RSDiffraction_INT()
# # U0 = prop(E1, z, x, y, s, t, lam)
# from RS import RSDiffraction_GPU
# prop = RSDiffraction_GPU(z, x, y, s, t, lam, 'cuda:1')
# start = time.time()
# U0 = prop(E1)
# end = time.time()
# print(f'Time elapsed for RS: {end-start:.2f}')
# plt.figure(figsize=(10,10))
# plt.tight_layout()
# plt.title(fr'RS: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta}$^\circ$, zo={zo}')
# plt.imshow(np.abs(U0), cmap='gray')
# plt.savefig(f'results/RS{n}-{theta}-lensless.png')
# plt.close()

# print('-------------- Propagating with shift BEASM --------------')
# from shift_BEASM_cpu import shift_BEASM2d
# prop = shift_BEASM2d(s0, t0, z, x, y, lam)
# start = time.time()
# U1 = prop(E1)
# end = time.time()
# print(f'Time elapsed for Shift-BEASM: {end-start:.2f}')
# plt.figure(figsize=(10,10))
# plt.tight_layout()
# plt.title(fr'BEASM: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta}$^\circ$, zo={zo}')
# plt.imshow(np.abs(U1), cmap='gray')
# plt.savefig(f'results/BEASM{n}-{theta}-lensless.png')
# plt.close()

print('----------------- Propagating with ASMMM -----------------')
from ASASM import AdpativeSamplingASM, AngularSpectrumMethodMM
device = 'cpu' #'cuda:2' # 
prop = AdpativeSamplingASM((s0, t0), z, x, y, s, t, lam, device)
# prop = AngularSpectrumMethodMM((s0, t0), z, x, y, s, t, lam, device)
E1_res = E1 / get_plane_wave((x0, y0), np.stack((xx, yy), axis=0), zo)
start = time.time()
U2 = prop(E1, effB)
end = time.time()
print(f'Time elapsed for ASMMM: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'MM: r={r*pitch:.2e}, z={z:.1e}, $\theta$={thetaX}$^\circ$, zo={zo}')
plt.imshow(np.abs(U2), cmap='gray')
plt.savefig(f'results1/MM{N}-{thetaX}.png')
plt.close()
