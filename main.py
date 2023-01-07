'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import compute_bandwidth, save_image
from vortex_plate import VortexPlate


def get_spherical_wave(k, x0, y0, to_xx, to_yy, distance):
    ''' 
    Get the phase shift of the spherical wave from a single point source 
    
    :param x0, y0: spatial coordinate of the source point
    :param to_xx, to_yy: coordinate grid at the destination plane
    :param distance: scalar tensor, travel distance
    :return: the spherical wave at destination
    '''

    radius = np.sqrt(distance**2 + (to_xx - x0)**2 + (to_yy - y0)**2)
    phase = k * radius

    lam = 2 * np.pi / k
    # normalize the total energy of input light to 1
    amplitude = pupil * distance / lam / radius**2
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return amplitude * np.exp(1j * phase)


def get_plane_wave(k, x0, y0, to_xx, to_yy, distance):

    vec = np.array([-x0, -y0, distance])
    kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))

    phase = k * (kx * to_xx + ky * to_yy + kz)

    # normalize the total energy of input light to 1
    amplitude = pupil * np.ones_like(phase)
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return np.exp(1j * phase)


def lens_transfer(k, f, to_xx, to_yy):

    phase = k/2 * (-1/f) * (to_xx**2 + to_yy**2)
    return np.exp(1j * phase)


def get_aperture_px(lam, D, f, x0, y0, z0):
    '''
    Calculate the minimum number of aperture pixels 
    required to satisfy the sampling theorem.
    '''

    s = 1.1  # oversampling factor for safety

    # combined
    dxmin = abs(x0) - D/2 if abs(x0) > D/2 else 0
    dymin = abs(y0) - D/2 if abs(y0) > D/2 else 0
    Rxp = np.sqrt((D/2 + x0)**2 + dymin**2 + z0**2)
    Rxm = np.sqrt((D/2 - x0)**2 + dymin**2 + z0**2)
    Ryp = np.sqrt((D/2 + y0)**2 + dxmin**2 + z0**2)
    Rym = np.sqrt((D/2 - y0)**2 + dxmin**2 + z0**2)
    Nx1 = D**2 / lam * abs((-1 - 2 * x0 / D) / Rxp + 1 / f)
    Nx2 = D**2 / lam * abs((1 - 2 * x0 / D) / Rxm - 1 / f)
    Nx = int(max(Nx1, Nx2) * s)
    Ny1 = D**2 / lam * abs((-1 - 2 * y0 / D) / Ryp + 1 / f)
    Ny2 = D**2 / lam * abs((1 - 2 * y0 / D) / Rym - 1 / f)
    Ny = int(max(Ny1, Ny2) * s)

    print(f'The lens has {Nx, Ny} pixels.')

    return Nx, Ny


# parameters
lam = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / lam  # wavenumebr

f = 35e-3  # focal length of lens (if applicable)
zo = 1.7  # source-aperture distance
z = 1/(1/f - 1/zo)  # aperture-sensor distance, at focal plane
r = f / 16 / 2  # radius of aperture

# define incident wave
thetaX = 0
thetaY = 0
R = zo / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
x0, y0 = R * np.sin(thetaX / 180 * np.pi), R * np.sin(thetaY / 180 * np.pi)
s0, t0 = -x0 / zo * z, -y0 / zo * z
print(f'aperture diameter = {2*r}, offset = {x0:.4f}')

# input field sampling
# Nx, Ny = get_aperture_px(lam, 2*r, f, x0, y0, zo)  # number of source points
Nx = Ny = 1024  # number of source points
pitchx = 2 * r / (Nx - 1)
pitchy = 2 * r / (Ny - 1)

# coordinates of aperture
x = np.linspace(-r, r, Nx)
y = np.linspace(-r, r, Ny)
xx, yy = np.meshgrid(x, y, indexing='xy')

# coordinates of observation plane
# energy spreads at larger angle, so use larger observation window.
Mx, My = Nx, Ny
# if max(thetaX, thetaY) <= 15:
#     l = r * .5
# elif max(thetaX, thetaY) <= 20:
#     l = r * 1.5 
# elif max(thetaX, thetaY) <= 25:
#     l = r * 2.5
# elif max(thetaX, thetaY) >= 30:
#     l = r * 5
l = r * 5

s = np.linspace(-l / 2 + s0, l / 2 + s0, Mx)
t = np.linspace(-l / 2 + t0, l / 2 + t0, My)

# circular aperture
pupil = np.where(xx**2 + yy**2 <= r**2, 1, 0)

# input field
exps = 1. # np.inf
phase_plate = VortexPlate(Nx, Ny, m=3)
E1 = pupil * get_plane_wave(k, x0, y0, xx, yy, zo)
# E1 = pupil * get_spherical_wave(k, x0, y0, xx, yy, zo) * lens_transfer(k, f, xx, yy)
# E1 = phase_plate.forward(E1)
B = compute_bandwidth(True, 2*r, lam, pitchx, pitchy, s=exps)  # plane
# B = compute_bandwidth(False, 2*r, lam, pitchx, pitchy, l1=1/(1/f-1/zo), s=exps)  # others


# print('-------------- Propagating with shift BEASM --------------')
# path = f'results1/BEASM-single{Nx, Ny}-{thetaX, thetaY}'
# from shift_BEASM import shift_BEASM2d
# prop = shift_BEASM2d((s0, t0), z, x, y, s, t, lam)
# start = time.time()
# U1 = prop(E1)
# # U1 = prop(E1, save_path=path)
# end = time.time()
# print(f'Time elapsed for Shift-BEASM: {end-start:.2f}')
# save_image(abs(U1), f'{path}.png')
# phase = np.angle(U1) % (2*np.pi)
# # save_image(phase, f'{path}-Phi.png')


print('----------------- Propagating with ASASM -----------------')
path = f'results2/ASASM-plane{Nx, Ny}-{thetaX, thetaY}-{exps}'
from ASASM import AdpativeSamplingASM
device = 'cpu'
# device = 'cuda:3'
prop = AdpativeSamplingASM(thetaX, thetaY, z, x, y, s, t, lam, B, 
                        device, crop_bandwidth=False)
# E1_res = E1 / get_plane_wave(k, x0, y0, xx, yy, zo)
start = time.time()
# U2 = prop(E1, decomposed=False)
U2 = prop(E1, decomposed=False, save_path=path)
# U2 = prop(E1_res, decomposed=True)
end = time.time()
print(f'Time elapsed for ASASM: {end-start:.2f}')
save_image(abs(U2), f'{path}.png')
phase = np.angle(U2) % (2*np.pi)
# save_image(phase, f'{path}-Phi.png')


# print('-------------- Propagating with RS integral --------------')
# path = f'results2/RS{Nx, Ny}-{thetaX, thetaY}'
# # from RS import RSDiffraction_INT  # cpu, super slow
# # prop = RSDiffraction_INT()
# # U0 = prop(E1, z, x, y, s, t, lam)
# from RS import RSDiffraction_GPU
# prop = RSDiffraction_GPU(z, x, y, s, t, lam, 'cuda:3')
# start = time.time()
# U0 = prop(E1)
# end = time.time()
# print(f'Time elapsed for RS: {end-start:.2f}')
# save_image(abs(U0), f'{path}.png')
# phase = np.angle(U0) % (2*np.pi)
# save_image(phase, f'{path}-Phi.png')