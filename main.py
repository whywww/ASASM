'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import effective_bandwidth, save_image, remove_linear_phase, get_spherical_wave, lens_transfer, get_plane_wave
from vortex_plate import VortexPlate
from tqdm import tqdm
import math


def spatial_sampling_with_LPS(D, exps, B=0, is_plane_wave=False):
    '''
    Calculate the minimum number of aperture pixels 
    required to satisfy the sampling theorem with Linear Phase Separation
    using a thin lens modulation
    '''

    if is_plane_wave:
        assert B != 0, "Please provide effective bandwidth!"
        # using effective bandwidth (S7)
        Nx = Ny = int(B * D)
    else:
        # using approximated phase gradient analysis
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        tau_x = D / lam * (abs((y0**2 + z0**2) / r0**3 - 1 / f) + x0 * y0 / r0**3)
        tau_y = D / lam * (abs((x0**2 + z0**2) / r0**3 - 1 / f) + x0 * y0 / r0**3)
        Nx = int(tau_x * D * exps)
        Ny = int(tau_y * D * exps)

    print(f'spatial sampling number = {Nx, Ny}.')
    return Nx, Ny


def spatial_sampling_without_LPS(D, exps):
    '''
    Calculate the minimum number of aperture pixels 
    required to satisfy the sampling theorem without Linear Phase Separation
    '''

    # using approximated phase gradient analysis
    r0 = np.sqrt(x0**2 + y0**2 + z0**2)
    tau_x = D / lam * (abs((y0**2 + z0**2) / r0**3 - 1 / f) + x0 * y0 / r0**3) + abs(2 / lam * x0 / r0)
    tau_y = D / lam * (abs((x0**2 + z0**2) / r0**3 - 1 / f) + x0 * y0 / r0**3) + abs(2 / lam * y0 / r0)
    Nx = int(tau_x * D * exps)
    Ny = int(tau_y * D * exps)

    print(f'spatial sampling number = {Nx, Ny}.')

    return Nx, Ny


# def conventional_spatial_sampling(D, exps):

#     # using approximated phase gradient analysis (S3, S10)
#     r0 = np.sqrt(x0**2 + y0**2 + z0**2)

#     # oblique wave without linear phase separation
#     tau_x1 = D / lam * (abs((y0**2 + z0**2) / r0**3) + x0 * y0 / r0**3) + abs(2 / lam * x0 / r0)
#     tau_y1 = D / lam * (abs((x0**2 + z0**2) / r0**3) + x0 * y0 / r0**3) + abs(2 / lam * y0 / r0)
    
#     # lens sampling
#     tau2 = D / lam / f
    
#     tau_x = max(tau_x1, tau2)
#     tau_y = max(tau_y1, tau2)
#     Nx = int(tau_x * D * exps)
#     Ny = int(tau_y * D * exps)

#     print(f'spatial sampling number = {Nx, Ny}.')

#     return Nx, Ny


def prepare_input_field(Nx, Ny, r, x0, y0, z0, is_plane_wave, decompose=False):

    # coordinates of aperture
    x = np.linspace(-r, r, Nx, endpoint=False)
    y = np.linspace(-r, r, Ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # circular aperture
    pupil = np.where(xx**2 + yy**2 <= r**2, 1, 0)

    # input field
    if is_plane_wave:
        E1 = pupil * get_plane_wave(k, x0, y0, xx, yy, z0)
    else:
        E1 = pupil * get_spherical_wave(k, x0, y0, xx, yy, z0) * lens_transfer(k, f, xx, yy)
    
    # phase_plate = VortexPlate(Nx, Ny, m=3)
    # E1 = phase_plate.forward(E1)

    if decompose:
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        E1 = E1 * np.exp(1j * k * (x0 / r0 * xx + y0 / r0 * yy))

    return x, y, E1


# hyperparameters
lam = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / lam  # wavenumebr
f = 35e-3  # focal length of lens (if applicable)
z0 = 1.7  # source-aperture distance
zf = 1/(1/f - 1/z0)  # image-side focal distance
z = zf  # aperture-sensor distance
r = f / 16 / 2  # radius of aperture
thetaX = thetaY = 6  # incident angle

e_ASASM = e_BEASM = 1.5  # expansion factor
e_RS = 1.5
is_plane_wave = False # false for spherical wave
decompose = True
times = 1  # number of times to run for each method
use_BEASM = False
use_ASASM = False
use_RS = True
result_folder = 'results-visual'


# define incident wave
R = z0 / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
x0, y0 = R * np.sin(thetaX / 180 * np.pi), R * np.sin(thetaY / 180 * np.pi)
print(f'aperture diameter = {2*r}, offset = {x0:.4f}, theta = {thetaX}.')

# define observation window
Mx, My = 1024, 1024
l = r * 1
s0, t0 = -x0 / z0 * z, -y0 / z0 * z
s = np.linspace(-l / 2 + s0, l / 2 + s0, Mx, endpoint=False)
t = np.linspace(-l / 2 + t0, l / 2 + t0, My, endpoint=False)

# temporal, used for comparing results
B = effective_bandwidth(2 * r, lam, is_plane_wave = is_plane_wave, zf = zf, exps = e_ASASM)
# Nx, Ny = spatial_sampling_with_LPS(2 * r, exps = e_ASASM, B = B, is_plane_wave = is_plane_wave)
Nx, Ny = spatial_sampling_without_LPS(2 * r, exps=e_BEASM)  # used for baseline methods


if use_BEASM:
    print('-------------- Propagating with shift BEASM --------------')
    # Nx, Ny = spatial_sampling_without_LPS(2 * r, exps=e_BEASM)  # used for baseline methods
    x, y, E1 = prepare_input_field(Nx, Ny, r, x0, y0, z0, is_plane_wave, decompose=False)

    from shift_BEASM import shift_BEASM2d
    prop = shift_BEASM2d((s0, t0), z, x, y, s, t, lam)
    path = f'{result_folder}/BEASM({Nx},{prop.Lfx})-{thetaX}-{e_BEASM:.1f}'
    runtime = 0
    for i in tqdm(range(times)):
        start = time.time()
        U1, Fu = prop(E1)
        end = time.time()
        runtime += end - start
    print(f'Time elapsed for Shift-BEASM: {runtime / times:.2f}')
    save_image(abs(U1), f'{path}.png')
    # phase = np.angle(U1) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U1), thetaX, thetaY, s, t, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    save_image(Fu, f'{path}-FU.png')
    # np.save(f'{path}', U1)


if use_ASASM:
    print('----------------- Propagating with ASASM -----------------')
    # Nx, Ny = spatial_sampling_with_LPS(2 * r, exps = e_ASASM, B = B, is_plane_wave = is_plane_wave)
    x, y, E1 = prepare_input_field(Nx, Ny, r, x0, y0, z0, is_plane_wave, decompose=decompose)

    from ASASM import AdpativeSamplingASM
    device = 'cpu'
    # device = 'cuda:3'
    prop = AdpativeSamplingASM(thetaX, thetaY, z, x, y, s, t, zf, lam, B, device, crop_bandwidth=True)
    path = f'{result_folder}/ASASM({Nx},{len(prop.fx)})-{thetaX}-{e_ASASM:.1f}'
    runtime = 0
    for i in tqdm(range(times)):
        start = time.time()
        U2, Fu = prop(E1, decomposed=decompose)
        end = time.time()
        runtime += end - start
    print(f'Time elapsed for ASASM: {runtime / times:.2f}')
    save_image(abs(U2), f'{path}.png')
    # phase = np.angle(U2) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U2), thetaX, thetaY, s, t, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    save_image(Fu, f'{path}-FU.png')
    # np.save(f'{path}', U2)


if use_RS:
    print('-------------- Propagating with RS integral --------------')
    Nx, Ny = spatial_sampling_without_LPS(2*r, e_RS)
    x, y, E1 = prepare_input_field(Nx, Ny, r, x0, y0, z0, is_plane_wave, decompose=False)

    # from RS import RSDiffraction_INT  # cpu, super slow
    # prop = RSDiffraction_INT()
    # U0 = prop(E1, z, x, y, s, t, lam)
    from RS import RSDiffraction_GPU
    prop = RSDiffraction_GPU(z, x, y, s, t, lam, 'cuda:1')
    path = f'RS/RS({Nx})-{thetaX}-{e_RS:.1f}'
    start = time.time()
    U0 = prop(E1)
    end = time.time()
    print(f'Time elapsed for RS: {end-start:.2f}')
    save_image(abs(U0), f'{path}.png')
    # phase = np.angle(U0) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U0), thetaX, thetaY, s, t, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    np.save(f'{path}', U0)
