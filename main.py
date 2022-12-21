'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import compute_bandwidth


def get_exact_spherical_wave(k, src_point, dest_plane, distance):
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

    lam = 2 * np.pi / k
    # normalize the total energy of input light to 1
    amplitude = pupil * distance / lam / radius**2
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return amplitude * np.exp(1j * phase)


def get_plane_wave(k, src_point, dest_plane, distance):
    x = src_point[0]
    y = src_point[1]
    vec = np.array([-x, -y, distance])
    kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))

    phase = k * (kx * dest_plane[0] + ky * dest_plane[1] + kz)

    # normalize the total energy of input light to 1
    amplitude = pupil * np.ones_like(phase)
    amplitude /= np.sqrt(np.sum(amplitude**2, axis=(-2,-1), keepdims=True))
    return np.exp(1j * phase)


def get_lens_phase_shift(k, f, ap_coords):
    phase = k/2 * (-1/f) * (ap_coords[0]**2 + ap_coords[1]**2)
    return np.exp(1j * phase)


def get_aperture_px(lam, D, f, x0, y0, z0):
    '''
    Calculate the minimum number of aperture pixels 
    required to satisfy the sampling theorem.
    '''

    s = 1.5  # oversampling factor for safety

    # paraxial spherical wave sampling requirement
    # Nl1 = ap_diam**2/zo.min()/wvls.min()
    # Exact spherical wave sampling requirement
    Nl1 = D**2 / lam / np.sqrt(D**2 / 4 + z0**2)
    # Lens phase shift sampling requirement
    Nl2 = D**2 / (lam * f)
    # ap_px = int(max(Nl1, Nl2) * s) 

    # combined
    dxmin = abs(x0) - D/2 if abs(x0) < D/2 else 0
    dymin = abs(y0) - D/2 if abs(y0) < D/2 else 0
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

    # ap_px = int(max(Nx, Ny) * s)

    print(f'The lens has {Nx, Ny} pixels.')

    return Nx, Ny


# parameters
lam = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / lam  # wavenumebr

f = 25e-3  # focal length of lens (if applicable)
zo = 1.7  # source-aperture distance
# z = 5e-3  # aperture-sensor distance
z = 1/(1/f - 1/zo) / 1.7  # aperture-sensor distance
# r = 0.4e-3  # radius of aperture 0.4 mm
r = f / 8 / 2  # radius of aperture

# define incident wave
alphaX = 5  # incident angle
alphaY = 0  # incident angle
x0, y0 = np.tan(alphaX / 180 * np.pi) * zo, np.tan(alphaY / 180 * np.pi) * zo  # object offset
s0, t0 = -np.tan(alphaX / 180 * np.pi) * z, -np.tan(alphaY / 180 * np.pi) * z  # image offset
R = np.sqrt(x0**2 + y0**2 + zo**2)
thetaX = np.arccos(x0 / R)
thetaY = np.arccos(y0 / R)
print(f'aperture diameter = {2*r}, offset = {x0}')

# input field sampling
Nx, Ny = get_aperture_px(lam, 2*r, f, x0, y0, zo)  # number of source points
# N = 1431  # number of source points
pitchx = 2 * r / (Nx - 1)
pitchy = 2 * r / (Ny - 1)

# coordinates of aperture
x = np.linspace(-r, r, Nx)
y = np.linspace(-r, r, Ny)
xx, yy = np.meshgrid(x, y, indexing='xy')

# coordinates of observation plane
# M = max(Nx, Ny)
Mx, My = Nx, Ny
# M = 500
l = r * 2
s = np.linspace(-l / 2 + s0, l / 2 + s0, Mx)
t = np.linspace(-l / 2 + t0, l / 2 + t0, My)

# circular aperture
pupil = np.where(xx**2 + yy**2 <= r**2, 1, 0)

# input field
# E1 = pupil * get_plane_wave(k, (x0, y0), np.stack((xx, yy), axis=0), zo)
# B = compute_bandwidth(True, 2*r, lam, pitch, l1=1/(1/f-1/zo), s=5)
# E0 = pupil * get_exact_spherical_wave(k, (0, 0), np.stack((xx, yy), axis=0), zo)
E1 = pupil * get_exact_spherical_wave(k, (x0, y0), np.stack((xx, yy), axis=0), zo) * get_lens_phase_shift(k, f, np.stack((xx, yy), axis=0))
B = compute_bandwidth(False, 2*r, lam, pitchx, pitchy, l1=1/(1/f-1/zo), s=1.5)

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
# plt.title(fr'RS: r={r:.2e}, z={z:.1e}, $\alpha$=({alphaX}$^\circ$, {alphaY}$^\circ$), zo={zo}')
# plt.imshow(np.abs(U0), cmap='gray')
# plt.savefig(f'results1/RS{N}-{alphaX, alphaY}.png')
# plt.close()

print('-------------- Propagating with shift BEASM --------------')
from shift_BEASM import shift_BEASM2d
prop = shift_BEASM2d((s0, t0), z, x, y, s, t, lam)
start = time.time()
U1 = prop(E1)  # cannot use different pitch?
end = time.time()
print(f'Time elapsed for Shift-BEASM: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'BEASM: r={r:.2e}, z={z:.1e}, $\alpha$=({alphaX}$^\circ$, {alphaY}$^\circ$), zo={zo}')
plt.imshow(np.abs(U1), cmap='gray')
plt.savefig(f'results1/BEASM{Nx, Ny}-{alphaX, alphaY}.png')
plt.close()

print('----------------- Propagating with ASMMM -----------------')
from ASASM import AdpativeSamplingASM
device = 'cpu'
# device = 'cuda:3'
prop = AdpativeSamplingASM(thetaX, thetaY, z, x, y, s, t, lam, B, device)
E1_res = E1 / get_plane_wave(k, (x0, y0), np.stack((xx, yy), axis=0), zo)
start = time.time()
U2 = prop(E1, decomposed=False)
# U2 = prop(E1_res, decomposed=True)
end = time.time()
print(f'Time elapsed for ASMMM: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'MM: r={r:.2e}, z={z:.1e}, $\alpha$=({alphaX}$^\circ$, {alphaY}$^\circ$), zo={zo}')
plt.imshow(np.abs(U2), cmap='gray')
plt.savefig(f'results1/MM{Nx, Ny}-{alphaX, alphaY}.png')
plt.close()