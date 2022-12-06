'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client simple_test.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time


def get_exact_spherical_wave_np(src_point, dest_plane, distance):
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
    phase = k * (radius - distance)

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
    return amplitude * np.exp(1j * phase)


def get_lens_phase_shift(f, ap_coords):
    phase = k/2 * (-1/f) * (ap_coords[0]**2 + ap_coords[1]**2)
    return np.exp(1j * phase)




# parameters
n = 1024  # number of source points
r = int(0.8/1.024 * n) // 2  # radius of aperture
pad = 2
N = n * pad

pitch = 1e-6
z = 5e-3  #1/(1/25e-3 - 1/1.7) # 
l = N * pitch

lam = 500e-9
k = 2 * np.pi / lam

theta = 5  # incident angle
zo = 1.7  # point source distance
x0, y0 = -np.tan(theta / 180 * np.pi) * zo, 0  # object offset
s0, t0 = np.tan(theta / 180 * np.pi) * z, 0  # image offset

f = 25e-3   # focal

# the source points
x = np.linspace(-l / 2, l / 2, N)
y = np.linspace(-l / 2, l / 2, N)
xx, yy = np.meshgrid(x, y, indexing='ij')

# the dest points
s = np.linspace(-l / 2 + s0, l / 2 + s0, N)
t = np.linspace(-l / 2 + t0, l / 2 + t0, N)

# the aperture
c = np.linspace(-N//2, N//2, N)
uu, vv = np.meshgrid(c, c, indexing='ij')
pupil = np.where(uu**2+vv**2<=r**2, 1, 0)

# the input field
E1 = pupil * get_plane_wave((x0, y0), np.stack((xx, yy), axis=0), zo)
# E1 = pupil * get_exact_spherical_wave_np((x0, y0), np.stack((xx, yy), axis=0), zo) * get_lens_phase_shift(f, np.stack((xx, yy), axis=0))
# E1 = get_exact_spherical_wave_np((xo, yo), np.stack((xx, yy), axis=0), zo)

print('-------------- Propagating with RS integral --------------')
# from RS import RSDiffraction_INT
# prop = RSDiffraction_INT()
# U0 = prop(E1, z, x, y, s, t, lam)
from RS import RSDiffraction_GPU
prop = RSDiffraction_GPU(z, x, y, s, t, lam, 'cuda:2')
start = time.time()
U0 = prop(E1)
end = time.time()
print(f'Time elapsed for RS: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'RS: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta}$^\circ$, zo={zo}')
plt.imshow(np.abs(U0), cmap='gray')
plt.savefig(f'results/RS{n}-{theta}.png')
plt.close()

print('-------------- Propagating with shift BEASM --------------')
from shift_BEASM_cpu import shift_BEASM2d
prop = shift_BEASM2d(s0, t0, z, x, y, lam)
start = time.time()
U1 = prop(E1)
end = time.time()
print(f'Time elapsed for Shift-BEASM: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'BEASM: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta}$^\circ$, zo={zo}')
plt.imshow(np.abs(U1), cmap='gray')
plt.savefig(f'results/BEASM{n}-{theta}.png')
plt.close()

print('----------------- Propagating with ASMMM -----------------')
from svASM import AngularSpectrumMethodMM
device = 'cpu' #'cuda:2' # 
prop = AngularSpectrumMethodMM((x0, y0, zo), z, x, y, s, t, lam, device)
start = time.time()
U2 = prop(E1)
end = time.time()
print(f'Time elapsed for ASMMM: {end-start:.2f}')
plt.figure(figsize=(10,10))
plt.tight_layout()
plt.title(fr'MM: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta}$^\circ$, zo={zo}')
plt.imshow(np.abs(U2)[0], cmap='gray')
plt.savefig(f'results/MM{n}-{theta}.png')
plt.close()