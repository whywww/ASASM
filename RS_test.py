import numpy as np
import matplotlib.pyplot as plt
import finufft
import numpy.fft as fft
from scipy import integrate
import math


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
z = 0.005 # 1/(1/25e-3 - 1/1.7) # 
l = N * pitch

lam = 500e-9
k = 2 * np.pi / lam

theta = 5 / 180 * np.pi  # incident angle
zo = 1.7  # point source distance
xo, yo = -np.tan(theta) * zo, 0  # object offset
xs, ys = np.tan(theta) * z, 0  # image offset

f = 25e-3   # focal

# the source points
x = np.linspace(-l / 2, l / 2, N)
y = np.linspace(-l / 2, l / 2, N)
xx, yy = np.meshgrid(x, y)

# create input field
# the aperture
c = np.linspace(-N//2, N//2, N)
uu, vv = np.meshgrid(c, c)
pupil = np.where(uu**2+vv**2<=r**2, 1, 0)

# E1 = pupil * get_plane_wave((xo, yo), np.stack((xx, yy), axis=0), zo)
E1 = pupil #* get_exact_spherical_wave_np((xo, yo), np.stack((xx, yy), axis=0), zo) * get_lens_phase_shift(f, np.stack((xx, yy), axis=0))
# E1 = get_exact_spherical_wave_np((xo, yo), np.stack((xx, yy), axis=0), zo)

from RS import RSDiffraction_INT, RSDiffraction_GPU
# prop = RSDiffraction_INT()
# U0 = prop(E1, z, x, y, x, y, lam)
prop = RSDiffraction_GPU()
U0 = prop(E1, z, x, y, x, y, lam, 'cuda:1')
plt.title(fr'RS: r={r*pitch:.2e}, z={z:.1e}, $\theta$={theta:.2e}, zo={zo}')
plt.imshow(np.abs(U0), cmap='gray')
plt.savefig('RS.png')