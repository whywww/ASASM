import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image 


def draw_ellipse(image, center, axes):

    thickness = 2
    image = np.repeat(np.array(image)[..., None], 3, axis=-1)
    image /= image.max()
    image = (image * 255).astype(np.uint8)
    # color = (215, 164, 0)  # BGR
    # color = (237, 228, 86)  # light yellow RGB
    color = (102, 145, 59)  # deep green RGB
    
    image = cv2.ellipse(image, center, axes, 0., 0., 360., color, thickness=thickness)
    
    return image


def draw_bandwidth(spectrum, fx, fy, bandwidth, save_path):
    dfx = fx[-1] - fx[-2]
    dfy = fy[-1] - fy[-2]
    lx = len(fx)
    ly = len(fy)
    rx = int(bandwidth / 2 / dfx)  # in pixel
    ry = int(bandwidth / 2 / dfy)  # in pixel

    circled_spectrum = draw_ellipse(abs(spectrum).cpu(), (lx//2, ly//2), (rx, ry))
    save_image(circled_spectrum, save_path)


def effective_bandwidth(D, wvls, is_plane_wave=False, zf=None, exps=1.5):
    if is_plane_wave:
        bandwidth = 129.3 * exps / np.pi / D
    else:
        assert zf is not None, "Wave origin should be provided!"
        bandwidth = exps * D / wvls / zf

    return bandwidth


def save_image(image, save_path):
    im = Image.fromarray(np.array(image / image.max() * 255).astype(np.uint8))
    im.save(save_path)


def image_err(img_hat, img_ref):
    nom = np.sum(np.square(img_hat - img_ref))
    denom = np.sum(np.square(img_ref))
    return nom / denom


def remove_linear_phase(phi, thetaX, thetaY, s, t, k):

    linear_phiX = -np.sin(thetaX / 180 * np.pi) * k
    linear_phiY = -np.sin(thetaY / 180 * np.pi) * k

    ss, tt = np.meshgrid(s, t, indexing='xy')

    phi_new = phi - ss * linear_phiX - tt * linear_phiY

    return phi_new % (2 * np.pi)


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

    amplitude = 1 / radius
    return amplitude * np.exp(1j * phase)


def get_plane_wave(k, x0, y0, to_xx, to_yy, distance):

    vec = np.array([-x0, -y0, distance])
    kx, ky, kz = vec / np.sqrt(np.dot(vec, vec))
    phase = k * (kx * to_xx + ky * to_yy + kz)

    return np.exp(1j * phase)


def lens_transfer(k, f, to_xx, to_yy):

    phase = k/2 * (-1/f) * (to_xx**2 + to_yy**2)
    return np.exp(1j * phase)