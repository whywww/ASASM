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
    color = (237, 228, 86)  # light yellow RGB
    # color = (102, 145, 59)  # deep green RGB
    
    image = cv2.ellipse(image, center, axes, 0., 0., 360., color, thickness=thickness)
    
    return image


def draw_bandwidth(spectrum, fx, fy, bandwidths, save_path):
    dfx = fx[-1] - fx[-2]
    dfy = fy[-1] - fy[-2]
    lx = len(fx)
    ly = len(fy)
    rx = int(bandwidths[0] / 2 / dfx)  # in pixel
    ry = int(bandwidths[1] / 2 / dfy)  # in pixel

    circled_spectrum = draw_ellipse(abs(spectrum).cpu(), (lx//2, ly//2), (rx, ry))
    save_image(circled_spectrum, save_path)


def compute_bandwidth(is_plane_wave, D, wvls, pitchx, pitchy, l1=None, s=5):
    if is_plane_wave:
        bandwidth = 129.3 * s / np.pi / D
    else:
        assert l1 is not None, "Wave origin should be provided!"
        bandwidth = s * D / wvls / l1  # physical

    bandwidthX = min(1 / pitchx, bandwidth)
    bandwidthY = min(1 / pitchy, bandwidth)

    return bandwidthX, bandwidthY


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