import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image 


def draw_ellipse(image, center, radius):

    thickness = 10
    image = np.repeat(np.array(image)[..., None], 3, axis=-1)
    image /= image.max()
    image = (image * 255).astype(np.uint8)
    # color = (215, 164, 0)  # BGR
    # color = (237, 228, 86)  # light yellow RGB
    color = (102, 145, 59)  # deep green RGB
    
    # image = cv2.ellipse(image, center, radius, 0., 0., 360., color, thickness=thickness)
    image = cv2.circle(image, center, radius=10, color=color, thickness=-1)
    image = cv2.rectangle(image, (center[0]-radius[0], center[1]-radius[1]), 
                (center[0]+radius[0], center[1]+radius[1]), color, thickness)
    
    return image


def draw_bandwidth(spectrum, fx, fy, fc, fb, save_path):
    dfx = fx[-1] - fx[-2]
    dfy = fy[-1] - fy[-2]
    rx = int(fb / 2 / dfx)  # in pixel
    ry = int(fb / 2 / dfy)  # in pixel
    cx = int(abs((fc - fx[0]) / dfx))
    cy = int(abs((fc - fy[0]) / dfy))

    circled_spectrum = draw_ellipse(abs(spectrum).cpu(), (cx, cy), (rx, ry))
    save_image(circled_spectrum, save_path)


def effective_bandwidth(D, wvls, is_plane_wave=False, zf=None, s=1.5):
    if is_plane_wave:
        bandwidth = 129.3 * s / np.pi / D
    else:
        assert zf is not None, "Wave origin should be provided!"
        bandwidth = s * D / wvls / zf

    return bandwidth


def save_image(image, save_path):
    im = Image.fromarray(np.array(image / image.max() * 255).astype(np.uint8))
    im.save(save_path)


def remove_linear_phase(phi, thetaX, thetaY, x, y, k):

    linear_phiX = -np.sin(thetaX / 180 * np.pi) * k
    linear_phiY = -np.sin(thetaY / 180 * np.pi) * k

    xx, yy = np.meshgrid(x, y, indexing='xy')
    phi_new = phi - xx * linear_phiX - yy * linear_phiY

    return np.remainder(phi_new, 2 * np.pi)


def snr(u_hat, u_ref):
    u_hat /= abs(u_hat).max()
    u_ref /= abs(u_ref).max()
    signal = np.sum(abs(u_hat)**2)
    alpha = np.sum(u_hat * np.conjugate(u_ref)) / np.sum(abs(u_ref)**2)
    snr = signal / np.sum(abs(u_hat - alpha * u_ref)**2)
    return 10 * np.log10(snr)