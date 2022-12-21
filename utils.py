import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_ellipse(image, center, axes):

    thickness = 2
    image = np.repeat(np.array(image)[..., None], 3, axis=-1)
    image /= image.max() 
    color = (215, 164, 0)  # BGR
    
    image = cv2.ellipse(image, center, axes, 0., 0., 360., color, thickness=thickness)
    
    return image * 255


def draw_bandwidth(spectrum, fx, fy, bandwidths, save_path):
    dfx = fx[-1] - fx[-2]
    dfy = fy[-1] - fy[-2]
    lx = len(fx)
    ly = len(fy)
    rx = int(bandwidths[0] / 2 / dfx)  # in pixel
    ry = int(bandwidths[1] / 2 / dfy)  # in pixel

    circled_spectrum = draw_ellipse(abs(spectrum).cpu(), (lx//2, ly//2), (rx, ry))
    cv2.imwrite(save_path, circled_spectrum)


def compute_bandwidth(is_plane_wave, D, wvls, pitchx, pitchy, l1=None, s=5):
    if is_plane_wave:
            bandwidth = 2 * 1.22 * s / D
    else:
        assert l1 is not None, "Wave origin should be provided!"
        bandwidth = s * D / wvls / l1  # physical

    bandwidthX = min(1 / pitchx, bandwidth)
    bandwidthY = min(1 / pitchy, bandwidth)

    return bandwidthX, bandwidthY