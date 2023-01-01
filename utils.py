import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image 


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


def save_image(image, save_path):
    image /= image.max()
    im = Image.fromarray(np.uint8(image*255))
    im.save(save_path)


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


def psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))