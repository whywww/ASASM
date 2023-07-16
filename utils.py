import numpy as np
from PIL import Image 
from matplotlib import cm


def effective_bandwidth(D, wvls=None, is_plane_wave=False, zf=None, s=1.):
    if is_plane_wave:
        bandwidth = 41.2 * s / D
    else:
        assert zf is not None, "Wave origin should be provided!"
        bandwidth = s * D / wvls / zf

    return bandwidth


def save_image(image, save_path, cmap='gray'):

    imarray = np.array(image / image.max())  # 0~1
    if cmap == 'viridis':
        imarray = cm.viridis(imarray)
    elif cmap == 'twilight':
        imarray = cm.twilight(imarray)
    elif cmap == 'magma':
        imarray = cm.magma(imarray)
    elif cmap == 'plasma':
        imarray = cm.plasma(imarray)
    im = Image.fromarray(np.uint8(imarray * 255))
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