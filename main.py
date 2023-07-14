'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import numpy as np
import time
from utils import save_image, remove_linear_phase, snr
import glob
from input_field import InputField


############################### hyperparameters ############################

wvls = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / wvls  # wavenumebr
f = 35e-3  # focal length of lens (if applicable)
z0 = 1.7  # source-aperture distance
zf = 1/(1/f - 1/z0)  # image-side focal distance
z = zf  # aperture-sensor distance
r = f / 16 / 2  # radius of aperture
thetaX = thetaY = 0  # incident angle in degree

s_LSASM = 1.5  # oversampling factor for LSASM
s_RS = 4  # oversampling factor for Rayleigh-Sommerfeld
compensate = True  # LPC
use_LSASM = True
use_RS = False
result_folder = 'results'
RS_folder = 'RS'
calculate_SNR = False

# define observation window
Mx, My = 512, 512
l = r * 0.25
# l = 0.0136/1.5  # first term in Eq6 scaled by 1/1.5 to estimate OW size, used for diffuser
# l = r * 8.  # 35 degrees
xc = - z * np.sin(thetaX / 180 * np.pi) / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
yc = - z * np.sin(thetaX / 180 * np.pi) / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)

x = np.linspace(-l / 2 + xc, l / 2 + xc, Mx, endpoint=True)
y = np.linspace(-l / 2 + yc, l / 2 + yc, My, endpoint=True)
print(f'observation window diamter = {l}.')

if use_LSASM:
    print('----------------- Propagating with ASASM -----------------')
    # use "12" for thin lens + spherical wave
    # use "3" for diffuser
    Uin = InputField("12", wvls, (thetaX, thetaY), r, z0, f, zf, s_LSASM)

    from LSASM import LeastSamplingASM
    device = 'cuda:0'
    # device = 'cpu'
    prop2 = LeastSamplingASM(Uin, x, y, z, device)
    path = f'{result_folder}/LSASM({len(Uin.xi)},{len(prop2.fx)})-{thetaX}-{s_LSASM:.2f}'

    start = time.time()
    U2 = prop2(Uin.E0)
    end = time.time()
    runtime = end - start
    print(f'Time elapsed for LSASM: {runtime:.2f}')

    save_image(abs(U2), f'{path}.png', cmap='gray')
    phase = remove_linear_phase(np.angle(U2), thetaX, thetaY, x, y, k) # for visualization
    save_image(phase, f'{path}-Phi.png', cmap='twilight')

    if calculate_SNR:
        if glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy') != []:
            u_GT = np.load(glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy')[0])
            print(f'SNR is {snr(U2, u_GT):.2f}')


if use_RS:
    print('-------------- Propagating with RS integral --------------')
    Uin = InputField("12", wvls, (thetaX, thetaY), r, z0, f, zf, s_RS)

    from RS import RSDiffraction_GPU
    prop = RSDiffraction_GPU(z, Uin.xi, Uin.eta, x, y, wvls, 'cuda:0')
    path = f'{RS_folder}/RS({len(Uin.xi)})-{thetaX}-{s_RS:.1f}'
    start = time.time()
    U0 = prop(Uin.E0)
    end = time.time()
    print(f'Time elapsed for RS: {end-start:.2f}')
    save_image(abs(U0), f'{path}.png', cmap='gray')
    phase = remove_linear_phase(np.angle(U0), thetaX, thetaY, x, y, k) # for visualization
    save_image(phase, f'{path}-Phi.png', cmap='twilight')
    np.save(f'{path}', U0)