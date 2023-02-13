'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import save_image, remove_linear_phase, snr
from tqdm import tqdm
import glob
from input_field import InputField


# hyperparameters
wvls = 500e-9  # wavelength of light in vacuum
k = 2 * np.pi / wvls  # wavenumebr
f = 35e-3  # focal length of lens (if applicable)
z0 = 1.7  # source-aperture distance
zf = 1/(1/f - 1/z0)  # image-side focal distance
z = zf  # aperture-sensor distance
r = f / 16 / 2  # radius of aperture
thetaX = thetaY = 0  # incident angle

s_ASASM = 2.5  # expansion factor
s_BEASM = 1.  # expansion factor
s_RS = 4.
compensate = True
times = 1  # number of times to run for each method
use_BEASM = False
use_ASASM = True
use_RS = False
result_folder = 'results'
RS_folder = 'RS'
device_RS = 'cuda:2'
calculate_SNR = False

# define observation window
# Mx, My = 512, 512
# l = r * 0.5
Mx, My = 1024, 1024
l = r * 0.25
xc = - z * np.sin(thetaX / 180 * np.pi) / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)
yc = - z * np.sin(thetaX / 180 * np.pi) / np.sqrt(1 - np.sin(thetaX / 180 * np.pi)**2 - np.sin(thetaY / 180 * np.pi)**2)

x = np.linspace(-l / 2 + xc, l / 2 + xc, Mx, endpoint=False)
y = np.linspace(-l / 2 + yc, l / 2 + yc, My, endpoint=False)
print(f'observation window diamter = {l}.')

if use_ASASM:
    print('----------------- Propagating with ASASM -----------------')
    Uin = InputField(2, wvls, thetaX, r, z0, f, zf, s_ASASM, compensate)

    from ASASM import AdpativeSamplingASM
    device = 'cpu'  # or 'cuda'
    prop2 = AdpativeSamplingASM(Uin, x, y, z, device)
    path = f'{result_folder}/ASASM({len(Uin.xi)},{len(prop2.fx)})-{thetaX}-{s_ASASM:.2f}'
    runtime = 0
    for i in tqdm(range(times)):
        start = time.time()
        U2, Fu = prop2(Uin.E0, compensate, path)
        end = time.time()
        runtime += end - start
    print(f'Time elapsed for ASASM: {runtime / times:.2f}')
    save_image(abs(U2), f'{path}.png')
    # phase = np.angle(U2) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U2), thetaX, thetaY, x, y, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    save_image(Fu, f'{path}-FU.png')
    np.save(f'{path}', U2)
    if calculate_SNR:
        if glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy') != []:
            u_GT = np.load(glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy')[0])
            print(f'SNR is {snr(U2, u_GT):.2f}')


if use_BEASM:
    print('-------------- Propagating with shift BEASM --------------')
    Uin = InputField(2, wvls, thetaX, r, z0, f, zf, s_BEASM, compensate=False)

    from shift_BEASM import shift_BEASM2d
    prop1 = shift_BEASM2d(Uin, x, y, z) #, len(prop2.fx)
    path = f'{result_folder}/BEASM({len(Uin.xi)},{prop1.Lfx})-{thetaX}-{s_BEASM:.3f}'
    runtime = 0
    for i in tqdm(range(times)):
        start = time.time()
        U1, Fu = prop1(Uin.E0)
        end = time.time()
        runtime += end - start
    print(f'Time elapsed for Shift-BEASM: {runtime / times:.2f}')
    save_image(abs(U1), f'{path}.png')
    # phase = np.angle(U1) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U1), thetaX, thetaY, x, y, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    save_image(Fu, f'{path}-FU.png')
    np.save(f'{path}', U1)
    if calculate_SNR:
        if glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy') != []:
            u_GT = np.load(glob.glob(f'{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy')[0])
            print(f'SNR is {snr(U1, u_GT):.2f}')


if use_RS:
    print('-------------- Propagating with RS integral --------------')
    Uin = InputField(1, wvls, thetaX, r, z0, f, zf, s_RS, compensate=False)

    # from RS import RSDiffraction_INT  # cpu, super slow
    # prop = RSDiffraction_INT()
    # U0 = prop(E1, z, x, y, s, t, lam)
    from RS import RSDiffraction_GPU
    prop = RSDiffraction_GPU(z, Uin.xi, Uin.eta, x, y, wvls, device_RS)
    path = f'{RS_folder}/RS({len(Uin.xi)})-{thetaX}-{s_RS:.1f}'
    start = time.time()
    U0 = prop(Uin.E0)
    end = time.time()
    print(f'Time elapsed for RS: {end-start:.2f}')
    save_image(abs(U0), f'{path}.png')
    # phase = np.angle(U0) % (2*np.pi)
    phase = remove_linear_phase(np.angle(U0), thetaX, thetaY, x, y, k) # for visualization
    save_image(phase, f'{path}-Phi.png')
    np.save(f'{path}', U0)