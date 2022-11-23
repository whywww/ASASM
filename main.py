'''
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
'''

import os
import configparser
from collections import namedtuple
import matplotlib.pyplot as plt

gpus = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # do not move

import torch
import torchvision.utils as vutils

from camera import LensCamera


def create_dirs(dir_result):
    
    try:
        if not os.path.exists(dir_result):
            os.makedirs(dir_result)
            print('Created result directory')
    except OSError:
        pass


def save_fig(img, pic_name, title='', vmin=None, vmax=None):
    'This works for single image with small values and gives the true range. [1,c,h,w]'
    
    img = img.to('cpu')
    if img.size(-3) == 1:
        c = 'gray'
        img = img[0,0,...]  # h,w
    else:
        c = None
        img = img[0].permute(1,2,0)  # h,w,c
    plt.imshow(img, cmap=c, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.savefig(pic_name)
    plt.close()
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device} with {torch.cuda.device_count()} GPUs: # {gpus}')

    config = configparser.ConfigParser(inline_comment_prefixes=";")
    config.read('config.ini')
    print('Using config.ini')
    param_camera = namedtuple('Camera', config['Camera'].keys())(**config['Camera'])
    dir_result = config['General']['dir_result']
    create_dirs(dir_result)

    cam = LensCamera(param_camera, device)
    theta = float(param_camera.angle)
    psf = cam.get_PSF(theta = theta)
    save_fig(torch.sqrt(psf), f'{dir_result}/psf_amp.png', fr'PSF from $\theta={theta}^\circ$ at z={cam.zs:.3f}m')

    with open(f'{dir_result}/config.ini', 'w') as configfile:
        config.write(configfile)