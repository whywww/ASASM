import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from utils import signaltonoise_dB


def save_image(image, save_path):
    image /= image.max()
    im = Image.fromarray(np.uint8(image*255))
    im.save(save_path)


N = 2048
thetaX = 10
thetaY = 10


################ Single Accuracy #####################

# read_path = f"results/BEASM{N,N}-{thetaX,thetaY}"

im_GT = Image.open(f'results/RS{N,N}-{thetaX,thetaY}.png')
im_BEASM = Image.open(f'results/BEASM{N,N}-{thetaX,thetaY}.png')
im_ASASM = Image.open(f'results/ASASM{N,N}-{thetaX,thetaY}.png')

db = signaltonoise_dB()