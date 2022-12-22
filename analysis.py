import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 

def save_image(image, save_path):
    image /= image.max()
    im = Image.fromarray(np.uint8(image*255))
    im.save(save_path)


N = 2048
thetaX = 10
thetaY = 10

read_path = f"results/BEASM{N,N}-{thetaX,thetaY}"
