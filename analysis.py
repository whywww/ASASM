import numpy as np
import matplotlib.pyplot as plt
import cv2


N = 1431
thetaX = 5
thetaY = 5

read_path = f"results1/BEASM{N}-{thetaX, thetaY}.csv"
Uout = np.genfromtxt(read_path, delimiter=',')

# save phase image
save_path = f"results1/BEASM{N}-{thetaX, thetaY}-phi.png"
cv2.imwrite(save_path, (np.angle(Uout)/2/np.pi+1)*255)
