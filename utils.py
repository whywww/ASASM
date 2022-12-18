import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_ellipse(image, center, axes):

    thickness = 1
    image = np.repeat(np.array(image)[..., None], 3, axis=-1)
    image /= image.max() 
    color = (215, 164, 0)  # BGR
    
    image = cv2.ellipse(image, center, axes, 0., 0., 360., color, thickness=thickness)
    
    return image * 255


# def effective_bandwidth(l1, R, lam, fx, fy, plane_wave=False):

#     s = 1.5  # expansion factor

#     df = fx[-1] - fx[-2]
#     lx = len(fx)
#     ly = len(fy)

#     if plane_wave:
#         bandwidth = 1.22 * s / 2 / R
#     else:
#         bandwidth = s * 2 * R / lam / l1  # physical

#     radius = int(bandwidth / 2 / df)  # in pixel

#     # circled = drawcircle(abs(Fu[0]), (lx//2, ly//2), radius)

#     # crop and mask
#     fx = fx[lx//2 - radius: lx // 2 + radius]
#     fy = fy[ly//2 - radius: ly // 2 + radius]

#     # if torch.is_tensor(fx):
#     #     fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
#     #     mask = torch.where(abs(fxx)**2 + abs(fyy)**2 <= (bandwidth/2)**2, 1, 0)
#     # else:
#     #     fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
#     #     mask = np.where(abs(fxx)**2 + abs(fyy)**2 <= (bandwidth/2)**2, 1, 0)

#     return fx, fy


class Effective_Bandwidth():
    def __init__(self, is_plane_wave, D, wvls, l1=None, s=1.5) -> None:
        
        if is_plane_wave:
            self.bandwidth = 2 * 1.22 * s / D
        else:
            assert l1 is not None, "Wave origin should be provided!"
            self.bandwidth = s * D / wvls / l1  # physical
    

    def crop_bandwidth(self, fx, fy):

        dfx = fx[-1] - fx[-2]
        dfy = fy[-1] - fy[-2]
        lx = len(fx)
        ly = len(fy)

        rx = int(self.bandwidth / 2 / dfx)  # in pixel
        ry = int(self.bandwidth / 2 / dfy)  # in pixel
        assert rx <= lx and ry <= ly, 'the expansion factor is too large!'

        # crop and mask
        fx = fx[lx // 2 - rx: lx // 2 + rx]
        fy = fy[ly // 2 - ry: ly // 2 + ry]

        if torch.is_tensor(fx):
            fxx, fyy = torch.meshgrid(fx-fx[len(fx)//2], fy-fy[len(fy)//2], indexing='xy')
            mask = torch.where(abs(fxx)**2 + abs(fyy)**2 <= (self.bandwidth/2)**2, 1., 0.)
        else:
            fxx, fyy = np.meshgrid(fx-fx[len(fx)//2], fy-fy[len(fy)//2], indexing='xy')
            mask = np.where(abs(fxx)**2 + abs(fyy)**2 <= (self.bandwidth/2)**2, 1., 0.)

        return fx, fy, mask

    
    def draw_bandwidth(self, fx, fy, spectrum, save_path):
        
        dfx = fx[-1] - fx[-2]
        dfy = fy[-1] - fy[-2]
        lx = len(fx)
        ly = len(fy)
        rx = int(self.bandwidth / 2 / dfx)  # in pixel
        ry = int(self.bandwidth / 2 / dfy)  # in pixel

        circled_spectrum = draw_ellipse(abs(spectrum), (lx//2, ly//2), (rx, ry))
        cv2.imwrite(save_path, circled_spectrum)
        # plt.imshow(circled_spectrum, cmap='gray')
        # plt.savefig(save_path)
        # plt.close()
