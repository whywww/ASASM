import torch
import math
from utils import draw_bandwidth, save_image


def mdft(in_matrix, x, y, fx, fy):
    'x,fx: vertical; y,fy: horizontal'
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-2)
    fx = fx.unsqueeze(-2)
    fy = fy.unsqueeze(-1)
    mx = torch.exp(-2 * torch.pi * 1j * torch.matmul(x, fx))
    my = torch.exp(-2 * torch.pi * 1j * torch.matmul(fy, y))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lx = torch.numel(x)
    ly = torch.numel(y)
    if lx == 1:
        dx = 1
    else:
        dx = (torch.squeeze(x)[-1] - torch.squeeze(x)[0]) / (lx - 1)

    if ly == 1:
        dy = 1
    else:
        dy = (torch.squeeze(y)[-1] - torch.squeeze(y)[0]) / (ly - 1)

    out_matrix = out_matrix * dx * dy  # the result is only valid for uniform sampling
    return out_matrix


def midft(in_matrix, x, y, fx, fy):
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-1)
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-2)
    mx = torch.exp(2 * torch.pi * 1j * torch.matmul(fx, x))
    my = torch.exp(2 * torch.pi * 1j * torch.matmul(y, fy))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lfx = torch.numel(fx)
    lfy = torch.numel(fy)
    if lfx == 1:
        dfx = 1
    else:
        dfx = (torch.squeeze(fx)[-1] - torch.squeeze(fx)[0]) / (lfx - 1)

    if lfy == 1:
        dfy = 1
    else:
        dfy = (torch.squeeze(fy)[-1] - torch.squeeze(fy)[0]) / (lfy - 1)

    out_matrix = out_matrix * dfx * dfy  # the result is only valid for uniform sampling
    return out_matrix


class AdpativeSamplingASM():
    def __init__(self, Uin, xvec, yvec, z, device):
        '''
        :param Uin: input field object
        :param xvec, yvec: vectors of destination coordinates
        :param z: propagation distance
        '''
        
        super().__init__()

        dtype = torch.double
        complex_dtype = torch.complex128

        xivec, etavec = torch.as_tensor(Uin.xi, device=device), torch.as_tensor(Uin.eta, device=device)
        xvec, yvec = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        z = torch.as_tensor(z, device=device)
        wavelength = torch.as_tensor(Uin.wvls, device=device)

        # maximum wavelength
        n = 1
        k = 2 * math.pi / wavelength * n

        # bandwidth of aperture
        Lfx = Uin.fbX
        Lfy = Uin.fbY
        # Lfx = Lfy = 54400

        # off-axis offset
        xc, yc = xvec[len(xvec) // 2], yvec[len(yvec) // 2]
        wx = xvec[-1] - xvec[0]
        wy = yvec[-1] - yvec[0]
        offx = torch.as_tensor(Uin.fcX, device=device)
        offy = torch.as_tensor(Uin.fcY, device=device)

        # shifted frequencies
        fxmax = Lfx / 2 + abs(offx)
        fymax = Lfy / 2 + abs(offy)

        # if frequencies exceed this range, some information is lost because of evanescent wave
        # fxmax, fymax < 1 / wavelength
        thetax_max = torch.asin(1 - wavelength * Lfx / 2) / math.pi * 180
        thetay_max = torch.asin(1 - wavelength * Lfy / 2) / math.pi * 180
        print(f'The oblique angle should not exceed ({thetax_max:.1f}, {thetay_max:.1f}) degrees.')

        # drop the evanescent wave
        fxmax = torch.clamp(fxmax, -1 / wavelength, 1 / wavelength)  
        fymax = torch.clamp(fymax, -1 / wavelength, 1 / wavelength)

        # combined phase gradient analysis
        gx1, gy1 = self.grad_H(wavelength, z, Lfx / 2 + offx, Lfy / 2 + offy)
        gx2, gy2 = self.grad_H(wavelength, z, -Lfx / 2 + offx, -Lfy / 2 + offy)
        FHcx = (gx1 + gx2) / (4 * torch.pi)
        FHcy = (gy1 + gy2) / (4 * torch.pi)

        if Uin.type == "12":
            hx = k * Uin.zf * wavelength**2 * Lfx / 2
            hy = k * Uin.zf * wavelength**2 * Lfy / 2
            FUHbx = abs((hx + gx1) - (-hx + gx2)) / (2 * torch.pi)
            FUHby = abs((hy + gy1) - (-hy + gy2)) / (2 * torch.pi)

            # to verify the extrema of g+h
            # fx = fy = torch.linspace(-Lfx / 2, Lfx / 2 , 100, device=device)
            # ghx, ghy = self.grad_H(wavelength, z, fx+offx, fy+offy)
            # ghx += k * Uin.zf * wavelength**2 * fx
            # ghy += k * Uin.zf * wavelength**2 * fy
            # FUHbx = (max(ghx) - min(ghx)) / (2 * torch.pi)
            # FUHby = (max(ghy) - min(ghy)) / (2 * torch.pi)

            deltax = self.compute_shift_of_H(FHcx, FUHbx + 2 * Uin.D, xc, wx)
            deltay = self.compute_shift_of_H(FHcy, FUHby + 2 * Uin.D, yc, wy)
            FUHcx_shifted = FHcx + deltax
            FUHcy_shifted = FHcy + deltay
            
            tau_UHx = 2 * abs(FUHcx_shifted) + FUHbx + 2 * Uin.D
            tau_UHy = 2 * abs(FUHcy_shifted) + FUHby + 2 * Uin.D
        else:
            tau_UHx = tau_UHy = torch.inf

        # upper bound
        FHbx = abs(gx1 - gx2) / (2 * torch.pi)
        FHby = abs(gy1 - gy2) / (2 * torch.pi)

        deltax = self.compute_shift_of_H(FHcx, FHbx + Uin.D, xc, wx)
        deltay = self.compute_shift_of_H(FHcy, FHby + Uin.D, yc, wy)
        FHcx_shifted = FHcx + deltax
        FHcy_shifted = FHcy + deltay

        tau_fx_bound = 2 * abs(FHcx_shifted) + FHbx + Uin.D
        tau_fy_bound = 2 * abs(FHcy_shifted) + FHby + Uin.D

        # final phase gradient
        tau_UHx = min(tau_UHx, tau_fx_bound)
        tau_UHy = min(tau_UHy, tau_fy_bound)

        dfxMax1 = 1 / tau_UHx
        dfyMax1 = 1 / tau_UHy

        # maximum sampling interval limited by OW
        dfxMax2 = 1 / (2 * abs(xc - deltax) + wx)
        dfyMax2 = 1 / (2 * abs(yc - deltay) + wy)

        # maximum sampling interval limited by diffraction limit
        # dfxMax3 = 1 / (41.2 * (xvec[-1] - xvec[-2]))
        # dfyMax3 = 1 / (41.2 * (yvec[-1] - yvec[-2]))

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax1, dfxMax2)
        dfy = min(dfyMax1, dfyMax2)
        print(f'Sampling interval limited by UH: {dfxMax1 <= dfxMax2}.')
        
        LRfx = math.ceil(Lfx / dfx * Uin.s)
        LRfy = math.ceil(Lfy / dfy * Uin.s)
        
        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        print(f'frequency sampling number = {LRfx, LRfy}, bandwidth = {Lfx:.2f}.')
        
        # spatial frequency coordinates
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)
        fx_shift, fy_shift = fx + offx, fy + offy

        fxx, fyy = torch.meshgrid(fx_shift, fy_shift, indexing='xy')
        # self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2))
        # shifted H
        self.H = torch.exp(1j * k * (wavelength * fxx * deltax + wavelength * fyy * deltay 
                        + z * torch.sqrt(1 - (fxx * wavelength)**2 - (fyy * wavelength)**2)))
        
        self.xi = xivec.to(dtype = complex_dtype)
        self.eta = etavec.to(dtype = complex_dtype)
        self.x = xvec.to(dtype = complex_dtype) - deltax  # shift the observation window back to origin
        self.y = yvec.to(dtype = complex_dtype) - deltay
        self.offx, self.offy = offx, offy
        self.device = device
        self.fx = fx_shift
        self.fy = fy_shift
        self.fbX = Uin.fbX
        self.fbY = Uin.fbY


    def __call__(self, E0, compensate=True, save_path=None):
        '''
        :param E0: input field torch.angle(E0)
        :param z: propagation distance
        :param xo, yo, zo: point source location, used to calculate frequency sampling
        :param xd, yd: source plane vectors
        :param xs, ys: destination plane vectors
        :param z: travel distance
        '''

        E0 = torch.as_tensor(E0, dtype=torch.complex128, device=self.device)

        fx = self.fx.unsqueeze(0)
        fy = self.fy.unsqueeze(0)

        if compensate:
            Fu = mdft(E0, self.xi, self.eta, fx - self.offx, fy - self.offy)
        else:
            Fu = mdft(E0, self.xi, self.eta, fx, fy)
        
        if save_path is not None:
            if compensate:
                draw_bandwidth(Fu[0], self.fx, self.fy, self.offx, self.offy, self.fbX, self.fbY, f'{save_path}-Fb.png')
            else:
                draw_bandwidth(Fu[0], self.fx - self.offx, self.fy - self.offy, self.offx, self.offy, self.fbX, self.fbY, f'{save_path}-Fb.png')

        Eout = midft(Fu * self.H, self.x, self.y, fx, fy)
        # Eout /= abs(Eout).max() # we dont need to normalize using MTP.

        return Eout[0].cpu().numpy(), abs(Fu[0]).cpu().numpy()


    def grad_H(self, lam, z, fx, fy):

        eps = torch.tensor(1e-9, device = fx.device)
        denom = torch.max(1 - (lam * fx)**2 - (lam * fy) ** 2, eps)
        gradx = - z * 2 * torch.pi * lam * fx / torch.sqrt(1 - (lam * fx)**2 - (lam * fy)**2)
        grady = - z * 2 * torch.pi * lam * fy / torch.sqrt(1 - (lam * fx)**2 - (lam * fy)**2)
        return gradx, grady

    
    def compute_shift_of_H(self, C1, C2, pc, w):

        if (w > -2 * C1 - 2 * pc + C2) and (w < 2 * C1 + 2 * pc + C2):
            delta = pc / 2 + w / 4 - C1 / 2 - C2 / 4
        elif (w > 2 * C1 + 2 * pc + C2) and (w < -2 * C1 - 2 * pc + C2):
            delta = pc / 2 - w / 4 - C1 / 2 + C2 / 4
        elif (w > 2 * C1 + 2 * pc + C2) and (w > -2 * C1 - 2 * pc + C2):
            delta = pc
        elif (w < 2 * C1 + 2 * pc + C2) and (w < -2 * C1 - 2 * pc + C2):
            delta = -C1

        return delta