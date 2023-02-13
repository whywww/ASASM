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
        :param xvec, yvec: vectors of source coordinates
        :param wavelengths: wavelengths
        '''
        
        super().__init__()

        dtype = torch.double
        complex_dtype = torch.complex128
        eps = torch.tensor(1e-9, device=device)

        xivec, etavec = torch.as_tensor(Uin.xi, device=device), torch.as_tensor(Uin.eta, device=device)
        xvec, yvec = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        z = torch.as_tensor(z, device=device)
        wavelength = torch.as_tensor(Uin.wvls, device=device)
        # thetaX = torch.as_tensor(thetaX, device=device, dtype=dtype)
        # thetaY = torch.as_tensor(thetaY, device=device, dtype=dtype)

        # maximum wavelength
        n = 1
        k = 2 * math.pi / wavelength * n

        # bandwidth of aperture
        # pitchx = xvec[-1] - xvec[-2]
        # pitchy = yvec[-1] - yvec[-2]
        # fftmax_X = 1 / (2 * pitchx)
        # fftmax_Y = 1 / (2 * pitchy)
        # Lfx = fftmax_X * 2
        # Lfy = fftmax_Y * 2
        Lfx = Uin.fb 
        Lfy = Uin.fb 

        # off-axis offset
        xc, yc = xvec[len(xvec) // 2], yvec[len(yvec) // 2]
        offx = torch.as_tensor(Uin.fc, device=device)
        offy = torch.as_tensor(Uin.fc, device=device)

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

        # maximum sampling interval limited by TF & Spectrum (Eq. S25)
        denom = max(1 - (wavelength * fxmax)**2 - (wavelength * fymax) ** 2, eps)
        # tau_u = 2 * abs(zf * wavelength * Lfx / 2)
        # tau_H = 2 * abs(-wavelength * z / torch.sqrt(denom) * fxmax + s0)
        # tau_x = tau_y = max(tau_u, tau_H)
        tau_x = 2 * abs(Lfx / 2 * wavelength * (Uin.zf - z / torch.sqrt(denom))) + 2 * abs(-z * wavelength * offx / torch.sqrt(denom) + xc)
        tau_y = 2 * abs(Lfy / 2 * wavelength * (Uin.zf - z / torch.sqrt(denom))) + 2 * abs(-z * wavelength * offy / torch.sqrt(denom) + yc)
        dfxMax1 = 1 / tau_x
        dfyMax1 = 1 / tau_y

        # maximum sampling interval limited by observation plane
        dfxMax2 = 1 / 2 / (xvec.max() - xvec.min())
        dfyMax2 = 1 / 2 / (yvec.max() - yvec.min())

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax2, dfxMax1)
        dfy = min(dfyMax2, dfyMax1)
        print(f'Sampling interval limited by UH: {dfxMax1 <= dfxMax2}.')
        
        LRfx = math.ceil(Lfx / dfx * Uin.s)
        LRfy = math.ceil(Lfy / dfy * Uin.s)
        # LRfx = LRfy = 2 * len(Uin.xi)
        
        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        print(f'frequency sampling number = {LRfx, LRfy}, bandwidth = {Lfx:.2f}.')
        # print(f'using bandwidth cropping saves ({int((fmax_fftX*2-effective_bandwidth)/dfx2)}, {int((fmax_fftY*2-effective_bandwidth)/dfy2)}) freq samplings.')
        
        # spatial frequency coordinates
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)
        fx_shift, fy_shift = fx + offx, fy + offy

        fxx, fyy = torch.meshgrid(fx_shift, fy_shift, indexing='xy')
        # self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2))
        # shifted H
        self.H = torch.exp(1j * k * (wavelength * fxx * xc + wavelength * fyy * yc 
                        + z * torch.sqrt(1 - (fxx * wavelength)**2 - (fyy * wavelength)**2)))
        
        self.xi = xivec.to(dtype = complex_dtype)
        self.eta = etavec.to(dtype = complex_dtype)
        self.x = xvec.to(dtype=complex_dtype) - xc  # shift the observation window back to origin
        self.y = yvec.to(dtype=complex_dtype) - yc
        self.offx, self.offy = offx, offy
        self.device = device
        # self.mask = torch.where(abs(fxx-offx)**2 / (Lfx / 2)**2 + abs(fyy-offy)**2 / (Lfy / 2)**2 <= 1, 1., 0.)
        self.fx = fx_shift
        self.fy = fy_shift
        self.fb = Uin.fb


    def __call__(self, E0, compensate=True, save_path=None):
        '''
        :param E0: input field
        :param z: propagation distance
        :param xo, yo, zo: point source location, used to calculate frequency sampling
        :param xd, yd: source plane vectors
        :param xs, ys: destination plane vectors
        :param z: travel distance
        '''

        E0 = torch.as_tensor(E0, dtype=torch.complex128, device=self.device)

        fx = self.fx.unsqueeze(0)
        fy = self.fy.unsqueeze(0)

        Fu = mdft(E0, self.xi, self.eta, fx - self.offx, fy - self.offy)
        # Fu = mdft(E0, self.xi, self.eta, fx, fy)  # uncomment this to get the correct uncompensated result
        
        if save_path is not None:
            # save_image(torch.angle(self.H), f'{save_path}-H.png')
            # save_image(torch.angle(Fu * self.H)[0], f'{save_path}-FuH.png')
            # save_image(torch.angle(Fu)[0], f'{save_path}-Fu.png')
            if compensate:
                draw_bandwidth(Fu[0], self.fx, self.fy, self.offx, self.fb, f'{save_path}-Fb.png')
            else:
                draw_bandwidth(Fu[0], self.fx - self.offx, self.fy - self.offy, self.offx, self.fb, f'{save_path}-Fb.png')

        Eout = midft(Fu * self.H, self.x, self.y, fx, fy)
        Eout /= abs(Eout).max()

        return Eout[0].cpu().numpy(), abs(Fu[0]).cpu().numpy()