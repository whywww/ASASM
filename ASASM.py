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
    def __init__(self, thetaX, thetaY, z, xvec, yvec, svec, tvec, zf,
                wavelength, effective_bandwidths, device, crop_bandwidth=True):
        '''
        :param xvec, yvec: vectors of source coordinates
        :param wavelengths: wavelengths
        '''
        
        super().__init__()

        dtype = torch.double
        complex_dtype = torch.complex128
        eps = torch.tensor(1e-9, device=device)

        xvec, yvec = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        svec, tvec = torch.as_tensor(svec, device=device), torch.as_tensor(tvec, device=device)
        z = torch.as_tensor(z, device=device)
        wavelength = torch.as_tensor(wavelength, device=device)
        thetaX = torch.as_tensor(thetaX, device=device, dtype=dtype)
        thetaY = torch.as_tensor(thetaY, device=device, dtype=dtype)

        # maximum wavelength
        n = 1
        k = 2 * math.pi / wavelength * n

        # bandwidth of aperture
        pitchx = xvec[-1] - xvec[-2]
        pitchy = xvec[-1] - xvec[-2]
        fmax_fftX = 1 / (2 * pitchx)
        fmax_fftY = 1 / (2 * pitchy)
        if crop_bandwidth:
            fbx, fby = effective_bandwidths  # use effective bandwidth
        else:
            fbx = fmax_fftX * 2  # use full bandwidth
            fby = fmax_fftY * 2  # use full bandwidth
        Lfx = fbx
        Lfy = fby

        # off-axis offset
        s0, t0 = svec[len(svec)//2], tvec[len(tvec)//2]
        offx = -torch.sin(thetaX / 180 * math.pi) / wavelength
        offy = -torch.sin(thetaY / 180 * math.pi) / wavelength

        # shifted frequencies
        fxmax = fbx / 2 + abs(offx)
        fymax = fby / 2 + abs(offy)

        # if frequencies exceed this range, some information is lost because of evanescent wave
        # fxmax, fymax < 1 / wavelength
        thetax_max = torch.asin(1 - wavelength * fbx / 2) / math.pi * 180
        thetay_max = torch.asin(1 - wavelength * fby / 2) / math.pi * 180
        print(f'The oblique angle should not exceed ({thetax_max:.1f}, {thetay_max:.1f}) degrees.')

        # drop the evanescent wave
        fxmax = torch.clamp(fxmax, -1 / wavelength, 1 / wavelength)  
        fymax = torch.clamp(fymax, -1 / wavelength, 1 / wavelength)
        denom = max(1 - (wavelength * fxmax)**2 - (wavelength * fymax) ** 2, eps)

        # maximum sampling interval limited by TF
        # dfxMax1 = dfyMax1 = torch.sqrt(1 - (wavelength * fmax_fft) ** 2) / (2 * wavelength * z * fmax_fft)  # on-axis
        # s_f = abs(2 * wavelength * z * fxmax) / torch.sqrt(denom) - 2 * abs(s0)
        # t_f = abs(2 * wavelength * z * fymax) / torch.sqrt(denom) - 2 * abs(t0)
        s_f = abs(2 * wavelength * fxmax * (z / torch.sqrt(denom) - zf) - 2 * abs(s0))
        t_f = abs(2 * wavelength * fymax * (z / torch.sqrt(denom) - zf) - 2 * abs(t0))
        dfxMax1 = 1 / s_f
        dfyMax1 = 1 / t_f

        # maximum sampling interval limited by observation plane
        dfxMax2 = 1 / 2 / abs(svec).max()
        dfyMax2 = 1 / 2 / abs(tvec).max()

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax2, dfxMax1)
        dfy = min(dfyMax2, dfyMax1)
        print(f'Sampling interval limited by UH: {dfxMax1:.2f}, limited by observation window {dfxMax2:.2f}.')
        # dfx = dfy = min(dfx, dfy) # uncomment to make FU a squared image!
        
        oversampling = 1.5  # oversampling factor
        LRfx = int(Lfx / dfx * oversampling)
        LRfy = int(Lfy / dfy * oversampling)
        
        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        print(f'frequency sampling number = {LRfx, LRfy}.')
        print(f'using bandwidth cropping saves ({int((fmax_fftX*2-effective_bandwidths[0])/dfx2)}, {int((fmax_fftY*2-effective_bandwidths[1])/dfy2)}) freq samplings.')
        
        # spatial frequency coordinates
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)
        fx_shift, fy_shift = fx + offx, fy + offy

        fxx, fyy = torch.meshgrid(fx_shift, fy_shift, indexing='xy')
        # self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2))
        # shifted H
        self.H = torch.exp(1j * k * (wavelength * fxx * s0 + wavelength * fyy * t0 
                        + z * torch.sqrt(1 - wavelength ** 2 * (fxx ** 2 + fyy ** 2))))

        self.x = xvec.to(dtype=complex_dtype)
        self.y = yvec.to(dtype=complex_dtype)
        self.s = svec.to(dtype=complex_dtype) - s0  # shift the observation window back to origin
        self.t = tvec.to(dtype=complex_dtype) - t0
        self.offx, self.offy = offx, offy
        self.device = device
        self.mask = torch.where(abs(fxx-offx)**2 / (fbx/2)**2 + abs(fyy-offy)**2 / (fby/2)**2 <= 1, 1., 0.)
        self.fx = fx_shift
        self.fy = fy_shift
        self.eB = effective_bandwidths


    def __call__(self, E0, decomposed=False, save_path=None):
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
        if decomposed:  # the oblique wave is removed beforehand
            Fu = mdft(E0, self.x, self.y, fx - self.offx, fy - self.offy)
        else:
            Fu = mdft(E0, self.x, self.y, fx, fy)
        
        # visualize the eb region
        if save_path is not None:
            draw_bandwidth(Fu[0], self.fx, self.fy, self.eB, f'{save_path}-EB.png')
        
        Fu *= self.mask
        Eout = midft(Fu * self.H, self.s, self.t, fx, fy)

        if save_path is not None:
            save_image(abs(Fu[0])/torch.max(abs(Fu[0])), f'{save_path}-FU.png')

        return Eout[0].cpu().numpy()