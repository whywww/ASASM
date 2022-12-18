import torch
import math
# from utils import effective_bandwidth


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
    def __init__(self, st_offset, z, xvec, yvec, svec, tvec, wavelength, device):
        '''
        :param xvec, yvec: vectors of source coordinates
        :param wavelengths: wavelengths
        '''
        
        super().__init__()

        dtype = torch.double
        complex_dtype = torch.complex128
        eps = 1e-9

        xvec, yvec = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        svec, tvec = torch.as_tensor(svec, device=device), torch.as_tensor(tvec, device=device)
        z = torch.as_tensor(z, device=device)
        wavelength = torch.as_tensor(wavelength, device=device)

        # maximum wavelength
        # n = 1
        # k = 2 * math.pi / wavelength * n

        # bandwidth of aperture
        pitch = xvec[-1] - xvec[-2]
        fmax_fft = 1 / (2 * pitch)
        Lfx = 2 * fmax_fft
        Lfy = 2 * fmax_fft

        # off-axis offset
        s0, t0 = st_offset
        offx = s0 / torch.sqrt(s0**2 + t0**2 + z**2) / wavelength
        offy = t0 / torch.sqrt(s0**2 + t0**2 + z**2) / wavelength

        # maximum sampling interval limited by TF
        # dfMax1 = torch.sqrt(1 - (maxLambda * fmax_fft) ** 2) / (2 * maxLambda * self.z * fmax_fft)
        fxmax = fmax_fft + abs(offx)
        fymax = fmax_fft + abs(offy)
        fxmax = torch.clamp(fxmax, -1/wavelength, 1/wavelength)  # drop the evanescent wave
        fymax = torch.clamp(fymax, -1/wavelength, 1/wavelength)  # drop the evanescent wave

        denom = max(1 - (wavelength * fxmax)**2 - (wavelength * fymax) ** 2, eps)
        checkterm = -z * offx * wavelength / denom 
        print(f'adding H shift will decrease sampling rate: { abs(checkterm) > abs(checkterm + s0)}')
        s_f = (2 * wavelength * z * fxmax) / torch.sqrt(denom) + 2 * abs(s0)
        t_f = (2 * wavelength * z * fymax) / torch.sqrt(denom) + 2 * abs(t0)
        dfxMax1 = 1 / s_f
        dfyMax1 = 1 / t_f

        # maximum sampling interval limited by observation plane
        dfxMax2 = 1 / 2 / abs(svec).max()
        dfyMax2 = 1 / 2 / abs(tvec).max()

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax2, dfxMax1)
        dfy = min(dfyMax2, dfyMax1)
        
        s = 1  # oversampling factor
        LRfx = int(Lfx / dfx * s)
        LRfy = int(Lfy / dfy * s)
        
        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        # spatial frequency coordinate
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)

        # fxx, fyy = torch.meshgrid(fx + offx, fy + offy, indexing='xy')
        # self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2))
        
        print(f'interval: {torch.abs(fx)[-1]-torch.abs(fx)[-2]:.2f}, length: {LRfx,LRfy}, bandwidth: {Lfx, Lfy}')

        # shifting H window
        # self.H = torch.exp(1j * k * (wavelength * fxx * s0 + wavelength * fyy * t0 
        #                 + z * torch.sqrt(1 - wavelength ** 2 * (fxx ** 2 + fyy ** 2))))

        self.x = xvec.to(dtype=complex_dtype)
        self.y = yvec.to(dtype=complex_dtype)
        self.s = svec.to(dtype=complex_dtype)
        self.t = tvec.to(dtype=complex_dtype)
        self.offx, self.offy = offx, offy
        self.fx, self.fy = fx, fy
        self.lam = wavelength
        self.device = device
        self.z = z
        self.s0, self.t0 = s0, t0


    def __call__(self, E0, effective_bandwidth):
        '''
        :param E0: input field
        :param z: propagation distance
        :param xo, yo, zo: point source location, used to calculate frequency sampling
        :param xd, yd: source plane vectors
        :param xs, ys: destination plane vectors
        :param z: travel distance
        '''

        E0 = torch.as_tensor(E0, dtype=torch.complex128, device=self.device)
        
        fx = self.fx + self.offx
        fy = self.fy + self.offy

        # fxe, fye, mask = effective_bandwidth.crop_bandwidth(fx, fy)
        fxe, fye, mask = fx, fy, torch.ones(len(fy), len(fx))  # without cropping

        fxx, fyy = torch.meshgrid(fxe, fye, indexing='xy')
        self.H = torch.exp(1j * 2 * math.pi / self.lam * (self.lam * fxx * self.s0 + self.lam * fyy * self.t0 
                            + self.z * torch.sqrt(1 - (self.lam * fxx) ** 2 - (self.lam * fyy) ** 2)))

        fxe = fxe.unsqueeze(0)
        fye = fye.unsqueeze(0)
        Fu = mdft(E0, self.x, self.y, fxe, fye)
        effective_bandwidth.draw_bandwidth(fx, fy, Fu[0], 'results1/effective_bandwidth.png')
        Fu *= mask
        Eout = midft(Fu * self.H, self.s - self.s0, self.t - self.t0, fxe, fye)

        # abs(Fu)/torch.max(abs(Fu)) # torch.angle(E0)
        # abs(Eout) # torch.angle(self.H)

        return Eout[0].cpu().numpy()


class AngularSpectrumMethodMM():
    def __init__(self, st_offset, z, xvec, yvec, svec, tvec, wavelengths, device):
        '''
        :param xvec, yvec: vectors of source coordinates
        :param wavelengths: wavelengths
        '''

        super(AngularSpectrumMethodMM, self).__init__()

        dtype = torch.double
        complex_dtype = torch.complex128
        eps = torch.tensor(1e-9)

        self.device = device
        self.x, self.y = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        self.s, self.t = torch.as_tensor(svec, device=device), torch.as_tensor(tvec, device=device)
        

        z = torch.as_tensor(z, device=device)
        s0, t0 = st_offset
        # x0, y0, zo = torch.as_tensor(o_loc)
        # s0, t0 = x0 / zo * z, y0 / zo * z
        self.s, self.t = self.s - s0, self.t - t0
        wavelengths = torch.as_tensor(wavelengths, device=device)

        # maximum wavelength
        n = 1
        k = 2 * math.pi / wavelengths * n
        maxLambda = wavelengths.max()

        # bandwidth of aperture (restricted by dx)
        pitch = self.x[-1] - self.x[-2]
        fmax_fft = 1 / (2 * pitch)

        # effective bandwidth of input field
        f_b = fmax_fft  # replace it with the effective bandwidth
        Lfx = 2 * f_b
        Lfy = 2 * f_b

        # off-axis spectrum offset
        # self.offx = x0 / torch.sqrt(x0**2 + y0**2 + zo**2) / maxLambda
        # self.offy = y0 / torch.sqrt(x0**2 + y0**2 + zo**2) / maxLambda
        self.offx = -s0 / torch.sqrt(s0**2 + t0**2 + z**2) / maxLambda
        self.offy = -t0 / torch.sqrt(s0**2 + t0**2 + z**2) / maxLambda
        
        # maximum sampling interval limited by TF
        fx_l, fx_u = -f_b-self.offx, f_b-self.offx  # bound after frequency shift
        fy_l, fy_u = -f_b-self.offy, f_b-self.offy

        theta_max = torch.asin(1-maxLambda*f_b) / math.pi * 180
        print(f'The oblique angle should not exceed {theta_max:.2f} degrees!')

        fx_l = torch.clamp(fx_l, -1/maxLambda, 1/maxLambda)  # drop the evanescent wave
        fx_u = torch.clamp(fx_u, -1/maxLambda, 1/maxLambda)
        fy_l = torch.clamp(fy_l, -1/maxLambda, 1/maxLambda)
        fy_u = torch.clamp(fy_u, -1/maxLambda, 1/maxLambda)

        fx_abs_max = max(torch.abs(fx_l), torch.abs(fx_u))
        fy_abs_max = max(torch.abs(fy_l), torch.abs(fy_u))
        # fx_abs_max = 0  # 1D case
        # fy_abs_max = 0  # 1D case

        fr_xl = max(1-(maxLambda * fx_l) ** 2-(maxLambda * fy_abs_max) ** 2, eps)
        fr_xu = max(1-(maxLambda * fx_u) ** 2-(maxLambda * fy_abs_max) ** 2, eps)
        fr_yl = max(1-(maxLambda * fy_l) ** 2-(maxLambda * fx_abs_max) ** 2, eps)
        fr_yu = max(1-(maxLambda * fy_u) ** 2-(maxLambda * fx_abs_max) ** 2, eps)
        s_fl = maxLambda*z*fx_l/torch.sqrt(fr_xl)
        s_fu = maxLambda*z*fx_u/torch.sqrt(fr_xu)
        t_fl = maxLambda*z*fy_l/torch.sqrt(fr_yl)
        t_fu = maxLambda*z*fy_u/torch.sqrt(fr_yu)

        # include observation window offset into H (It is proved that this operation can reduce the samplings of spectrum)
        dfxMax1 = min(1/(2*torch.abs(s0 - s_fl)), 1/(2*torch.abs(s0 - s_fu)))
        dfyMax1 = min(1/(2*torch.abs(t0 - t_fl)), 1/(2*torch.abs(t0 - t_fu)))

        # not include observation window offset into H
        # dfxMax1 = min(1/(2*torch.abs(s_Hl)), 1/(2*torch.abs(s_Hu)))
        # dfyMax1 = min(1/(2*torch.abs(t_Hl)), 1/(2*torch.abs(t_Hu)))

        # maximum sampling interval limited by observation plane
        dfxMax2 = 1 / (2*torch.abs(self.s).max())
        dfyMax2 = 1 / (2*torch.abs(self.t).max())

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax2, dfxMax1)
        dfy = min(dfyMax2, dfyMax1)

        s = 1  # oversampling factor
        LRfx = int(Lfx / dfx * s)
        LRfy = int(Lfy / dfy * s)

        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        # spatial frequency coordinate
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)

        # convert linear phase of input field into position offset of spatial-frequency spectrum
        self.fx, self.fy = fx - self.offx, fy - self.offy

        fxx, fyy = torch.meshgrid(self.fx, self.fy, indexing='xy')
        self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelengths * fxx) ** 2 - (wavelengths * fyy) ** 2))

        # convert observation window position offset into linear phase of H
        self.H = self.H*torch.exp(1j*k*wavelengths*(fxx*s0 + fyy*t0))

        print(f'max freq: {torch.abs(self.fx).max():.2f}, interval: {torch.abs(self.fx)[-1]-torch.abs(self.fx)[-2]:.2f}, length: {LRfx,LRfy}, bandwidth: {Lfx, Lfy}')

        self.x = self.x.to(dtype=complex_dtype)
        self.y = self.y.to(dtype=complex_dtype)
        self.s = self.s.to(dtype=complex_dtype)
        self.t = self.t.to(dtype=complex_dtype)


    def __call__(self, E0):
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
        Fu = mdft(E0, self.x, self.y, fx + self.offx, fy + self.offy)  # assume an on-axis wave
        Eout = midft(Fu * self.H, self.s, self.t, fx, fy)

        # abs(Fu)/torch.max(abs(Fu))
        # abs(Eout) # torch.angle(self.H)

        return Eout[0].cpu().numpy()