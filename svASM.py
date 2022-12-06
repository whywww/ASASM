import torch
import math


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


class AngularSpectrumMethodMM():
    def __init__(self, o_loc, z, xvec, yvec, svec, tvec, wavelengths, device):
        '''
        :param xvec, yvec: vectors of source coordinates
        :param wavelengths: wavelengths
        '''
        
        super(AngularSpectrumMethodMM, self).__init__()

        dtype = torch.double
        complex_dtype = torch.complex128

        self.device = device
        self.x, self.y = torch.as_tensor(xvec, device=device), torch.as_tensor(yvec, device=device)
        self.s, self.t = torch.as_tensor(svec, device=device), torch.as_tensor(tvec, device=device)
        z = torch.as_tensor(z, device=device)
        o_loc = torch.as_tensor(o_loc)
        wavelengths = torch.as_tensor(wavelengths, device=device)

        # maximum wavelength
        n = 1
        k = 2 * math.pi / wavelengths * n
        maxLambda = wavelengths.max()

        # bandwidth of aperture
        pitch = self.x[-1] - self.x[-2]
        fmax_fft = 1 / (2 * pitch)
        Lfx = 2 * fmax_fft
        Lfy = 2 * fmax_fft

        # off-axis spectrum offset
        x0, y0, zo = o_loc
        offx = -x0 / torch.sqrt(x0**2 + y0**2 + zo**2) / maxLambda
        offy = -y0 / torch.sqrt(x0**2 + y0**2 + zo**2) / maxLambda
        # offx = torch.remainder(offx, Lfx)
        # offy = torch.remainder(offy, Lfy)
        # offx = offy = 0

        # maximum sampling interval limited by TF
        dfMax1 = torch.sqrt(1 - (maxLambda * fmax_fft) ** 2) / (2 * maxLambda * z * fmax_fft)
        # fmin = -maxLambda * z * (fmax_fft + offx) / torch.sqrt(1 - (maxLambda * (fmax_fft + offx))**2)
        # fmax = -maxLambda * z * (-fmax_fft + offy) / torch.sqrt(1 - (maxLambda * (-fmax_fft + offy))**2)
        # dfMax1 = 1 / abs(fmax - fmin)

        # maximum sampling interval limited by observation plane
        Lx = svec[-1] - svec[0]
        Ly = tvec[-1] - tvec[0]
        dfxMax2 = 1 / Lx
        dfyMax2 = 1 / Ly

        # minimum requirements of sampling interval in k space
        dfx = min(dfxMax2, dfMax1)
        dfy = min(dfyMax2, dfMax1)
        
        s = 1  # oversampling factor
        LRfx = int(Lfx / dfx * s)
        LRfy = int(Lfy / dfy * s)
        
        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        # spatial frequency coordinate
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)
        self.fx, self.fy = fx + offx, fy + offy

        fxx, fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')
        self.H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelengths * fxx) ** 2 - (wavelengths * fyy) ** 2))
        
        print(f'max freq: {torch.abs(self.fx).max():.2f}, interval: {torch.abs(self.fx)[-1]-torch.abs(self.fx)[-2]:.2f}, length: {LRfx,LRfy}, bandwidth: {Lfx, Lfy}')

        # s0, t0 = x0 / zo * z, y0 / zo * z
        # self.H = torch.exp(1j * k * (wavelengths * fxx * s0 + wavelengths * fyy * t0 + z * torch.sqrt(1 - wavelengths ** 2 * (fxx ** 2 + fyy ** 2))))

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
        Fu = mdft(E0, self.y, self.x, fy, fx)
        Eout = midft(Fu * self.H, self.t, self.s, fy, fx)

        # abs(Fu)/torch.max(abs(Fu))
        # abs(Eout) # torch.angle(self.H)

        return Eout.cpu().numpy()