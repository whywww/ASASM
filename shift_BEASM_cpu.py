import numpy as np
import finufft


class BEASM1d():
    def __init__(self, n, pitch, z, wvls) -> None:
        '''
        :param n: number of samples in source spatial domain
        :param pitch: sampling pitch in source spatial domain
        :param z: propagation distance
        :param wvls: wavelengths

        '''

        N = n * 2  # padding
        l = N * pitch
        k = 2 * np.pi / wvls

        # the source points
        self.x = np.linspace(-l / 2, l / 2 - pitch, N)

        # the target points
        self.s = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N)

        K = N / 2 / np.max(np.abs(self.s))
        fcn = 1 / 2 * np.sqrt(N / wvls / z)  # f_extend
        ss = fcn / np.max(np.abs(self.s))
        zc = N * pitch**2 / wvls
        if z < zc:
            fxn = self.s
        else:
            fxn = self.s * (ss - 0.0)
        
        self.H = np.exp(1j * k * (z * np.sqrt(1 - (fxn * wvls)**2)))
        self.fx = fxn * K


    def __call__(self, c):
        
        iflag = -1
        eps = 1e-12

        t_asmNUFT = finufft.nufft1d3(self.x / np.max(np.abs(self.x)) * np.pi, c, self.fx, isign=iflag, eps=eps)
        t_3 = finufft.nufft1d3(self.x / (np.max(np.abs(self.x))) * np.pi, self.H * t_asmNUFT, self.fx, isign=-iflag, eps=eps)

        t_3 = t_3 / np.max(np.abs(t_3))
        N = len(self.x)
        t_3 = t_3[N//2 - N//4 + 1 : N // 2 + N // 4 + 1]

        # phase_asm_ex = np.angle(t_3)
        amplitude_asm_ex = np.abs(t_3)

        return amplitude_asm_ex



class BEASM2d():
    def __init__(self, z, xvec, yvec, wvls, device) -> None:
        '''
        :param xvec, yvec: source and destination window grid
        :param z: propagation distance
        :param wvls: wavelengths
        '''

        dtype = np.double
        device = 'cpu'
        # xvec, yvec = xvec.to(device).numpy(), yvec.to(device).numpy()
        # wvls = wvls.to(device).numpy().flatten()
        # z = z.to(device).numpy()

        k = 2 * np.pi / wvls
        N = len(xvec)
        pitch = xvec[-1] - xvec[-2]
        l = N * pitch

        # the source & target points (same window)
        xx, yy = np.meshgrid(xvec, yvec)
        self.x = xx.flatten()
        self.y = yy.flatten()

        # the frequency points
        fx = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
        fy = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy)
        self.fx = fxx.flatten()
        self.fy = fyy.flatten()

        K1 = N / 2 / np.max(np.abs(fx))
        K2 = N / 2 / np.max(np.abs(fy))
        fcn = 1 / 2 * np.sqrt(N / wvls / z)  # f_e/xtend
        # fcn = 1 / 2 / pitch  # use the full band

        fx_ = fcn / np.max(np.abs(fx))
        fy_ = fcn / np.max(np.abs(fy))
        zc = N * pitch**2 / wvls
        if z < zc:
            fxn = self.fx
            fyn = self.fy
        else:
            fxn = self.fx * (fx_ - 0.0)
            fyn = self.fy * (fy_ - 0.0)
        
        self.H = np.exp(1j * k * (z * np.sqrt(1 - (fxn * wvls)**2 - (fyn * wvls)**2)))
        self.fx = fxn * K1
        self.fy = fyn * K2
        print(f'max freq: {self.fx.max():.2f}, interval: {self.fx[-1]-self.fx[-2]:.2f}, length: {N,N}')


    def __call__(self, E0):
        
        iflag = -1
        eps = 1e-12
        # E0 = E0.to(device='cpu').numpy().astype(np.complex128)
        E0 = E0.astype(np.complex128)

        shape = E0.shape
        E0 = E0.flatten()
        t_asmNUFT = finufft.nufft2d3(self.x / np.max(np.abs(self.x)) * np.pi, self.y / np.max(np.abs(self.y)) * np.pi, E0, self.fx, self.fy, isign=iflag, eps=eps)
        t_3 = finufft.nufft2d3(self.x / np.max(np.abs(self.x)) * np.pi, self.y / np.max(np.abs(self.y)) * np.pi, self.H * t_asmNUFT, self.fx, self.fy, isign=-iflag, eps=eps).reshape(shape)

        t_3 = t_3 / np.max(np.abs(t_3))
        # t_3 = t_3[N1//2 - N1//4 + 1 : N1 // 2 + N1 // 4 + 1, N2//2 - N2//4 + 1 : N2 // 2 + N2 // 4 + 1]

        return t_3

    
# TODO:
# 1. torch+cuda implementation
# 2. shift-BEASM 2d
# 3. test other parameters


class shift_BEASM2d:
    def __init__(self, x0, y0, z, xvec, yvec, wvls) -> None:
        '''
        :param x0, y0: destination window shift 
        :param xvec, yvec: source and destination window grid
        :param z: propagation distance
        :param wvls: wavelengths
        '''

        dtype = np.double
        # xvec, yvec = xvec.to(device).numpy(), yvec.to(device).numpy()
        # wvls = wvls.to(device).numpy().flatten()
        # z = z.to(device).numpy()

        k = 2 * np.pi / wvls
        N = len(xvec)
        pitch = xvec[-1] - xvec[-2]
        l = N * pitch

        # the source & target points (same window)
        xx, yy = np.meshgrid(xvec, yvec)
        self.x = xx.flatten()
        self.y = yy.flatten()

        # the frequency points
        fmax_fft = 1 / 2 / pitch
        fx = np.linspace(-fmax_fft, fmax_fft - 1 / l, N, dtype=dtype)
        fy = np.linspace(-fmax_fft, fmax_fft - 1 / l, N, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy)
        self.fx = fxx.flatten()
        self.fy = fyy.flatten()

        R = np.sqrt(wvls * z / N / pitch**2)
        K1 = N / 2 / np.max(np.abs(fx))
        K2 = N / 2 / np.max(np.abs(fy))
        
        zc = N * pitch**2 / wvls
        if z < zc:
            fxn = self.fx
            fyn = self.fy
        else:
            lim = R * N/2 * pitch

            fx_emax = min(1 / wvls / np.sqrt((z / (x0 + lim))**2 + 1), fmax_fft)
            # fx_emin = min(1 / wvls / ((z / (x0 + lim))**2 + 1), fmax_fft)
            fy_emax = min(1 / wvls / np.sqrt((z / (y0 + lim))**2 + 1), fmax_fft)
            # fy_emin = min(1 / wvls / ((z / (y0 + lim))**2 + 1), fmax_fft)
   
            # fRx = fx_emax if x0 > -lim else -fx_emax
            # fLx = fx_emin if x0 >= lim else -fx_emin
            # fRy = fy_emax if y0 > -lim else -fy_emax
            # fLy = fy_emin if y0 >= lim else -fy_emin

            # dfx = (fRx - fLx) / N
            # dfy = (fRy - fLy) / N
            # fxn = np.linspace(fLx, fRx - dfx, N)
            # fyn = np.linspace(fLy, fRy - dfy, N)
            # fxx, fyy = np.meshgrid(fxn, fyn)
            # fxn, fyn = fxx.flatten(), fyy.flatten()

            fxn = self.fx / max(abs(self.fx)) * fx_emax
            fyn = self.fy / max(abs(self.fy)) * fy_emax

        self.H = np.exp(1j * k * (wvls * fxn * x0 + wvls * fyn * y0 + z * np.sqrt(1 - (fxn * wvls)**2 - (fyn * wvls)**2)))
        self.fx = fxn * K1
        self.fy = fyn * K2

        print(f'max freq: {self.fx.max():.2f}, interval: {self.fx[-1]-self.fx[-2]:.2f}, length: {N,N}')


    def __call__(self, E0):
        
        iflag = -1
        eps = 1e-12
        # E0 = E0.to(device='cpu').numpy().astype(np.complex128)
        E0 = E0.astype(np.complex128)

        shape = E0.shape
        E0 = E0.flatten()
        t_asmNUFT = finufft.nufft2d3(self.x / np.max(np.abs(self.x)) * np.pi, self.y / np.max(np.abs(self.y)) * np.pi, E0, self.fx, self.fy, isign=iflag, eps=eps)
        t_3 = finufft.nufft2d3(self.x / np.max(np.abs(self.x)) * np.pi, self.y / np.max(np.abs(self.y)) * np.pi, self.H * t_asmNUFT, self.fx, self.fy, isign=-iflag, eps=eps).reshape(shape)

        t_3 = t_3 / np.max(np.abs(t_3))

        return t_3