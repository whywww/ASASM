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
        xx, yy = np.meshgrid(xvec, yvec, indexing='ij')
        self.x = xx.flatten() / np.max(np.abs(xx)) * np.pi
        self.y = yy.flatten() / np.max(np.abs(yy)) * np.pi

        # the frequency points
        fx = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
        fy = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
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
        t_asmNUFT = finufft.nufft2d3(self.x, self.y, E0, self.fx, self.fy, isign=iflag, eps=eps)
        t_3 = finufft.nufft2d3(self.x, self.y, self.H * t_asmNUFT, self.fx, self.fy, isign=-iflag, eps=eps).reshape(shape)

        t_3 = t_3 / np.max(np.abs(t_3))

        return t_3

    
# TODO:
# 1. torch+cuda implementation
# 3. test other parameters


class shift_BEASM2d:
    def __init__(self, s0, t0, z, xvec, yvec, wvls) -> None:
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
        pad = 2
        l = N * pitch
        fftmax = 1 / 2 / pitch

        # the source & target points (same window)
        xx, yy = np.meshgrid(xvec, yvec, indexing='ij')
        self.x = xx.flatten() / np.max(np.abs(xx)) * np.pi
        self.y = yy.flatten() / np.max(np.abs(yy)) * np.pi

        # the frequency points
        R = np.sqrt(wvls * z / N / pitch**2)
        # R = 1  # this is shift-BLASM
        Lx = Ly = R * N/pad * pitch
        fx_limP = 1 / wvls / np.sqrt((z / (s0 + Lx))**2 + 1)
        fx_limN = 1 / wvls / np.sqrt((z / (s0 - Lx))**2 + 1)
        fy_limP = 1 / wvls / np.sqrt((z / (t0 + Ly))**2 + 1)
        fy_limN = 1 / wvls / np.sqrt((z / (t0 - Ly))**2 + 1)

        fx_ue = fx_limP if s0 > -Lx else -fx_limP
        fx_le = fx_limN if s0 >= Lx else -fx_limN
        fy_ue = fy_limP if t0 > -Ly else -fy_limP
        fy_le = fy_limN if t0 >= Ly else -fy_limN

        fx_ue = np.clip(fx_ue, -fftmax, fftmax)
        fx_le = np.clip(fx_le, -fftmax, fftmax)
        fy_ue = np.clip(fy_ue, -fftmax, fftmax)
        fy_le = np.clip(fy_le, -fftmax, fftmax)

        dfx = (fx_ue - fx_le) / N
        dfy = (fy_ue - fy_le) / N

        if dfx <= 0 or dfy <= 0:
            s0_lim1 = z * wvls / np.sqrt(4*pitch**2-wvls**2) + Lx
            s0_lim2 = z * wvls / np.sqrt(4*pitch**2-wvls**2) - Lx
            s0_lim = max(abs(s0_lim1), abs(s0_lim2))
            theta_max = np.arctan2(s0_lim, z) * 180 / np.pi
            raise Exception(f"The oblique angle should not exceed {theta_max:.2f} degrees!")

        fx = np.linspace(fx_le, fx_ue - dfx, N, dtype=dtype)
        fy = np.linspace(fy_le, fy_ue - dfy, N, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy, indexing='ij')

        K1 = N / (fx_ue - fx_le - dfx)
        K2 = N / (fy_ue - fy_le - dfy)
        # K1 = N / pad / np.max(np.abs(fx))
        # K2 = N / pad / np.max(np.abs(fy))
        # K1 = N / pad / (fftmax - 1/l)  # not sure!!!!
        # K2 = N / pad / (fftmax - 1/l)
        
        # zc = N * pitch**2 / wvls
        # if z < zc:
        # fxn = fx
        # fyn = fy
        # else:
            # lim = R * N/2 * pitch

            # fx_emax = 1 / wvls / np.sqrt((z / (s0 + lim))**2 + 1)
            # fx_emin = 1 / wvls / np.sqrt((z / (s0 - lim))**2 + 1)
            # fy_emax = 1 / wvls / np.sqrt((z / (t0 + lim))**2 + 1)
            # fy_emin = 1 / wvls / np.sqrt((z / (t0 - lim))**2 + 1)

            # fx_ue = fx_emax if s0 > -lim else -fx_emax
            # fx_le = fx_emin if s0 >= lim else -fx_emin
            # fy_ue = fy_emax if t0 > -lim else -fy_emax
            # fy_le = fy_emin if t0 >= lim else -fy_emin

            # fx_ue, fx_le = fx_emax, -fx_emin
            # fy_ue, fy_le = fy_emax, -fy_emin

            # dfx = (fx_ue - fx_le) / N
            # dfy = (fy_ue - fy_le) / N
            # fxn = np.linspace(fx_le, fx_ue - dfx, N)
            # fyn = np.linspace(fy_le, fy_ue - dfy, N)
            # fxx, fyy = np.meshgrid(fxn, fyn, indexing='ij')
            # fxn, fyn = fxx.flatten(), fyy.flatten()

            # fxn = self.fx / max(abs(self.fx)) * fx_ue
            # fyn = self.fy / max(abs(self.fy)) * fy_ue

            # fxn = self.fx / max(abs(self.fx)) * fx_emax
            # fyn = self.fy / max(abs(self.fy)) * fy_emax

        self.fx = fxx.flatten() * K1
        self.fy = fyy.flatten() * K2

        fxx, fyy = fxx.astype(np.complex128), fyy.astype(np.complex128)
        # self.H = np.exp(1j * k * (wvls * fxx * s0 + wvls * fyy * t0 + z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2)))
        self.H = np.exp(1j * k * z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2))
        # self.H = np.exp(1j * k * (wvls * fxx * s0 + wvls * fyy * t0 + z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2)))
        self.H = self.H.flatten()
        
        print(f'fx range: {fx.min():.2f} to {fx.max():.2f}, interval: {dfx:.2f}&{dfy:.2f}, '\
                f'length: {N,N}, bandwidth: {fx[-1]-fx[0]:.2f}&{fy[-1]-fy[0]:.2f}')


    def __call__(self, E0):
        
        iflag = -1
        eps = 1e-12
        # E0 = E0.to(device='cpu').numpy().astype(np.complex128)
        E0 = E0.astype(np.complex128)

        shape = E0.shape
        E0 = E0.flatten()
        t_asmNUFT = finufft.nufft2d3(self.x, self.y, E0, self.fx, self.fy, isign=iflag, eps=eps)
        t_3 = finufft.nufft2d3(self.x, self.y, self.H * t_asmNUFT, self.fx, self.fy, isign=-iflag, eps=eps).reshape(shape)

        t_3 = t_3 / np.max(np.abs(t_3))

        # abs(t_asmNUFT.reshape(shape))
        # np.angle(self.H.reshape(shape))
        # np.angle(t_3) np.abs(t_3)
        return t_3