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
    def __init__(self, st_offset, z, xvec, yvec, svec, tvec, wvls) -> None:
        '''
        :param x0, y0: destination window shift 
        :param xvec, yvec: source and destination window grid
        :param z: propagation distance
        :param wvls: wavelengths
        '''

        dtype = np.double

        k = 2 * np.pi / wvls
        Nx, Ny = len(xvec), len(yvec)  # no padding
        self.Mx, self.My = len(svec), len(tvec)
        pad = 1
        self.Nx, self.Ny = Nx, Ny
        pitchx = xvec[-1] - xvec[-2]
        pitchy = yvec[-1] - yvec[-2]
        fftmaxX = 1 / 2 / pitchx
        fftmaxY = 1 / 2 / pitchy

        # the source points
        s0, t0 = st_offset
        xx, yy = np.meshgrid(xvec, yvec, indexing='xy')
        self.x = xx.flatten() / np.max(np.abs(xx)) * np.pi
        self.y = yy.flatten() / np.max(np.abs(yy)) * np.pi
        self.xmax, self.ymax = xvec.max(), yvec.max()
        self.smax, self.tmax = svec.max(), tvec.max()

        # the target points
        ss, tt = np.meshgrid(svec - s0, tvec - t0, indexing='xy')
        self.s, self.t = ss.flatten(), tt.flatten()

        # the frequency points
        Rx = np.sqrt(wvls * z / Nx / pitchx**2)
        Ry = np.sqrt(wvls * z / Ny / pitchy**2)
        # Rx = Ry = 1  # this is shift-BLASM
        Lx = Rx * Nx * pitchx  # maximum half width of observation window after band extending
        Ly = Ry * Ny * pitchy  # maximum half width of observation window after band extending
        fx_limP = 1 / wvls / np.sqrt((z / (s0 + Lx))**2 + 1)
        fx_limN = 1 / wvls / np.sqrt((z / (s0 - Lx))**2 + 1)
        fy_limP = 1 / wvls / np.sqrt((z / (t0 + Ly))**2 + 1)
        fy_limN = 1 / wvls / np.sqrt((z / (t0 - Ly))**2 + 1)

        fx_ue = fx_limP if s0 > -Lx else -fx_limP
        fx_le = fx_limN if s0 >= Lx else -fx_limN
        fy_ue = fy_limP if t0 > -Ly else -fy_limP
        fy_le = fy_limN if t0 >= Ly else -fy_limN

        fx_ue = np.clip(fx_ue, -fftmaxX, fftmaxX)
        fx_le = np.clip(fx_le, -fftmaxX, fftmaxX)
        fy_ue = np.clip(fy_ue, -fftmaxY, fftmaxY)
        fy_le = np.clip(fy_le, -fftmaxY, fftmaxY)

        dfx = (fx_ue - fx_le) / (Nx * pad)
        dfy = (fy_ue - fy_le) / (Ny * pad)

        s0_lim1 = z * wvls / np.sqrt(4*pitchx**2-wvls**2) + Lx
        s0_lim2 = z * wvls / np.sqrt(4*pitchx**2-wvls**2) - Lx
        s0_lim = max(abs(s0_lim1), abs(s0_lim2))
        theta_max = np.arctan2(s0_lim, z) * 180 / np.pi
        print(f"The oblique angle should not exceed {theta_max:.2f} degrees!")
        assert dfx > 0 and dfy > 0

        fx = np.linspace(fx_le, fx_ue - dfx, Nx * pad, dtype=dtype)
        fy = np.linspace(fy_le, fy_ue - dfy, Ny * pad, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy, indexing='xy')

        K1 = Nx / 2 / fftmaxX
        K2 = Ny / 2 / fftmaxY

        self.fx = fxx.flatten() * K1
        self.fy = fyy.flatten() * K2

        fxx, fyy = fxx.astype(np.complex128), fyy.astype(np.complex128)
        self.H = np.exp(1j * k * (wvls * fxx * s0 + wvls * fyy * t0 
                        + z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2)))
        # self.H = np.exp(1j * k * z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2))
        self.H = self.H.flatten()
        
        print(f'frequency sampling number = {Nx*pad, Ny*pad}')


    def __call__(self, E0):
        
        iflag = -1
        eps = 1e-12
        E0 = E0.astype(np.complex128)
        E0 = E0.flatten()
        
        Fu = finufft.nufft2d3(self.x, self.y, E0, self.fx, self.fy, isign=iflag, eps=eps)
        Eout = finufft.nufft2d3(self.fx/self.Nx * np.pi, self.fy/self.Ny * np.pi, self.H * Fu, 
                        self.s/self.xmax * self.Mx, self.t/self.ymax * self.My, 
                        isign=-iflag, eps=eps).reshape(self.My, self.Mx)

        Eout /= np.max(np.abs(Eout))

        # abs(Fu.reshape(self.Ny*1, self.Nx*1))
        # np.angle(self.H.reshape(shape))
        # np.abs(Eout) np.angle(Eout) 
        return Eout