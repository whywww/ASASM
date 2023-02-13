import numpy as np
import finufft
from utils import save_image


# class BEASM1d():
#     def __init__(self, n, pitch, z, wvls) -> None:
#         '''
#         :param n: number of samples in source spatial domain
#         :param pitch: sampling pitch in source spatial domain
#         :param z: propagation distance
#         :param wvls: wavelengths

#         '''

#         N = n * 2  # padding
#         l = N * pitch
#         k = 2 * np.pi / wvls

#         # the source points
#         self.x = np.linspace(-l / 2, l / 2 - pitch, N)

#         # the target points
#         self.s = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N)

#         K = N / 2 / np.max(np.abs(self.s))
#         fcn = 1 / 2 * np.sqrt(N / wvls / z)  # f_extend
#         ss = fcn / np.max(np.abs(self.s))
#         zc = N * pitch**2 / wvls
#         if z < zc:
#             fxn = self.s
#         else:
#             fxn = self.s * (ss - 0.0)
        
#         self.H = np.exp(1j * k * (z * np.sqrt(1 - (fxn * wvls)**2)))
#         self.fx = fxn * K


#     def __call__(self, c):
        
#         iflag = -1
#         eps = 1e-12

#         t_asmNUFT = finufft.nufft1d3(self.x / np.max(np.abs(self.x)) * np.pi, c, self.fx, isign=iflag, eps=eps)
#         t_3 = finufft.nufft1d3(self.x / (np.max(np.abs(self.x))) * np.pi, self.H * t_asmNUFT, self.fx, isign=-iflag, eps=eps)

#         t_3 = t_3 / np.max(np.abs(t_3))
#         N = len(self.x)
#         t_3 = t_3[N//2 - N//4 + 1 : N // 2 + N // 4 + 1]

#         # phase_asm_ex = np.angle(t_3)
#         amplitude_asm_ex = np.abs(t_3)

#         return amplitude_asm_ex



# class BEASM2d():
#     def __init__(self, z, xvec, yvec, wvls, device) -> None:
#         '''
#         :param xvec, yvec: source and destination window grid
#         :param z: propagation distance
#         :param wvls: wavelengths
#         '''

#         dtype = np.double

#         k = 2 * np.pi / wvls
#         N = len(xvec)
#         pitch = xvec[-1] - xvec[-2]
#         l = N * pitch

#         # the source & target points (same window)
#         xx, yy = np.meshgrid(xvec, yvec, indexing='ij')
#         self.x = xx.flatten() / np.max(np.abs(xx)) * np.pi
#         self.y = yy.flatten() / np.max(np.abs(yy)) * np.pi

#         # the frequency points
#         fx = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
#         fy = np.linspace(-1 / 2 / pitch, 1 / 2 / pitch - 1 / l, N, dtype=dtype)
#         fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
#         self.fx = fxx.flatten()
#         self.fy = fyy.flatten()

#         K1 = N / 2 / np.max(np.abs(fx))
#         K2 = N / 2 / np.max(np.abs(fy))
#         fcn = 1 / 2 * np.sqrt(N / wvls / z)  # f_extend
#         # fcn = 1 / 2 / pitch  # use the full band

#         fx_ = fcn / np.max(np.abs(fx))
#         fy_ = fcn / np.max(np.abs(fy))
#         zc = N * pitch**2 / wvls
#         if z < zc:
#             fxn = self.fx
#             fyn = self.fy
#         else:
#             fxn = self.fx * (fx_ - 0.0)
#             fyn = self.fy * (fy_ - 0.0)
        
#         self.H = np.exp(1j * k * (z * np.sqrt(1 - (fxn * wvls)**2 - (fyn * wvls)**2)))
#         self.fx = fxn * K1
#         self.fy = fyn * K2
#         print(f'max freq: {self.fx.max():.2f}, interval: {self.fx[-1]-self.fx[-2]:.2f}, length: {N,N}')


#     def __call__(self, E0):
        
#         iflag = -1
#         eps = 1e-12
#         E0 = E0.astype(np.complex128)

#         shape = E0.shape
#         E0 = E0.flatten()
#         t_asmNUFT = finufft.nufft2d3(self.x, self.y, E0, self.fx, self.fy, isign=iflag, eps=eps)
#         t_3 = finufft.nufft2d3(self.x, self.y, self.H * t_asmNUFT, self.fx, self.fy, isign=-iflag, eps=eps).reshape(shape)

#         t_3 = t_3 / np.max(np.abs(t_3))

#         return t_3


class shift_BEASM2d:
    def __init__(self, Uin, xvec, yvec, z, Nf=None) -> None:
        '''
        :param x0, y0: destination window shift 
        :param xvec, yvec: source and destination window grid
        :param z: propagation distance
        :param wvls: wavelengths
        '''

        dtype = np.double

        wvls = Uin.wvls
        k = 2 * np.pi / wvls
        Nx, Ny = len(Uin.xi), len(Uin.eta)  # no padding
        self.Nx, self.Ny = Nx, Ny
        self.Mx, self.My = len(xvec), len(yvec)
        pitchx = Uin.xi[-1] - Uin.xi[-2]
        pitchy = Uin.eta[-1] - Uin.eta[-2]
        fftmaxX = 1 / 2 / pitchx
        fftmaxY = 1 / 2 / pitchy

        # the source points
        self.xi = Uin.xi_.flatten()
        self.eta = Uin.eta_.flatten()
        self.xmax, self.ymax = abs(Uin.xi).max(), abs(Uin.eta).max()

        # the target points
        xc, yc = xvec[len(xvec) // 2], yvec[len(yvec) // 2]
        xx, yy = np.meshgrid(xvec - xc, yvec - yc, indexing='xy')
        self.x, self.y = xx.flatten(), yy.flatten()

        # the frequency points
        if Nf is None:
            pad = 2
            self.Lfx = Nx * pad
            self.Lfy = Ny * pad
        else:
            self.Lfx = self.Lfy = Nf
        Rx = np.sqrt(wvls * z / Nx / pitchx**2)
        Ry = np.sqrt(wvls * z / Ny / pitchy**2)
        # Rx = Ry = 1  # this is shift-BLASM
        Lx = Rx * Nx * pitchx  # maximum half width of observation window after band extending
        Ly = Ry * Ny * pitchy
        fx_limP = 1 / wvls / np.sqrt((z / (xc + Lx))**2 + 1)
        fx_limN = 1 / wvls / np.sqrt((z / (xc - Lx))**2 + 1)
        fy_limP = 1 / wvls / np.sqrt((z / (yc + Ly))**2 + 1)
        fy_limN = 1 / wvls / np.sqrt((z / (yc - Ly))**2 + 1)

        fx_ue = fx_limP if xc > -Lx else -fx_limP
        fx_le = fx_limN if xc >= Lx else -fx_limN
        fy_ue = fy_limP if yc > -Ly else -fy_limP
        fy_le = fy_limN if yc >= Ly else -fy_limN

        fx_ue = np.clip(fx_ue, -fftmaxX, fftmaxX)
        fx_le = np.clip(fx_le, -fftmaxX, fftmaxX)
        fy_ue = np.clip(fy_ue, -fftmaxY, fftmaxY)
        fy_le = np.clip(fy_le, -fftmaxY, fftmaxY)

        dfx = (fx_ue - fx_le) / self.Lfx
        dfy = (fy_ue - fy_le) / self.Lfy

        assert dfx > 0 and dfy > 0, "The oblique angle is too large."

        fx = np.linspace(fx_le, fx_ue - dfx, self.Lfx, dtype=dtype)
        fy = np.linspace(fy_le, fy_ue - dfy, self.Lfy, dtype=dtype)
        fxx, fyy = np.meshgrid(fx, fy, indexing='xy')

        self.fxmax = fx.max()
        self.fymax = fy.max()
        self.fx = fxx.flatten() 
        self.fy = fyy.flatten() 

        fxx, fyy = fxx.astype(np.complex128), fyy.astype(np.complex128)
        self.H = np.exp(1j * k * (wvls * fxx * xc + wvls * fyy * yc 
                        + z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2)))
        # self.H = np.exp(1j * k * z * np.sqrt(1 - (fxx * wvls)**2 - (fyy * wvls)**2))
        self.H = self.H.flatten()
        
        print(f'frequency sampling number = {self.Lfx, self.Lfy}, bandwidth = {fx_ue - fx_le:.2f}')
        # self.B = fx_ue - fx_le


    def __call__(self, E0, save_path=None):
        
        iflag = -1
        eps = 1e-12
        E0 = E0.astype(np.complex128)
        E0 = E0.flatten()
        
        Fu = finufft.nufft2d3(self.xi / self.xmax * np.pi, self.eta / self.ymax * np.pi, E0, 
                            self.fx * self.xmax * 2, self.fy * self.ymax * 2, isign=iflag, eps=eps)
        Eout = finufft.nufft2d3(self.fx / self.fxmax * np.pi, self.fy / self.fymax * np.pi, self.H * Fu, 
                            self.x * self.fxmax * 2, self.y * self.fymax * 2, 
                            isign=-iflag, eps=eps).reshape(self.My, self.Mx)

        Eout /= np.max(np.abs(Eout))

        if save_path is not None:
            H1 = self.H.reshape(self.Lfy, self.Lfx)
            Fu1 = Fu.reshape(self.Lfy, self.Lfx)
            save_image(np.angle(H1), f'{save_path}-H.png')
            save_image(np.angle(Fu1 * H1), f'{save_path}-FuH.png')
            save_image(np.angle(Fu1), f'{save_path}-Fu.png')

        return Eout, abs(Fu.reshape(self.Lfy, self.Lfx))