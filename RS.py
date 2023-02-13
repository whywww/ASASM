import numpy as np
from tqdm import tqdm


class RSDiffraction_INT():
    def __init__(self) -> None:
        pass

    def __call__(self, E0, z, xvec, yvec, svec, tvec, wavelengths):
        
        k = 2 * np.pi / wavelengths
        xx, yy = np.meshgrid(xvec, yvec, indexing='ij')
        # ss, tt = np.meshgrid(svec, tvec)
        # xx_, yy_ = xx[...,None,None], yy[...,None,None]
        # block_sz = 8

        # LX, LY = len(xvec), len(yvec)
        LS, LT = len(svec), len(tvec)

        # Eout = []
        # for bs in tqdm(range(LS // block_sz), desc='svec', position=0):
        #     Erow = []
        #     for bt in tqdm(range(LT // block_sz), desc='tvec', position=1, leave=False):
        #         ss_ = ss[bs*block_sz : (bs+1)*block_sz]
        #         tt_ = tt[bt*block_sz : (bt+1)*block_sz]
        #         ss_xy = np.broadcast_to(ss_, (LX, LY, *ss_.shape))
        #         tt_xy = np.broadcast_to(tt_, (LX, LY, *tt_.shape))
        #         r = np.sqrt((ss_xy - xx_)**2 + (tt_xy - yy_)**2 + z**2)
        #         h = -1 / (2 * np.pi) * (1j * k - 1 / r) * np.exp(1j * k * r) * z / r**2
        #         block_sum = np.einsum('ij, ijst', E0, h)
        #         Erow.append(block_sum)
        #     Eout.append(np.hstack(Erow))
        # Eout = np.vstack(Eout)

        Eout = np.empty((LT, LS), dtype=np.complex)
        for j in tqdm(range(LT), desc='tvec', position=0):
            for i in tqdm(range(LS), desc='svec', position=1, leave=False):
                r = np.sqrt((svec[i] - xx)**2 + (tvec[j] - yy)**2 + z**2)
                h = -1 / (2 * np.pi) * (1j * k - 1 / r) * np.exp(1j * k * r) * z / r**2
                Eout[j,i] = np.sum(E0 * h)

        return Eout



import torch
import math

class RSDiffraction_GPU():
    def __init__(self, z, xvec, yvec, svec, tvec, wavelengths, device) -> None:
        '''
        x,s are horizontal. y,t are vertical.
        '''

        self.device = device
        self.k = 2 * torch.pi / wavelengths
        self.z = z

        xvec, yvec = torch.tensor(xvec), torch.tensor(yvec)
        svec, tvec = torch.tensor(svec), torch.tensor(tvec)
        xx, yy = torch.meshgrid(xvec, yvec, indexing='xy')
        ss, tt = torch.meshgrid(svec, tvec, indexing='xy')
        self.ss, self.tt = ss.to(device), tt.to(device)
        self.xx, self.yy = xx.to(device), yy.to(device)
        
        self.block_sz = 120  # depends on your memory, e.g., 128 needs 24GB GPU memory

    
    def __call__(self, E0):
        
        E0 = torch.tensor(E0, dtype=torch.complex128, device=self.device)

        LX, LY = E0.shape[-2:]
        LS, LT = self.ss.shape

        Eout = []
        for bt in tqdm(range(math.ceil(LT / self.block_sz)), desc='tvec', position=0):
            Erow = []
            for bs in tqdm(range(math.ceil(LS / self.block_sz)), desc='svec', position=1, leave=False):
                ss_ = self.ss[bt*self.block_sz : (bt+1)*self.block_sz, bs*self.block_sz : (bs+1)*self.block_sz]
                tt_ = self.tt[bt*self.block_sz : (bt+1)*self.block_sz, bs*self.block_sz : (bs+1)*self.block_sz]
                block_sum = torch.zeros_like(ss_, dtype=E0.dtype)
                for by in tqdm(range(math.ceil(LY / self.block_sz)), desc='yvec', position=2, leave=False):
                    for bx in tqdm(range(math.ceil(LX / self.block_sz)), desc='xvec', position=3, leave=False):
                        E0_ = E0[by*self.block_sz : (by+1)*self.block_sz, bx*self.block_sz : (bx+1)*self.block_sz]
                        xx_ = self.xx[by*self.block_sz : (by+1)*self.block_sz, bx*self.block_sz : (bx+1)*self.block_sz]
                        yy_ = self.yy[by*self.block_sz : (by+1)*self.block_sz, bx*self.block_sz : (bx+1)*self.block_sz]
                        xx_st = xx_[..., None, None]
                        yy_st = yy_[..., None, None]
                        xy_ss = ss_.expand(*xx_.shape, *ss_.shape)
                        xy_tt = tt_.expand(*xx_.shape, *tt_.shape)
                        r = torch.sqrt((xy_ss - xx_st)**2 + (xy_tt - yy_st)**2 + self.z**2)
                        h = -1 / (2 * torch.pi) * (1j * self.k - 1 / r) * torch.exp(1j * self.k * r) * self.z / r**2
                        block_sum += torch.einsum('xy, xyst', E0_, h)
                Erow.append(block_sum)
            Eout.append(torch.hstack(Erow))
        Eout = torch.vstack(Eout)

        return Eout.cpu().numpy()


    # def __call__(self, E0):
        
    #     E0 = torch.tensor(E0, dtype=torch.complex128, device=self.device)

    #     LX, LY = E0.shape[-2:]
    #     LS, LT = self.ss.shape

    #     Eout = []
    #     for bs in tqdm(range(math.ceil(LS / self.block_sz)), desc='svec', position=0):
    #         Erow = []
    #         for bt in tqdm(range(math.ceil(LT / self.block_sz)), desc='tvec', position=1, leave=False):
    #             ss_ = self.ss[bs*self.block_sz : (bs+1)*self.block_sz, bt*self.block_sz : (bt+1)*self.block_sz]
    #             tt_ = self.tt[bs*self.block_sz : (bs+1)*self.block_sz, bt*self.block_sz : (bt+1)*self.block_sz]
    #             ss_xy = ss_.expand(LX, LY, *ss_.shape)
    #             tt_xy = tt_.expand(LX, LY, *tt_.shape)
    #             r = torch.sqrt((ss_xy - self.xx_)**2 + (tt_xy - self.yy_)**2 + self.z**2)
    #             h = -1 / (2 * torch.pi) * (1j * self.k - 1 / r) * torch.exp(1j * self.k * r) * self.z / r**2
    #             block_sum = torch.einsum('ij, ijst', E0, h)
    #             Erow.append(block_sum)
    #         Eout.append(torch.hstack(Erow))
    #     Eout = torch.vstack(Eout)

    #     return Eout.cpu().numpy()