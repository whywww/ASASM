import numpy as np
from tqdm import tqdm


class RSDiffraction_INT():
    def __init__(self) -> None:
        pass

    def __call__(self, E0, z, xvec, yvec, svec, tvec, wavelengths):
        
        k = 2 * np.pi / wavelengths
        xx, yy = np.meshgrid(xvec, yvec)
        ss, tt = np.meshgrid(svec, tvec)
        xx_, yy_ = xx[...,None,None], yy[...,None,None]
        block_sz = 8

        LX, LY = len(xvec), len(yvec)
        LS, LT = len(svec), len(tvec)

        Eout = []
        for bs in tqdm(range(LS // block_sz), desc='svec', position=0):
            Erow = []
            for bt in tqdm(range(LT // block_sz), desc='tvec', position=1, leave=False):
                ss_ = ss[bs*block_sz : (bs+1)*block_sz]
                tt_ = tt[bt*block_sz : (bt+1)*block_sz]
                ss_xy = np.broadcast_to(ss_, (LX, LY, *ss_.shape))
                tt_xy = np.broadcast_to(tt_, (LX, LY, *tt_.shape))
                r = np.sqrt((ss_xy - xx_)**2 + (tt_xy - yy_)**2 + z**2)
                h = -1 / (2 * np.pi) * (1j * k - 1 / r) * np.exp(1j * k * r) * z / r**2
                block_sum = np.einsum('ij, ijst', E0, h)
                Erow.append(block_sum)
            Eout.append(np.hstack(Erow))
        Eout = np.vstack(Eout)

        # Eout = np.empty((LS, LT), dtype=np.complex)
        # for i in tqdm(range(LS), desc='svec', position=0):
        #     for j in tqdm(range(LT), desc='tvec', position=1, leave=False):
        #         r = np.sqrt((svec[i] - xx)**2 + (tvec[j] - yy)**2 + z**2)
        #         h = -1 / (2 * np.pi) * (1j * k - 1 / r) * np.exp(1j * k * r) * z / r**2
        #         Eout[i,j] = np.sum(E0 * h)

        return Eout



import torch
import math

class RSDiffraction_GPU():
    def __init__(self) -> None:
        pass

    def __call__(self, E0, z, xvec, yvec, svec, tvec, wavelengths, device):
        
        k = 2 * torch.pi / wavelengths
        E0 = torch.tensor(E0, dtype=torch.complex128, device=device)
        xvec, yvec = torch.tensor(xvec), torch.tensor(yvec)
        svec, tvec = torch.tensor(svec), torch.tensor(tvec)
        xx, yy = torch.meshgrid(xvec, yvec)
        ss, tt = torch.meshgrid(svec, tvec)
        ss, tt = ss.to(device), tt.to(device)
        xx_, yy_ = xx[...,None,None].to(device), yy[...,None,None].to(device)
        block_sz = 8  # depends on your memory

        LX, LY = len(xvec), len(yvec)
        LS, LT = len(svec), len(tvec)

        Eout = []
        for bs in tqdm(range(math.ceil(LS / block_sz)), desc='svec', position=0):
            Erow = []
            for bt in tqdm(range(math.ceil(LT / block_sz)), desc='tvec', position=1, leave=False):
                ss_ = ss[bs*block_sz : (bs+1)*block_sz, bt*block_sz : (bt+1)*block_sz]
                tt_ = tt[bs*block_sz : (bs+1)*block_sz, bt*block_sz : (bt+1)*block_sz]
                ss_xy = ss_.expand(LX, LY, *ss_.shape)
                tt_xy = tt_.expand(LX, LY, *tt_.shape)
                r = torch.sqrt((ss_xy - xx_)**2 + (tt_xy - yy_)**2 + z**2)
                h = -1 / (2 * torch.pi) * (1j * k - 1 / r) * torch.exp(1j * k * r) * z / r**2
                block_sum = torch.einsum('ij, ijst', E0, h)
                Erow.append(block_sum)
            Eout.append(torch.hstack(Erow))
        Eout = torch.vstack(Eout)

        print(Eout.shape)
        return Eout.cpu().numpy()