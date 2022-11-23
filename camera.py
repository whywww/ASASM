import math

import torch
import torch.nn.functional as F

from svASM import AngularSpectrumMethodMM
from shift_BEASM import BEASM2d


class BaseCamera():
    def __init__(self, hparams, device) -> None:
        self.hparams = hparams
        self.device = device

        self.wvls = torch.tensor(list(map(float, hparams.wvls.split(','))))[None,:,None,None].to(device)
        self.k = 2 * math.pi / self.wvls  # wavenumber
        self._init_camera()

    def _init_camera(self):
        
        ''' Initialize camera parameters
        '''
        
        self.zo = torch.tensor(float(self.hparams.zo), device=self.device)  # object distance
        assert float(self.hparams.zs) >= 0, 'propagation distance should be positive!'
        self.zs = torch.tensor(float(self.hparams.zs), device=self.device)  # sensor distance

        # aperture
        self.ap_diam = float(self.hparams.aperture_diam)  # physical diameter
        self.ap_px = self._get_aperture_px(self.ap_diam) 
        self.ap_pitch = self.ap_diam / self.ap_px
        
        # zero-padded
        pad = 1
        x = torch.linspace(-1.*pad, 1.*pad, self.ap_px*pad) # center is 0
        xx, yy = torch.meshgrid(x, x, indexing='xy')  # aperture coord.
        self.pupil = (torch.where( xx**2 + yy**2 <= 1., 1, 0 )[None,None,...]).to(self.device)  # binary
        
        self.avec = self.ap_pitch * torch.linspace(-(self.ap_px * pad - 1) / 2, (self.ap_px * pad - 1) / 2, self.ap_px * pad, dtype=torch.double, device=self.device)
        uu, vv = torch.meshgrid(self.avec, self.avec, indexing='xy')
        self.ap_grid = torch.stack((uu, vv))

        # sensor
        self.ss_pitch = self.ap_pitch / 2 
        self.ss_diam = float(self.hparams.sensor_diameter)
        self.ss_px = int(self.ss_diam / self.ss_pitch)
        psf_upsampling_rate = float(self.hparams.psf_upsampling_rate)
        self.ss_vec = self.ss_pitch * torch.linspace(-(self.ss_px-1)/2, (self.ss_px-1)/2, 
                        int(self.ss_px * psf_upsampling_rate), device=self.device, dtype=torch.double)
        ss_virpitch = self.ss_pitch / psf_upsampling_rate

        self.psf_sz = int(self.hparams.psf_size)  # sensor pitch psf
        self.high_res_psf_sz = int(self.psf_sz * psf_upsampling_rate)  # high resolution psf
        window_pad_rate = float(self.hparams.window_padding_rate)
        self.ss_window_sz = int(self.high_res_psf_sz * window_pad_rate)  # the window is used to compute the full-PSF and oof-energy  
        self.ss_window_diam = self.ss_window_sz * ss_virpitch


    def _get_aperture_px(self, ap_diam):
        '''
        Calculate the minimum number of aperture pixels required to satisfy the sampling theorem.
        '''

        # paraxial spherical wave sampling requirement
        # Nl1 = ap_diam**2/zo.min()/wvls.min()
        # Exact spherical wave sampling requirement
        Nl1 = (self.k * ap_diam**2 / (2 * math.pi) / torch.sqrt(ap_diam**2 / 4 + self.zo**2)).max()
        # Lens phase shift sampling requirement
        Nl2 = (self.k * ap_diam**2 / (2 * math.pi * self.f)).max()
        
        s = 1.1  # oversampling factor for safety
        ap_px = int(max(Nl1, Nl2) * s) 
        print(f'aperture uses {ap_px} pixels, pitch={ap_diam/ap_px}')

        return ap_px


    def init_propagation_function(self, xo, yo, zo, ssvecx, ssvecy):

        if self.hparams.prop == 'ASMMM':
            prop = AngularSpectrumMethodMM((xo, yo, zo), self.zs, self.lvec, self.lvec, 
                        ssvecx, ssvecy, self.wvls, self.device)
        elif self.hparams.prop == 'shift-BEASM':
            prop = BEASM2d(self.zs, self.lvec, self.lvec, self.wvls, self.device)
        else: 
            raise NotImplementedError

        print(f'Propagation method: {prop.__class__.__name__}')
        return prop


    def get_exact_spherical_phase(self, src_points, dest_grid, distance):
        ''' 
        Get the phase shift of the spherical wave from a single point source 
        
        :param src_points: tuple (xvec,yvec), planar coordinate of the sources
        :param dest_grid: tensor [uu,vv], coordinate grid at the destination. The grid diameter must == aperture diamter
        :param distance: scalar tensor, travel distance
        :return: (DxCxUxV) amplitude and phase of the spherical wave
        '''

        z = distance.reshape(-1,1,1,1)
        x = src_points[0].reshape(-1,1,1,1)
        y = src_points[1].reshape(-1,1,1,1)
        radius = torch.sqrt(z**2 + (dest_grid[0]-x)**2 + (dest_grid[1]-y)**2)
        phase = self.k * (radius - z)
        
        # normalize the total energy of input light to 1
        # s.t. torch.sum(amplitude**2, dim=(-2,-1)) == ones
        amplitude = self.pupil * z / self.wvls / radius**2
        amplitude /= torch.sqrt(torch.sum(amplitude**2, dim=(-2,-1), keepdim=True))

        return amplitude, phase

    
    def get_parallel_wave(self, src_point, dest_plane, distance):
        x = src_point[0]
        y = src_point[1]
        vec = torch.tensor([-x, -y, distance], device=self.device)
        kx, ky, kz = vec / torch.sqrt(torch.dot(vec, vec))

        # radius = np.sqrt((distance**2 + dest_plane[0]-x)**2 + (dest_plane[1]-y)**2)
        phase = self.k * (kx * dest_plane[0] + ky * dest_plane[1] + kz)

        # normalize the total energy of input light to 1
        amplitude = self.pupil * torch.ones_like(phase)
        amplitude /= torch.sqrt(torch.sum(amplitude**2, axis=(-2,-1), keepdims=True))
        
        return amplitude * torch.exp(1j * phase)

    
    def get_center_at_image_plane(self, xo, yo, zo, vecx, vecy):
        '''
        Get the image center in a plane parameterized by (vecx, vecy) 
        at distance zi from the lens, given a point source at (xo, yo, zo)
        '''

        pitchx = vecx[-1] - vecx[-2]
        pitchy = vecy[-1] - vecy[-2]
        Lx, Ly = len(vecx), len(vecy)

        # map from object to image plane
        xi, yi = -xo * self.zs / zo, -yo * self.zs / zo

        # pixel offset from the image center
        p_offset, q_offset = xi / pitchx, yi / pitchy

        # pixel location at the image
        ps = (Lx//2 + p_offset).to(torch.int)
        qs = (Ly//2 + q_offset).to(torch.int)

        if ps.min() < 0 or qs.min() < 0 or ps.max() > Lx or qs.max() > Ly:
            import warnings
            warnings.warn(f'The image center is outside the boundary!')

        return ps, qs


    def at_sensor(self, Us):

        intensity = torch.square(torch.abs(Us))
        # no need to normalize

        c = self.ss_window_sz // 2
        psf_crop = intensity[...,
                        c - self.high_res_psf_sz//2 : c + self.high_res_psf_sz//2 + self.high_res_psf_sz%2,
                        c - self.high_res_psf_sz//2 : c + self.high_res_psf_sz//2 + self.high_res_psf_sz%2] # odd/even ok

        return psf_crop

    
    def get_psf_from_xyz(self, xo, yo):
        pass

    def get_PSF(self, theta):
        pass


class LensCamera(BaseCamera):
    def __init__(self, hparams, device) -> None:
        super().__init__(hparams, device)
        self._init_lens()


    def _init_lens(self):
        
        ''' Initialize camera parameters
        '''
        
        self.f = float(self.hparams.focal_length)  # focal length
        zof = float(self.hparams.focal_distance) # object focal distance
        zsf = 1 / (1 / self.f - 1 / zof)  # sensor focal distance
        if float(self.hparams.zs) < 0:
            self.zs = torch.tensor(zsf, device=self.device)  # sensor distance
        else:
            self.zs = torch.tensor(float(self.hparams.zs), device=self.device)  # sensor distance

        # aperture
        fnum = float(self.hparams.fnum)
        self.ap_diam = self.f / fnum  # physical diameter
        self.ap_px = self._get_aperture_px(self.ap_diam) 
        self.ap_pitch = self.ap_diam / self.ap_px
        
        obj_diam = self.ss_diam * self.zo / zsf  # the diameter at max non-jittered depth
        print(f'FoV is {torch.atan2(obj_diam/2, self.zo.max())/math.pi*360:.2f} degrees and sensor size is {self.ss_px} pixels.')
        
    
    def get_lens_phase_shift(self):
        '''
        The lens phase transfer function
        
        :return: (1xCxUxV) phase shift by lens
        '''
        
        return self.k/2 * (-1/self.f) * (self.ap_grid[0]**2 + self.ap_grid[1]**2)


    def get_psf_from_xyz(self, xo, yo):
        ''' 
        Generate the PSF of a set of 3D object points along the same x, y coords.
        
        :param xov, yov: vert, horiz scene location of the virtual object
        :param scene_depths: tensor list of scene depths
        :param gb: Gaussian smoothing of depth-wise PSF
        :return: size(n_depths, n_wvls, psfsz, psfsz), PSF kernels at different depths
        '''

        amp, phi_in = self.get_exact_spherical_phase((xo, yo), self.ap_grid, self.zo)
        phi_lens = self.get_lens_phase_shift()
        phi_after_lens = (phi_in + phi_lens)
        Ul = amp * torch.exp(1j * phi_after_lens)
        
        # get psf window
        ps, qs = self.get_center_at_image_plane(xo, yo, self.zo, self.ss_vec, self.ss_vec)
        ss_vecx = self.ss_vec[ps - self.ss_window_sz//2 : ps + self.ss_window_sz//2 + self.ss_window_sz%2]
        ss_vecy = self.ss_vec[qs - self.ss_window_sz//2 : qs + self.ss_window_sz//2 + self.ss_window_sz%2]
        self.prop = self.init_propagation_function(xo, yo, self.zo, ss_vecx, ss_vecy)
        Us = self.prop(Ul)

        psf = self.at_sensor(Us)

        return psf.to(torch.float)

    
    def get_PSF(self, theta):

        radian = theta / 180 * math.pi
        r = math.tan(radian) * self.zo
        xo = r * math.sin(math.pi/4)
        yo = r * math.cos(math.pi/4)

        psf_upsampled = self.get_psf_from_xyz(xo, yo)

        # downsample to match sensor size
        psf = F.interpolate(psf_upsampled, size=self.psf_sz, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        return psf



class LenslessCamera(BaseCamera):
    def __init__(self, hparams, device) -> None:
        super().__init__(hparams, device)
    
    
    def get_psf_from_xyz(self, xo, yo):
        ''' 
        Generate the PSF of a set of 3D object points along the same x, y coords.
        
        :param xov, yov: vert, horiz scene location of the virtual object
        :param scene_depths: tensor list of scene depths
        :param gb: Gaussian smoothing of depth-wise PSF
        :return: size(n_depths, n_wvls, psfsz, psfsz), PSF kernels at different depths
        '''

        amp, phi_in = self.get_exact_spherical_phase((xo, yo), self.ap_grid, self.zo)
        Ul = amp * torch.exp(1j * phi_in)

        # get psf window
        ps, qs = len(self.ss_vec) // 2
        ss_vecx = self.ss_vec[ps - self.ss_window_sz//2 : ps + self.ss_window_sz//2 + self.ss_window_sz%2]
        ss_vecy = self.ss_vec[qs - self.ss_window_sz//2 : qs + self.ss_window_sz//2 + self.ss_window_sz%2]
        
        # prop
        self.prop = self.init_propagation_function(xo, yo, self.zo, ss_vecx, ss_vecy)
        Us = self.prop(Ul)

        psf = self.at_sensor(Us)

        return psf.to(torch.float)

    
    def get_PSF(self, theta):

        radian = theta / 180 * math.pi
        r = math.tan(radian) * self.zo
        xo = r * math.sin(math.pi/4)
        yo = r * math.cos(math.pi/4)

        psf_upsampled = self.get_psf_from_xyz(xo, yo)

        # downsample to match sensor size
        psf = F.interpolate(psf_upsampled, size=self.psf_sz, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        return psf