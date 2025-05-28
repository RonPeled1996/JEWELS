## Modules:
import numpy as np
import emcee
from astropy.cosmology import LambdaCDM
from astropy.stats import SigmaClip
from photutils.background import Background2D, SExtractorBackground
from scipy.signal import fftconvolve


## Standard cosmology:
H0 = 70     # Hubble parameter; (km/s) / Mpc
Om0 = 0.3   # omega matter
Ode0 = 0.7  # omega lambda


def d_A_z(z):  # distance function to a source at redshift z.
    return LambdaCDM(H0, Om0, Ode0).angular_diameter_distance(z).value  # Mpc


def d_A_z1z2(z1, z2):  # distance function between sources at redshifts z1, z2.
    return LambdaCDM(H0, Om0, Ode0).angular_diameter_distance_z1z2(z1, z2).value  # Mpc


def Sersic_2D(grid_x, grid_y, params):
    
    
    """A galaxy model (Sersic profile, from https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Sersic2D.html)
    
    Args:
    eps (int or float; range [0, 1]): eps = 1 - b/a, where a and b are the semi-major and semi-minor axes of the Sersic ellipse, respectively.
    a (int or float; a > 0): the semi-major axis of the Sersic ellipse, in pixels.
    phi (int or float; range [-pi/2, pi/2]): the inclination angle of the PIEMD ellipse, in radians.
    sers_ind (int or float; range [0.06, 10]): the Sersic index.
    x0, y0 (int or float): the central coordinates of the Sersic ellipse, in pixels.
    
    Returns:    
    Sersic model (2d ndarray)
    
    """

    
    eps, a, phi, sers_ind, x0, y0 = params

    
    if not isinstance(eps, (float, int)):
        raise TypeError("eps must be a float or an integer.")
    
    # if ((eps < 0) or (eps > 1)):
    #     raise ValueError("eps must be within the range [0, 1].")
       
    if not isinstance(a, (float, int)):
        raise TypeError("a must be a float or an integer.")
    
    # if not a > 0:
    #     raise ValueError("a must be positive.")

    if not isinstance(phi, (float, int)):
        raise TypeError("The inclination angle must be a float or an integer.")
        
    # if ((phi < - np.pi / 2) or (phi > np.pi / 2)):
    #     raise ValueError("The inclination angle must be within the range [-pi/2, pi/2].")
        
    if not isinstance(sers_ind, (float, int)):
        raise TypeError("The Sersic index must be a float or an integer.")
        
    # if ((sers_ind < 0.06) or (sers_ind > 10)):
    #     raise ValueError("The Sersic index must be within the range [0.06, 10].")
        
    if not isinstance(x0, (float, int)):
        raise TypeError("x0 must be a float or an integer.")
        
    if not isinstance(y0, (float, int)):
        raise TypeError("y0 must be a float or an integer.")
        
        
    A = (grid_x - x0) * np.cos(phi) + (grid_y - y0) * np.sin(phi)
    B = -(grid_x - x0) * np.sin(phi) + (grid_y - y0) * np.cos(phi)
    r = np.sqrt(A**2 + (B/(1 - eps))**2)
    n = sers_ind
    
    if n < 0.36:
        bn = 0.01945 - 0.8902*n + 10.95 * n**2 - 19.67 * n**3 + 13.43 * n**4
    else:
        bn = 2*n - 1/3 + 4/(405 * n) + 46/(25515 * n**2) + 131/(1148175 * n**3) - 2194697/(30690717750 * n**4)
    
    r_eff = a * (bn / np.log(2))**n  # assumes r with eps=0
    func = np.exp(-bn * ((r / r_eff)**(1/n) - 1)) * np.heaviside(5 * r_eff - r, 1)
    
    return func / np.max(func)


class PIEMD:
    
    
    """A galaxy cluster parametric model (from "glafic (2022)" appendix B.6 & B.7)
    
    *In the following, FOV = Field Of View, that is, the galaxy cluster image that you try to fit this model to. If you are not using the model for 
     fitting, you can stay with the defaults.
    
    Args:
    grid_x (int or 1d/2d ndarray): the x coordiante in pixels to calculate the model on, in pixels.
    grid_y (int or 1d/2d ndarray): the y coordiante in pixels to calculate the model on, in pixels.
    center_x (int or float): the x coordinate of the PIEMD ellipse center, in pixels.
    center_y (int or float): the y coordinate of the PIEMD ellipse center, in pixels.
    eps (int or float; range [0, 1]): eps = 1 - b/a, where a and b are the semi-major and semi-minor axes of the PIEMD ellipse, respectively.
    phi (int or float; range [-pi/2, pi/2]): the inclination angle of the PIEMD ellipse, in radians.
    sigma_v (float; sigma_v > 0): the velocity dispersion of the galaxies in the modeled galaxy cluster, in km/sec.
    r_core (float; r_core > 0): the core radius of the PIEMD ellipse, in Mpc.
    D_l (float; D_l > 0): the distance from the observer to the lens, in Mpc.
    D_s (float; D_s > 0): the distance from the observer to the source, in Mpc.
    D_ls (float; D_ls > 0): the distance from the lens to the source, in Mpc.
    pixel_in_DD (int or float; pixel_in_DD > 0): the FOV resolution, i.e., how many decimal degrees does one pixel represent. The defualt is 0.04/3600, 
                                                 that is, 1 pixel = 0.04".
    pixel_in_x (int or float; pixel_in_x > 0): for cases the grid you use has a lower resolution than the FOV. It sets the number of FOV pixels that fit 
                                               into a grid pixel, in the x direction. The default is 1 (i.e., the same resolution).
    pixel_in_y (int or float; pixel_in_y > 0): for cases the grid you use has a lower resolution than the FOV. It sets the number of FOV pixels that fit 
                                               into a grid pixel, in the y direction. The default is 1 (i.e., the same resolution).
                                               
    """
    
    
    c_cm = 29979245800  # cm / s
    pc = 3.08567758128 * 10**18  # cm
    c = c_cm / 10**5  # km / s


    __slots__ = ('grid_x', 'grid_y', 'center_x', 'center_y', 'eps', 'phi', 'sigma_v', 'r_core',
                 'D_l', 'D_s', 'D_ls', 'pixel_in_DD', 'pixels_in_x', 'pixels_in_y')

    
    def __init__(self, grid_x, grid_y, center_x, center_y, eps, phi, sigma_v, r_core,
                 D_l, D_s, D_ls, pixel_in_DD=0.04/3600, pixels_in_x=1, pixels_in_y=1):
        
        
        if isinstance(grid_x, (int, np.ndarray)) and isinstance(grid_y, (int, np.ndarray)):
            if grid_x.shape == grid_y.shape:
                if len(grid_x.shape) == 1 or len(grid_x.shape) == 2:               
                            self.grid_x = grid_x  # pixels
                            self.grid_y = grid_y  # pixels
                else:
                    raise TypeError("grid_x and grid_y must be 0, 1 or 2 dimensional.")
            else:
                raise TypeError("grid_x and grid_y must have the same shape.")
        else:
            raise ValueError("grid_x and grid_y must be integers or ndarrays.")
            
            
        if isinstance(center_x, (float, int)):
            self.center_x = center_x  # pixels
        else:
            raise TypeError("center_x must be a float or an integer.")
            
            
        if isinstance(center_y, (float, int)):
            self.center_y = center_y  # pixels
        else:
            raise TypeError("center_y must be a float or an integer.")
            
            
        if isinstance(eps, (float, int)):
            if eps >= 0 and eps <= 1:
                self.eps = eps
            else:
                raise ValueError("eps must be within the range [0, 1].")
        else:
            raise TypeError("eps must be a float or an integer.")
            
            
        if isinstance(phi, (float, int)):
            if phi >= - np.pi / 2 and phi <= np.pi / 2:
                self.phi = phi  # radians
            else:
                raise ValueError("The inclination angle must be within the range [-pi/2, pi/2].")
        else:
            raise TypeError("The inclination angle must be a float or an integer.")
            
            
        if isinstance(sigma_v, float):
            if sigma_v > 0:
                self.sigma_v = sigma_v  # km/sec
            else:
                raise ValueError("The velocity dispersion must be positive.")
        else:
            raise TypeError("The velocity dispersion must be a float.")
            
            
        if isinstance(r_core, float):
            if r_core > 0:
                self.r_core = r_core  # Mpc
            else:
                raise ValueError("The core radius must be positive.")
        else:
            raise TypeError("The core radius must be a float.")
            
            
        if isinstance(D_l, float):
            if D_l > 0:
                self.D_l = D_l  # Mpc
            else:
                raise ValueError("D_l must be positive.")
        else:
            raise TypeError("D_l must be a float.")
            
            
        if isinstance(D_s, float):
            if D_s > 0:
                self.D_s = D_s  # Mpc
            else:
                raise ValueError("D_s must be positive.")
        else:
            raise TypeError("D_s must be a float.")
            
            
        if isinstance(D_ls, float):
            if D_ls > 0:
                self.D_ls = D_ls  # Mpc
            else:
                raise ValueError("D_ls must be positive.")
        else:
            raise TypeError("D_ls must be a float.")
            
            
        if isinstance(pixel_in_DD, (float, int)):
            if pixel_in_DD > 0:
                self.pixel_in_DD = pixel_in_DD
            else:
                raise ValueError("pixel_in_DD must be positive.")
        else:
            raise TypeError("pixel_in_DD must be a float or an integer.")
            
        
        if isinstance(pixels_in_x, (float, int)):
            if pixels_in_x > 0:
                self.pixels_in_x = pixels_in_x
            else:
                raise ValueError("pixels_in_x must be positive.")
        else:
            raise TypeError("pixels_in_x must be a float or an integer.")
            
            
        if isinstance(pixels_in_y, (float, int)):
            if pixels_in_y > 0:
                self.pixels_in_y = pixels_in_y
            else:
                raise ValueError("pixels_in_y must be positive.")
        else:
            raise TypeError("pixels_in_y must be a float or an integer.")

        
    def PIEMD_second_derivatives(self): 
                
            
        delta_x = self.grid_x - self.center_x  # grid cells
        delta_y = self.grid_y - self.center_y  # grid cells

        x = delta_x * self.pixels_in_x * self.pixel_in_DD * (np.pi / 180) * self.D_l * 10**6 * self.pc  # cm
        y = delta_y * self.pixels_in_y * self.pixel_in_DD * (np.pi / 180) * self.D_l * 10**6 * self.pc  # cm
        # it's D_l because the images are in the lens plane
        
        
        ## working in the rotated frame:
        x_tilde = x * np.cos(self.phi) - y * np.sin(self.phi)
        y_tilde = x * np.sin(self.phi) + y * np.cos(self.phi)
        q = 1 - self.eps

        if q == 0 or q < 0:
            b_PIEMD = 0
            s = 0
        else:
            b_PIEMD = 4 * np.pi * (self.sigma_v / self.c)**2 * (self.D_ls / self.D_s) / np.sqrt(q) * self.D_l * 10**6 * self.pc  # cm
            s = self.r_core / np.sqrt(q) * 10**6 * self.pc  # cm
        zeta = np.sqrt(q**2 * (s**2 + x_tilde**2) + y_tilde**2)  # cm

        factor = b_PIEMD * q / zeta
        denominator = (1 + q**2) * s**2 + 2 * zeta * s + x_tilde**2 + y_tilde**2
        psi_xx_tilde = factor / denominator * (q**2 * s**2 + y_tilde**2 + s * zeta)
        psi_yy_tilde = factor / denominator * (s**2 + x_tilde**2 + s * zeta)
        psi_xy_tilde = - factor / denominator * x_tilde * y_tilde
    
    
        ## transforming from the rotated frame back to the unrotated frame: 
        psi_xx = psi_xx_tilde * np.cos(self.phi)**2 + 2 * psi_xy_tilde * np.sin(self.phi) * np.cos(self.phi) \
                 + psi_yy_tilde * np.sin(self.phi)**2
        psi_yy = psi_xx_tilde * np.sin(self.phi)**2 - 2 * psi_xy_tilde * np.sin(self.phi) * np.cos(self.phi) \
                 + psi_yy_tilde * np.cos(self.phi)**2
        psi_xy = (psi_yy_tilde - psi_xx_tilde) * np.sin(self.phi) * np.cos(self.phi) \
                 + psi_xy_tilde * (np.cos(self.phi)**2 - np.sin(self.phi)**2)

        
        return psi_xx, psi_yy, psi_xy  

    
    def kappa(self):  # the convergence
        psi_xx, psi_yy, psi_xy = self.PIEMD_second_derivatives()
        return 0.5 * (psi_xx + psi_yy)  
    
    
    def gamma_1(self):  # the shear's 1st component
        psi_xx, psi_yy, psi_xy = self.PIEMD_second_derivatives()
        return 0.5 * (psi_xx - psi_yy)  
    
    
    def gamma_2(self):  # the shear's 2nd component
        psi_xx, psi_yy, psi_xy = self.PIEMD_second_derivatives()
        return psi_xy 
    
    
    def gamma(self):  # shear's magnitude
        gamma_1 = self.gamma_1()
        gamma_2 = self.gamma_2()
        return np.sqrt(gamma_1**2 + gamma_2**2)  
      
        
    def angle(self):  # shear's inclination angle
        gamma_1 = self.gamma_1()
        gamma_2 = self.gamma_2()
        return 0.5 * np.arctan2(gamma_2, gamma_1)  
    
    
class light_sources:
    
    
    """ A class for the catalog of a galaxy cluster
    
    Args:
    x (1d ndarray): the x coordinates of the catalog galaxies, in pixels.
    y (1d ndarray): the y coordinates of the catalog galaxies, in pixels.
    a (1d ndarray): the semi-major axis of the catalog galaxies, in pixels.
    b (1d ndarray): the semi-minor axis of the catalog galaxies, in pixels.
    phi (1d ndarray): the inclination angle of the catalog galaxies, in radians.
    
    *All the above arrays must have the same length.
    
    cluster_image (2d ndarray): the FOV image.
    seg_map (2d ndarray): the segmantation map.
    
    *cluster_image.shape == seg_map.shape
        
    """
    

    __slots__ = ('x', 'y', 'a', 'b', 'phi', 'cluster_image', 'seg_map')

    
    def __init__(self, x, y, a, b, phi, cluster_image, seg_map):
        
        
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
        and isinstance(phi, np.ndarray):
            if x.shape == y.shape == a.shape == b.shape == phi.shape:
                if len(x.shape) == 1:                
                            self.x = x  # pixels
                            self.y = y  # pixels
                            self.a = a  # pixels
                            self.b = b  # pixels
                            self.phi = phi  # radians
                else:
                    raise TypeError("x, y, a, b, phi be 1 dimensional.")
            else:
                raise TypeError("x, y, a, b, phi must must have the same shape.")
        else:
            raise ValueError("x, y, a, b, phi must be ndarrays.")
    
    
        if isinstance(cluster_image, np.ndarray) and isinstance(seg_map, np.ndarray):
            if cluster_image.shape == seg_map.shape:
                if len(cluster_image.shape) == 2:                
                            self.cluster_image = cluster_image
                            self.seg_map = seg_map
                else:
                    raise TypeError("cluster_image and seg_map must be 2 dimensional.")
            else:
                raise TypeError("cluster_image and seg_map must have the same shape.")
        else:
            raise ValueError("cluster_image and seg_map be ndarrays.")
                     
         
    def size_filter(self, size=20):
        
        
        """A method that filters the galaxies according to np.size > size
        
        Args:
        size (int, size > 0): the array size bar below which galaxies are filtered. Default is 20.
        
        Returns:
        ind_size (1d ndarray): the indexes of the galaxies with np.size > size.
        
        """
        
        
        if not isinstance(size, int):
            raise TypeError("size must be an integer.")
            
        if not size > 0:
            raise ValueError("size must be positive.")

            
        ind_size = np.zeros(1)

        for ID_ind in range(len(self.x)):
            half_box = np.round(3 * self.a[ID_ind]).astype(int)  # pixels
            x_gal = self.x[ID_ind].astype(int)  # pixels
            y_gal = self.y[ID_ind].astype(int)  # pixels
            ind_gal = self.seg_map[y_gal, x_gal]

            min_x = x_gal - half_box  # pixels
            max_x = x_gal + 1 + half_box  # pixels
            min_y = y_gal - half_box  # pixels
            max_y = y_gal + 1 + half_box  # pixels

            galaxy_map = self.seg_map[min_y:max_y, min_x:max_x]

            if np.where(galaxy_map == ind_gal)[0].size > size:
                ind_size = np.append(ind_size, ID_ind)
            else:
                continue

        ind_size = ind_size[1:].astype(int)
        
        
        return ind_size
    
    
    def deconvolver(self, ID_ind, PSF, nsteps=10000, MCMC_progress=False):
        
        
        """A method that deconvolves each galaxy image from the aperture's PSF, using MCMC.
        Can be parallelized. It assumes a Sersic profile for the intrinsic galaxy, and accounts for the model bias.
        
        Args:
        ID_ind (int): the galaxy's index to be deconvolved.
        PSF (2d ndarray): the aperture's PSF. Must be normlized such that np.allclose(np.sum(PSF), 1, 1e-1, 1e-1).
        nsteps (int): the number of steps for the MCMC walkers to take. Default is 10000.
        MCMC_progress (bool): whether to show the progress bar or not. Default is False.
        
        *Below, ellipticity is e = (a - b)/(a + b) that is need for WL; not to be confused with eps = 1 - b/a.
        
        Returns:
        ID_ind (float): the deconvolved galaxy's index.
        e1 (float): the deconvolved galaxy's first ellipticity component, that is, e = cos(2*phi), where phi is the inclination angle.
        e2 (float): the deconvolved galaxy's second ellipticity component, that is, e = sin(2*phi), where phi is the inclination angle.
        ind_gal (float): the deconvolved galaxy's segmantation map index (as it might not be the same as ID_ind).
        
        """
        
        
        if not isinstance(ID_ind, int):
            raise TypeError("ID_ind must be an integer.")
        
        if not len(PSF.shape) == 2:
            raise TypeError("PSF must be 2 dimensional.")
            
        if not isinstance(PSF, np.ndarray):
            raise TypeError("PSF must be an ndarray.")
        
        if not np.allclose(np.sum(PSF), 1, 1e-1, 1e-1):
            raise ValueError("PSF must be normalised such that np.allclose(np.sum(PSF), 1, 1e-1, 1e-1)")            
            
        if not isinstance(nsteps, int):
            raise TypeError("nsteps must be an integer.")
               
        if not isinstance(MCMC_progress, bool):
            raise TypeError("MCMC_progress is a boolean argument.")
            
        
        ## galaxy borders (the 'xy' of the graph are opposite to the 'ij' of the matrix):
        half_box = np.round(3 * self.a[ID_ind]).astype(int)  # pixels
        x_gal = self.x[ID_ind].astype(int)  # pixels
        y_gal = self.y[ID_ind].astype(int)  # pixels
        ind_gal = self.seg_map[y_gal, x_gal]

        min_x = x_gal - half_box  # pixels
        max_x = x_gal + 1 + half_box  # pixels
        min_y = y_gal - half_box  # pixels
        max_y = y_gal + 1 + half_box  # pixels

        galaxy_map = self.seg_map[min_y:max_y, min_x:max_x]
        galaxy = self.cluster_image[min_y:max_y, min_x:max_x]
                
        
        ## add bkg:     
        sigma_clip = SigmaClip(sigma=5., maxiters=15)

        box_x = int(0.2 * galaxy_map.shape[1])  # the 'xy' of the graph are opposite to the 'ij' of the matrix
        if box_x <= 1:
            box_x = 2
        box_y = int(0.2 * galaxy_map.shape[0])
        if box_y <= 1:
            box_y = 2
        bkg = Background2D(galaxy, (box_x, box_y), filter_size=(11, 11), sigma_clip=sigma_clip,
                           bkg_estimator=SExtractorBackground())

        galaxy_sub = np.copy(galaxy)
        galaxy_sub[np.where(galaxy_map!=ind_gal)] = 0
        galaxy_sub += np.random.normal(loc=bkg.background_median, scale=bkg.background_rms_median, size=galaxy.shape)
        galaxy_normed = galaxy_sub / np.max(galaxy_sub)


        image_uns = bkg.background_rms_median 


        ## galaxy initial parameters (the 'xy' of the graph are opposite to the 'ij' of the matrix):
        eps_init = 1 - self.b[ID_ind] / self.a[ID_ind]
        a_init = self.a[ID_ind]
        phi_init = self.phi[ID_ind]
        sers_init = 0.5
        x_0 = (self.x[ID_ind] - min_x - 1)  # "-1" is added b-c of the array syntax
        y_0 = (self.y[ID_ind] - min_y - 1)
        init_params = np.array([eps_init, a_init, phi_init, sers_init, x_0, y_0])


        ## grid adjustments:
        len_gal_x = len(galaxy[0, :])
        len_gal_y = len(galaxy[:, 0])
        axis_x = np.arange(len_gal_x)
        axis_y = np.arange(len_gal_y)
        xx, yy = np.meshgrid(axis_x, axis_y, indexing='xy')  # 'xy' is to match the graph


        ## MCMC:
        def log_likelihood(params, image, uncertainty):
            model_normed = Sersic_2D(xx, yy, params)
            if len(np.where(model_normed > 0)) < 1:
                return -np.inf
            convolution = fftconvolve(model_normed, PSF, mode='same')
            convolution /= np.max(convolution)
            arr = (convolution - image) ** 2 / uncertainty ** 2
            chi2 = np.sum(arr)
            return - 0.5 * chi2


        def log_prior(params):
            eps, a, phi, sers_ind, x0, y0 = params
            if 0 < eps < 0.7\
            and 0.5 * a_init < a < 3 * a_init\
            and -0.5 * np.pi < phi < 0.5 * np.pi\
            and 0.06 < sers_ind < 2\
            and 0.95 * x_0 < x0 < 1.05 * x_0\
            and 0.95 * y_0 < y0 < 1.05 * y_0:  # bn is a good aproximation only for n > 0.06
                return 0.0
            return -np.inf


        def log_posterior(params, image, uncertainty): 
            lp = log_prior(params)
            ll = log_likelihood(params, image, uncertainty)
            if not np.isfinite(lp):
                return -np.inf
            if np.isnan(ll):
                return -np.inf
            return lp + ll


        ndim = len(init_params)  # number of parameters in the model
        nwalkers = 2*ndim + 1  # number of MCMC walkers
        starting_guesses = init_params + 1e-2 * np.random.random((nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(galaxy_normed, image_uns))
        sampler.run_mcmc(starting_guesses, nsteps, progress=MCMC_progress)


        ## extracting intrinsic (fitted) shear:
        emcee_trace = sampler.get_chain(discard=int(0.7 * nsteps), flat=True) 
        eps_bias = np.percentile(emcee_trace[:, 0], 50)
        phi_bias = np.percentile(emcee_trace[:, 2], 50)
        
        
        ## accounting for model bias:
        add_bias = -0.032
        mult_bias = 0.914
        eps = - add_bias + (1 / mult_bias) * eps_bias
        phi = phi_bias
        
        
        ## transforming to ellipticity:
        e = eps / (2 - eps)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)
        
        
        return ID_ind, e1, e2, ind_gal
    
    
    def shear(self, e1, e2, x, y, divx, divy, SN):
        
        
        """A method that calculates the shear over a grid overlaying the galaxy cluster's FOV.
        
        Args:
        e1 (1d ndarray): 1st ellipticity component of all galaxies to be considered (see 'deconvolver').
        e2 (1d ndarray): 2st ellipticity component of all galaxies to be considered (see 'deconvolver').
        x (1d ndarray): x coordinate of all galaxies to be considered.
        y (1d ndarray): y coordinate of all galaxies to be considered.
        divx (int, divx > 0): the number of x axis divisions.
        divy (int, divy > 0): the number of y axis divisions.
        SN (1d ndarray): the flux S/N values of all galaxies to be considered.
        
        *divx and divy set the grid over which the averaging and calculation of the shear takes place. Only cells with at least 8 sources are considered, in order to get reliable averages.
        
        Returns:
        g1 (2d ndarray): 1st shear component grid.
        g1_std (2d ndarray): 1st shear component std grid.
        g2 (2d ndarray): 2st shear component grid.
        g2_std (2d ndarray): 2st shear component std grid.
        g (2d ndarray): shear magnitude grid.
        phi_mean (2d ndarray): shear inclination angle grid.
        jacobian_xy (float): the jacobian transforming from the FOV resolution to the shear grid resolution.
        x_axis_cells (1d ndarray): shear grid's x axis.
        y_axis_cells (1d ndarray): shear grid's y axis.

        *shear is defined similarly to ellipticity (see 'deconvolver').
                
        """
        
        
        if not isinstance(e1, np.ndarray):
            raise TypeError("e1 must be an ndarray.")
        
        if not len(e1.shape) == 1:
            raise TypeError("e1 must be 1 dimensional.")
        
        if not isinstance(e2, np.ndarray):
            raise TypeError("e2 must be an ndarray.")
            
        if not len(e2.shape) == 1:
            raise TypeError("e2 must be 1 dimensional.")
            
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be an ndarray.")
        
        if not len(x.shape) == 1:
            raise TypeError("x must be 1 dimensional.")
            
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be an ndarray.")
        
        if not len(y.shape) == 1:
            raise TypeError("y must be 1 dimensional.")

        if not isinstance(SN, np.ndarray):
            raise TypeError("SN must be an ndarray.")
        
        if not len(SN.shape) == 1:
            raise TypeError("SN must be 1 dimensional.")
            
        if not e1.shape == e2.shape == x.shape == y.shape == SN.shape:
            raise ValueError("e1, e2, x, y, SN must have the same shape.")
            
        if not isinstance(divx, int):
            raise TypeError("divx must be an integer.")
            
        if not divx > 0:
            raise ValueError("divx must be positive.")
            
        if not isinstance(divy, int):
            raise TypeError("divy must be an integer.")
            
        if not divy > 0:
            raise ValueError("divy must be positive.")
        
        
        ## grid adjustments:
        x_min = 0
        x_max = self.cluster_image.shape[1]
        y_min = 0
        y_max = self.cluster_image.shape[0]
    
        x_div = (x_max - x_min) / divx
        y_div = (y_max - y_min) / divy
        jacobian_xy = x_div * y_div

        x_axis_cells = np.linspace(x_min, x_max, divx)
        y_axis_cells = np.linspace(y_min, y_max, divy)
        
        
        ## averaging the shear on a grid:
        g1 = np.zeros((divy, divx))
        g1_std = np.zeros((divy, divx))
        g2 = np.zeros((divy, divx))
        g2_std = np.zeros((divy, divx))
        g = np.zeros((divy, divx))
        phi_mean = np.zeros((divy, divx))
        
        for nx in range(divx):
            for ny in range(divy):
                                
                # finding the indexes of the sources in each cell:
                ind_x_div = np.where((x >= x_min + nx * x_div) & (x < x_min + (nx + 1) * x_div))[0]
                y_x = y[ind_x_div]
                ind_y_x = np.where((y_x >= y_min + ny * y_div) & (y_x < y_min + (ny + 1) * y_div))[0]
                y_xx = y_x[ind_y_x]
                ind_m = np.array([])
                for i in range(len(y_xx)):
                    ind_m = np.append(ind_m, np.where(y == y_xx[i])[0])
                ind_m = ind_m.astype(int)
                                
                if len(ind_m) > 7:  # minimum 8 sources are required in a cell to get a reliable average
                    g1[ny, nx] = np.average(e1[ind_m], weights=SN[ind_m])
                    g1_std[ny, nx] = np.std(e1[ind_m])
                    g2[ny, nx] = np.average(e2[ind_m], weights=SN[ind_m])
                    g2_std[ny, nx] = np.std(e2[ind_m])
                    g[ny, nx] = np.sqrt(g1[ny, nx]**2 + g2[ny, nx]**2)
                    phi_mean[ny, nx] = 0.5 * np.arctan2(g2[ny, nx], g1[ny, nx])  # rad
                else:
                    continue
                    
        
        return g1, g1_std, g2, g2_std, g, phi_mean, jacobian_xy, x_axis_cells, y_axis_cells
    
    
    def convergence_map(self, axis_1, axis_2, shear1, shear2, jacobian, divx, divy):
        
        
        """A method that computes the convergence over the same grid that the shear was calculate on.
        
        Agrs:
        axis_1 (1d ndarray): shear grid's x axis.
        axis_2 (1d ndarray): shear grid's y axis.
        shear1 (2d ndarray): 1st shear component grid (see 'shear').
        shear2 (2d ndarray): 2st shear component grid (see 'shear').
        jacobian (float): the jacobian transforming from the FOV resolution to the shear grid resolution.
        divx (int, divx > 0): the number of x axis divisions.
        divy (int, divy > 0): the number of y axis divisions.
        
        Returns:
        kappa (2d ndarray): the convergence grid.
        
        """
        
        
        if not isinstance(axis_1, np.ndarray) and not isinstance(axis_2, np.ndarray) and not isinstance(shear1, np.ndarray) \
        and not isinstance(shear2, np.ndarray):
            raise TypeError("axis_1, axis_2, shear1, shear2 must be ndarrays.")
            
        if not len(axis_1.shape) == len(axis_2.shape) == 1:
            raise ValueError("axis_1 and axis_2 should 1 dimensional.")
        
        if not axis_1.shape[0] == divx:
            raise ValueError("the length of axis_1 should equal divx.")
            
        if not axis_2.shape[0] == divy:
            raise ValueError("the length of axis_2 should equal divy.")
            
        if not len(shear1.shape) == len(shear2.shape) == 2:
            raise ValueError("shear1 and shear2 must be 2 dimensional.")
            
        if not shear1.shape == shear2.shape:
            raise ValueError("shear1 and shear2 must have the same shape.")
               
        if not isinstance(jacobian, float):
            raise TypeError("jacobian must be a float.")
            
        if not isinstance(divx, int):
            raise TypeError("divx must be an integer.")
            
        if not divx > 0:
            raise ValueError("divx must be positive.")
            
        if not isinstance(divy, int):
            raise TypeError("divy must be an integer.")
            
        if not divy > 0:
            raise ValueError("divy must be positive.")
            
            
        mesh_1, mesh_2 = np.meshgrid(axis_1, axis_2)
        kappa = np.zeros((divy, divx))
        for nx in range(divx):  # iterating over theta
            for ny in range(divy):
                integrand = np.zeros((divy, divx))
                for ix in range(divx):  # iterating over theta_tag
                    delta_1 = mesh_1[0, nx] - axis_1[ix]
                    for iy in range(divy):
                        delta_2 = mesh_2[ny, 0] - axis_2[iy]
                        delta = np.sqrt(delta_1**2 + delta_2**2)
                        if delta == 0:
                            D1 = 0
                            D2 = 0
                        else:
                            D1 = (delta_1**2 - delta_2**2) / delta**4
                            D2 = 2 * delta_1 * delta_2 / delta**4
                        integrand[iy, ix] = D1 * shear1[iy, ix] + D2 * shear2[iy, ix]
                kappa[ny, nx] = np.sum(integrand) * jacobian
        
        
        return - kappa / np.pi
    

convergence_uncertainty = 0.1
    
    
def mass(convergence, mask, D_l, D_s, D_ls, x_min, x_max, y_min, y_max, divx, divy, pixel_in_DD=0.04/3600):
    
    
    """A function that estimates the mass of the galaxy cluster, based on its convergence map
    
    Args:
    convergence (2d ndarray): the convergence map.
    mask (tuple of ndarrays, ndarray or int): the array elements you wish to sum over.
    D_l (float; D_l > 0): the distance from the observer to the lens, in Mpc.
    D_s (float; D_s > 0): the distance from the observer to the source, in Mpc.
    D_ls (float; D_ls > 0): the distance from the lens to the source, in Mpc.
    x_min (int): the minimal value of the x axis of the convergence map.
    x_max (int): the maximal value of the x axis of the convergence map.
    y_min (int): the minimal value of the y axis of the convergence map.
    y_max (int): the maximal value of the y axis of the convergence map.
    divx (int, divx > 0): the number of x axis divisions.
    divy (int, divy > 0): the number of y axis divisions.
    pixel_in_DD (int or float; pixel_in_DD > 0): the FOV resolution, i.e., how many decimal degrees does one pixel represent. The defualt is 0.04/3600, 
                                                 that is, 1 pixel = 0.04".    
    
    Returns:
    mass (float): the mass estimate of the cluster, in solar masses.
    mass_uns (float): the uncertainty of the above value, taken to be 15% of it, following Zitrin et al. (2015)*.

    *Adi Zitrin et al 2015 ApJ 801 44. DOI 10.1088/0004-637X/801/1/44.
    
    """
    
    
    if not isinstance(convergence, np.ndarray):
            raise TypeError("convergence must be an ndarray.")
    
    if not len(convergence.shape) == 2:
            raise ValueError("convergence must be 2 dimensional.")       
            
    if not isinstance(D_l, float):
        raise TypeError("D_l must be a float.")

    if not D_l > 0:
        raise ValueError("D_l must be positive.")
        
    if not isinstance(D_s, float):
        raise TypeError("D_s must be a float.")

    if not D_s > 0:
        raise ValueError("D_s must be positive.")
        
    if not isinstance(D_ls, float):
        raise TypeError("D_ls must be a float.")

    if not D_ls > 0:
        raise ValueError("D_ls must be positive.")
            
    if not isinstance(x_min, int):
        raise TypeError("x_min must be an integer.")
        
    if not isinstance(x_max, int):
        raise TypeError("x_max must be an integer.")
        
    if not isinstance(y_min, int):
        raise TypeError("y_min must be an integer.")
        
    if not isinstance(y_max, int):
        raise TypeError("y_max must be an integer.")
            
    if not isinstance(divx, int):
        raise TypeError("divx must be an integer.")
            
    if not divx > 0:
        raise ValueError("divx must be positive.")

    if not isinstance(divy, int):
        raise TypeError("divy must be an integer.")

    if not divy > 0:
        raise ValueError("divy must be positive.")
        
    if not isinstance(pixel_in_DD, (float, int)):
        raise TypeError("pixel_in_DD must be a float or an integer.")
        
    if not pixel_in_DD > 0:
        raise ValueError("pixel_in_DD must be positive.")
        
        
    G = 4.3016 * 10**(-9)  # (km / s)^2 * (Mpc / M_sun)
    c_cm = 29979245800  # cm / s
    c = c_cm / 10**5  # km / s
    sigma_c = c**2 / (4 * np.pi * G) * D_s / (D_l * D_ls)  # M_sun / Mpc^2

    x_div = (x_max - x_min) / divx
    y_div = (y_max - y_min) / divy
    jacobian = D_l**2 * (pixel_in_DD**2 * x_div * y_div) * (np.pi / 180)**2  # Mpc^2
    
    if np.min(convergence[np.where(~ np.isnan(convergence))]) < 0:
        mass = sigma_c * np.sum(convergence[mask] + np.abs(np.min(convergence[np.where(~ np.isnan(convergence))]))) * jacobian  # M_sun
    else:
        mass = sigma_c * np.sum(convergence[mask]) * jacobian  # M_sun
        
    mass_uns = 0.15 * mass  # M_sun
    
    
    return mass, mass_uns