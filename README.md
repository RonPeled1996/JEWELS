# WLensingtool
A Weak Lensing mass mapping pipeline of galaxy cluster images.

It has been written using the modules: Numpy, Scipy, emcee, astropy and photutils.

### Documentation:
d_A_z(z): **function.** Returns the distance to a source at redshift z in Mpc.

d_A_z1z2(z1, z2): **function.** Returns the between sources at redshifts z1, z2.

Sersic_2D(grid_x, grid_y, params): **function.** A galaxy model (Sersic profile, from https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Sersic2D.html)
    
    Args:
    eps (int or float; range [0, 1]): eps = 1 - b/a, where a and b are the semi-major and semi-minor axes of the Sersic ellipse, respectively.
    a (int or float; a > 0): the semi-major axis of the Sersic ellipse, in pixels.
    phi (int or float; range [-pi/2, pi/2]): the inclination angle of the PIEMD ellipse, in radians.
    sers_ind (int or float; range [0.06, 10]): the Sersic index.
    x0, y0 (int or float): the central coordinates of the Sersic ellipse, in pixels.
    
    Returns:    
    Sersic model (2d ndarray)

PIEMD(grid_x, grid_y, center_x, center_y, eps, phi, sigma_v, r_core, D_l, D_s, D_ls, pixel_in_DD=0.04/3600, pixels_in_x=1, pixels_in_y=1): 
**class.** A galaxy cluster parametric model (from "glafic (2022)" appendix B.6 & B.7).
    
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

.PIEMD_second_derivatives(): **method.** Returns the second derivatives psi_xx, psi_yy, psi_xy. Their shape is the same as the shape of grid_x and grid_y.
.kappa(self):  **method.**  Returns the convergence: 0.5 * (psi_xx + psi_yy)  
.gamma_1(self):  **method.**  Returns the shear's 1st component: 0.5 * (psi_xx - psi_yy)  
.gamma_2(self):  **method.**  Returns the shear's 2nd component: psi_xy 
.gamma(self):  **method.** Returns the shear's magnitude: sqrt(gamma_1^2 + gamma_2^2)  
.angle(self):  **method.** Returns the shear's inclination angle: 0.5 * arctan2(gamma_2, gamma_1)

light_sources(x, y, a, b, phi, cluster_image, seg_map, PSF): **class.** A class for the catalog of a galaxy cluster.
    
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
    
    PSF (2d ndarray): the aperture's PSF. Must be normlized such that np.allclose(np.sum(PSF), 1, 1e-1, 1e-1).

.size_filter(self, size=20): **method.** A method that filters the galaxies according to np.size > size.
        
        Args:
        size (int, size > 0): the array size bar below which galaxies are filtered. Default is 20.
        
        Returns:
        ind_size (1d ndarray): the indexes of the galaxies with np.size > size.

.deconvolver(self, ID_ind, nsteps=10000, MCMC_progress=False): **method.** A method that deconvolves each galaxy image from the aperture's PSF, using MCMC.
        Can be parallelized (see example notebook).
        
        Args:
        ID_ind (int): the galaxy's index to be deconvolved.
        nsteps (int): the number of steps for the MCMC walkers to take. Default is 10000.
        MCMC_progress (bool): whether to show the progress bar or not. Default is False.
        
        *Below, ellipticity is (a - b)/(a + b) that is need for WL; not to be confused with eps = 1 - b/a.
        
        Returns:
        ID_ind (float): the deconvolved galaxy's index.
        e1 (float): the deconvolved galaxy's first ellipticity component.
        e2 (float): the deconvolved galaxy's second ellipticity component.
        ind_gal (float): the deconvolved galaxy's segmantation map index (as it might not be the same as ID_ind).

.shear(self, e1, e2, x, y, divx, divy): **method.** A method that calculates the shear over a grid overlaying the galaxy cluster's FOV.
        
        Args:
        e1 (1d ndarray): 1st ellipticity component of all galaxies to be considered.
        e2 (1d ndarray): 2st ellipticity component of all galaxies to be considered.
        x (1d ndarray): x coordinate of all galaxies to be considered.
        y (1d ndarray): y coordinate of all galaxies to be considered.
        divx (int, divx > 0): the number of x axis divisions.
        divy (int, divy > 0): the number of y axis divisions.
        
        *divx and divy set the grid over which the averaging and calculation of the shear takes place.
        
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

.convergence_map(self, axis_1, axis_2, shear1, shear2, jacobian, divx, divy): **method.** A method that computes the convergence over the same grid that the shear was calculate on.
        
        Agrs:
        axis_1 (1d ndarray): shear grid's x axis.
        axis_2 (1d ndarray): shear grid's y axis.
        shear1 (2d ndarray): 1st shear component grid.
        shear2 (2d ndarray): 2st shear component grid.
        jacobian (float): the jacobian transforming from the FOV resolution to the shear grid resolution.
        divx (int, divx > 0): the number of x axis divisions.
        divy (int, divy > 0): the number of y axis divisions.
        
        Returns:
        kappa (2d ndarray): the convergence grid.

mass(convergence, mask, D_l, D_s, D_ls, x_min, x_max, y_min, y_max, divx, divy, pixel_in_DD=0.04/3600): **function.** A function that estimates the mass of the galaxy cluster, based on its convergence map
    
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
    mass (float): the mass estimate of the cluster.
    mass_uns (float): the uncertainty of the above value, taken to be 15% of it.
