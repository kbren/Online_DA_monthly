import sys
import numpy as np
import pickle
import xarray as xr
import datetime


def load_L(sourcename): 
    """Loads pre-trained L's for a linear inverse model. 
    
       INPUTS: 
       sourcename: string defining the training data used. 
       
       OUTPUTS: 
       LIMd: dictionary containing all the information of the LIM 
             Keys include: ['vec', 'veci', 'val', 'lam_L', 'Gt', 'lam_L_adj', 
                            'npos_eigenvalues', 'E3', 'W_all', 'standard_factor', 
                            'E3_all', 'E_sic', 'var_dict', 'P_train', 'exp_setup', 
                            'frac_neg_eigenvals']     
    """
    
    if 'CESM_LME_40' in sourcename: 
        data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/master/'
        filename = 'LIMd_cesm_lme_002_ntrain_850_1650_validyrs_tas50L40_psl50L40_zg50L40_tos50L40_sit50L40_sic50L40.pkl'
        
    elif 'cesm_lme_nh' in sourcename: 
        data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/master/'
        filename = ('LIMcast_cesm_lme_ntrain_850_1650_cesm_lme_validy_1651_1850_tas50L0.1_psl50L0.1_'+
                    'zg50L0.1_tos50L0.1_sit50L40_sic50L40_20211202.pkl')
        
    elif 'cesm_lme_Amon' in sourcename: 
        data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/master/'
#         filename = ('LIMcast_cesm_lme_Amon_ntrain_850_1650_cesm_lme_Amon_validy_1651_1850_tas50L0.1_psl50L0.1_'+
#                     'zg50L0.1_tos50L0.1_sit50L40_sic50L40_20211202.pkl')
        filename = ('LIMcast_cesm_lme_Amon_ntrain_850_1650_cesm_lme_Amon_validy_1651_1850_tas50L0.1_psl50L0.1_'+
                    'zg50L0.1_tos50L0.1_sit50L40_sic50L40_20211202_sqrtWt.pkl')
    
    elif 'multimod_CESM1_MPI_GFDL_HadGEM3_CanESM_Amon' in sourcename: 
        data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/multimod/'
        filename = ('LIMcast_multimod_CESM1_MPI_GFDL_HadGEM3_CanESM_hist_Amon_ntrain29_202110_cesm_lme_Amon_'+
                    'validy_1851_1950_tas50L0.1_psl50L0.1_zg50L0.1_tos50L0.1_sit50L40_sic50L40_20220114.pkl')
        
    elif 'multimod_MPI_GFDL_HadGEM3_CanESM_Amon' in sourcename: 
        data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/multimod/'
        filename = ('LIMcast_multimod_MPI_GFDL_HadGEM3_CanESM_hist_Amon_ntrain29_202110_cesm_lme_Amon_'+
                    'validy_1851_1950_tas50L0.1_psl50L0.1_zg50L0.1_tos50L0.1_sit50L40_sic50L40_20220114.pkl')
    
    print('Loading: '+str(data_dir+filename))
    LIMd = pickle.load(open(data_dir+filename, "rb" ))
    
    return LIMd

def calc_lim_covariance(Cov, Gt, N, t): 
    """Calculates for the initial covariance (n=0, dt=0))
    """
#    Gt = LIMd['Gt']
#    Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam_L']*t))),LIMd['veci'])
    
    Pb = np.matmul(np.matmul(Gt,Cov),Gt.T) + N
    
    return Pb

def coefficient_efficiency(ref,test,valid=None):
    """
    Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

    Inputs:
    test:  test array
    ref:   reference array, of same size as test
    valid: fraction of valid data required to calculate the statistic 

    Note: Assumes that the first dimension in test and ref arrays is time!!!

    Outputs:
    CE: CE statistic calculated following Nash & Sutcliffe (1970)
    """

    # check array dimensions
    dims_test = test.shape
    dims_ref  = ref.shape
    #print('dims_test: ', dims_test, ' dims_ref: ', dims_ref)

    if len(dims_ref) == 3:   # 3D: time + 2D spatial
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2: # 2D: time + 1D spatial
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1: # 0D: time series
        dims = 1
    else:
        print('In coefficient_efficiency(): Problem with input array dimension! Exiting...')
        SystemExit(1)

    CE = np.zeros(dims)

    # error
    error = test - ref

    # CE
    numer = np.nansum(np.power(error,2),axis=0)
    denom = np.nansum(np.power(ref-np.nanmean(ref,axis=0),2),axis=0)
    CE    = 1. - np.divide(numer,denom)

    if valid:
        nbok  = np.sum(np.isfinite(ref),axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok,nball)
        indok  = np.where(ratio >= valid)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if isinstance(dims,(tuple,list)):
                CE[indbad] = np.nan
            else:
                CE = np.nan

    return CE


def calc_error_covariance(C_0,Gt,t): 
    """Calculates the noise forcing term algebraically (Pendlan 1989, eqn 11)
    """
#     C_0 = LIMd['C_0']
#     Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam_L']*t))),LIMd['veci'])
    
    N = C_0 - np.matmul(np.matmul(Gt,C_0),Gt.T)
    
    return N

def get_min_distance(ob_lat,ob_lon,ref_lat,ref_lon):
    """Calculates circle distances between lat-lon points on the Earth
    
    INPUTS: 
    ob_lat, ob_lon: single lat/lon for observation
    ref_lat, ref_lon: lat/lon fields of reference data
    
    OUTPUS: 
    min_latarg: index of ref_lat closest to ob_lat
    min_lonarg: index of ref_lon closest to ob_lon
    """
    ob_lat = np.deg2rad(ob_lat)
    ob_lon = np.deg2rad(ob_lon)
    ref_lat = np.deg2rad(ref_lat)
    ref_lon = np.deg2rad(ref_lon)
    
    dlat = ref_lat.flatten() - ob_lat
    dlon = ref_lon.flatten() - np.abs(ob_lon)
    
    min_latdist = np.nanmin(np.abs(dlat))
    min_londist = np.nanmin(np.abs(dlon))

    min_latarg = np.argmin(np.abs(dlat))
    min_lonarg = np.argmin(np.abs(dlon))

    return min_latarg, min_lonarg

def build_hadcrut_mask(had_tas, had_lat,had_lon,ref_lat,ref_lon):
    """Calculates circle distances between lat-lon points on the Earth
    
    INPUTS: 
    had_tas: temperature observations (time, nlat, nlon)
    had_lat, had_lon: 2D lat/lon for observations
    ref_lat, ref_lon: lat/lon fields of reference data
    
    OUTPUS: 
    had_refgrid: mask of reference gridcells that are closest to 
                 HadCRUT observations (time, nlat, nlon)
    """
    had_refgrid = np.zeros((had_tas.shape[0],ref_lat.shape[0],ref_lon.shape[0]))
    
    for t in range(had_tas.shape[0]):
        had_lat_finite = had_lat[np.isfinite(had_tas[t,:,:])]
        had_lon_finite = had_lon[np.isfinite(had_tas[t,:,:])]

        for i in range(len(had_lat_finite)):
            min_latarg,min_lonarg = get_min_distance(had_lat_finite[i],had_lon_finite[i],lat,lon)
            had_refgrid[t,min_latarg,min_lonarg] = had_refgrid[t,min_latarg,min_lonarg]+1
            
    return had_refgrid

def build_hadcrut_mask_loc(had_tas, hac_unc, had_lat,had_lon,ref_lat,ref_lon):
    """Calculates circle distances between lat-lon points on the Earth
    
    INPUTS: 
    had_tas: temperature observations (time, nlat, nlon)
    had_lat, had_lon: 2D lat/lon for observations
    ref_lat, ref_lon: lat/lon fields of reference data
    
    OUTPUS: 
    had_refgrid: mask of reference gridcells that are closest to 
                 HadCRUT observations (time, nlat, nlon)
    had_uncertainty_refgrid: uncertainty values of observations associated with 
                 each HadCRUT observation in had_refgrid
    """
    had_refgrid = np.zeros((had_tas.shape[0],ref_lat.shape[0],ref_lon.shape[0]))
    had_uncertainty_refgrid = np.zeros((had_tas.shape[0],ref_lat.shape[0],ref_lon.shape[0]))
    
    all_locations = {}
    
    for t in range(had_tas.shape[0]):
        locations = {}
        had_lat_finite = had_lat[np.isfinite(had_tas[t,:,:])]
        had_lon_finite = had_lon[np.isfinite(had_tas[t,:,:])]
        
        min_latarg_all = np.zeros_like(had_lat_finite)
        min_lonarg_all = np.zeros_like(had_lon_finite)

        for i in range(len(had_lat_finite)):
            min_latarg,min_lonarg = get_min_distance(had_lat_finite[i],had_lon_finite[i],ref_lat,ref_lon)
            had_refgrid[t,min_latarg,min_lonarg] = had_refgrid[t,min_latarg,min_lonarg]+1
            
            min_latarg_all[i] = min_latarg
            min_lonarg_all[i] = min_lonarg                                 
        
        locations['reference_arg_lat'] = min_latarg_all
        locations['reference_arg_lon'] = min_lonarg_all
        locations['had_lat_finite'] = had_lat_finite
        locations['had_lon_finite'] = had_lon_finite
        all_locations[t] = locations
            
    return had_refgrid,all_locations

def build_obs_grid(lat,lon,dlat,dlon,latcutoff=None,latmax=None):
    """Finds the observation location on a regular lat/lon grid
    """
    if latcutoff is None: 
        latcut = lat
        latmin = 0
    else:
        latcut = lat[(lat>latcutoff)]
        latmin = int(np.where(lat>latcutoff)[0][0])
        
    if latmax is None: 
        latend = lat.shape[0]
    else: 
        latend = int(np.where(lat<latmax)[0][-1])

    nlat = lat.shape[0]
    nlon = lon.shape[0]

    obs_grid = np.zeros((nlat,nlon))

    latdiff = np.abs(latcut[:-1]-latcut[1:])[0]
    londiff = np.abs(lon[:-1]-lon[1:])[0]

    skiplat = int(np.ceil(dlat/latdiff))
    skiplon = int(np.ceil(dlon/londiff))

    obs_grid[latmin:latend:skiplat,0::skiplon] = obs_grid[latmin:latend:skiplat,0::skiplon]+1
    
    return obs_grid 

def build_H_time(had_mask): 
    """Builds H for a given time. 
       INPUTS: 
       had_mask: matrix mask of hadCRUT observations on reference grid at given time (nlat,nlon)
       
       OUTPUTS: 
       H: matrix whose columns pick out one observation location (nobs,nlalo)
    """
    ndof = had_mask.shape[0]*had_mask.shape[1]
    nobs = (had_mask>0).sum()
    mask_2d = np.reshape(had_mask,(ndof))
    
    H = np.zeros((nobs,ndof))
    h = 0
    for i in range(ndof):
        if mask_2d[i]>0: 
            H[h,i] =H[h,i]+1
            h =h+1
            
    return H, nobs, ndof

def project_H(H,U):
    """Projects H from lat/lon space to EOF space. 
    
       INPUTS: 
       H: forward operator vector (nlat,nlon)
       U: projection matrix (nlatlon,neofs
       
       OUTPUTS:
       H_eof: observation locations in eof space (nobs,neofs)
    """
    H1d = H.flatten()
    
    H_eof = np.matmul(H1d[np.newaxis,:],U)

def solver_KF_update(Xb, Pb, Y, R, H):
    """Applies a Kalman Filter solver.
    
       INPUTS: 
       Xb: prior mean (ndof,ntime)
       Pb: prior covariance 
       Y: observations (nobs,ntime)
       R: obervation error variance 
       
       OUTPUT: 
       Xa: analysis mean
       Pa: analysis covariance 
    """
    K_num = np.matmul(Pb,H.T)
    K_den = np.matmul(np.matmul(H,Pb),H.T)+R
    K_den_inv = np.linalg.inv(K_den)
    K = np.matmul(K_num,K_den_inv)
    
    # Solve the update equation:
    diff = Y - np.matmul(H,Xb)
    innov = np.matmul(K,diff)
    
    Xa = Xb + innov
    
    return Xa, K, K_den, diff

def solver_KF_cov(K,H,Pb): 
    """Applies a Kalman Filter solver to solve for covariance analysis.
    
       INPUTS: 
       Pb: prior covariance 
       R: obervation error variance 
       K: Kalman gain (ndof,nobs)
       
       OUTPUT: 
       Pa: analysis covariance 
    """  
    ndof = Pb.shape[0]
    
    # Solve the covariance update: 
    part = np.identity(ndof) - np.matmul(K,H)
    
    Pa = np.matmul(part,Pb)
    
    return Pa
    

def covariance(X,Y):
    """This function takes two time series of the same size and calculates 
       the covariance. First it removes the mean and then multiplies the 
       two timeseries and sums them. ONLY WORKS FOR 1D ARRAYS? 
       
       X = array (time,lat,lon)
       Y = array (time,lat,lon)
    """

    X_mean = np.nanmean(X,axis=0)
    X_anom = X - X_mean

    Y_mean = np.nanmean(Y,axis=0)
    Y_anom = Y - Y_mean

    cov = (np.dot(X_anom,Y_anom))/(X.shape[0]-1)

    return cov

def covariance_nans(X,Y,anomalize=False):
    """This function takes two time series of the same size and calculates 
       the covariance. First it removes the mean and then multiplies the 
       two timeseries and sums them. ONLY WORKS FOR 1D ARRAYS? 
       
       X = array (time,lat,lon)
       Y = array (time,lat,lon)
    """
    
    if anomalize: 
        X_mean = np.nanmean(X,axis=0)
        X_anom = X - X_mean

        Y_mean = np.nanmean(Y,axis=0)
        Y_anom = Y - Y_mean
    else: 
        X_anom = X
        Y_anom = Y 
    
    if np.isnan(Y_anom).sum()>0: 
        Y_anom_nonan = np.nan_to_num(Y_anom)
    else: 
        Y_anom_nonan = Y_anom
        
    if np.isnan(X_anom).sum()>0: 
        X_anom_nonan = np.nan_to_num(X_anom)
    else: 
        X_anom_nonan = X_anom

    cov = (np.dot(X_anom_nonan,Y_anom_nonan))/(X.shape[0]-1)

    return cov

def LIM_forecast(LIMd,x,Pa,lag,adjust=True):
    """
    # There is a bug with this forecast function function: It uses the eigenvectors and 
    #        values to calculate Gt, but it's giving the same value for all lags in the forecast
    
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * adj: True/False, if True removes negative eigenvalues 
    
    Outputs (in a dictionary):
    *'error' - error variance as a function of space and forecast lead time (ndof,ntims)
    *'x_forecast' - the forecast states (ndof,ntims)
    *'x_truth_phys_space' - true state in physical space (nlat*nlon,*ntims)
    *'x_forecast_phys_space' - forecast state in physical space (nlat*nlon,*ntims)
    """
    
    ndof = x.shape[0]
#    ntims = x.shape[1]
    LIMfd = {}
    
    max_eigenval = np.real(LIMd['lam_L']).max()
    
    if adjust: 
        print('Adjust is True...')
        if max_eigenval >0: 
            print('YES negative eigenvalue found...adjusting')
            LIMd['frac_neg_eigenvals'] = ((LIMd['lam_L']>0).sum())/(LIMd['lam_L'].shape[0])
            LIMd['lam_L_adj'] = LIMd['lam_L'] - (max_eigenval+0.01)
        else: 
            print('NO negative eigenvalue found...')
            LIMd['frac_neg_eigenvals'] = 0
            LIMd['lam_L_adj'] = LIMd['lam_L']
    else: 
        print('Adjust is False...')
        LIMd['frac_neg_eigenvals'] = np.nan
        LIMd['lam_L_adj'] = LIMd['lam_L']
    
    print('lag=',lag)
    # make the propagator for this lead time
    #Gt = np.real(np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam_L_adj']*lag))),LIMd['veci']))
    Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam_L_adj']*lag))),LIMd['veci'])

    # forecast mean state X:
    if lag == 0:
        # need to handle this time separately, or the matrix dimension is off
        x_predict = np.matmul(Gt,x)
#           x_predict_save[k,:,:] = x_predict
    else:
        x_predict = np.matmul(Gt,x)
#            x_predict_save[k,:,t:] = x_predict
    
    # forecast covariance P:
    N = calc_error_covariance(LIMd['C_0'],Gt,lag)
    Cov_predict = calc_lim_covariance(Pa, Gt, N, lag)

    Ld = {}
    Ld['Gt'] = Gt
    LIMfd[lag] = Ld

    LIMfd['x_forecast'] = np.squeeze(x_predict)  
    LIMfd['cov_forecast'] = np.real(Cov_predict)
    LIMfd['Noise'] = N
        
    return LIMfd

def decompress_eof_separate_sic(X_proj, LIMd):
    """
    INPUTS: 
    ========
    P_train
    nmodes
    nmodes_sic
    E
    E_sic
    limvars
    var_dict
    W_all
    Weights=True
    sic_separate=False
    
    OUTPUTS: 
    ========
    X_train_dcomp
    """
    E3_all = np.zeros((LIMd['E3_all'].shape[0]+LIMd['E_sic'].shape[0], 
                       LIMd['E3_all'].shape[1]+LIMd['E_sic'].shape[1]))

    E3_all[0:LIMd['E3_all'].shape[0],0:LIMd['E3_all'].shape[1]] = LIMd['E3_all']
    E3_all[LIMd['E3_all'].shape[0]:,LIMd['E3_all'].shape[1]:] = LIMd['E_sic']

    X_dcomp = np.matmul(E3_all,X_proj)

    if LIMd['exp_setup']['Weight'] is True: 
        X_train_dcomp = unweight_decompressed_vars(X_dcomp, LIMd['exp_setup']['limvars'], 
                                                   LIMd['var_dict'], LIMd['W_all'])
    else: 
        X_train_dcomp = x_train_dcomp
    
    return X_train_dcomp, E3_all

def unweight_decompressed_vars(x_train_dcomp, limvars, var_dict, W_all):
    """
    INPUTS: 
    ========
    x_train_dcomp
    limvars
    var_dict
    W_all
    
    OUTPUTS: 
    ========
    X_out 
    """
    X_out = np.zeros_like(x_train_dcomp)
    
    start=0
    if len(x_train_dcomp.shape)<2:
        for var in (limvars):
            inds_end = var_dict[var]['var_ndof']
            X_out[start:start+inds_end] = x_train_dcomp[start:start+inds_end]/W_all[var][:]
            start = start+inds_end
    else:
        for var in (limvars):
            inds_end = var_dict[var]['var_ndof']
            X_out[start:start+inds_end,:] = x_train_dcomp[start:start+inds_end,:]/W_all[var][:,np.newaxis]
            start = start+inds_end
        
    return X_out

def count_ndof_all(limvars, E3, sic_separate=False): 
    #Count total degrees of freedom: 
    if sic_separate is True: 
        if len(limvars) <2: 
            ndof_all = 0
            neof_all = 0
        else: 
            limvars_nosic = [l for l in limvars if l not in 'sic']
            for v,var in enumerate(limvars_nosic):
                if v == 0: 
                    ndof_all = E3[var].shape[0]
                    neof_all = E3[var].shape[1]
                else: 
                    ndof_all = ndof_all+E3[var].shape[0]
                    neof_all = neof_all+E3[var].shape[1]
    else:      
        for v,var in enumerate(limvars):
            if v == 0: 
                ndof_all = E3[var].shape[0]
                neof_all = E3[var].shape[1]
            else: 
                ndof_all = ndof_all+E3[var].shape[0]
                neof_all = neof_all+E3[var].shape[1]
#            print(ndof_all)
            
    return ndof_all, neof_all

def get_var_indices(limvars, var_dict): 
    """Get indices for each variable
    """
    start = 0
    for k, var in enumerate(limvars): 
        print('working on '+var)
        inds = var_dict[var]['var_ndof']
        var_inds = np.arange(start,start+inds,1)
        start = inds+start

        var_dict[var]['var_inds'] = var_inds
        
    return var_dict

def project_validation_var_og(X, E3, standard_factor, W, Weights=False): 
    """
    """
    if Weights is True: 
        if len(W.shape)<2:
            W_new = W[:,np.newaxis]
        else: 
            W_new = W
        eofs_out = E3/standard_factor
        # projection
        P_var = np.matmul(eofs_out.T,W_new*np.nan_to_num(X))
    else: 
        eofs_out = E3/standard_factor
        # projection
        P_var = np.matmul(eofs_out.T,np.nan_to_num(X))

    Ptrunc = P_var/standard_factor
        
    return Ptrunc

def project_validation_var(X, E3, standard_factor, W, Weights=False): 
    """
    """
    if Weights is True: 
        if len(W.shape)<2:
            W_new = W[:,np.newaxis]
        else: 
            W_new = W
        eofs_out = E3
        # projection
        P_var = np.matmul(eofs_out.T,W_new*np.nan_to_num(X))
    else: 
        eofs_out = E3/standard_factor
        # projection
        P_var = np.matmul(eofs_out.T,np.nan_to_num(X))

    standard_factor = np.sqrt(np.var(P_var))
    Ptrunc = P_var/standard_factor
        
    return Ptrunc, standard_factor


def stack_variable_eofs(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                        var_dict, verbose=False): 
    start = 0
    nvars = len(limvars)
    
    E3_all = np.zeros((ndof_all,int(ntrunc*nvars)))

    for v,var in enumerate(limvars):
        if v == 0: 
            Ptrunc_all = Ptrunc[var]
        else: 
            Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
        E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]

        if verbose is True:
            print(E3_all[18430:18435,0])
            print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']
      
    return Ptrunc_all, E3_all

def stack_variable_eofs_standardized2(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                                      standard_factor, var_dict, verbose=False): 
    start = 0
    nvars = len(limvars)
    
    E3_all = np.zeros((ndof_all,int(ntrunc*nvars)))
    E3_all_stand_to = np.zeros((ndof_all,int(ntrunc*nvars)))
    E3_all_stand_from = np.zeros((ndof_all,int(ntrunc*nvars)))

    for v,var in enumerate(limvars):
        if v == 0: 
            Ptrunc_all = Ptrunc[var]
        else: 
            Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
        E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
        E3_all_stand_to[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]/(standard_factor[var]**2)
        E3_all_stand_from[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]*(standard_factor[var]**2)

        if verbose is True:
            print(E3_all[18430:18435,0])
            print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']
      
    return Ptrunc_all, E3_all, E3_all_stand_to, E3_all_stand_from

def stack_variable_eofs_standardized(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                                     standard_factor, var_dict, verbose=False): 
    start = 0
    nvars = len(limvars)
    
    E3_all = np.zeros((ndof_all,int(ntrunc*nvars)))
    E3_all_stand_to = np.zeros((ndof_all,int(ntrunc*nvars)))
    E3_all_stand_from = np.zeros((ndof_all,int(ntrunc*nvars)))

    for v,var in enumerate(limvars):
        if v == 0: 
            Ptrunc_all = Ptrunc[var]
        else: 
            Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
        E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
        E3_all_stand_to[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]/standard_factor[var]
        E3_all_stand_from[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]*standard_factor[var]

        if verbose is True:
            print(E3_all[18430:18435,0])
            print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']
      
    return Ptrunc_all, E3_all, E3_all_stand_to, E3_all_stand_from

def stack_variable_eofs_og(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                        var_dict,nmonths=1,sic_separate=False, 
                        verbose=False): 
    start = 0
    if sic_separate is True: 
        limvars_nosic = [l for l in limvars if l not in 'sic']
        nvars = len(limvars_nosic)
        E3_all = np.zeros([ndof_all,int(ntrunc*(nvars)),nmonths])
        if nmonths == 1: 
            E3_all = np.squeeze(E3_all)

        for v,var in enumerate(limvars_nosic):
            print(str(v) + ', '+var)
            if v == 0: 
                Ptrunc_all = Ptrunc[var]
            else: 
                Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
                
            E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
            
            if verbose is True: 
                print(E3_all[18430:18435,0])
                print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']

        Ptrunc_sic = Ptrunc['sic']
        E_sic = E3['sic']
    else: 
        nvars = len(limvars)
        E3_all = np.zeros([ndof_all,int(ntrunc*(nvars))])
        if nmonths == 1: 
            E3_all = np.squeeze(E3_all)

        for v,var in enumerate(limvars):
            if v == 0: 
                Ptrunc_all = Ptrunc[var]
            else: 
                Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
            E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
            
            if verbose is True:
                print(E3_all[18430:18435,0])
                print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']
            
    if sic_separate == True: 
        return Ptrunc_all, E3_all, Ptrunc_sic, E_sic
    else: 
        return Ptrunc_all, E3_all
    
def calc_tot_si_checks(var, areacell, units, lat, nlon, lat_cutoff=0.0): 
    """Calculates total NH sea ice area. 
    
    INPUTS: 
    var: sea ice concentration (ndof,ntime), (ndarray)
    areacell: grid cell area, 1D or 2D, (ndarray)
    units: string ('km^2' or 'm2')
    lat: 
    lat_cutoff
    
    OUTPUTS: 
    tot_nh_var
    """
    
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell

    if units == 'm2':
        cellarea = (areacell_1d*1e-6)[:,np.newaxis]
    else: 
        cellarea = areacell_1d[:,np.newaxis]

    if len(var.shape)<=1:
        var_3d = np.reshape(var,(lat.shape[0],nlon))
    elif len(var.shape) ==2:     
        var_3d = np.reshape(var,(lat.shape[0],nlon,var.shape[1]))
    else: 
        var_3d = var

    if len(lat.shape)<=1:
        var_nh_3d = var_3d[(lat>0),:,:]
        test_var = var_nh_3d[np.isfinite(var_nh_3d)]
    elif len(lat.shape)>1:
        var_nh_3d = var_3d[(lat>0),:]
        test_var = var_nh_3d[np.isfinite(var_nh_3d)]

    if np.nanmax(test_var)>5:
        print('Max concentration is '+str(np.round(np.nanmax(test_var),2))+
              ' ...dividing concentration by 100.')
        Var = var/100.0
    else: 
        Var = var

    nh_var = Var*cellarea

    if len(lat.shape)<=1:
        if len(var.shape)<=1:
            nh_var_3d = np.reshape(nh_var,(lat.shape[0],nlon))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_var = np.nansum(np.nansum(nh_var_3d[lat_inds,:].squeeze(),axis=0),axis=0)
        else:     
            nh_var_3d = np.reshape(nh_var,(lat.shape[0],nlon,var.shape[1]))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_var = np.nansum(np.nansum(nh_var_3d[lat_inds,:,:].squeeze(),axis=0),axis=0)
    else:
        lat_1d = np.reshape(lat,(var.shape[0]))
        lat_inds = np.where(lat_1d>lat_cutoff)
        tot_nh_var = np.nansum(nh_var[lat_inds,:].squeeze(),axis=0)
        
    return tot_nh_var

def calc_tot_sivol_checks(sit, areacell, units, lat, nlon, lat_cutoff=0.0): 
    """Calculates total NH sea ice volume. 
    
    INPUTS: 
    var: sea ice thickness (ndof,ntime), units = 'sivol per unit grid cell area', [ndarray]
    areacell: grid cell area, 1D or 2D, [ndarray]
    units: string ('km^2' or 'm2')
    lat: 
    lat_cutoff
    
    OUTPUTS: 
    tot_nh_vol
    """
    
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell

    if units == 'm2':
        print('changing units of cell area from '+units+' to km^2')
        cellarea = (areacell_1d*1e-6)[:,np.newaxis]
    else: 
        print('not changing units of cell area')
        cellarea = areacell_1d[:,np.newaxis]

    if len(sit.shape)<=1:
        sit_3d = np.reshape(sit,(lat.shape[0],nlon))
    elif len(sit.shape) ==2:     
        sit_3d = np.reshape(sit,(lat.shape[0],nlon,sit.shape[1]))
    else: 
        sit_3d = sit

    if len(lat.shape)<=1:
        sit_nh_3d = sit_3d[(lat>0),:,:]
        test_sit = sit_nh_3d[np.isfinite(sit_nh_3d)]
    elif len(lat.shape)>1:
        sit_nh_3d = sit_3d[(lat>0),:]
        test_sit = sit_nh_3d[np.isfinite(sit_nh_3d)]

    nh_vol = sit*cellarea

    if len(lat.shape)<=1:
        if len(sit.shape)<=1:
            nh_vol_3d = np.reshape(nh_vol,(lat.shape[0],nlon))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_vol = np.nansum(np.nansum(nh_vol_3d[lat_inds,:].squeeze(),axis=0),axis=0)
        else:     
            nh_vol_3d = np.reshape(nh_vol,(lat.shape[0],nlon,sit.shape[1]))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_vol = np.nansum(np.nansum(nh_vol_3d[lat_inds,:,:].squeeze(),axis=0),axis=0)
    else:
        lat_1d = np.reshape(lat,(sit.shape[0]))
        lat_inds = np.where(lat_1d>lat_cutoff)
        tot_nh_vol = np.nansum(nh_vol[lat_inds,:].squeeze(),axis=0)
        
    return tot_nh_vol

def global_mean_noarea(var,lat): 
    """Assumes var is dimensions (time,nlat,nlon)
    """
    var_shape = len(var.shape)
    var_nan_mask = np.where(np.isnan(var),np.nan,1)
    
    weights = np.cos(np.deg2rad(lat))
    weights_2d = weights[:,np.newaxis]*np.ones((var[0,:,:].shape))
    
    tot_var = np.nansum(np.nansum(var*weights[np.newaxis,:,np.newaxis],axis=1),axis=1)
    wt_sum = np.nansum(np.nansum(weights_2d,axis=0),axis=0)
    
    var_mn = tot_var/wt_sum
    
    return var_mn

def global_mean(var, areacell): 
    """Assumes var is dimensions (nlat*nlon,time)
    """
    var_shape = len(var.shape)
    var_nan_mask = np.where(np.isnan(var),np.nan,1)
    
    if (var_shape<2)&(len(areacell.shape)>1): 
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    elif (var_shape>1)&(len(areacell.shape)>1): 
        areacell_1d_temp = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
        areacell_1d = areacell_1d_temp[:,np.newaxis]
    elif (var_shape>1)&(len(areacell.shape)<=1): 
        areacell_1d = areacell[:,np.newaxis]
    else: 
        areacell_1d = areacell
        
    tot_nh_var = var*areacell_1d
    
    tot_var = np.nansum(tot_nh_var,axis=0)
    wt_sum = np.nansum(areacell_1d*var_nan_mask,axis=0)
    
    var_mn = tot_var/wt_sum
    
    return var_mn

def arctic_mean(var, areacell, cutoff=0.0): 
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell
        
    tot_nh_var = var*areacell_1d
    
    if len(lat.shape)<=1:
        lat_inds = np.where(var_dict[var]['lat']>cutoff)
        tot_nh_var = np.nansum(np.nansum(tot_nh_var[:,lat_inds,:],axis=1),axis=1)
    
        wt_sum = np.nansum(np.nansum(cellarea[lat_inds,:],axis=0),axis=0)
    else:
        lat_inds = np.where(var_dict[var]['lat']>cutoff)
        tot_nh_var = np.nansum(np.nansum(tot_nh_var[:,lat_inds],axis=1),axis=1)
    
        wt_sum = np.nansum(np.nansum(cellarea[lat_inds],axis=0),axis=0)
    
    var_mn = tot_nh_var/wt_sum
    
    return var_mn

def calc_validataion_stats(var, truth_anom, forecast_anom, var_dict,areacell,areacell_dict,
                           areawt_name,LIMd,lat_cutoff=False,iplot=False):
    """
    """ 
    units = areacell_dict[var][areawt_name[var]]['units']

    if 'km' in units:
        acell = areacell[var]
    elif 'centi' in units: 
        print('changing cellarea units from '+
              str(areacell_dict[var][areawt_name[var]]['units'])+' to km^2')
        acell = areacell[var]*(1e-10)
        units = 'km^2'
    else: 
        print('changing cellarea units from '+
              str(areacell_dict[var][areawt_name[var]]['units'])+' to km^2')
        acell = areacell[var]*(1e-6)
        units = 'km^2'
        
    if var == 'sic':
        nlon = int(var_dict[var]['var_ndof']/var_dict[var]['lat'].shape[0])
        tot_var_forecast = statskb.calc_tot_si_checks(forecast_anom,acell,units,var_dict[var]['lat'],nlon,lat_cutoff=0.0)
        tot_var_truth = statskb.calc_tot_si_checks(truth_anom,acell,units,var_dict[var]['lat'],nlon,lat_cutoff=0.0)
    else: 
        tot_var_forecast = statskb.global_mean(forecast_anom,acell)
        tot_var_truth = statskb.global_mean(truth_anom,acell)
    
    
    if iplot==True: 
        time = var_dict[var]['time']
        ntime = tot_var_truth.shape[0]
        ttime = time.shape[0]
        
        plt.figure(figsize=(6,4))
        plt.plot(tot_var_truth,label='truth')
        plt.plot(tot_var_forecast,label='forecast')
        plt.xlim(0,20)
        plt.legend()
        plt.show()

    corr_tot = np.corrcoef(tot_var_truth,np.nan_to_num(tot_var_forecast))[0,1]
    ce_tot = oda.coefficient_efficiency(tot_var_truth,np.nan_to_num(tot_var_forecast))
    
     ## New error var calculations: 10/06/21
    forecast_nan_mask = np.where(np.isclose(np.nanvar(LIMd['E3'][var],axis=1),0,atol=1e-5),np.nan,1)

    rmse = np.sqrt(np.nanmean((truth_anom-forecast_anom)**2,axis=1))
    gm_rmse = statskb.global_mean(rmse*forecast_nan_mask,acell)
    gsum_rmse = np.nansum(gm_rmse*forecast_nan_mask)
    
    return corr_tot, ce_tot, tot_var_forecast, tot_var_truth, gm_rmse, gsum_rmse, rmse

def calc_corr_ce(truth,recon):

    corr = np.corrcoef(truth,np.nan_to_num(recon))[0,1]
    ce = coefficient_efficiency(truth,np.nan_to_num(recon))

    return corr, ce

def load_HadCRUT5(): 
    """HadCRUT5 tas data units are in Kelvin. 
    """
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/HadCRUT/'
    hadcruname = 'HadCRUT.5.0.1.0.anomalies.ensemble_mean.nc'
    had_ds = xr.open_dataset(data_dir+hadcruname)
    
    HadCRUT_tas = had_ds.tas_mean.values
    HadCRUT_lat = had_ds.latitude.values
    HadCRUT_lon = had_ds.longitude.values
    HadCRUT_time = had_ds.time.values
    
    return HadCRUT_tas, HadCRUT_lat, HadCRUT_lon, HadCRUT_time

def load_HadCRUT5_uncertainty(): 
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/HadCRUT/'
    hadcruname = 'HadCRUT.5.0.1.0.uncorrelated.nc'
    had_ds = xr.open_dataset(data_dir+hadcruname)
    
    HadCRUT_unc_tas = had_ds.tas_unc.values
    HadCRUT_unc_lat = had_ds.latitude.values
    HadCRUT_unc_lon = had_ds.longitude.values
    HadCRUT_unc_time = had_ds.time.values
    
    return HadCRUT_unc_tas, HadCRUT_unc_lat, HadCRUT_unc_lon, HadCRUT_unc_time

def load_HadCRUT5_regridLME(): 
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/HadCRUT/'
    hadcruname = 'HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean_regridLME.nc'
    had_ds = xr.open_dataset(data_dir+hadcruname)
    
    HadCRUT_tas = had_ds.tas_mean.values
    HadCRUT_lat = had_ds.lat.values
    HadCRUT_lon = had_ds.lon.values
    HadCRUT_time = had_ds.time.values
    
    return HadCRUT_tas, HadCRUT_lat, HadCRUT_lon, HadCRUT_time

def load_HadSST4(): 
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/HadCRUT/'
    hadsstname = 'HadSST.4.0.1.0_median.nc'
    sst_ds = xr.open_dataset(data_dir+hadsstname)
    
    HadSST_tas = sst_ds.tos.values
    HadSST_lat = sst_ds.latitude.values
    HadSST_lon = sst_ds.longitude.values
    HadSST_time = sst_ds.time.values
    
    return HadSST_tas, HadSST_lat, HadSST_lon, HadSST_time

def load_CRUTEMP5(): 
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/HadCRUT/'
    cruname = 'CRUTEM.5.0.1.0.anomalies.nc'
    cru_ds = xr.open_dataset(data_dir+cruname)
    
    CRUT_tas = cru_ds.tas.values
    CRUT_lat = cru_ds.latitude.values
    CRUT_lon = cru_ds.longitude.values
    CRUT_time = cru_ds.time.values
    
    return CRUT_tas, CRUT_lat, CRUT_lon, CRUT_time


def load_GISTEMP(): 
    data_dir = '/home/disk/kalman3/rtardif/LMR/data/analyses/GISTEMP/'
    gisname = 'gistemp1200_ERSSTv5.nc'
    gis_ds = xr.open_dataset(data_dir+gisname)
    
    GIS_tas = gis_ds.tempanomaly.values
    GIS_lat = gis_ds.lat.values
    GIS_lon = gis_ds.lon.values
    GIS_time = gis_ds.time.values
    
    return GIS_tas, GIS_lat, GIS_lon, GIS_time

def load_BerkeleyEarth(): 
    data_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/BerkeleyEarth/'
    BEname = 'Land_and_Ocean_LatLong1.nc'
    BE_ds = xr.open_dataset(data_dir+BEname)
    
    BE_tas = BE_ds.temperature.values
    BE_lat = BE_ds.latitude.values
    BE_lon = BE_ds.longitude.values
    
    nyears = 2021-1850+1
    
    t_total = nyears*12
    years = int(1850+np.floor((t_total-1)/12))
    BE_time = np.array([datetime.datetime(y, m, 15) for y in np.arange(1850,1850+nyears,1) for m in np.arange(1,13,1)])
    
    return BE_tas, BE_lat, BE_lon, BE_time

def load_GFDLECDA(): 
    GFDLECDA_folder = '/home/disk/katabatic/wperkins/data/LMR/data/analyses/GFDLECDA/'
    GFDLECDA_file = 'sst_GFDLECDAv3.1_196101-201012.nc'
    
    GFDLECDA_data = xr.open_dataset(GFDLECDA_folder+GFDLECDA_file)
    
    GFDLECDA_lon = GFDLECDA_data.lon.values
    GFDLECDA_lat = GFDLECDA_data.lat.values
    GFDLECDA_sst = GFDLECDA_data.sst.values

    nyears = 2010-1961+1

    t_total = nyears*12
    years = int(1961+np.floor((t_total-1)/12))
    GFDLECDA_time = np.array([datetime.datetime(y, m, 15) for y in np.arange(1961,1961+nyears,1) for m in np.arange(1,13,1)])
    
    return GFDLECDA_sst, GFDLECDA_lat, GFDLECDA_lon, GFDLECDA_time

def load_HadleyEN4():
    HadleyEN4_folder = '/home/disk/katabatic/wperkins/data/LMR/data/analyses/HadleyEN4/'
    HadleyEN4_file = 'sst_HadleyEN4.2.1g10_190001-201012.nc'
    
    HadleyEN4_data = xr.open_dataset(HadleyEN4_folder+HadleyEN4_file)
    
    HadleyEN4_lon = HadleyEN4_data.lon.values
    HadleyEN4_lat = HadleyEN4_data.lat.values
    HadleyEN4_sst = HadleyEN4_data.sst.values
    HadleyEN4_time = HadleyEN4_data.time.values

    return HadleyEN4_sst, HadleyEN4_lat, HadleyEN4_lon, HadleyEN4_time

def load_ORA20C():
    ORA20C_folder = '/home/disk/katabatic/wperkins/data/LMR/data/analyses/ORA20C/'
    ORA20C_file = 'sst_ORA20C_ensemble_mean_190001-200912.nc'
    ORA20C_spread = 'sst_ORA20C_ensemble_spread_190001-200912.nc'
    
    ORA20C_data = xr.open_dataset(ORA20C_folder+ORA20C_file)
    
    ORA20C_lon = ORA20C_data.lon.values
    ORA20C_lat = ORA20C_data.lat.values
    ORA20C_sst = ORA20C_data.sst.values
    ORA20C_time = ORA20C_data.time.values

    return ORA20C_sst, ORA20C_lat, ORA20C_lon, ORA20C_time

def load_annual_satellite(): 
    """Loads annual satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    #fet_file = 'Fetterer_data_v3_annual_1978_2017.npz'
    fet_file = 'Fetterer_data_v3_annual_78_17.npz'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent']
    fet_sia = fet_data['si_area']
    fet_time = fet_data['time']
    
    return fet_sia, fet_sie, fet_time

def load_annual_satellite_anom(ANOM_END): 
    """Loads annual satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    #fet_file = 'Fetterer_data_v3_annual_1978_2017.npz'
    fet_file = 'Fetterer_data_v3_annual_78_17.npz'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent'][1:]
    fet_sia = fet_data['si_area'][1:]
    fet_sia_adj = fet_data['si_area_adj']
    fet_time = fet_data['time'][1:]

    # Find anomalies: 

    # Calculate mean 
    fet_anom_cent_sia = np.nanmean(fet_sia[np.where(fet_time<=ANOM_END)],axis=0)
    fet_anom_cent_sia_adj = np.nanmean(fet_sia_adj[np.where(fet_time[1:]<=ANOM_END)],axis=0)
    fet_anom_cent_sie = np.nanmean(fet_sie[np.where(fet_time<=ANOM_END)],axis=0)

    # Find anomalies:  
    fet_sia_anom = fet_sia - fet_anom_cent_sia
    fet_sia_anom_adj = fet_sia_adj - fet_anom_cent_sia_adj
    fet_sie_anom = fet_sie - fet_anom_cent_sie
    
    return fet_sia_anom, fet_sia_anom_adj, fet_sie_anom, fet_time

def load_bren2020_full(): 
    data_dir = ('/home/disk/p/mkb22/Documents/si_analysis_kb/'+
                'instrumental_assimilation_experiments/Brennan_etal_2020/data/')
    
    filename = 'Brennan_etal_2020_sie_recons.nc'

    import xarray as xr

    data_bren2020 = xr.open_dataset(data_dir+filename)
    
    return data_bren2020

def load_bren2020_data(mod_list,temp_list):
    """INPUTS:
       mod_list = list of strings, all caps (ex: ['MPI','CCSM4'])
       temp_list = list of temperature datasets (ex: ['HadCRUT4','Berkeley_Earth','GISTEMP'])
    """
    data_bren2020 = load_bren2020_full()
    sie_bren2020 = {}
    sie_97_5_bren2020 = {}
    sie_2_5_bren2020 = {}
    
    for m in mod_list: 
        for d in temp_list: 
            sie_name = 'sie_'+m+'_'+d
            bren2020_time = data_bren2020[sie_name][d+'_time'].values
            sie_bren2020[sie_name] = np.nanmean(np.reshape(data_bren2020[sie_name].values,(169,1000)),
                                                axis=1)
            sie_97_5_bren2020[sie_name+'_97_5'] = data_bren2020[sie_name+'_97_5'].values
            sie_2_5_bren2020[sie_name+'_2_5'] = data_bren2020[sie_name+'_2_5'].values
            
    return bren2020_time, sie_bren2020, sie_97_5_bren2020, sie_2_5_bren2020



