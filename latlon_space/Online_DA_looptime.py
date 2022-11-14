import sys
import numpy as np
import xarray as xr
import datetime

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pickle 

import Online_DA_utils as oda

t =0
# Select number of years to reconstruct: 
nyears = 1

# Generate times: 
t_total = nyears*12+1
years = int(1850+np.floor((t_total-1)/12))
time = np.array([datetime.datetime(y, m, 15) for y in np.arange(1850,years,1) for m in np.arange(1,13,1)])

limvars = ['tas','psl','zg','tos','sit','sic']

## ---------------------------------------------------------
## LOAD L: 
## ---------------------------------------------------------
print('Loading L....')
LIM = oda.load_L('cesm_lme_Amon')
LIMd = LIM['LIMd']
LIMd.keys()

Projector_tas = LIMd['E3']['tas']

## ---------------------------------------------------------
## LOAD HadCRUT5: 
## ---------------------------------------------------------
print('Loading HadCRUT5....')
HadCRUT_tas, HadCRUT_lat, HadCRUT_lon, HadCRUT_time = oda.load_HadCRUT5()
HadCRUT_lon[:36] = HadCRUT_lon[:36] +360

HadCRUT_tas_nh = HadCRUT_tas[:,18:,:]
HadCRUT_lat_nh = HadCRUT_lat[18:]

nobs_had = np.isfinite(HadCRUT_tas_nh[0,:,:]).sum()

HadCRUT_lat_2d = HadCRUT_lat_nh[:,np.newaxis]*np.ones((HadCRUT_lat_nh.shape[0],HadCRUT_lon.shape[0]))
HadCRUT_lon_2d = np.ones((HadCRUT_lat_nh.shape[0],HadCRUT_lon.shape[0]))*HadCRUT_lon[np.newaxis,:]

HadCRUT_obs_1d = HadCRUT_tas_nh[t,np.isfinite(HadCRUT_tas_nh[t,:,:])]
HadCRUT_obs_lat = HadCRUT_lat_2d[np.isfinite(HadCRUT_tas_nh[t,:,:])]
HadCRUT_obs_lon = HadCRUT_lon_2d[np.isfinite(HadCRUT_tas_nh[t,:,:])]

## ---------------------------------------------------------
## BUILD R: 
## ---------------------------------------------------------
print('Building R....')
HadCRUT_unc_tas, HadCRUT_unc_lat, HadCRUT_unc_lon, HadCRUT_unc_time = oda.load_HadCRUT5_uncertainty()
HadCRUT_unc_tas_nh = HadCRUT_unc_tas[:,18:,:]
HadCRUT_unc_lat_nh = HadCRUT_unc_lat[18:]

HadCRUT_obs_unc_1d = HadCRUT_unc_tas_nh[t,np.isfinite(HadCRUT_unc_tas_nh[t,:,:])]

R_had = oda.covariance_nans(HadCRUT_obs_unc_1d[:,np.newaxis],HadCRUT_obs_unc_1d[:,np.newaxis].T,anomalize=False)

## ---------------------------------------------------------
## LOAD INITIAL CONDITIONS IN LAT/LON SPACE: 
## ---------------------------------------------------------
print('Loading initial conditions....')
priorpath = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
#priorname = 'Prior_state_tas_tos_psl_zg500_sit_sic_latcut0_1_CESM_LME_Amon_time_1650.pkl.npz'
priorname = 'Prior_state_tas_tos_psl_zg500_latcut0_1_sit_sic_latcut40_CESM_LME_Amon_time_1650_1850.pkl.npz'
#priorcov_name = 'Prior_covariance_tas_tos_psl_zg500_sit_sic_latcut0_1_CESM_LME_Amon_time_1650_1850.pkl.npz'
priorcov_name = 'Prior_covariance_tas_tos_psl_zg500_latcut0_1_sit_sic_latcut40_CESM_LME_Amon_time_1650_1850.pkl.npz'
#priorcov_name = 'Prior_covariance_tas_tos_psl_zg500_latcut0_1_sit_sic_latcut40_CESM_LME_Amon_time_1850_01.pkl.npz'

Xb_allvars = np.load(priorpath+priorname)
#Select January of 1850 for initial prior:
Xb_allvars_initial = Xb_allvars['Xb_allvars'][:,-12]
Xb_initial = Xb_allvars_initial

Pb_data = np.load(priorpath+priorcov_name)
Pb_initial = Pb_data['Pb_initial']#/10

## ---------------------------------------------------------
## BUILD H: 
## ---------------------------------------------------------
print('Building H....')
obsmask_filename = 'Hadcrut_data_mask_CESMLME_grid.pkl'
obsmask_loc_filename = 'Hadcrut_data_mask_with_locations_CESMLME_grid.pkl'
obsmask_loc_filename_nh = 'Hadcrut_data_mask_with_locations_CESMLME_grid_NH.pkl'
obscesm_filename_nh = 'Hadcrut_data_on_CESMLME_grid_NH.pkl'

had_ds = pickle.load(open(obsmask_loc_filename_nh,"rb"))
had_mask_nh = had_ds['hadcrut_closest_CESM_gridpoint']
locations = had_ds['lat_lon_locations']

## ---------------------------------------------------------
## Initialize loop over time: 
## ---------------------------------------------------------
obs_assimilated = {}
obs = {}
Xa_alltime = np.zeros((t_total,Xb_initial.shape[0]))
mse_xb = np.zeros((t_total))
mse_xa = np.zeros((t_total))
ratios = {}

for t in np.arange(0,t_total,1):
    print('----------------------------')
    print('Time = '+str(t))
    print('----------------------------')
    
    ## Observations to assimilate: 
    HadCRUT_obs_1d = HadCRUT_tas_nh[t,np.isfinite(HadCRUT_tas_nh[t,:,:])]
    HadCRUT_obs_lat = HadCRUT_lat_2d[np.isfinite(HadCRUT_tas_nh[t,:,:])]
    HadCRUT_obs_lon = HadCRUT_lon_2d[np.isfinite(HadCRUT_tas_nh[t,:,:])]
    
    HadCRUT_obs_unc_1d = HadCRUT_unc_tas_nh[t,np.isfinite(HadCRUT_unc_tas_nh[t,:,:])]
    R_had = oda.covariance_nans(HadCRUT_obs_unc_1d[:,np.newaxis],
                                HadCRUT_obs_unc_1d[:,np.newaxis].T,anomalize=False)
    
    print('Nobs assimilated = '+str(HadCRUT_obs_1d.shape[0]))
          
    obs['obs'] = HadCRUT_obs_1d 
    obs['obs_lat'] = HadCRUT_obs_lat
    obs['obs_lon'] = HadCRUT_obs_lon
    obs_assimilated[t] = obs

    ## BUILD H: 
    H_cap, nobs, ndof = oda.build_H_time(had_mask_nh[t,:,:])
    H = np.zeros((nobs,Xb_initial.shape[0]))
    H[:,0:6912] = H_cap

    ## ---------------------------------------------------------
    ## INITIAL UPDATE STEP: KALMAN FILTER 
    ## ---------------------------------------------------------
    print('Solving update equation....')
    print('Updating mean state.')
    Xb_initial = np.nan_to_num(Xb_initial)
    Y = HadCRUT_obs_1d

    Xa, K, K_den, diff = oda.solver_KF_update(Xb_initial, Pb_initial, HadCRUT_obs_1d, R_had, H)
    
    Xa_alltime[t,:] = Xa
    mse_xb[t] = np.nanmean(diff**2)
    mse_xa[t] = np.nanmean((Y - np.matmul(H,Xa))**2)
    print('Prior MSE     = '+str(mse_xb[t]))
    print('Posterior MSE = '+str(mse_xa[t]))
    
    diffcov = np.dot(diff[:,np.newaxis],diff[:,np.newaxis].T)/diff.shape[0]
    ratio = (np.diagonal(diffcov)/np.diagonal(K_den))
    ratios[t] = ratio
    print('Median Ratio = '+str(np.median(ratio)))
    
    ndof = Pb_initial.shape[0]

    print('Updating covariance.')
    # Solve the covariance update: 
    # part = np.identity(ndof) - np.matmul(K,H)
    # Pa = np.matmul(part,Pb_initial)
    
    Pa = oda.solver_KF_cov(K,H,Pb_initial)
    
    timestep = {}
    timestep['HK_ratio'] = ratio
    timestep['Xa'] = Xa
#     timestep['Pa'] = Pa
    
    savename = ('Time_'+str(t)+'.pkl')
    saveloc = '/home/disk/kalman2/mkb22/Online_DA/experiments/ODA_hadcrut_CESM_LME_ogprior/'
    print('Saving timestep here: '+saveloc+savename)
    pickle.dump(timestep, open(saveloc+savename, "wb" ) )

    ## ---------------------------------------------------------
    ## PROJECT ANALYSIS INTO EOF SPACE: 
    ## ---------------------------------------------------------
    print('Projecting analysis into EOF space...')
    Ptrunc_initial = {}
    
    for var in limvars: 
        Ptrunc_initial[var] = oda.project_validation_var(Xa[LIMd['var_dict'][var]['var_inds']][:,np.newaxis], 
                                                         LIMd['E3'][var],LIMd['standard_factor'][var],
                                                         LIMd['W_all'][var], Weights=LIMd['exp_setup']['Weight'])

    [ndof_all_initial, neof_all_initial] = oda.count_ndof_all(limvars, LIMd['E3'], sic_separate=False)

    [Ptrunc_all_initial, 
     E3_all_initial] = oda.stack_variable_eofs(limvars, ndof_all_initial, LIMd['exp_setup']['ntrunc'],
                                               Ptrunc_initial,LIMd['E3'],LIMd['var_dict'],verbose=False)

#     X_intial = np.concatenate((Ptrunc_all_initial, Ptrunc_sic_intial),axis=0)
    Xa_trunc = Ptrunc_all_initial
    
    Pa_trunc = np.matmul(np.matmul(E3_all_initial.T,Pa),E3_all_initial)

    ## ---------------------------------------------------------
    ## MAKE FORECAST: 
    ## ---------------------------------------------------------
    print('Performing forecast...')
    LIM_Xfcast = oda.LIM_forecast(LIMd,Xa_trunc,Pa_trunc,1,adjust=False)

    ## ---------------------------------------------------------
    ## PROJECT FORECAST INTO FULL FIELD SPACE: 
    ## ---------------------------------------------------------
    print('Projecting forecast into full field space...')
    nmodes = LIMd['E3_all'].shape[1]

    Xb_initial, E3_all = oda.decompress_eof_separate_sic(LIM_Xfcast['x_forecast'],LIMd)
    
    Pb_initial = np.matmul(np.matmul(E3_all,LIM_Xfcast['cov_forecast']),E3_all.T)
    
    
experiment = {}
experiment['obs_assimilated'] = obs_assimilated
experiment['Xa'] = Xa_alltime
experiment['mse_xb'] = mse_xb
experiment['mse_xa'] = mse_xa
experiment['ratios'] = ratios
experiment['time'] = time
    
savename = ('ODA_hadcrut_CESM_LME_t_0_'+str(t_total)+'cov_forecasted_sqrtWt.pkl')
saveloc = '/home/disk/kalman2/mkb22/Online_DA/experiments/'

print('Saving output here: '+saveloc+savename)
pickle.dump(experiment, open(saveloc+savename, "wb" ) )

