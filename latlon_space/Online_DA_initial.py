import sys
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pickle 

import Online_DA_utils as oda

t=0
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
priorname = 'Prior_state_tas_tos_psl_zg500_latcut0_1_sit_sic_latcut40_CESM_LME_Amon_time_1650.pkl.npz'
#priorcov_name = 'Prior_covariance_tas_tos_psl_zg500_sit_sic_latcut0_1_CESM_LME_Amon_time_1650_1850.pkl.npz'
priorcov_name = 'Prior_covariance_tas_tos_psl_zg500_latcut0_1_sit_sic_latcut40_CESM_LME_Amon_time_1650_1850.pkl.npz'

Xb_allvars = np.load(priorpath+priorname)
Xb_allvars_initial = Xb_allvars['Xb_allvars'][:,-1]
Xb_initial = Xb_allvars_initial

Pb_data = np.load(priorpath+priorcov_name)
Pb_initial = Pb_data['Pb_initial']

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

H_cap, nobs, ndof = oda.build_H_time(had_mask_nh[0,:,:])
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

mse_xb = np.nanmean(diff**2)
mse_xa = np.nanmean((Y - np.matmul(H,Xa))**2)
print('Prior MSE     = '+str(mse_xb))
print('Posterior MSE = '+str(mse_xa))

ndof = Pb_initial.shape[0]

print('Updating covariance.')
# Solve the covariance update: 
# part = np.identity(ndof) - np.matmul(K,H)

# Pa = np.matmul(part,Pb_initial)
Pa = oda.solver_KF_cov(K,H,Pb_initial)

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

[Ptrunc_all_initial, _] = oda.stack_variable_eofs(limvars, ndof_all_initial, LIMd['exp_setup']['ntrunc'],
                                                  Ptrunc_initial, LIMd['E3'], LIMd['var_dict'],verbose=False)

#     X_intial = np.concatenate((Ptrunc_all_initial, Ptrunc_sic_intial),axis=0)
X_initial = Ptrunc_all_initial

## ---------------------------------------------------------
## MAKE FORECAST: 
## ---------------------------------------------------------
print('Performing forecast...')
LIM_Xfcast = oda.LIM_forecast(LIMd,X_initial,1,adjust=LIMd['exp_setup']['adj'])

## ---------------------------------------------------------
## PROJECT FORECAST INTO FULL FIELD SPACE: 
## ---------------------------------------------------------
print('Projecting forecast into full field space...')
nmodes = LIMd['E3_all'].shape[1]

Xb_intitial, E3_all = oda.decompress_eof_separate_sic(LIM_Xfcast['x_forecast'],LIMd)

Pb_initial = np.matmul(np.matmul(E3_all,LIM_Xfcast['cov_forecast']),E3_all.T)
