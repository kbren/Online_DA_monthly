import sys,os,copy
import numpy as np
import pickle
import xarray as xr

from time import time
from spharm import Spharmt, getspecindx, regrid
from netCDF4 import Dataset
from scipy import stats

sys.path.insert(2,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_lite_utils as LMRlite
import LMR_config
import LMR_utils as lmr

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils

import persistence_utils as pu
import LIM_forecaster

#USER PARAMETERS: -----------------------------------
# reconstruct sea ice using these instrumental temperature datasets
dset_chosen = ['CRU']
ref_dset = 'CRU'

annual_mean = False

# define an even (lat,lon) grid for ob locations:
dlat = 10.
dlon = dlat

# set the ob error variance (uniform for now)
r = 0.1
rstr = '0_1'

iter = 1

# inflate the sea ice variable here (can inflate whole state here too)
inflate = 2.6
inf_name = '2_6'
prior_name = 'ccsm4'

cfile = './configs/config_lite.yml.monthly'

loc = 15000

# set variable that will be sampled for pseudoproxies
truth_var = 'tas_sfc_Amon'

recon_start = 1681
recon_end = 1750
#------------------------------------------------------

# check for user-specified config file; otherwise, use the one in the SRC directory

yaml_file = os.path.join(LMR_config.SRC_DIR,cfile)

cfg,cfg_dict = LMRlite.load_config(yaml_file)

# fetch information from datasets.yml and attach to the cfg
dataset_info = LMR_config._DataInfo.data
cfg.info = dataset_info

# Create name of output file
month = cfg.core.recon_months[0]
savedir = '/home/disk/p/mkb22/Documents/si_analysis_kb/persistence_prior/data/'
savename = ('sic_tas_pseudo_hadobs_P3_'+prior_name+'_monthly_'+str(recon_start)+'_'+str(recon_end)+
            '_r'+rstr+'_inf'+inf_name+'_loc'+str(loc)+'_iter'+str(iter)+'.pkl')

#------------------------------------------------------
# load prior
X, Xb_one = LMRlite.load_prior(cfg)
Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)

# Save original lat/lon values for later: 
prior_lat_full = X.prior_dict['tas_sfc_Amon']['lat']
prior_lon_full = X.prior_dict['tas_sfc_Amon']['lon']

prior_sic_lat_full = X.prior_dict['sic_sfc_OImon']['lat']
prior_sic_lon_full = X.prior_dict['sic_sfc_OImon']['lon']

# check if config is set to regrid the prior
if cfg.prior.regrid_method:
    print('regridding prior...')
    # this function over-writes X, even if return is given a different name
    [X_regrid,Xb_one_new] = LMRlite.prior_regrid(cfg,X,Xb_one,verbose=True)
else:
    X_regrid.trunc_state_info = X.full_state_info
    
    
# make a grid object for the prior
grid = LMRlite.Grid(X_regrid)

# locate position of variables in Xb_one_new: 
tas_pos = X_regrid.trunc_state_info['tas_sfc_Amon']['pos']
sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']

# Load full monthly prior to draw observations from --------------
truth_data = {}
prior_dir = '/home/disk/kalman3/rtardif/LMR/data/model/ccsm4_last_millenium/'
suffix = 'CCSM4_past1000_085001-185012.nc'
varnames = ['tas_sfc_Amon', 'sic_sfc_OImon']
filenames = [os.path.join(prior_dir, f'{var}_{suffix}') for var in varnames]
for var in varnames: 
    filename = (os.path.join(prior_dir, f'{var}_{suffix}'))
    truth_data[var] = xr.open_dataset(filename)
    
truth_tas = truth_data['tas_sfc_Amon']['tas'].values
truth_sic = truth_data['sic_sfc_OImon']['sic'].values

# Prep full monthly prior for regridding 
truth_tas_lat = truth_data['tas_sfc_Amon']['lat'].values
truth_tas_lon = truth_data['tas_sfc_Amon']['lon'].values
truth_sic_lat = truth_data['sic_sfc_OImon']['lat'].values
truth_sic_lon = truth_data['sic_sfc_OImon']['lon'].values
t = np.ones((truth_tas_lat.shape[0],truth_tas_lon.shape[0]))
s = np.ones((truth_sic_lat.shape[0],truth_sic_lon.shape[0]))

truth_tas_lat = truth_tas_lat[:,np.newaxis]*t
truth_tas_lon = t*truth_tas_lon[np.newaxis,:]
truth_sic_lat = truth_sic_lat[:,np.newaxis]*s
truth_sic_lon = s*truth_sic_lon[np.newaxis,:]

tas_mo = np.reshape(truth_tas,(1001,12,truth_tas.shape[1],truth_tas.shape[2]))
truth_tas_anom = tas_mo - np.nanmean(tas_mo,axis=0)

tas_shape = truth_tas_anom.shape
sic_shape = truth_sic.shape
truth_tas_anom_regrid_prep = truth_tas_anom.reshape(tas_shape[0]*tas_shape[1],
                                                    tas_shape[2]*tas_shape[3]).T
truth_sic_anom_regrid_prep = truth_sic.reshape(sic_shape[0],sic_shape[1]*sic_shape[2]).T

nyears = 1001*12

# Regrid full tas prior for proxy selection and verification: 
[truth_tas_regrid,lat_tas_new,lon_tas_new] = lmr.regrid_esmpy(cfg.prior.esmpy_grid_def['nlat'],
                                                        cfg.prior.esmpy_grid_def['nlon'],
                                                        nyears,
                                                        truth_tas_anom_regrid_prep,
                                                        truth_tas_lat,
                                                        truth_tas_lon,
                                                        truth_tas_lat.shape[0],
                                                        truth_tas_lat.shape[1],
                                                        method=cfg.prior.esmpy_interp_method)

truth_tas_regrid = np.reshape(truth_tas_regrid.T,
                              (nyears,cfg.prior.esmpy_grid_def['nlat'],
                               cfg.prior.esmpy_grid_def['nlon']))

# Regrid full tas prior for proxy selection and verification: 
[truth_sic_regrid,lat_sic_new,lon_sic_new] = lmr.regrid_esmpy(cfg.prior.esmpy_grid_def['nlat'],
                                                        cfg.prior.esmpy_grid_def['nlon'],
                                                        nyears,
                                                        truth_sic_anom_regrid_prep,
                                                        truth_sic_lat,
                                                        truth_sic_lon,
                                                        truth_sic_lat.shape[0],
                                                        truth_sic_lat.shape[1],
                                                        method=cfg.prior.esmpy_interp_method)

truth_sic_regrid = np.reshape(truth_sic_regrid.T,
                              (nyears,cfg.prior.esmpy_grid_def['nlat'],
                               cfg.prior.esmpy_grid_def['nlon']))
print(truth_sic_regrid.shape)
#------------------------------------------------------
# NH surface area in M km^2 from concentration in percentage
nharea = 2*np.pi*(6380**2)/1e8

## Set up reconstruction time: 
# Loop over all years available in this reference dataset
recon_years = range(recon_start,recon_end)
nyears = len(recon_years)

# Find recon time indices in prior 
prior_time = []
for i in range(850,1851):
    b = np.tile(i,12)
    prior_time.append(b)   
prior_time = np.reshape(np.array(prior_time),(1001*12))
recon_ind_start = np.min(np.where(prior_time>=recon_start))
recon_ind_end = np.max(np.where(prior_time<=recon_end))

cfg.core.recon_period = (recon_start,recon_end)

# Prior anomalies to draw pseudo proxies from: 
prior_tas_regrid_anom = truth_tas_regrid[recon_ind_start:recon_ind_end,:,:]
sic_prior_regrid_truth = truth_sic_regrid[recon_ind_start:recon_ind_end,:,:]

# Prep for pseudo obs draw ------------------------------------
# make a grid object for the reference data
grid_pseudo = LMRlite.Grid()
pseudolon,pseudolat = np.meshgrid(grid.lon[0,:],grid.lat[:,0])
grid_pseudo.lat = pseudolat
grid_pseudo.lon = pseudolon
grid_pseudo.nlat = grid_pseudo.lat.shape[0]
grid_pseudo.nlon = grid_pseudo.lon.shape[1]

#Load finite HadCRUT lat/lon locations to draw pseudo obs from: 
# if annual_mean: 
#     hadcrut_finite = pickle.load( open( "Had_obs_loc_files/hadcrut_annual_finite_latlon.pkl", "rb" ) )

#     hadcrut_ob_lat = hadcrut_finite['hadcrut_ob_lat']
#     hadcrut_ob_lon = hadcrut_finite['hadcrut_ob_lon']
# else: 
#     month = cfg.core.recon_months[0]
#     hadcrut_finite = pickle.load( open(("Had_obs_loc_files/hadcrut_mo_"+
#                                         str(month)+"_finite_latlon.pkl"), "rb" ) )

#     hadcrut_ob_lat = hadcrut_finite['hadcrut_ob_lat']
#     hadcrut_ob_lon = hadcrut_finite['hadcrut_ob_lon']
#     print('month loaded =',month)
    
# inflate the entire state vector------------------------------------
if 2 == 1:
    print('inflating full state vector by '+str(inflate))
    xbm = np.mean(Xb_one_new,1)
    xbp = Xb_one_new - xbm[:,None]
    Xb_one_inflate = np.copy(Xb_one_new)
    Xb_one_inflate = np.add(inflate*xbp,xbm[:,None])
else:
    # inflate sea ice only
    print('inflating only sea ice by '+str(inflate))
    xb_sicm = np.mean(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:],1)
    xb_sicp = np.squeeze(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:])-xb_sicm[:,None]
    sic_new = np.add(inflate*xb_sicp,xb_sicm[:,None])
    Xb_one_inflate = np.copy(Xb_one_new)
    Xb_one_inflate[sic_pos[0]:sic_pos[1]+1,:] = sic_new
    
Xb_one_original = Xb_one_inflate
Xbm_one_original = np.nanmean(Xb_one_inflate,axis=1)
Xbp_one_original = Xb_one_original - Xbm_one_original[:,np.newaxis]

# set localization length scale ------------------------------------
# option to change the localization radius in the config file (used by DA code)
print('Original localization radius = '+str(cfg.core.loc_rad))
cfg_params = lmr.param_cfg_update('core.loc_rad',loc)
cfg_new = LMR_config.Config(**cfg_params)
print('New localization radius = '+str(cfg_new.core.loc_rad))


# DA ---------------------------------------------------------------
# These are the time indices for the reference dataset; useful later
iyr_ref = np.zeros(nyears,dtype=np.int16)

sic_save = []
sic_save_lalo = np.zeros((nyears*12,grid.nlat,grid.nlon))
tas_save_lalo = np.zeros((nyears*12,grid.nlat,grid.nlon))
sic_save_2_5 = np.zeros((nyears*12,grid.nlat,grid.nlon))
sic_save_97_5 = np.zeros((nyears*12,grid.nlat,grid.nlon))
sic_lalo_full = []
sic_full_Nens = []
sie_full_Nens = []
sic_full_Sens = []
sie_full_Sens = []
obs_size = np.zeros((nyears*12))
obs_full = {}

sic_ndim = sic_pos[1]-sic_pos[0]+1

# Option to save the full ensemble
#Xap_save = np.zeros([nyears,sic_ndim,cfg.core.nens])
#Xap_var_save = np.zeros([nyears,sic_ndim])

obs_shape = np.zeros((nyears))

begin_time = time()
yk = -1

for y, target_year in enumerate(recon_years):
    print('Reconstructing year '+str(target_year))
    for im, mo in enumerate(range(1,13)):
        yk = yk+1
        print('Reconstructing month '+str(mo))
        
        hadcrut_finite = pickle.load( open(("Had_obs_loc_files/hadcrut_mo_"+
                                            str(mo)+"_finite_latlon.pkl"), "rb" ) )

        hadcrut_ob_lat = hadcrut_finite['hadcrut_ob_lat']
        hadcrut_ob_lon = hadcrut_finite['hadcrut_ob_lon']
  #      print('month loaded =',mo)
        
        nobs = hadcrut_ob_lat[y].shape[0]
        obs_size[yk] = nobs
        print('Number of obs for this month is = '+str(nobs))
        
        sic_lalo_full_P = []
        sic_lalo_full = []

        vY = np.zeros((nobs))
        vYe = np.zeros([nobs,grid.nens])
        vYe_coords = []
        vR = []
        k = -1

        obs_loc = []

         # Break up prior by two variables 
        Xb_one_tas = Xb_one_inflate[tas_pos[0]:tas_pos[1]+1,:]
        Xb_one_sic = Xb_one_inflate[tas_pos[1]+1:Xb_one.shape[0],:]

        # this is used for drawing prior ensemble estimates of pseudoproxy (Ye)
        Xb_sampler = np.reshape(Xb_one_tas,[grid.nlat,grid.nlon,grid.nens])

        # Draw pseudo observations from instrumental Hadcrut lat/lon locations for this recon year
        for iob in range(nobs):
    #        print('assimilating ob '+str(i))
            [pseudoproxy, itlat, itlon] = pu.draw_pseudoproxy(grid_pseudo, hadcrut_ob_lat[y][iob],
                                                              hadcrut_ob_lon[y][iob], 
                                                              prior_tas_regrid_anom[yk,:,:], r)

            vY[iob] = pseudoproxy
            vYe[iob,:] = Xb_sampler[itlat,itlon,:]
            vR.append(r)
            obs_loc = np.append(obs_loc,[grid_pseudo.lat[itlat,0],grid_pseudo.lon[0,itlon]])

            # make vector of Ye coordinates for localization
            if iob is 0: 
                vYe_coords = np.array([grid_pseudo.lat[itlat,0],grid_pseudo.lon[0,itlon]])[np.newaxis,:]
            else: 
                new = np.array([grid_pseudo.lat[itlat,0],grid_pseudo.lon[0,itlon]])[np.newaxis,:]
                vYe_coords = np.append(vYe_coords,new, axis=0)

        obs_full[yk] = np.reshape(obs_loc,(nobs,2))

        # Do data assimilation
        #xam,Xap,_ = LMRlite.Kalman_optimal(obs_QC,R_QC,Ye_QC,Xb_one_inflate)
        xam,Xap = LMRlite.Kalman_ESRF(cfg_new,vY,vR,vYe,
                                      Xb_one_inflate,X=X_regrid,
                                      vYe_coords=vYe_coords,verbose=False)

        # Save full fields and total area for later. 
        tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1],[grid.nlat,grid.nlon])
        tas_save_lalo[yk,:,:] = tas_lalo

        # this saves sea-ice area for the entire ensemble
        sic_Nens = []
        sie_Nens = []
        sic_Sens = []
        sie_Sens = []
        for k in range(grid.nens):
            sic_lalo = np.reshape(xam[sic_pos[0]:sic_pos[1]+1]+Xap[sic_pos[0]:sic_pos[1]+1,k],
                                  [grid.nlat,grid.nlon])
            sic_lalo_P = (xam[sic_pos[0]:sic_pos[1]+1]+Xbp_one_original[sic_pos[0]:sic_pos[1]+1,k])
            if 'full' in cfg.prior.state_variables['sic_sfc_OImon']:
                sic_lalo = np.where(sic_lalo<0.0,0.0,sic_lalo)
                sic_lalo = np.where(sic_lalo>100.0,100.0,sic_lalo)
                sic_lalo_P = np.where(sic_lalo_P<0.0,0.0,sic_lalo_P)
                sic_lalo_P = np.where(sic_lalo_P>100.0,100.0,sic_lalo_P)
                # Calculate extent: 
                sie_lalo = siutils.calc_sea_ice_extent(sic_lalo,15.0)
            else: 
                sic_lalo = sic_lalo
                sie_lalo = np.nan*np.ones(sic_lalo.shape)

            _,nhmic,shmic = lmr.global_hemispheric_means(sic_lalo,grid.lat[:, 0])
            _,sie_nhmic,sie_shmic = lmr.global_hemispheric_means(sie_lalo,grid.lat[:, 0])
            sic_Nens.append(nhmic)
            sie_Nens.append(sie_nhmic)
            sic_Sens.append(shmic)
            sie_Sens.append(sie_shmic)
            sic_lalo_full.append(sic_lalo)
            sic_lalo_full_P.append(sic_lalo_P)
            
        #Method 3: 
        Xb_one_inflate = Xb_one_inflate*0.0
        Xb_one_inflate[tas_pos[0]:tas_pos[1]+1,:] = (xam[tas_pos[0]:tas_pos[1]+1][:,np.newaxis]+
                                                     Xbp_one_original[tas_pos[0]:tas_pos[1]+1,:])
        Xb_one_inflate[tas_pos[0]:tas_pos[1]+1,:] = Xbp_one_original[tas_pos[0]:tas_pos[1]+1,:]
        Xb_one_inflate[sic_pos[0]:sic_pos[1]+1,:] = np.array(sic_lalo_full_P).T

        sic_save_lalo[yk,:,:] = np.nanmean(np.array(sic_lalo_full),axis=0)
        sic_save_97_5[yk,:,:] = np.percentile(sic_lalo_full,97.5,axis=0)
        sic_save_2_5[yk,:,:] = np.percentile(sic_lalo_full,2.5,axis=0)
        sic_full_Nens.append(sic_Nens)
        sie_full_Nens.append(sie_Nens)
        sic_full_Sens.append(sic_Sens)
        sie_full_Sens.append(sie_Sens)
        
        limvars = ['tas','sic']
        forecast = LIM_forecaster.run_LIM_forecast(limvars, 50, 'cesm_lme', verbose=False)

    print('done reconstructing: ',target_year,'\n')

# ---------------------------------------------------------------

tas_dict = {}
sice_dict = {}
obs_dict = {}

tas_dict['tas_ensmn_lalo'] = tas_save_lalo
tas_dict['tas_prior_ens_inf'] = Xb_one_inflate[tas_pos[0]:tas_pos[1],:]
tas_dict['tas_truth_anom'] = prior_tas_regrid_anom
tas_dict['time'] = recon_years
tas_dict['lat'] = grid.lat
tas_dict['lon'] = grid.lon

sice_dict['sic_ensmn_lalo'] = sic_save_lalo
sice_dict['sic_97_5_lalo'] = sic_save_97_5
sice_dict['sic_2_5_lalo'] = sic_save_2_5
# sice_dict['sic_tot_nens'] = np.squeeze(np.array(sic_full_ens))
# sice_dict['sie_tot_nens'] = np.squeeze(np.array(sie_full_ens))
sice_dict['sic_Ntot_nens'] = np.squeeze(np.array(sic_full_Nens))
sice_dict['sie_Ntot_nens'] = np.squeeze(np.array(sie_full_Nens))
sice_dict['sic_Stot_nens'] = np.squeeze(np.array(sic_full_Sens))
sice_dict['sie_Stot_nens'] = np.squeeze(np.array(sie_full_Sens))
sice_dict['sic_prior_ens_inf'] = Xb_one_inflate[sic_pos[0]:sic_pos[1]+1,:]
sice_dict['sic_truth_full'] = sic_prior_regrid_truth

obs_dict['nobs'] = obs_size
obs_dict['obs_loc'] = obs_full

recon_dict = {}
recon_dict['tas'] = tas_dict
recon_dict['sice'] = sice_dict
recon_dict['obs'] = obs_dict

print('Saving to: ',savedir+savename)
pickle.dump(recon_dict,open(savedir+savename, "wb"))


