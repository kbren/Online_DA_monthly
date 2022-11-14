import sys
import numpy as np
import xarray as xr
import datetime
import scipy.stats as spy
import random

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pickle 

sys.path.append("../")
import Online_DA_utils as oda

t =0
# Select number of years to reconstruct: 
nyears = 155

# Generate times: 
t_total = nyears*12
years = int(1851+np.floor((t_total-1)/12))
time = np.array([datetime.datetime(y, m, 15) for y in np.arange(1851,1851+nyears,1) for m in np.arange(1,13,1)])

limvars = ['tas','psl','zg','tos','sit','sic']
neofs = 300
Nobs = 50
it_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

inf = 2
inf_fact = '2_siconly'

limname = 'multimod_MPI_GFDL_HadGEM3_CanESM_Amon'
modname = 'cesm_lme_Amon'
tname = 'multimod_MPI_GFDL_HadGEM3_CanESM_hist_Amon'
obsname = 'hadCRUT_all'
#obsname = 'cesm_lme_Amon'
# obsname = 'cesm_lme_Amon'

## ---------------------------------------------------------
## LOAD L: 
## ---------------------------------------------------------
print('Loading L....')
LIM = oda.load_L(limname)
LIMd = LIM['LIMd']
#LIMd.keys()

Projector_tas = LIMd['E3']['tas']

## ---------------------------------------------------------
## LOAD HadCRUT5 Obs: 
## ---------------------------------------------------------
obsdir = '/home/disk/p/mkb22/Documents/si_analysis_kb/Online_DA_monthly/observations/'
#obsfilename = 'TAS_pseudo_obs_'+obsname+'_1851_2005.pkl'
obsfilename = 'Hadcrut_data_on_CESMLME_grid_NH.pkl'
#obsfilename = 'Hadcrut_data_on_CESMLME_grid_NH_fixed_network36.pkl'
#obsfilename = 'TAS_pseudo_obs_'+obsname+'_hadCRUT_locations_1851_2005.pkl'

had_obs_data = pickle.load(open(obsdir+obsfilename,"rb"))

had_obs_data_full = had_obs_data['had_on_refgrid'][12:1872,:,:]
# had_obs_data_og = had_obs_data['had_on_refgrid'][12:1872,:,:]
# had_obs_data_og = had_obs_data['observations']

#Select obs mask from given timestep in Hadcrut to use throughout:
# had_155_mask = np.where(np.isfinite(had_obs_data_og[0,:,:]),1,0)
# had_obs_data_full = had_155_mask[np.newaxis,:,:]*np.nan_to_num(had_obs_data_og[:,:,:])
# had_obs_data_full[np.logical_not(np.abs(had_obs_data_full)>0)]=np.nan

# had_obs_full_2d = np.reshape(had_obs_data_full, (had_obs_data_full.shape[0]*had_obs_data_full.shape[1],
#                                                 had_obs_data_full.shape[2]))
had_obs_full_2d = np.reshape(had_obs_data_full, (had_obs_data_full.shape[0],
                                                 had_obs_data_full.shape[1]*had_obs_data_full.shape[2])).T

# Project HadCRUT to eof space:
eofs_out = LIMd['E3']['tas']/LIMd['standard_factor']['tas']
P_var = np.matmul(eofs_out.T, (LIMd['W_all']['tas'][:,np.newaxis]*np.nan_to_num(had_obs_full_2d)))
Ptrunc = P_var/LIMd['standard_factor']['tas']

## ---------------------------------------------------------
## LOAD INITIAL CONDITIONS IN EOF SPACE: 
## ---------------------------------------------------------
priordir = '/home/disk/p/mkb22/Documents/si_analysis_kb/Online_DA_monthly/priors/'
priorname = 'Xb_initial_'+modname+'_300ndof_1651_1850.pkl'
trainname = 'Xb_initial_'+tname+'_300ndof_1850_2473.pkl'

Xb_initial_data = pickle.load(open(priordir+priorname,"rb"))
Xb_initial_allt = Xb_initial_data['Xb_initial']

Xb_train_data = pickle.load(open(priordir+trainname,"rb"))
Xb_train_allt = Xb_train_data['Xb_initial']

ndof_total = 0
limvars = ['tas','psl','zg','tos','sit','sic']
for var in limvars: 
    ndof_total = ndof_total+LIMd['var_dict'][var]['var_ndof']
    
Pb_initial = LIMd['C_0']

## ---------------------------------------------------------
## BUILD H: 
## ---------------------------------------------------------

# had_mask = np.zeros_like(had_obs_data_full)
# had_mask[np.isfinite(had_obs_data_full)] = had_mask[np.isfinite(had_obs_data_full)]+1

for it in it_list:
    print('-------------------------------------')
    print('ITERATION '+str(it))
    print('-------------------------------------')
    
    ## ---------------------------------------------------------
    ## Randomly sample X number of obs at each timestep: 
    had_mask_rand = np.zeros_like((had_obs_full_2d.T))

    for t in range(1860):
        true_inds = np.where(np.isfinite(np.reshape(had_obs_data_full[t,:,:],
                                                    (had_obs_data_full.shape[1]*had_obs_data_full.shape[2]))))

        if true_inds[0].shape[0]>=Nobs:
            true_rand = random.sample(list(true_inds[0]),Nobs)
        else: 
            true_rand = true_inds[0]

        had_mask_rand[t,true_rand] = had_mask_rand[t,true_rand]+1

    had_mask = np.reshape(had_mask_rand,(had_mask_rand.shape[0],had_obs_data_full.shape[1],
                                         had_obs_data_full.shape[2]))
    ## ---------------------------------------------------------


    E3_all = np.zeros((ndof_total,neofs))
    ntrunc = 50

    for v,var in enumerate(limvars): 
        E3_all[LIMd['var_dict'][var]['var_inds'],
               int(v*ntrunc):int((v+1)*ntrunc)] = LIMd['E3'][var]/np.sqrt(LIMd['standard_factor'][var])

    ## ---------------------------------------------------------
    ## Initialize loop over time: 
    ## ---------------------------------------------------------
    obs_assimilated = {}
    Xa_alltime = np.zeros((t_total,neofs))
    diff_alltime = {}
    K_den_alltime = {}
    mse_xb = np.zeros((t_total))
    mse_xa = np.zeros((t_total))
    Pb_trace = np.zeros((t_total))
    ratios = {}

    Xb = np.nan_to_num(Xb_initial_allt[:,-1])
    Pb = Pb_initial

    for t in np.arange(0,t_total,1):
        print('----------------------------')
        print('Time = '+str(t))
        print('----------------------------')

        ## ---------------------------------------------------------
        ## BUILD H: 
        ## ---------------------------------------------------------
        H_cap, nobs, ndof = oda.build_H_time(had_mask[t,:,:])

        H = np.zeros((nobs,ndof_total))
        H[:,0:6912] = H_cap

        U = E3_all
        H_eof = np.matmul(H,U)

        ## ---------------------------------------------------------
        ## BUILD R: 
        ## ---------------------------------------------------------
        Hx = np.matmul(H_eof,Xb_train_allt[:,-had_obs_full_2d.shape[1]:])
        Hx_cap = np.matmul(H_cap,np.nan_to_num(had_obs_full_2d))

        epsilon = Hx_cap - Hx

        slope = np.zeros(Hx.shape[0])
        for i in range(Hx.shape[0]):
            slope[i],_,_,_,_ = spy.linregress(Hx[i,:],epsilon[i,:])
        e = epsilon - slope[:,np.newaxis]*Hx

        R = np.matmul(epsilon,epsilon.T)/(epsilon.shape[1]-1)
        R_e = np.matmul(e,e.T)/(e.shape[1]-1)

        ## ---------------------------------------------------------

        R_hack = R_e
        H_final = H_eof

        Y = np.matmul(H_cap,np.nan_to_num(had_obs_full_2d[:,t]))

        obs = {}
        obs['obs'] = Y
    #     obs['obs_lat'] = obs_lat
    #     obs['obs_lon'] = obs_lon
        obs_assimilated[t] = obs

        ## ---------------------------------------------------------
        ## INITIAL UPDATE STEP: KALMAN FILTER 
        ## ---------------------------------------------------------
        print('Solving update equation....')
        print('Updating mean state.')
        print('Nobs assimilated = '+str(Y.shape[0])) 

        Xa, K, K_den, diff = oda.solver_KF_update(Xb, Pb, Y, R_hack, H_final)

        Xa_alltime[t,:] = Xa
        mse_xb[t] = np.nanmean(diff**2)
        mse_xa[t] = np.nanmean((Y - np.matmul(H_final,Xa))**2)
        diff_alltime[t] = diff 
        K_den_alltime[t] = K_den
        print('Prior MSE     = '+str(mse_xb[t]))
        print('Posterior MSE = '+str(mse_xa[t]))

        ndof = Pb.shape[0]

        print('Updating covariance.')

        Pa = oda.solver_KF_cov(K,H_final,Pb)
        
        inflation = np.ones_like(Pa)
        inflation[-50:,-50:] = inflation[-50:,-50:]*inf
        Pa = Pa*inflation

        ## ---------------------------------------------------------
        ## MAKE FORECAST: 
        ## ---------------------------------------------------------
        print('Performing forecast...')
        LIM_Xfcast = oda.LIM_forecast(LIMd,Xa,Pa,1,adjust=False)

        Xb = LIM_Xfcast['x_forecast']
        Pb = LIM_Xfcast['cov_forecast']

        Pb_trace[t] = np.trace(LIM_Xfcast['cov_forecast'])

        print('R = '+str(R_hack.max()))
        print('HPbHT = '+str(np.max(np.matmul(np.matmul(H_final,Pb),H_final.T))))

    experiment = {}
    experiment['obs_assimilated'] = obs_assimilated
    experiment['Xa'] = Xa_alltime
    experiment['diff'] = diff_alltime
    experiment['K_den'] = K_den_alltime
    experiment['mse_xb'] = mse_xb
    experiment['mse_xa'] = mse_xa
    experiment['ratios'] = ratios
    experiment['time'] = time
    experiment['Pb_trace'] = Pb_trace

    savename = ('ODA_prior_'+modname+'_LIM_'+limname+'_obs_'+obsname+'_hadCRUT_locations_rand'+str(Nobs)+
                '_inf'+inf_fact+'_Pa_t_0_'+str(t_total)+'_nomj_trainRe_it'+str(it)+'.pkl')
    saveloc = '/home/disk/kalman2/mkb22/Online_DA/experiments/hadcrut/nobs50/'

    print('Saving output here: '+saveloc+savename)
    pickle.dump(experiment, open(saveloc+savename, "wb" ) )

