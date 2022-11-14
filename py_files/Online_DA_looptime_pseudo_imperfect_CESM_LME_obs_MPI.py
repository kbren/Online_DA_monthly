import sys
import numpy as np
import xarray as xr
import datetime
import scipy.stats as spy

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
years = int(1850+np.floor((t_total-1)/12))
time = np.array([datetime.datetime(y, m, 15) for y in np.arange(1850,1850+nyears,1) for m in np.arange(1,13,1)])

limvars = ['tas','psl','zg','tos','sit','sic']
neofs = 300

limname = 'cesm_lme_Amon'
modname = 'cesm_lme_Amon'
obsname = 'cmip6_mpi_hist_regridlme_Amon'

## ---------------------------------------------------------
## LOAD L: 
## ---------------------------------------------------------
print('Loading L....')
LIM = oda.load_L(limname)
LIMd = LIM['LIMd']
LIMd.keys()

Projector_tas = LIMd['E3']['tas']

## ---------------------------------------------------------
## LOAD Pseudo Obs: 
## ---------------------------------------------------------
obsdir = '/home/disk/p/mkb22/Documents/si_analysis_kb/Online_DA_monthly/observations/'
obsfilename = 'TAS_pseudo_obs_'+obsname+'_1851_2005.pkl'

pseudo_obs_data = pickle.load(open(obsdir+obsfilename,"rb"))

pseudo_obs_full = pseudo_obs_data['observations']
obs_lat = pseudo_obs_data['obs_lat']
obs_lon = pseudo_obs_data['obs_lon']
obs_mask = pseudo_obs_data['H']

pseudo_obs_2d = np.reshape(pseudo_obs_full, (pseudo_obs_full.shape[0]*pseudo_obs_full.shape[1],
                           pseudo_obs_full.shape[2]))

tas_3d = pseudo_obs_data['X_var_3d']
tas_2d = np.reshape(tas_3d,(tas_3d.shape[0]*tas_3d.shape[1],tas_3d.shape[2]))

## ---------------------------------------------------------
## LOAD INITIAL CONDITIONS IN EOF SPACE: 
## ---------------------------------------------------------
priordir = '/home/disk/p/mkb22/Documents/si_analysis_kb/Online_DA_monthly/priors/'
priorname = 'Xb_initial_'+modname+'_300ndof_1651_1850.pkl'
#validname = 'Xb_initial_'+modname+'_300ndof_1851_2005.pkl'
trainname = 'Xb_initial_'+modname+'_300ndof_850_1650.pkl'

Xb_initial_data = pickle.load(open(priordir+priorname,"rb"))
Xb_initial_allt = Xb_initial_data['Xb_initial']

Xb_train_data = pickle.load(open(priordir+trainname,"rb"))
Xb_train_allt = Xb_train_data['Xb_initial']

ndof_total = 0
limvars = ['tas','psl','zg','tos','sit','sic']
for var in limvars: 
    ndof_total = ndof_total+LIMd['var_dict'][var]['var_ndof']
    
Pb_initial = LIMd['C_0']

inflator = np.ones_like(Pb_initial)
inflator[250:,250:] = inflator[250:,250:]*2

## ---------------------------------------------------------
## BUILD H: 
## ---------------------------------------------------------
H_cap, nobs, ndof = oda.build_H_time(obs_mask[:,:,t])

H = np.zeros((nobs,ndof_total))
H[:,0:6912] = H_cap

E3_all = np.zeros((ndof_total,neofs))
ntrunc = 50

for v,var in enumerate(limvars): 
    E3_all[LIMd['var_dict'][var]['var_inds'],
           int(v*ntrunc):int((v+1)*ntrunc)] = LIMd['E3'][var]/np.sqrt(LIMd['standard_factor'][var])
    
U = E3_all
H_eof = np.matmul(H,U)

## ---------------------------------------------------------
## BUILD R: 
## ---------------------------------------------------------
Hx = np.matmul(H_eof,Xb_train_allt[:,-tas_2d.shape[1]:])
Hx_cap = np.matmul(H_cap,tas_2d)

epsilon = Hx_cap - Hx

slope = np.zeros(Hx.shape[0])
for i in range(Hx.shape[0]):
    slope[i],_,_,_,_ = spy.linregress(Hx[i,:],epsilon[i,:])
e = epsilon - slope[:,np.newaxis]*Hx

R = np.matmul(epsilon,epsilon.T)/(epsilon.shape[1]-1)
R_e = np.matmul(e,e.T)/(e.shape[1]-1)

## ---------------------------------------------------------
## Initialize loop over time: 
## ---------------------------------------------------------
obs_assimilated = {}
obs = {}
Xa_alltime = np.zeros((t_total,neofs))
diff_alltime = np.zeros((t_total,pseudo_obs_2d.shape[0]))
K_den_alltime = np.zeros((t_total,pseudo_obs_2d.shape[0],pseudo_obs_2d.shape[0]))
mse_xb = np.zeros((t_total))
mse_xa = np.zeros((t_total))
ratios = {}

Xb = np.nan_to_num(Xb_train_allt[:,0])
Pb = Pb_initial
R_hack = R_e
H_final = H_eof

for t in np.arange(0,t_total,1):
    print('----------------------------')
    print('Time = '+str(t))
    print('----------------------------')
    
    Y = pseudo_obs_2d[:,t]
     
    obs['obs'] = Y
    obs['obs_lat'] = obs_lat
    obs['obs_lon'] = obs_lon
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
    diff_alltime[t,:] = diff 
    K_den_alltime[t,:,:] = K_den
    print('Prior MSE     = '+str(mse_xb[t]))
    print('Posterior MSE = '+str(mse_xa[t]))

    print('Median Ratio = '+str(np.trace(R_hack)))
    print('Mean Ratio = '+str(np.trace(Pb)))
    
    ndof = Pb.shape[0]

    print('Updating covariance.')
    
    Pa = oda.solver_KF_cov(K,H_final,Pb)
    
#     timestep = {}
#     timestep['HK_ratio'] = ratio
#     timestep['Xa'] = Xa
#     timestep['Pa'] = Pa
    
#     savename = ('Time_'+str(t)+'.pkl')
#     saveloc = '/home/disk/kalman2/mkb22/Online_DA/experiments/ODA_hadcrut_CESM_LME_ogprior/'
#     print('Saving timestep here: '+saveloc+savename)
#     pickle.dump(timestep, open(saveloc+savename, "wb" ) )

    ## ---------------------------------------------------------
    ## MAKE FORECAST: 
    ## ---------------------------------------------------------
    print('Performing forecast...')
    LIM_Xfcast = oda.LIM_forecast(LIMd,Xa,Pa,1,adjust=False)
    
    Xb = LIM_Xfcast['x_forecast']
    Pb = LIM_Xfcast['cov_forecast']*inflator
    
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
    
savename = ('ODA_pseudo_imperfect_'+limname+'_obs_'+obsname+'_t_0_'+str(t_total)+'_nomj_trainRe_infsic2.pkl')
saveloc = '/home/disk/kalman2/mkb22/Online_DA/experiments/pseudo/'

print('Saving output here: '+saveloc+savename)
pickle.dump(experiment, open(saveloc+savename, "wb" ) )

