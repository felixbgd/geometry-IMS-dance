# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:05:39 2023

@author: fbigand
"""

#%% 
##############################################################
############  IMPORT LIBRARIES AND SET PARAMETERS ############
##############################################################

# Private libraries (available with this code on the GitHub repo)
# PLmocap: my own library for mocap processing/visualization
from PLmocap.viz import *
from PLmocap.preprocessing import *
from PLmocap.classif import *
from PLmocap.stats import *
# MNE Python (Gramfort et al.) with minor bug fixed for cluster-based permutation
import mne_fefe

# Public libraries (installable with anaconda)
from ezc3d import c3d
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal, interpolate, sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn import metrics
import time
import numpy as np
import os
from pylab import *
import seaborn as sns
import pycwt as wavelet


######################## PARAMETERS ########################
fps = 25                # frame rate to be used for analyses
DUR = 60                # duration of each trial
NB_MARKERS = 22         # number of body markers
NB_DYADS = 35           # number of dyads
NB_TRIALS = 32          # number of trials per dyad
NB_CONDITIONS = 4       # number of conditions
NB_SONGS = 8            # number of songs


## STRUCTURE OF SKELETON (FOR VISUALIZATION)
# Marker names
list_markers = np.array(['LB Head','LF Head','RF Head','RB Head','Sternum','L Shoulder','R Shoulder',\
                            'L Elbow','L Wrist','L Hand','R Elbow','R Wrist','R Hand','Pelvis','L Hip',\
                            'R Hip','L Knee','L Ankle','L Foot','R Knee','R Ankle','R Foot'])
    
# Segments to link markers as stick figure
liaisons = [(0,1),(0,3),(1,2),(2,3),(4,5),(4,6),(5,6),(5,7),(7,8),(8,9),(6,10),(10,11),(11,12), \
            (13,14),(13,15),(14,15),(14,16),(16,17),(17,18),(15,19),(19,20),(20,21)]
    
# 3D axes names
dim_name = ["x","y","z"]


## INFORMATION ABOUT SONGS AND THEIR METRICAL STRUCTURE
song_bpms = np.array([111.03,116.07,118.23,118.95,120.47,125.93,128.27,129.06])
musParts_names = ['DRUMS','DRUMS+BASS','DRUMS+BASS+KEYBOARDS','DRUMS+BASS+KEYBOARDS+VOICE']
musParts_beats = [0,16,32,48,80] # start drums , bass, harmony, voice, end
beats_tFrames = np.zeros( (NB_SONGS,81) )
musParts_tFrames = np.zeros( (NB_SONGS,len(musParts_beats)) )
for i in range(NB_SONGS):
    periodbeat = (60/song_bpms[i])
    beats_tFrames[i,:] = np.linspace(0,80*periodbeat,81) # Because 80 beats for each song
    musParts_tFrames[i,:] = beats_tFrames[i,musParts_beats]
    

#%% 
##############################################################
############             IMPORT DATA              ############
##############################################################

print('-------------------------------------------')
print('LOADING DATA...')
print('-------------------------------------------')

NB_DIM = 3

# Decide whether to import csv or npy files
# datform = 'npy'
datform = 'csv'

## DIRECTORIES
# Input dir (to load motion data)
if datform == 'csv':  input_dir = os.path.normpath(( os.getcwd() + "/DATA/motion-csv" ))
if datform == 'npy':  input_dir = os.path.normpath(( os.getcwd() + "/DATA/motion-npy" ))
folders   = os.listdir(input_dir);    folders=[x for i,x in enumerate(folders) if (x.startswith("Dyad"))]
folders   = sorted(folders)

# Output dir (to store results)
output_dir = os.path.normpath( os.getcwd() + "/results_supp_IMS-XYZ" )
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)


## LOAD LOG FILES (CONDITIONS/SONGS)
cond_mus = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/cond_mus.csv"), index_col=0).to_numpy().astype(int)
cond_vis = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/cond_vis.csv" ), index_col=0).to_numpy().astype(int)
song_subj_LEFT = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/song_subj_LEFT.csv"), index_col=0).to_numpy().astype(int)
song_subj_RIGHT = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/song_subj_RIGHT.csv"), index_col=0).to_numpy().astype(int)
song = np.stack((song_subj_LEFT,song_subj_RIGHT),axis=0)


## PREPARE THE TIME REFERENCE SYSTEM
## (All trials are downsampled to the temporal scale of the shortest song.
## This way, all trials are comparable across time, in a metrical scale (in beats rather than in seconds))
# 1. Find number of frames of the shortest song (also considering 3 bars of silence before & after, for padding)
tStop_min = int( min(musParts_tFrames[: , -1]) * fps )  # Length of shortest song (in frames)
Nsamps_before_min = int( 3 * tStop_min / 20 )           # 3 bars of silence before/after
NB_T = tStop_min + Nsamps_before_min*2                  # Length of shortest song, considering the padding (in frames)

# 2. Convert time frames in metrical frames (from beat 1 to beat 105)
t_norm = np.arange(0,NB_T)/fps          # Length in seconds
bpm_max = max(song_bpms)                # Tempo of the shortest song (beat per minute)
t_norm_beat = t_norm/(60/bpm_max) + 1   # Length in beats (song 80 beats, before 12 beats, after 12 beats)


## PREPARE XWT (Cross-wavelet transform)
# 1. XWT parameters
NB_OCT = 5                  # Number of octaves   
NB_SCALES_PER_OCT=16        # Number of scales per octave (resolution)
dt=1/fps; dj=1/NB_SCALES_PER_OCT; J=NB_OCT/dj; NB_FREQ = int(J)+1

## 2. Init XWT matrix, to be entered subsequently into the ANOVA
## (timexfreq per DIM per dyad, trials averaged for each condition)
IMS_formatJASP = np.zeros( (NB_FREQ,NB_T,NB_DIM, NB_DYADS , NB_CONDITIONS) )

## LOOP FOR IMPORT AND XWT CALCULATION
# Dyads
for d in range(1, NB_DYADS+1):
    if d < 10 : numDyad = "0" + str(d)
    else : numDyad = str(d)
    print("Dyad " + numDyad)
    
    # Motion data
    if datform == 'csv':   
        folder = os.path.normpath(( os.getcwd() + "/DATA/motion-csv/" + folders[d-1] ))
        files  = os.listdir(os.path.normpath(folder + '/subj_LEFT'));      files=[x for i,x in enumerate(files) if (x.endswith(".csv"))]
        files  = sorted(files)
    if datform == 'npy':   
        folder = os.path.normpath(( os.getcwd() + "/DATA/motion-npy/" + folders[d-1] ))
        files  = os.listdir(os.path.normpath(folder + '/subj_LEFT'));      files=[x for i,x in enumerate(files) if (x.endswith(".npy"))]
        files  = sorted(files)
    
    # Init an intermediate IMS matrix, associated to the dyad
    IMS_dyad = np.zeros(( NB_FREQ, NB_T, NB_DIM , NB_TRIALS ))
    # Trials
    for tr in range(NB_TRIALS):
        if tr+1 < 10 : numTrial = "0" + str(tr+1)
        else : numTrial = str(tr+1)
        songsTrial = song[:,(d-1),tr]
        freqsBeat = 1/(60/song_bpms[songsTrial])
        
        # Subjects 1 and 2
        for subj in range(2):   
                
            ## LOAD motion data of the precise SUBJECT and TRIAL
            if subj == 0 : 
                if datform == 'csv':    xyz_vec = pd.read_csv(folder + "/subj_LEFT/tr" + numTrial + ".csv", index_col=0).to_numpy().astype(float)
                if datform == 'npy':    xyz_vec = np.load(folder + "/subj_LEFT/tr" + numTrial + ".npy")
            if subj == 1 : 
                if datform == 'csv':    xyz_vec = pd.read_csv(folder + "/subj_RIGHT/tr" + numTrial + ".csv", index_col=0).to_numpy().astype(float)
                if datform == 'npy':    xyz_vec = np.load(folder + "/subj_RIGHT/tr" + numTrial + ".npy")
            
            ## DOWN SAMPLE data if fps!=fps_orig (250) and there is no nan in the trial (just a sanity check)
            if (fps != 250) and (False in np.isnan(xyz_vec[:])) : 
                samps = int(DUR*fps)
                xyz_vec_ds=np.zeros((xyz_vec.shape[0],samps))  
                for i in range(xyz_vec_ds.shape[0]): 
                    xyz_vec_ds[i,:]=np.interp(np.linspace(0.0, 1.0, samps, endpoint=False), np.linspace(0.0, 1.0,  xyz_vec.shape[1], endpoint=False), xyz_vec[i,:])
                xyz_vec = xyz_vec_ds
                
            ## RESHAPE from (Nmarkers*3, Time) to (Nmarkers, 3, Time)
            sz = xyz_vec.shape
            xyz_vec_resh = np.reshape(xyz_vec, (sz[0]//3,3,sz[1]))
            
            ## REORIENT to have consistent orientation between subjects (which were facing each other)
            orient = np.sign(-1 * xyz_vec_resh[ 13 , 1 , 0])
            xyz_vec_resh[:,0,:] = xyz_vec_resh[:,0,:] * orient;   xyz_vec_resh[:,1,:] = xyz_vec_resh[:,1,:] * orient
            
            ## TRIM to measure XWT of the DIMs only 3 bars silence before - 20 bars music - 3 bars silence after
            # i.e. trim 7s before + 45s + 5s after (to have enough time before/after so that subsequently we can pad the song segment with 3-bar before/after, see below)
            if d==1: tBefore=3; tAfter=0    # exception of dyad 1 with a different amount of silence beofre/after
            else: tBefore=1; tAfter=2
            xyz_vec_resh = xyz_vec_resh[:,:,tBefore*fps:DUR*fps - tAfter*fps]
            
            ## RE-REFERENCE TO LOCAL ORIGIN, i.e. average position of point between feet for this trial (to avoid inter-trial offsets when concatenated)
            Ox = np.mean((xyz_vec_resh[18,0,:]+xyz_vec_resh[21,0,:])/2);   Oy = np.mean((xyz_vec_resh[18,1,:]+xyz_vec_resh[21,1,:])/2);   Oz = np.mean((xyz_vec_resh[18,2,:]+xyz_vec_resh[21,2,:])/2)
            xyz_vec_resh[:,0,:] -= Ox;   xyz_vec_resh[:,1,:] -= Oy;   xyz_vec_resh[:,2,:] -= Oz
            
            ## RESHAPE to come back to initial (Nmarkers*3, Time)
            sz_resh = xyz_vec_resh.shape
            xyz_vec = np.reshape(xyz_vec_resh, (sz_resh[0]*sz_resh[1] , sz_resh[2]))
            
            ## REMOVE ID MARKER, i.e. IdThighAdd (used only to distinguish between the two skeletons in the room)
            xyz_vec = xyz_vec[:66,:]
            
            ## DE-MEAN by the average posture of the trial
            lensong_diff = np.around(musParts_tFrames[songsTrial , -1] * fps).astype(int)    # length of each song
            lensong = min(lensong_diff)                                                      # minimum length between the two songs
            xyz_vec -= np.mean(xyz_vec[:,7*fps:7*fps+lensong],1).reshape((-1,1))
            
            ## COMPUTE VELOCITY of the DIMs
            idx_x = np.arange(0,66,3); idx_y = np.arange(1,66,3); idx_z = np.arange(2,66,3)
            if subj==0:     # for subject 1
                pos_subj1 = xyz_vec
                vel_subj1 = np.gradient( pos_subj1 , 1/fps , axis=1 )          # Compute velocity
                
                traj_subj1 = np.zeros((3,vel_subj1.shape[1]))
                traj_subj1[0,:] = vel_subj1[idx_x,:].mean(axis=0)
                traj_subj1[1,:] = vel_subj1[idx_y,:].mean(axis=0)
                traj_subj1[2,:] = vel_subj1[idx_z,:].mean(axis=0)
                
                # de-mean
                idx_norm = np.arange(7*fps,7*fps+lensong_diff[0])      # only taking into account when they danced (silence before and after is only kept for padding and zooming out)
                traj_subj1 = ( traj_subj1 - np.nanmean(traj_subj1[:,idx_norm] , axis = 1).reshape((-1,1)) ) / np.nanstd(traj_subj1[:,idx_norm] , axis = 1).reshape((-1,1))
               
            if subj==1:   # same but for subject 2
                pos_subj2 = xyz_vec
                traj_subj2 = np.gradient( pos_subj2 , 1/fps , axis=1 ) 
                
                vel_subj2 = np.gradient( pos_subj2 , 1/fps , axis=1 )          # Compute velocity
          
                traj_subj2 = np.zeros((3,vel_subj2.shape[1]))
                traj_subj2[0,:] = vel_subj2[idx_x,:].mean(axis=0)
                traj_subj2[1,:] = vel_subj2[idx_y,:].mean(axis=0)
                traj_subj2[2,:] = vel_subj2[idx_z,:].mean(axis=0)

                idx_norm = np.arange(7*fps,7*fps+lensong_diff[1])    # detrend and standardize only taking into account when they danced (silence before and after is only kept for padding and zooming out)
                traj_subj2 = ( traj_subj2 - np.nanmean(traj_subj2[:,idx_norm] , axis = 1).reshape((-1,1)) ) / np.nanstd(traj_subj2[:,idx_norm] , axis = 1).reshape((-1,1))

        ####### CROSS-WAVELET TRANSFORM #######
        if True not in np.isnan(traj_subj1[:]):
            print('XWT... Trial ' + str(numTrial))
            for dim in range(NB_DIM):
                # Nans sanity checks
                if True in np.isnan(traj_subj1[dim,:]): print('!!!WARNING!!! NaN')
                if True in np.isnan(traj_subj2[dim,:]): print('!!!WARNING!!! NaN')
                data1 = traj_subj1[dim,:]
                data2 = traj_subj2[dim,:]
                
                mother = wavelet.Morlet(6)      # Use Morlet wavelet of w0=6 to generate the wavelet transforms
                
                ## COMPUTE XWT considering whether the subjects listened to same vs. different song
                # The XWT is generated as a function of the metrical frequencies (from 0.5*beat to 16*beat)
                if cond_mus[(d-1),tr] == 0:         # Different songs
                    # If they listened to different songs (thus, different tempi), compute the XWT twice
                    # (first for the metrical freqs of subject1's song, then subject2's song) and then average the XWTs
                    W12_diff = [];  coi_diff = []   # to store XWT (and coi) associated to each of the 2 songs
                    for i in range(2):              # for each of the 2 songs
                        s0 = 1/(1.033043*(freqsBeat[i]*2))      # use scale s0 that is exactly the fastest frequency you're interested in, here 0.5*beat period (there is a 1.03 factor bt scale and freq)
                        # Compute continuous wavelet transforms (CWTs) of each subject for this song
                        W1, scales, freqs, coi1, fft, fftfreqs = wavelet.cwt(data1, dt, dj, s0, J, mother)
                        W2, scales, freqs, coi2, fft, fftfreqs = wavelet.cwt(data2, dt, dj, s0, J, mother)
                        
                        # Compute XWT from these CWTs
                        W12_diff.append( W1 * W2.conj() )   
                        coi_diff.append( np.minimum(coi1,coi2) )        # This is the Cone of Influence (defining where edge effects make the results hard to interpret)
                        
                    W12 = (W12_diff[0] + W12_diff[1] )/2                # average the 2 XWTs
                    coi = np.minimum( coi_diff[0] , coi_diff[1] )       # take the COI that they have in common (so the minimum period (or maximum frequency))
                
                else:                               # Same song
                    # If they listened to the same song (thus, same tempo), it's simpler: just compute the XWT associated to this song metrical levels
                    assert freqsBeat[0]==freqsBeat[1] , "bpms should be the same if same song"
                    s0 = 1/(1.033043*(freqsBeat[0]*2))      # use scale s0 that is exactly the fastest frequency you're interested in, here 0.5*beat period (there is a 1.03 factor bt scale and freq)
                    # Compute continuous wavelet transforms (CWTs) of each subject for the shared song
                    W1, scales, freqs, coi1, fft, fftfreqs = wavelet.cwt(data1, dt, dj, s0, J, mother)
                    W2, scales, freqs, coi2, fft, fftfreqs = wavelet.cwt(data2, dt, dj, s0, J, mother)
                    
                    # Compute the XWT from these CWTs
                    W12 = ( W1 * W2.conj() )      
                    coi = np.minimum(coi1,coi2)         # This is the Cone of Influence (defining where edge effects make the results hard to interpret)
                    
                    
                ## COMPUTE POWER OF THE XWT
                power = np.abs(W12);
                
                ## RETAIN 3-bar silence -- 20-bar music -- 3-bar silence, acting as padding
                Nsampes_before = int(3 * lensong / 20 )    # Length corresponding to 3 bars (the song has 20 bars)
                power = power[:,7*fps - Nsampes_before:7*fps + lensong + Nsampes_before]
                
                ## DOWN SAMPLE the data to beat-relative scale (i.e., to the number of frames of shortest song)
                power_ds = np.zeros((power.shape[0],NB_T)) 
                for freq in range(int(J+1)):        # Down sample every frequency band (row) across time (columns)
                    power_ds[freq,:] = np.interp(np.linspace(0.0, 1.0, NB_T), np.linspace(0.0, 1.0,  len(power[freq,:])), power[freq,:])
                
                ## SMOOTH IN TIME with rolling average window of 3 bars
                kernel_size = round(Nsamps_before_min)           # rolling window of 3 bars
                kernel = np.ones(kernel_size) / kernel_size      # setting the kernel to compute the average   
                for freq in range(int(J+1)):         # Smooth every frequency band (row) across time (columns)
                    power_ds[freq,:] = np.convolve(power_ds[freq,:], kernel, mode='same')
                    
                    
                ## STORE in IMS matrix of the dyad
                IMS_dyad[:,:,dim,tr] = power_ds
  
        else:       # for only one trial that is missing (dyad32, tr32): put nans
            IMS_dyad[:,:,:,tr] = np.nan
      
        
    ########### IMS MATRIX FOR ANOVA: FOR EACH DYAD, AVERAGE TRIALS WITHIN CONDITIONS ############ 
    # 1. Create mask of conditions
    YesVis_mask = (cond_vis[(d-1),:]==1); NoVis_mask = (cond_vis[(d-1),:]==0)
    SameMus_mask = (cond_mus[(d-1),:]==1); DiffMus_mask = (cond_mus[(d-1),:]==0)
    
    # 2. Retain IMS data of this dyad for each condition
    IMS_dyad_YesVisSameMus = IMS_dyad[:,:,:,YesVis_mask & SameMus_mask]
    IMS_dyad_YesVisDiffMus = IMS_dyad[:,:,:,YesVis_mask & DiffMus_mask]
    IMS_dyad_NoVisSameMus = IMS_dyad[:,:,:,NoVis_mask & SameMus_mask]
    IMS_dyad_NoVisDiffMus = IMS_dyad[:,:,:,NoVis_mask & DiffMus_mask]
    
    # 3. Average across trials within each of these conditions
    IMS_dyad_YesVisSameMus_mean = np.nanmean(IMS_dyad_YesVisSameMus,axis=3)
    IMS_dyad_YesVisDiffMus_mean = np.nanmean(IMS_dyad_YesVisDiffMus,axis=3)
    IMS_dyad_NoVisSameMus_mean = np.nanmean(IMS_dyad_NoVisSameMus,axis=3)
    IMS_dyad_NoVisDiffMus_mean = np.nanmean(IMS_dyad_NoVisDiffMus,axis=3)

    # 4. Store for ANOVA
    IMS_formatJASP[:,:,:,(d-1),0] = IMS_dyad_YesVisSameMus_mean
    IMS_formatJASP[:,:,:,(d-1),1] = IMS_dyad_YesVisDiffMus_mean
    IMS_formatJASP[:,:,:,(d-1),2] = IMS_dyad_NoVisSameMus_mean
    IMS_formatJASP[:,:,:,(d-1),3] = IMS_dyad_NoVisDiffMus_mean
        
    
    ## Convert frequency units of the XWT outcome in a scale realtive to the beat frequency (for further plots)
    freqs_per_beat = np.round( freqs / freqsBeat[1] , 4 )
    

#%% 
##############################################################
######      ANOVA ANALYSIS ACROSS TIME FOR EACH DIM      ######
##############################################################

print('-------------------------------------------')
print('RUNNING THE ANOVA ANALYSIS...')
print('-------------------------------------------')

NB_DIM = 3
NB_T = tStop_min + Nsamps_before_min*2

## FORMAT DATA CORRECTLY before entering into the ANOVA across time
# 1. Average the XWT power across periods before running ANOVA on the time-series
X = IMS_formatJASP.mean(axis=0)[:,:NB_DIM,:,:]   

# 2. Order the columns in proper format for MNE library
X = np.transpose(X, [2, 0, 1, 3])

# 3. Standardize the XWTs across conditions, within each dyad and DIM
for dim in range(NB_DIM):
    for d in range(NB_DYADS):
        X[d,:,dim,:] /= X[d,:,dim,:].std()

# 4. Convert X to a list of data across conditions (required by MNE cluster-stats libraries)
X_mne = [np.squeeze(x) for x in np.split(X, 4, axis=-1)]


## PREPARE STATS/ANOVA computation (and cluster analyses)
# 1. Specify design
factor_levels = [2, 2]      # 2x2 factorial design (Vis (Yes/No) x Mus (Same/Diff))

# 2. Create adjacency matrix for the temporal clusters (neighbours are adjacent bins in time) (only during the song segment, not the silence before/after used for padding of the XWT)
nei_mask_time = sparse.diags([1., 1.], offsets=(-1, 1), shape=(NB_T - 2*Nsamps_before_min, NB_T - 2*Nsamps_before_min))
adjacency = mne_fefe.stats.combine_adjacency( nei_mask_time )

# 3. Compute difference of means between the main factors, to make sure clusters have the same sign  
# (so "YesVis vs. NoVis"; "SameMus vs DiffMus"; "[(SameMus vs DiffMus) if YesVis] vs. [(SameMus vs DiffMus) if NoVis]")
diffMeans = np.zeros( (3 , NB_T, NB_DIM))        # for main effect 1 (vis), main effect 2 (mus), and interaction
diffMeans[0,:,:] = ( (X[:,:,:,0].mean(axis=0) + X[:,:,:,1].mean(axis=0))/2 ) - ( (X[:,:,:,2].mean(axis=0) + X[:,:,:,3].mean(axis=0))/2 )
diffMeans[1,:,:] = ( (X[:,:,:,0].mean(axis=0) + X[:,:,:,2].mean(axis=0))/2 ) - ( (X[:,:,:,1].mean(axis=0) + X[:,:,:,3].mean(axis=0))/2 )
diffMeans[2,:,:] = ( X[:,:,:,0].mean(axis=0) - X[:,:,:,1].mean(axis=0) ) - ( X[:,:,:,2].mean(axis=0) - X[:,:,:,3].mean(axis=0) )

# 4. Init cluster info + signed F values
# lists that store cluster timing (tStart and tStop) + the associated pValue, for main effects and interaction. "SIG" means the cluster is significant (over a threshold defined by 10,000 permutations)
cluster_start_vis = []; cluster_start_mus = []; cluster_start_int = []; clusterSIG_start_vis = []; clusterSIG_start_mus = []; clusterSIG_start_int = [];
cluster_stop_vis = []; cluster_stop_mus = []; cluster_stop_int = []; clusterSIG_stop_vis = []; clusterSIG_stop_mus = []; clusterSIG_stop_int = [];
p_vis = []; p_mus = []; p_int = []; pSIG_vis = []; pSIG_mus = []; pSIG_int = [];

# Matrix of signed F values (timeseries; one for each main effect + the interaction)
F_obs = np.zeros((3,NB_DIM,NB_T-2*Nsamps_before_min))

## RUN CLUSTER PERMUTATION TESTS, independently on each DIM (these independent tests are Bonferroni-corrected)
pthresh = 0.05 / NB_DIM           # Bonferroni correction
for dim in range(NB_DIM):     
    print('\n-------\nPRINCIPAL MOVEMENT DIM ' + str(dim+1) + '\n-------\n')
    X_dim = [x[:,Nsamps_before_min:-Nsamps_before_min,dim] for x in X_mne]        # analyze only when they listen to music

    for effect in range(3): 
        ## DEFINE the stat function depending on the effect (because we need to weigh the F value with the difference sign)
        if effect==0:
            print('-------\nMain effect Visual \n-------')
            effects = 'A'
            def stat_fun(*args):
                # get f-values only + weigh the Fvalue by the sign of the difference (to avoid clusters of different sign)
                diffMeansVis = (args[0].mean(axis=0) + args[1].mean(axis=0))/2 - (args[2].mean(axis=0) + args[3].mean(axis=0))/2 
                return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                                 effects=effects, return_pvals=False)[0] * np.sign(diffMeansVis)
        
        if effect==1:
            print('-------\nMain effect Music \n-------')
            effects = 'B'
            def stat_fun(*args):
                # get f-values only + weigh the Fvalue by the sign of the difference (to avoid clusters of different sign)
                diffMeansMus = (args[0].mean(axis=0) + args[2].mean(axis=0))/2 - (args[1].mean(axis=0) + args[3].mean(axis=0))/2 
                return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                                 effects=effects, return_pvals=False)[0] * np.sign(diffMeansMus)
            
        if effect==2:
            print('-------\nInteraction Visual x Music \n-------')
            effects = 'A:B'
            def stat_fun(*args):
                # get f-values only + weigh the Fvalue by the sign of the difference (to avoid clusters of different sign)
                diffMeans_IMS_mus_YesVis = args[0].mean(axis=0) - args[1].mean(axis=0)
                diffMeans_IMS_mus_NoVis = args[2].mean(axis=0) - args[3].mean(axis=0)
                diff_interact = diffMeans_IMS_mus_YesVis - diffMeans_IMS_mus_NoVis
                return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                                  effects=effects, return_pvals=False)[0] * np.sign(diff_interact)
        
        
        ## CLUSTERING
        # 1. Define threhsold Fvalue
        f_thresh = mne_fefe.stats.f_threshold_mway_rm(NB_DYADS, factor_levels, effects, pthresh)      
        
        # 2. Define number of permutations (10,000)
        n_permutations = 10001 
        
        # 3. Run MNE cluster test
        print('Clustering.')
        F_obs[effect,dim,:], clusters, cluster_p_values, H0  = \
            mne_fefe.stats.cluster_level.spatio_temporal_cluster_test(X_dim, adjacency=adjacency, n_jobs=None,
                                         threshold=f_thresh, stat_fun=stat_fun,
                                         n_permutations=n_permutations,t_power=1,
                                         buffer_size=None,out_type='indices',stat_cluster='sum')
        
        # 4. Store initial clusters
        if len(cluster_p_values)>0:
            first_vals = np.zeros(len(cluster_p_values),dtype=int); last_vals = np.zeros(len(cluster_p_values),dtype=int); 
            for c in range(len(cluster_p_values)):
                first_vals[c] = clusters[c][0][0] + Nsamps_before_min
                last_vals[c] =  clusters[c][0][-1] + Nsamps_before_min
            
            if effect==0: cluster_start_vis.append(first_vals); cluster_stop_vis.append(last_vals); p_vis.append(cluster_p_values)
            if effect==1: cluster_start_mus.append(first_vals); cluster_stop_mus.append(last_vals); p_mus.append(cluster_p_values)
            if effect==2: cluster_start_int.append(first_vals); cluster_stop_int.append(last_vals); p_int.append(cluster_p_values)
            
        # 5. Filter clusters that have only 1 temporal bin
        clusters_filt = [x for i, x in enumerate(clusters) if len(x[x == True]) > 1]
        idx = [i for i, x in enumerate(clusters) if len(x[x == True]) > 1]
        cluster_p_values_filt = cluster_p_values[idx]

        # 6. Retain only significant clusters
        idx_cluster_sig = np.where(cluster_p_values_filt<pthresh)[0]
        if len(idx_cluster_sig)>0:
            first_vals = np.zeros(len(idx_cluster_sig),dtype=int); last_vals = np.zeros(len(idx_cluster_sig),dtype=int); p_corr = np.zeros(len(idx_cluster_sig));
            for c in range(len(idx_cluster_sig)):
                first_vals[c] = clusters_filt[idx_cluster_sig[c]][0][0] + Nsamps_before_min
                last_vals[c] =  clusters_filt[idx_cluster_sig[c]][0][-1] + Nsamps_before_min
                p_corr[c] = cluster_p_values_filt[idx_cluster_sig[c]] 
            
            if effect==0: clusterSIG_start_vis.append(first_vals); clusterSIG_stop_vis.append(last_vals); pSIG_vis.append(p_corr)
            if effect==1: clusterSIG_start_mus.append(first_vals); clusterSIG_stop_mus.append(last_vals); pSIG_mus.append(p_corr)
            if effect==2: clusterSIG_start_int.append(first_vals); clusterSIG_stop_int.append(last_vals); pSIG_int.append(p_corr)
            
        else: 
            if effect==0: cluster_start_vis.append(np.empty(0)); cluster_stop_vis.append(np.empty(0)); p_vis.append(np.empty(0))
            if effect==1: cluster_start_mus.append(np.empty(0)); cluster_stop_mus.append(np.empty(0)); p_mus.append(np.empty(0))
            if effect==2: cluster_start_int.append(np.empty(0)); cluster_stop_int.append(np.empty(0)); p_int.append(np.empty(0))
            
            if effect==0: clusterSIG_start_vis.append(np.empty(0)); clusterSIG_stop_vis.append(np.empty(0)); pSIG_vis.append(np.empty(0))
            if effect==1: clusterSIG_start_mus.append(np.empty(0)); clusterSIG_stop_mus.append(np.empty(0)); pSIG_mus.append(np.empty(0))
            if effect==2: clusterSIG_start_int.append(np.empty(0)); clusterSIG_stop_int.append(np.empty(0)); pSIG_int.append(np.empty(0))


#%% 
##############################################################
######  PLOT THE RESULTS OF THE ANOVA WITH CLUSTERS     ######
######       GRAPH SHOWING THE DIM-SPECIFIC ANOVAS       ######
######     (with IMS curves, related to supp fig 4)     ######
##############################################################

import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
matplotlib.use('Agg')   # for very big figure
NB_DIM = 3

## SET PARAMETERS of the plot
bigtitle=['VISUAL','MUSIC','INTERACTION']
dim_label = ['Mediolateral', 'Anteroposterior', 'Vertical']

## COMPUTE IMS PERIODICITY spectrum (time-averaged XWT) of each DIM
dim_spectrums = IMS_formatJASP.mean(axis=1).mean(axis=2).mean(axis=2)

## PLOT
fig = plt.figure(figsize=(40,10))
gs = fig.add_gridspec(3 , 9, width_ratios=[0.5,1,10,0.5,1,10,0.5,1,10])
for effect in range(3):
    if effect==0: y=1        # Visual on first column
    if effect==1: y=4        # Music on second column
    if effect==2: y=7        # Interaction on third column
        
    for dim in range(NB_DIM):
        # Plot name of the dimension on the left
        if effect==0:
            ax0= fig.add_subplot(gs[dim, 0])
            ax0.set_aspect('equal')
            ax0.get_xaxis().set_visible(False)  
            ax0.get_yaxis().set_visible(False)
            ax0.set_axis_off()
            ax0.set_title(dim_label[dim],rotation='horizontal',x=-2,y=0.5,fontsize=20)
        
        if effect==0: clusterSIG_start = clusterSIG_start_vis[dim]; clusterSIG_stop = clusterSIG_stop_vis[dim]; p_corr = p_vis[dim]; label='visual'
        if effect==1: clusterSIG_start = clusterSIG_start_mus[dim]; clusterSIG_stop = clusterSIG_stop_mus[dim]; p_corr = p_mus[dim]; label='music'
        if effect==2: clusterSIG_start = clusterSIG_start_int[dim]; clusterSIG_stop = clusterSIG_stop_int[dim]; p_corr = p_int[dim]; label='music'
    
        ##### FIRST BLOCK: IMS PERIODICITY #####
        ax1= fig.add_subplot(gs[dim, y])
        
        means_dim = np.array([ dim_spectrums[:,dim] , dim_spectrums[:,dim] ])
        minSpec = means_dim.min();  maxSpec = means_dim.max()
        ax1.pcolormesh(np.arange(2),np.log2(freqs_per_beat),means_dim.T,vmin=minSpec,vmax=maxSpec)
        
        # Set params
        ax1.set_xlim((-0.5,0.5)); ax1.set_xticks([]);
        ax1.set_yticks( np.array([2,1,0,-1,-2,-3,-4]) ) 
        ax1.set_yticklabels( ['0.25','0.5','1','2','4','8','16'] , fontsize = 12)
        if dim == 0: ax1.set_title('SYNCHRONY\nSPECTRUM',fontsize=15,pad=15)
        if dim == 0: ax1.set_ylabel('Period = beat *' , fontsize = 15)
        ax1.set_ylim(np.log2([freqs_per_beat.min(),freqs_per_beat.max()]))
        
        ##### SECOND BLOCK: IMS CURVES and F VALUES #####
        ax2 = fig.add_subplot(gs[dim, y+1])
        
        # 1. Compute IMS curves across conditions
        if effect==0:
            mean_high = (X[:,:,dim,0].mean(axis=0) + X[:,:,dim,1].mean(axis=0))/2 ; label_high = 'YesVis'
            mean_low = (X[:,:,dim,2].mean(axis=0) + X[:,:,dim,3].mean(axis=0))/2 ; label_low = 'NoVis'
        
        if effect==1:
            mean_high = (X[:,:,dim,0].mean(axis=0) + X[:,:,dim,2].mean(axis=0))/2 ; label_high = 'SameMus'
            mean_low = (X[:,:,dim,1].mean(axis=0) + X[:,:,dim,3].mean(axis=0))/2 ; label_low = 'DiffMus'
            
        # 2. Plot them (plot all 4 conditions when it's the interaction)
        if effect in [0,1]:
            ax2.plot(t_norm_beat,mean_high,label=label_high,c='black')
            ax2.plot(t_norm_beat,mean_low,label=label_low,c='black', linestyle='dashed')
        if effect==2:
            ax2.plot(t_norm_beat,X[:,:,dim,0].mean(axis=0),label='YV-SM',c='crimson')
            ax2.plot(t_norm_beat,X[:,:,dim,1].mean(axis=0),label='YV-DM',c='crimson', linestyle='dashed')
            ax2.plot(t_norm_beat,X[:,:,dim,2].mean(axis=0),label='NV-SM',c='steelblue')
            ax2.plot(t_norm_beat,X[:,:,dim,3].mean(axis=0),label='NV-DM',c='steelblue', linestyle='dashed')
        
        if dim == 0: ax2.set_title(r'$\bf{' + bigtitle[effect] + '}$\n' + 'ANOVA CLUSTERS ON SYNCHRONY OVER TIME',fontsize=15,pad=30)
        ymin = 0; ymax = 3.5
            
        # 3. Plot F-values as colored background
        if effect==0: cmap = sns.diverging_palette(250, 38, s=100, l=70, as_cmap=True)
        if effect==1: cmap = sns.diverging_palette(280, 130, s=100, l=55, as_cmap=True)
        if effect==2: cmap = plt.cm.get_cmap('BrBG_r')
        c_clusters = np.array([ cmap(0) , cmap(cmap.N)])

        F_obs_to_plot= np.zeros((4,NB_T))
        F_obs_to_plot[:,:Nsamps_before_min] = np.nan; F_obs_to_plot[-Nsamps_before_min:,:] = np.nan
        F_obs_to_plot[:,Nsamps_before_min:-Nsamps_before_min] = np.array([ F_obs[effect,dim,:] , F_obs[effect,dim,:] , F_obs[effect,dim,:] , F_obs[effect,dim,:]  ])
        absMax = np.amax(abs(F_obs))
        im = ax2.pcolormesh(t_norm_beat,np.arange(4),F_obs_to_plot,cmap=cmap,vmin=-absMax,vmax=absMax,alpha=0.8)

        # 4. Plot the clusters as colored rectangles
        for c in range(len(clusterSIG_start)):
            sign = np.sign(diffMeans[effect,clusterSIG_start[c]:clusterSIG_stop[c],dim])[0]
            if sign == -1: color=c_clusters[0]
            if sign == 1: color=c_clusters[1]
            len_cluster = t_norm_beat[clusterSIG_stop[c]] - t_norm_beat[clusterSIG_start[c]]
            
            ax2.add_patch(Rectangle((t_norm_beat[clusterSIG_start[c]],0.4),len_cluster, 0.2,edgecolor = 'k',facecolor = color,lw=1,zorder=10))
    
        # 5. Set silence before/after as gray shaded area
        ax2.axvspan(t_norm_beat[0], t_norm_beat[Nsamps_before_min], facecolor='lightgray',zorder=10,alpha=0.6); ax2.axvspan(t_norm_beat[-Nsamps_before_min], t_norm_beat[-1], facecolor='lightgray',zorder=10,alpha=0.6);
        
        # 6. Set labels, lims, params...
        ax2.tick_params(axis='y', labelsize=12)
        ax2.set_ylim(ymin,ymax); 
        ax2.set_xticks( np.arange(1,105,4) ) 
        ax2.set_xlim( ([min(t_norm_beat),max(t_norm_beat)]))
       
        if dim == 0 : 
            x_labels = np.empty(np.arange(1,105,4).shape, dtype('<U21'))
            x_labels[1]='     SILENCE'; x_labels[-2]='       SILENCE';
            x_labels[5]+='DRUMS';  x_labels[9]+='+BASS';  x_labels[13]+='+KEYBOARD';  x_labels[17]+='+VOICE'; x_labels[21]+='+VOICE BIS';
            ax2.set_xticklabels( x_labels , fontsize=11 )
            ax2.tick_params(axis='x', which='both', length=0)
            ax2.xaxis.set_ticks_position('top')
            ax2.vlines(np.arange(13,94,16) , ymin,ymax*1.1, color='gray', alpha=0.6, zorder=8 , clip_on=False)
            ax2.legend(loc='upper left' , fontsize = 12).set_zorder(30)
        elif dim == 2 : 
            x_labels = (np.arange(1,105,4)//4 - 2).astype('<U21')
            x_labels[0] = x_labels[2] = x_labels[-2] = ''; x_labels[1]=''; x_labels[-1]=''
            ax2.set_xticklabels( x_labels , fontsize=11 ); ax2.set_xlabel('Bar', fontsize=11)
            ax2.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=8)
        else : 
            ax2.set_xticks([])
            ax2.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=8)
    
fig.tight_layout()
# fig.savefig(output_dir + '/XYZ-specific_IMS.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/XYZ-specific_IMS.png', dpi=300, bbox_inches='tight'); plt.close() 
  