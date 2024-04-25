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
output_dir = os.path.normpath( os.getcwd() + "/results_supp_QoM" )
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)


## LOAD LOG FILES (CONDITIONS/SONGS)
cond_mus = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/cond_mus.csv"), index_col=0).to_numpy().astype(int)
cond_vis = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/cond_vis.csv" ), index_col=0).to_numpy().astype(int)
song_subj_LEFT = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/song_subj_LEFT.csv"), index_col=0).to_numpy().astype(int)
song_subj_RIGHT = pd.read_csv(os.path.normpath( os.getcwd() + "/DATA/log/song_subj_RIGHT.csv"), index_col=0).to_numpy().astype(int)
song = np.stack((song_subj_LEFT,song_subj_RIGHT),axis=0)


###### LOAD & STORE MOTION DATA ######

## INIT DATA MATRIX CONTAINING ALL TRIALS PER PARTICIPANT
## (70 participants x 22*3D markers x 31142 time frames (all trials concatenated))
# Each trial is cropped to the duration of the shortest song within the dyad
# so in SameMus: all trials last each of the 8 songs (YesVis + NoVis --> *2)
# in DiffMus: half of trials last each of the 4 shortest songs of subj1, half these of subj2 (still YesVis + NoVis --> *4)
LEN_CONCAT_TIME = (musParts_tFrames[:,-1]*fps).astype(int).sum()*2 + (musParts_tFrames[4:,-1]*fps).astype(int).sum()*4
data_subj = np.zeros((NB_DYADS*2 , NB_MARKERS*3 , LEN_CONCAT_TIME))

## INIT MEAN/STD VECTORS FOR STANDARDIZATION
pmean_tr = np.zeros((NB_DYADS*2 , NB_MARKERS*3 , NB_TRIALS))    # mean posture per trial
std_tr = np.zeros(( NB_DYADS*2 , NB_TRIALS))                    # global std per trial

## LOOP FOR IMPORT
# Dyads
for d in range(1, NB_DYADS+1):
    if d < 10 : numDyad = "0" + str(d)
    else : numDyad = str(d)
    print("Dyad " + numDyad)
    
    # Motion data 
    if datform == 'csv':    # (.csv)
        folder = os.path.normpath(( os.getcwd() + "/DATA/motion-csv/" + folders[d-1] ))
        files  = os.listdir(os.path.normpath(folder + '/subj_LEFT'));      files=[x for i,x in enumerate(files) if (x.endswith(".csv"))]
        files  = sorted(files)
    if datform == 'npy':    # (.npy)
        folder = os.path.normpath(( os.getcwd() + "/DATA/motion-npy/" + folders[d-1] ))
        files  = os.listdir(os.path.normpath(folder + '/subj_LEFT'));      files=[x for i,x in enumerate(files) if (x.endswith(".npy"))]
        files  = sorted(files)
    
    # Subjects 1 and 2
    for subj in range(2):   
        iStart=0        # to concatenate trials
        # Trials
        for tr in range(NB_TRIALS):
            if tr+1 < 10 : numTrial = "0" + str(tr+1)
            else : numTrial = str(tr+1)  
            
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
            
            ## TRIM to extract PMs only when they listen to music
            # 1. find the song segment (i.e., remove the parts with silence before and after)
            if d==1: tBefore=10; tAfter=5   # exception of dyad 1 with a different amount of silence
            else: tBefore=8; tAfter=7
            xyz_vec_resh = xyz_vec_resh[:,:,tBefore*fps:DUR*fps - tAfter*fps]
            
            # 2. trim to the shortest song between the 2 subjects
            songsTrial = song[:,(d-1),tr]   # the two songs of this dyad and trial
            tStop = int( min(musParts_tFrames[songsTrial , -1]) * fps ) # take the end frame of the shortest song between two subjects    
            xyz_vec_resh = xyz_vec_resh[:,:,:tStop]
            
            ## RE-REFERENCE TO LOCAL ORIGIN, i.e. average position of point between feet for this trial (to avoid inter-trial offsets when concatenated)
            Ox = np.mean((xyz_vec_resh[18,0,:]+xyz_vec_resh[21,0,:])/2);   Oy = np.mean((xyz_vec_resh[18,1,:]+xyz_vec_resh[21,1,:])/2);   Oz = np.mean((xyz_vec_resh[18,2,:]+xyz_vec_resh[21,2,:])/2)
            xyz_vec_resh[:,0,:] -= Ox;   xyz_vec_resh[:,1,:] -= Oy;   xyz_vec_resh[:,2,:] -= Oz
            
            ## RESHAPE to come back to initial (Nmarkers*3, Time)
            sz_resh = xyz_vec_resh.shape
            xyz_vec = np.reshape(xyz_vec_resh, (sz_resh[0]*sz_resh[1] , sz_resh[2]))
            
            ## REMOVE ID MARKER, i.e. IdThighAdd (used only to distinguish between the two skeletons in the room)
            xyz_vec = xyz_vec[:66,:]
            
            ## STANDARDIZATION FOR MULTI-SUBJECT PM EXTRACTION (PCA)
            # 1. De-mean by the average posture of the trial
            pmean_tr[(d-1)*2+subj,:,tr] = np.mean(xyz_vec,1)
            xyz_vec -= pmean_tr[(d-1)*2+subj,:,tr].reshape((-1,1))
            
            # 2. Divide by the general std over all markers (this way, every subject contributes equally to the variance captured by PCA)
            std_tr[(d-1)*2+subj,tr] = np.std( xyz_vec[:] )
            xyz_vec /= std_tr[(d-1)*2+subj,tr].reshape((-1,1))
            
            ## STORE DATA
            data_subj[(d-1)*2+subj,:,iStart:iStart+tStop] = xyz_vec.copy()
            
            iStart+=tStop   # increment next location to store the data for this subject (ie concatenation one trial after the other)


#%% 
##############################################################
############     EXTRACT PRINCIPAL MOVEMENTS      ############
##############################################################

print('-------------------------------------------')
print('EXTRACTING PRINCIPAL MOVEMENTS...')
print('-------------------------------------------')

## CREATE DATA MATRIX "P", i.e. posture vectors concatenated over trials and participants 
# (2,177,912 frames × 66 (22 markers * 3 axes))  (1 trial missing in total, i.e. tr32 od fyad32)
P = data_subj[0,:,:]
for i in range(1,NB_DYADS*2):
    P = np.hstack((P,data_subj[i,:,:][:,np.where(~np.isnan(data_subj[i,0,:]))[0]]))
P = P.T
del data_subj

## PM EXTRACTION
U, S, V = np.linalg.svd(P, full_matrices=False)                                 # PCA applied to P, using Singular Value Decomposition
eigenval_PMs = S**2                                                             # Eigenvalues                              
PMs_var_explained = np.cumsum(eigenval_PMs) / np.sum(eigenval_PMs)              # Variance explained
nb_PMs_95 = [i for (i, val) in enumerate(PMs_var_explained) if val>0.95][0]     # Finds the first K PMs that explain >95% variance
PM_scores_time = (U*S)                                                          # PM score timeseries
PM_weights_vect = V                                                             # PM weight vector

## PLOT variance explained by the first 20 PMs
fig = plt.figure()
plt.bar(np.arange(20),PMs_var_explained[:20]*100,facecolor='w',edgecolor='k',width=0.7); plt.ylim((0,105))
# fig.savefig(output_dir + '/PMs_explained-var.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/PMs_explained-var.png', dpi=300, bbox_inches='tight'); plt.close() 


#%% 
##############################################################
############    FREQUENCY FILTERING OF THE PMs    ############
##############################################################

print('-------------------------------------------')
print('FILTERING of THE PMs...')
print('-------------------------------------------')

################ LOW-PASS FILTERING ################
# Prepare filter
fc = 6                                   # Cut-off frequency of the filter
w = fc / (fps / 2)                       # Normalize the frequency
b, a = signal.butter(2, w, 'low')        # 2d-order Butterwoth filter

# Apply filter (to each participant and trial)
iStart=0
for i in range(NB_DYADS*2):
    if i in [62,63]: nbTr = NB_TRIALS - 1
    else: nbTr = NB_TRIALS
    for tr in range(nbTr):
        songsTrial = song[:,i//2,tr]
        tStop = int( min(musParts_tFrames[songsTrial , -1]) * fps ) # take the end frame of the shortest song between two subjects    
        PM_scores_time[iStart:iStart+tStop,:] = signal.filtfilt(b, a, PM_scores_time[iStart:iStart+tStop,:], axis=0)
        
        iStart += tStop


#%% 
##############################################################
#########     COMPUTE QUANTITY OF MOVEMENT (QoM)     #########
##############################################################

print('------------------------')
print('COMPUTING QUANTITY OF MOVEMENT...')
print('------------------------')

NB_PM = 15

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


## PREPARE QoM
# Init QoM matrix, to be entered subsequently into the ANOVA
# (QoM timeseries per PM per dyad, trials averaged for each condition)
QoM_formatJASP = np.zeros( (NB_T, NB_PM, NB_DYADS , NB_CONDITIONS) )

## LOOP FOR QoM CALCULATION
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
    
    # Init an intermediate QoM matrix, associated to the dyad
    QoM_dyad = np.zeros(( NB_T, NB_PM , NB_TRIALS ))
    # Trials
    for tr in range(NB_TRIALS):
        if tr+1 < 10 : numTrial = "0" + str(tr+1)
        else : numTrial = str(tr+1)
        songsTrial = song[:,(d-1),tr]
        
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
            
            ## TRIM to measure XWT of the PMs only 3 bars silence before - 20 bars music - 3 bars silence after
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
            xyz_vec -= pmean_tr[(d-1)*2+subj,:,tr].reshape((-1,1))
            
            ## COMPUTE VELOCITY of the PMs
            lensong_diff = np.around(musParts_tFrames[songsTrial , -1] * fps).astype(int)    # length of each song
            lensong = min(lensong_diff)                                                      # minimum length between the two songs
            # lensong = np.mean(lensong_diff).astype(int)  
            if subj==0:     # for subject 1
                pos_subj1 = np.dot(xyz_vec.T , PM_weights_vect.T).T             # Get PM score by projecting the trajectory onto the eigen vector
                traj_subj1 = np.gradient( pos_subj1 , 1/fps , axis=1 )          # Compute velocity
                
                # de-mean
                idx_norm = np.arange(7*fps,7*fps+lensong_diff[0])      # only taking into account when they danced (silence before and after is only kept for padding and zooming out)
                traj_subj1 = ( traj_subj1 - np.nanmean(traj_subj1[:,idx_norm] , axis = 1).reshape((-1,1)) ) 


            if subj==1:   # same but for subject 2
                pos_subj2 = np.dot(xyz_vec.T , PM_weights_vect.T).T
                traj_subj2 = np.gradient( pos_subj2 , 1/fps , axis=1 ) 
                
                # de-mean
                idx_norm = np.arange(7*fps,7*fps+lensong_diff[1])    # only taking into account when they danced (silence before and after is only kept for padding and zooming out)
                traj_subj2 = ( traj_subj2 - np.nanmean(traj_subj2[:,idx_norm] , axis = 1).reshape((-1,1)) )

        
        
        ####### QoM (SUM WITHIN DYAD) #######
        if True not in np.isnan(traj_subj1[:]):
            print('QoM... Trial ' + str(numTrial))
            ## COMPUTE QoM
            QoM1 = abs(traj_subj1)
            QoM2 = abs(traj_subj2)
            
            ## RETAIN 3-bar silence -- 20-bar music -- 3-bar silence, acting as padding
            Nsamps_before = int(3 * lensong_diff[0] / 20 )    # Length corresponding to 3 bars (the song has 20 bars)
            QoM1 = QoM1[:,7*fps - Nsamps_before:7*fps + lensong_diff[0] + Nsamps_before]
            Nsamps_before = int(3 * lensong_diff[1] / 20 )    # Length corresponding to 3 bars (the song has 20 bars)
            QoM2 = QoM2[:,7*fps - Nsamps_before:7*fps + lensong_diff[1] + Nsamps_before]
            
            QoM1_ds = np.zeros(( NB_T , NB_PM ))
            QoM2_ds = np.zeros(( NB_T , NB_PM ))
            for pm in range(NB_PM):
                ## DOWN SAMPLE the data to beat-relative scale (i.e., to the number of frames of shortest song)
                QoM1_ds[:,pm] = np.interp(np.linspace(0.0, 1.0, NB_T), np.linspace(0.0, 1.0,  len(QoM1[pm,:])), QoM1[pm,:])
                ## DOWN SAMPLE the data to beat-relative scale (i.e., to the number of frames of shortest song)
                QoM2_ds[:,pm] = np.interp(np.linspace(0.0, 1.0, NB_T), np.linspace(0.0, 1.0,  len(QoM2[pm,:])), QoM2[pm,:])
                
                ## SMOOTH IN TIME with rolling average window of 3 bars
                kernel_size = round(Nsamps_before_min)           # rolling window of 3 bars
                kernel = np.ones(kernel_size) / kernel_size      # setting the kernel to compute the average   
                QoM1_ds[:,pm] = np.convolve(QoM1_ds[:,pm], kernel, mode='same')
                QoM2_ds[:,pm] = np.convolve(QoM2_ds[:,pm], kernel, mode='same')
                
            QoM_dyad[:,:,tr] = QoM1_ds + QoM2_ds
                
        else:       # for only one trial that is missing (dyad32, tr32): put nans
            QoM_dyad[:,:,tr] = np.nan
      
        
    ########### IMS MATRIX FOR ANOVA: FOR EACH DYAD, AVERAGE TRIALS WITHIN CONDITIONS ############ 
    # 1. Create mask of conditions
    YesVis_mask = (cond_vis[(d-1),:]==1); NoVis_mask = (cond_vis[(d-1),:]==0)
    SameMus_mask = (cond_mus[(d-1),:]==1); DiffMus_mask = (cond_mus[(d-1),:]==0)
    
    # 2. Retain QoM data of this dyad for each condition
    QoM_dyad_YesVisSameMus = QoM_dyad[:,:,YesVis_mask & SameMus_mask]
    QoM_dyad_YesVisDiffMus = QoM_dyad[:,:,YesVis_mask & DiffMus_mask]
    QoM_dyad_NoVisSameMus = QoM_dyad[:,:,NoVis_mask & SameMus_mask]
    QoM_dyad_NoVisDiffMus = QoM_dyad[:,:,NoVis_mask & DiffMus_mask]
    
    # 3. Average across trials within each of these conditions
    QoM_dyad_YesVisSameMus_mean = np.nanmean(QoM_dyad_YesVisSameMus,axis=2)
    QoM_dyad_YesVisDiffMus_mean = np.nanmean(QoM_dyad_YesVisDiffMus,axis=2)
    QoM_dyad_NoVisSameMus_mean = np.nanmean(QoM_dyad_NoVisSameMus,axis=2)
    QoM_dyad_NoVisDiffMus_mean = np.nanmean(QoM_dyad_NoVisDiffMus,axis=2)

    # 4. Store for ANOVA
    QoM_formatJASP[:,:,(d-1),0] = QoM_dyad_YesVisSameMus_mean
    QoM_formatJASP[:,:,(d-1),1] = QoM_dyad_YesVisDiffMus_mean
    QoM_formatJASP[:,:,(d-1),2] = QoM_dyad_NoVisSameMus_mean
    QoM_formatJASP[:,:,(d-1),3] = QoM_dyad_NoVisDiffMus_mean
    
    

#%% 
##############################################################
######      ANOVA ANALYSIS ACROSS TIME FOR EACH PM      ######
##############################################################

print('-------------------------------------------')
print('RUNNING THE ANOVA ANALYSIS...')
print('-------------------------------------------')

NB_PM = 15
NB_T = tStop_min + Nsamps_before_min*2 


## FORMAT DATA CORRECTLY before entering into the ANOVA across time
# 1. Take the QoM values
X = QoM_formatJASP[:,:NB_PM,:,:]
 
# 2. Order the columns in proper format for MNE library
X = np.transpose(X, [2, 0, 1, 3])   # reshape to good format for MNE library ()

# 3. Standardize the XWTs across conditions, within each dyad and PM
# Normalize to reduce inter-dyad differences
for pm in range(NB_PM):
    for d in range(NB_DYADS):
        X[d,:,pm,:] /= X[d,:,pm,:].std()

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
diffMeans = np.zeros( (3 , NB_T, NB_PM))        # for main effect 1 (vis), main effect 2 (mus), and interaction
diffMeans[0,:,:] = ( (X[:,:,:,0].mean(axis=0) + X[:,:,:,1].mean(axis=0))/2 ) - ( (X[:,:,:,2].mean(axis=0) + X[:,:,:,3].mean(axis=0))/2 )
diffMeans[1,:,:] = ( (X[:,:,:,0].mean(axis=0) + X[:,:,:,2].mean(axis=0))/2 ) - ( (X[:,:,:,1].mean(axis=0) + X[:,:,:,3].mean(axis=0))/2 )
diffMeans[2,:,:] = ( X[:,:,:,0].mean(axis=0) - X[:,:,:,1].mean(axis=0) ) - ( X[:,:,:,2].mean(axis=0) - X[:,:,:,3].mean(axis=0) )

# 4. Init cluster info + signed F values
# lists that store cluster timing (tStart and tStop) + the associated pValue, for main effects and interaction. "SIG" means the cluster is significant (over a threshold defined by 10,000 permutations)
cluster_start_vis = []; cluster_start_mus = []; cluster_start_int = []; clusterSIG_start_vis = []; clusterSIG_start_mus = []; clusterSIG_start_int = [];
cluster_stop_vis = []; cluster_stop_mus = []; cluster_stop_int = []; clusterSIG_stop_vis = []; clusterSIG_stop_mus = []; clusterSIG_stop_int = [];
p_vis = []; p_mus = []; p_int = []; pSIG_vis = []; pSIG_mus = []; pSIG_int = [];

# Matrix of signed F values (timeseries; one for each main effect + the interaction)
F_obs = np.zeros((3,NB_PM,NB_T-2*Nsamps_before_min))

## RUN CLUSTER PERMUTATION TESTS, independently on each PM (these independent tests are Bonferroni-corrected)
pthresh = 0.05 / NB_PM           # Bonferroni correction
for pm in range(NB_PM):     
    print('\n-------\nPRINCIPAL MOVEMENT PM ' + str(pm+1) + '\n-------\n')
    X_pm = [x[:,Nsamps_before_min:-Nsamps_before_min,pm] for x in X_mne]        # analyze only when they listen to music

    for effect in range(3):
        ## DEFINE the stat function depending on the effect (because we need to weigh the F value with the difference sign)
        if effect==0:
            print('-------\nMain effect Visual \n-------')
            effects = 'A'
            def stat_fun(*args):
                # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
                diffMeansVis = (args[0].mean(axis=0) + args[1].mean(axis=0))/2 - (args[2].mean(axis=0) + args[3].mean(axis=0))/2 
                return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                                 effects=effects, return_pvals=False)[0] * np.sign(diffMeansVis)
        
        if effect==1:
            print('-------\nMain effect Music \n-------')
            effects = 'B'
            def stat_fun(*args):
                # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
                diffMeansMus = (args[0].mean(axis=0) + args[2].mean(axis=0))/2 - (args[1].mean(axis=0) + args[2].mean(axis=0))/2 
                return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                                 effects=effects, return_pvals=False)[0] * np.sign(diffMeansMus)
            
        if effect==2:
            print('-------\nInteraction Visual x Music \n-------')
            effects = 'A:B'
            def stat_fun(*args):
                # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
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
        F_obs[effect,pm,:], clusters, cluster_p_values, H0 = clu = \
            mne_fefe.stats.cluster_level.spatio_temporal_cluster_test(X_pm, adjacency=adjacency, n_jobs=None,
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
        
        # # And filter diff-mus clusters that occur after the last bar, because of one subject stopping moving before the other
        # clusters_filt = [x for i, x in enumerate(clusters_filt) if x[0][0] < 800]
        # idx = [i for i, x in enumerate(clusters_filt) if x[0][0] < 800]
        # cluster_p_values_filt = cluster_p_values[idx]

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
######  1. PREPARE PM PLOTS BY DETECTING MIN AND MAX    ######
##############################################################

NB_PM=15

# Adjust the exaggeration of the rendering of the PM min/max manually, for PMs that are hard to visualize/interpret
EXAG = np.zeros(NB_PM)
PM_min = np.zeros( (NB_PM , NB_MARKERS , 3) ); PM_max = np.zeros( (NB_PM , NB_MARKERS , 3) )
for pm in range(NB_PM):
    EXAG[pm] = 1
    if pm in [2]: EXAG[pm] = 1.5
    if pm in [3,6,8,10,14]: EXAG[pm] = 2
    if pm in [7]: EXAG[pm] = 3
    
    # Project the PM back onto the high-dimensional 3D space
    eigenmov = np.outer(EXAG[pm]*PM_scores_time[:,pm] , PM_weights_vect[pm,:]).T
    iStart=0; 
    for i in range(NB_DYADS*2) :
        if i in [62,63]: nbTr = NB_TRIALS - 1
        else: nbTr = NB_TRIALS
        for tr in range(nbTr):
            songsTrial = song[:,i//2,tr]
            tStop = int( min(musParts_tFrames[songsTrial , -1]) * fps )   # take the end frame of the shortest song between two subjects    
            eigenmov[:,iStart:iStart+tStop] *= std_tr[i,tr]               # de-standardize it
            
            iStart += tStop
    
    # Generate a stick figure that look slike an "average skeleton" --> de-mean with the cross-participant average posture
    eigenmov += np.mean(np.nanmean(pmean_tr,2),0).reshape((-1,1))
    # Reshape in a data format compatible with the plot function (Markers, Space dimension, Time)
    eigenmov = np.reshape( eigenmov ,(NB_MARKERS , 3 , PM_scores_time.shape[0]) )
    
    # Find min and max of the PM postures
    i_min = argmin(PM_scores_time[:,pm]) ; i_max = argmax(PM_scores_time[:,pm])    
    PM_min[pm,:,:] = eigenmov[:,:,i_min];  PM_max[pm,:,:] = eigenmov[:,:,i_max]     # these are the min and max posture, per PM
    
    
       
#%% 
##############################################################
#####  PLOT THE RESULTS OF THE ANOVA WITH CLUSTERS       #####
#####     2. GRAPH SHOWING THE PM-SPECIFIC ANOVAS        #####
#####     (with F-value imagesc, related to fig S1)      #####
##############################################################

import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

matplotlib.use('Agg')   # for very big figure
NB_PM = 15


## SET PARAMETERS of the plot
bigtitle=['VISUAL','MUSIC','INTERACTION']
# Colors for the colormaps of the F-values
colors_vis=np.array([(255,207,0),(255,169,30),(255,140,0),(223,120,0)])/255
colors_mus=np.array([(164,251,166),(74,229,74),(48,203,0),(15,146,0),(0,98,3)])/255
colors_vis_mean = colors_vis.mean(axis=0)
colors_mus_mean = colors_mus.mean(axis=0)



for effect in range(3):   
    fig = plt.figure(figsize=(25,20))
    gs = fig.add_gridspec(8 , 2)
    
    for pm in range(NB_PM):
        m = pm%8; col=pm//8
                
        if effect==0: clusterSIG_start = clusterSIG_start_vis[pm]; clusterSIG_stop = clusterSIG_stop_vis[pm]; p_corr = p_vis[pm]; colors=[sns.diverging_palette(250, 30,l=65)[0],sns.diverging_palette(250, 30,l=65)[-1]]; label='visual'; 
        if effect==1: clusterSIG_start = clusterSIG_start_mus[pm]; clusterSIG_stop = clusterSIG_stop_mus[pm]; p_corr = p_mus[pm]; colors=['mediumpurple','forestgreen']; label='music'
        if effect==2: clusterSIG_start = clusterSIG_start_int[pm]; clusterSIG_stop = clusterSIG_stop_int[pm]; p_corr = p_int[pm]; colors=['mediumpurple','forestgreen']; label='music'
        
        # Scaling
        ymin = 0; ymax = 3.5
        if effect==2: ymin=-0.5; ymax=1
        
        ### PLOT SIGNED F VALUES AS COLORED BACKGROUND, for every PM ###
        ax = fig.add_subplot(gs[m, col])
        
        # Set color map
        if effect==0: cmap = sns.diverging_palette(250, 38, s=100, l=70, as_cmap=True)
        if effect==1: cmap = sns.diverging_palette(280, 130, s=100, l=55, as_cmap=True)
        if effect==2: cmap = plt.cm.get_cmap('BrBG_r')
        c_clusters = np.array([ cmap(0) , cmap(cmap.N)])
        
        # Plot F values as imagesc, with colorbar
        F_obs_to_plot= np.zeros((4,NB_T))
        F_obs_to_plot[:,:Nsamps_before_min] = np.nan; F_obs_to_plot[-Nsamps_before_min:,:] = np.nan
        F_obs_to_plot[:,Nsamps_before_min:-Nsamps_before_min] = np.array([ F_obs[effect,pm,:] , F_obs[effect,pm,:] , F_obs[effect,pm,:] , F_obs[effect,pm,:]  ])
        absMax = np.amax(abs(F_obs[:,:,:]))      # without the last bar issue in diffmus
        im = ax.pcolormesh(t_norm_beat,np.arange(4),F_obs_to_plot,cmap=cmap,vmin=-absMax,vmax=absMax,rasterized=True)
        divider = make_axes_locatable(ax); cax = divider.append_axes('right', size='5%', pad=0.05);  fig.colorbar(im,cax=cax)
        
        # Specify significant clusters with rectangles spanning the cluster time range
        for c in range(len(clusterSIG_start)):
            sign = np.sign(diffMeans[effect,clusterSIG_start[c]:clusterSIG_stop[c],pm])[0]
            if sign == -1: color=c_clusters[0]
            if sign == 1: color=c_clusters[1]
            len_cluster = t_norm_beat[clusterSIG_stop[c]] - t_norm_beat[clusterSIG_start[c]]            
            ax.add_patch(Rectangle((t_norm_beat[clusterSIG_start[c]],0.4),len_cluster, 0.2,edgecolor = 'k',facecolor = color,lw=1,zorder=10))
            
        # Color the silence before/after as gray
        ax.axvspan(t_norm_beat[0], t_norm_beat[Nsamps_before_min], facecolor='lightgray',zorder=10,alpha=0.6); ax.axvspan(t_norm_beat[-Nsamps_before_min], t_norm_beat[-1], facecolor='lightgray',zorder=10,alpha=0.6);
        
        # Set ticks, labels etc
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(ymin,ymax); 
        ax.set_xticks( np.arange(1,105,4) ) 
        ax.set_xlim( ([min(t_norm_beat),max(t_norm_beat)]))
        if m == 0 : 
            ax.set_title(r'$\bf{' + bigtitle[effect] + '}$\n' + 'ANOVA CLUSTERS ON SYNCHRONY OVER TIME',fontsize=15,pad=30)
            x_labels = np.empty(np.arange(1,105,4).shape, dtype('<U21'))
            x_labels[1]='     SILENCE'; x_labels[-2]='       SILENCE';
            x_labels[5]+='DRUMS';  x_labels[9]+='+BASS';  x_labels[13]+='+KEYBOARD';  x_labels[17]+='+VOICE'; x_labels[21]+='+VOICE BIS';
            ax.set_xticklabels( x_labels , fontsize=11 )
            ax.tick_params(axis='x', which='both', length=0)
            ax.xaxis.set_ticks_position('top')
            ax.vlines(np.arange(13,94,16) , ymin,ymax*1.1, color='gray', alpha=0.6, zorder=8 , clip_on=False)
        elif pm == 7 or pm == 14 : 
            x_labels = (np.arange(1,105,4)//4 - 2).astype('<U21')
            x_labels[0] = x_labels[2] = x_labels[-2] = ''; x_labels[1]=''; x_labels[-1]=''
            ax.set_xticklabels( x_labels , fontsize=11 ); ax.set_xlabel('Bar', fontsize=11)
            ax.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=8)
        else : 
            ax.set_xticks([])
            ax.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=8)
       
    
    # Save 
    fig.tight_layout()
    # fig.savefig(output_dir + '/PMspecific-QoM_' + bigtitle[effect] + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/PMspecific-QoM_' + bigtitle[effect] + '.png', dpi=300, bbox_inches='tight'); plt.close() 
    
    
    
