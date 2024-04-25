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
output_dir = os.path.normpath( os.getcwd() + "/results_main_IMS" )
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
print('PSD analysis and FILTERING of THE PMs...')
print('-------------------------------------------')

################ SPECTRAL ANALYSIS OF THE PMs ################
NB_PM = 15

## COMPUTE PSD (Power Spectral Density) using Welch method
# Window size 2s, overlap 75%
Nwin=fps*2
freqs, psd = signal.welch(PM_scores_time[:,:NB_PM],fps,axis=0,nperseg=Nwin,noverlap=3*Nwin//4)

## PLOT PSD
# Normalize every PM's PSD for plot consistency
for pm in range(NB_PM):
    psd /= psd.max(axis=0)[np.newaxis,:]

# Make the plot
matplotlib.use('Agg')
fig = plt.figure(figsize=(15,20));
st = fig.suptitle('Frequency content of the first ' + str(NB_PM) + ' common PMs')
iPlot = 1
for pm in [0,8,1,9,2,10,3,11,4,12,5,13,6,14,7]:
    ax = fig.add_subplot((NB_PM+1)//2,2,iPlot)
    plt.plot(freqs,psd[:,pm],c='tab:blue',label='mean');  
    plt.xticks(np.arange(min(freqs), max(freqs)+1, 5));  ax.set_xlim(0,min(20,fps//2)); 
    plt.title('PM ' + str(pm+1))
    if pm==0: plt.ylabel('PSD (amp**2/Hz)')
    if pm==6: plt.xlabel('Frequency (Hz)')
    iPlot+=1
fig.tight_layout()
# fig.savefig(output_dir + '/PMs_frequency_content.pdf', dpi=600, bbox_inches='tight'); 
fig.savefig(output_dir + '/PMs_frequency_content.png', dpi=300, bbox_inches='tight'); plt.close()  


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
##########               (OPTIONAL)                 ##########
##########  PLOT PMs AS 2D (min/max) STICK FIGURES  ##########
##########           (related to fig 1)             ##########
##############################################################
            
pos_viz=0  # optional
if pos_viz==1:
    print('-------------------------------------------')
    print('PLOTTING THE PMs AS 2D PLOTS...')
    print('-------------------------------------------')
    
    ## VISUALIZE THE FIRST K PMs (2-posture graph with the min and max PM postures)
    NB_PM = 15
    for pm in range(NB_PM):
        # Exaggerate min/max to ease visualization
        EXAG = 1
        if pm in [2]: EXAG = 1.5
        if pm in [3,6,8,10,14]: EXAG = 2
        if pm in [7]: EXAG = 3
        
        # Project the PM back onto the high-dimensional 3D space
        eigenmov = np.outer(EXAG*PM_scores_time[:,pm] , PM_weights_vect[pm,:]).T
        iStart=0; 
        for i in range(NB_DYADS*2) :
            if i in [62,63]: nbTr = NB_TRIALS - 1   # trial missing for dyad32
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
        
        # Find min and max PM postures
        i_min = argmin(PM_scores_time[:,pm]); i_max = argmax(PM_scores_time[:,pm])
        times=[i_min,i_max]
         
        # Call the plot functions of PLmocap (in both Frontal (XZ) and Sagittal (YZ) planes)
        plot_2frames(eigenmov,times,"XZ",liaisons=liaisons,center_sens=13,save_dir=output_dir + '/PM' + str(pm+1) + '_XZ.png'); plt.close()
        plot_2frames(eigenmov,times,"YZ",liaisons=liaisons,center_sens=13,save_dir=output_dir + '/PM' + str(pm+1) + '_YZ.png'); plt.close()


#%% 
##############################################################
#####   COMPUTE INTER-PERSONAL MVOEMENT SYNCHRONY (IMS)  #####
##############################################################

print('-------------------------------------------')
print('COMPUTING INTER-PERSONAL MOVEMENT SYNCHRONY...')
print('-------------------------------------------')

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


## PREPARE XWT (Cross-wavelet transform)
# 1. XWT parameters
NB_OCT = 5                  # Number of octaves   
NB_SCALES_PER_OCT=16        # Number of scales per octave (resolution)
dt=1/fps; dj=1/NB_SCALES_PER_OCT; J=NB_OCT/dj; NB_FREQ = int(J)+1

## 2. Init XWT matrix, to be entered subsequently into the ANOVA
## (timexfreq per PM per dyad, trials averaged for each condition)
IMS_formatJASP = np.zeros( (NB_FREQ,NB_T,NB_PM, NB_DYADS , NB_CONDITIONS) ) 

## LOOP FOR XWT CALCULATION
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
    IMS_dyad = np.zeros(( NB_FREQ, NB_T, NB_PM , NB_TRIALS ))
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
            if subj==0:     # for subject 1
                pos_subj1 = np.dot(xyz_vec.T , PM_weights_vect.T).T          # Get PM score by projecting the trajectory onto the eigen vector
                traj_subj1 = np.gradient( pos_subj1 , 1/fps , axis=1 )          # Compute velocity

                # de-mean
                idx_norm = np.arange(7*fps,7*fps+lensong_diff[0])      # only taking into account when they danced (i.e., not during silence before and after, used only for further padding)
                traj_subj1 = ( traj_subj1 - np.nanmean(traj_subj1[:,idx_norm] , axis = 1).reshape((-1,1)) ) / np.nanstd(traj_subj1[:,idx_norm] , axis = 1).reshape((-1,1))

            if subj==1:   # same but for subject 2
                pos_subj2 = np.dot(xyz_vec.T , PM_weights_vect.T).T
                traj_subj2 = np.gradient( pos_subj2 , 1/fps , axis=1 ) 
                
                # de-mean
                idx_norm = np.arange(7*fps,7*fps+lensong_diff[1])    # only taking into account when they danced (i.e., not during silence before and after, used only for further padding)
                traj_subj2 = ( traj_subj2 - np.nanmean(traj_subj2[:,idx_norm] , axis = 1).reshape((-1,1)) ) / np.nanstd(traj_subj2[:,idx_norm] , axis = 1).reshape((-1,1))

        ####### CROSS-WAVELET TRANSFORM #######
        if True not in np.isnan(traj_subj1[:]):
            print('XWT... Trial ' + str(numTrial))
            for pm in range(NB_PM):
                # Nans sanity checks
                if True in np.isnan(traj_subj1[pm,:]): print('!!!WARNING!!! NaN')
                if True in np.isnan(traj_subj2[pm,:]): print('!!!WARNING!!! NaN')
                data1 = traj_subj1[pm,:]
                data2 = traj_subj2[pm,:]
                
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
                Nsamps_before = int(3 * lensong / 20 )    # Length corresponding to 3 bars (the song has 20 bars)
                power = power[:,7*fps - Nsamps_before:7*fps + lensong + Nsamps_before]
                
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
                IMS_dyad[:,:,pm,tr] = power_ds
  
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
######      ANOVA ANALYSIS ACROSS TIME FOR EACH PM      ######
##############################################################

print('-------------------------------------------')
print('RUNNING THE ANOVA ANALYSIS...')
print('-------------------------------------------')

NB_PM = 15
NB_T = tStop_min + Nsamps_before_min*2

## FORMAT DATA CORRECTLY before entering into the ANOVA across time
# 1. Average the XWT power across periods before running ANOVA on the time-series
X = IMS_formatJASP.mean(axis=0)[:,:NB_PM,:,:]   

# 2. Order the columns in proper format for MNE library
X = np.transpose(X, [2, 0, 1, 3])

# 3. Standardize the XWTs across conditions, within each dyad and PM
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
        F_obs[effect,pm,:], clusters, cluster_p_values, H0  = \
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
######  PLOT THE RESULTS OF THE ANOVA WITH CLUSTERS     ######
######     2. GRAPH SHOWING THE PM-SPECIFIC ANOVAS      ######
######        (with IMS curves, related to fig 2)       ######
##############################################################

matplotlib.use('Agg')   # for very big figure
NB_PM = 15

## SET PARAMETERS of the plot
bigtitle=['VISUAL','MUSIC','INTERACTION']
# minX, minY for the PM stick figures plots
minDim = np.array([ min((PM_min[:,:,0].min(),PM_max[:,:,0].min())) , min((PM_min[:,:,1].min(),PM_max[:,:,1].min())) , 0])
maxDim = np.array([ max((PM_min[:,:,0].max(),PM_max[:,:,0].max())) , max((PM_min[:,:,1].max(),PM_max[:,:,1].max())) , max((PM_min[:,:,2].max(),PM_max[:,:,2].max()))])

## COMPUTE IMS PERIODICITY spectrum (time-averaged XWT) of each PM
PM_spectrums = IMS_formatJASP.mean(axis=1).mean(axis=2).mean(axis=2)

## PLOT
for effect in range(2):   # for main effects Vision or Music
    fig = plt.figure(figsize=(25,20))
    gs = fig.add_gridspec(8 , 7, width_ratios=[3,1,10,0.5,3,1,10])
    
    for pm in range(NB_PM):
        m = pm%8; col=(pm//8)*4
        
        if m == 0 and col==0: 
            axtitle = fig.add_subplot(gs[m, col+3]); axtitle.set_axis_off()
            axtitle.set_title(r'$\bf{' + bigtitle[effect] + '}$\n' + 'ANOVA CLUSTERS ON SYNCHRONY OVER TIME',fontsize=20,pad=30)
        
        if effect==0: clusterSIG_start = clusterSIG_start_vis[pm]; clusterSIG_stop = clusterSIG_stop_vis[pm]; p_corr = pSIG_vis[pm]; colors=[sns.diverging_palette(250, 30,l=65)[0],sns.diverging_palette(250, 30,l=65)[-1]]; label='visual'; 
        if effect==1: clusterSIG_start = clusterSIG_start_mus[pm]; clusterSIG_stop = clusterSIG_stop_mus[pm]; p_corr = pSIG_mus[pm]; colors=['mediumpurple','forestgreen']; label='music'

        ##### FIRST BLOCK: PMs PLOTS #####
        ax1 = fig.add_subplot(gs[m, col+0])
        
        # Adjust the scale along the 3 axis (to make XYZ proportions well visualized)
        Kx = (maxDim[0] - minDim[0])*0.2; Ky = (maxDim[1] - minDim[1])*0.2; Kz = (maxDim[2] - minDim[2])*0.1;

        ax1.set_aspect('equal')
        ax1.get_xaxis().set_visible(False)  
        ax1.get_yaxis().set_visible(False)
        
        # Choose frontal vs. sagittal plane to plot the PM
        if pm in [1,3,7,8,9,13]: dim1=1; dim2=2 # YZ plane (sagittal)
        else:                    dim1=0; dim2=2 # XZ plane (frontal)
        
        ## PLOT STICK FIGURES min/max
        # 1. Scatter plot of the body parts
        for i in range(NB_MARKERS):
            alpha=0.6
            # MIN
            ax1.scatter(PM_min[pm,i,dim1], PM_min[pm,i,dim2],  c='gray', marker='o', s=55,  alpha=alpha,rasterized=True)
            
            # MAX
            ax1.scatter(PM_max[pm,i,dim1], PM_max[pm,i,dim2],  c='black', marker='o', s=55, alpha=alpha,rasterized=True )
        
        # 2. Plot segments
        for l in liaisons:
            c1 = l[0];   c2 = l[1]  # get the two joints
            alpha=1
            # MIN
            ax1.plot([PM_min[pm,c1,dim1], PM_min[pm,c2,dim1]], [PM_min[pm,c1,dim2], PM_min[pm,c2,dim2]], '-', lw=1.5, c='gray',alpha=alpha,rasterized=True)   
            
            # MAX
            ax1.plot([PM_max[pm,c1,dim1], PM_max[pm,c2,dim1]], [PM_max[pm,c1,dim2], PM_max[pm,c2,dim2]], '-', lw=1.5, c='black',alpha=alpha,rasterized=True)   
        
        # 3. Set limits, title, etc
        if dim1==0:  ax1.set_xlim(minDim[dim1]-Kx,maxDim[dim1]+Kx); ax1.invert_xaxis()
        if dim1==1 : ax1.set_xlim(minDim[dim1]-Ky,maxDim[dim1]+Ky);  
        ax1.set_axis_off()
        ax1.set_title('PM ' + str(pm+1) + '\na=' + format(EXAG[pm],'g'),rotation='horizontal',x=-0.3,y=0.5,fontsize=20)
        
        
        ##### SECOND BLOCK: IMS PERIODICITY #####
        ax2 = fig.add_subplot(gs[m, col+1])
        
        means_pm = np.array([ PM_spectrums[:,pm] , PM_spectrums[:,pm] ])
        minSpec = means_pm.min();  maxSpec = means_pm.max()
        ax2.pcolormesh(np.arange(2),np.log2(freqs_per_beat),means_pm.T,vmin=minSpec,vmax=maxSpec,rasterized=True)
        
        # Set params
        ax2.set_xlim((-0.5,0.5)); ax2.set_xticks([]);
        ax2.set_yticks( np.array([2,1,0,-1,-2,-3,-4]) ) 
        ax2.set_yticklabels( ['0.25','0.5','1','2','4','8','16'] , fontsize = 12)
        if m == 0: ax2.set_title('SYNCHRONY\nSPECTRUM',fontsize=15,pad=15)
        if m==0: ax2.set_ylabel('Period = beat *' , fontsize = 15)
        ax2.set_ylim(np.log2([freqs_per_beat.min(),freqs_per_beat.max()]))
        
        
        ##### THIRD BLOCK: IMS CURVES and F VALUES #####
        ax3 = fig.add_subplot(gs[m, col+2])
        
        # 1. Plot IMS curves across conditions
        if effect==0:
            mean_high = (X[:,:,pm,0].mean(axis=0) + X[:,:,pm,1].mean(axis=0))/2 ; label_high = 'YesVis'
            mean_low = (X[:,:,pm,2].mean(axis=0) + X[:,:,pm,3].mean(axis=0))/2 ; label_low = 'NoVis'
        
        if effect==1:
            mean_high = (X[:,:,pm,0].mean(axis=0) + X[:,:,pm,2].mean(axis=0))/2 ; label_high = 'SameMus'
            mean_low = (X[:,:,pm,1].mean(axis=0) + X[:,:,pm,3].mean(axis=0))/2 ; label_low = 'DiffMus'
            
        ax3.plot(t_norm_beat,mean_high,label=label_high,c='black')
        ax3.plot(t_norm_beat,mean_low,label=label_low,c='black', linestyle='dashed')
        ymin = 0; ymax = max(np.hstack((mean_high , mean_low)))*1.1
        
        # 2. Plot the clusters as shaded regions
        for c in range(len(clusterSIG_start)):
            sign = np.sign(diffMeans[effect,clusterSIG_start[c]:clusterSIG_stop[c],pm])[0]
            if sign == -1: color=colors[0]
            if sign == 1: color=colors[1]
            ax3.axvspan(t_norm_beat[clusterSIG_start[c]], t_norm_beat[clusterSIG_stop[c]], facecolor=color, alpha=0.4)
            ax3.text(t_norm_beat[clusterSIG_start[c]], ymax/2, "p = " + str(p_corr[c]), rotation=90, verticalalignment='center',style='italic')
        if (m==0):    # for the legend
            ax3.axvspan(0,0,0 , facecolor=colors[0], alpha=0.6, label=label_high+' < '+label_low) 
            ax3.axvspan(0,0,0 , facecolor=colors[1], alpha=0.6, label=label_high+' > '+label_low) 
        ax3.axvspan(t_norm_beat[0], t_norm_beat[Nsamps_before_min], facecolor='lightgray',zorder=10,alpha=0.6); ax3.axvspan(t_norm_beat[-Nsamps_before_min], t_norm_beat[-1], facecolor='lightgray',zorder=10,alpha=0.6);
        
        # 3. Set labels, lims, params...
        ax3.tick_params(axis='y', labelsize=12)
        ax3.set_ylim(ymin,ymax); 
        ax3.set_xticks( np.arange(1,105,4) ) 
        ax3.set_xlim( ([min(t_norm_beat),max(t_norm_beat)]))
            
        if m == 0 : 
            x_labels = np.empty(np.arange(1,105,4).shape, dtype('<U21'))
            x_labels[1]='     SILENCE'; x_labels[-2]='       SILENCE';
            x_labels[5]+='DRUMS';  x_labels[9]+='+BASS';  x_labels[13]+='+KEYBOARD';  x_labels[17]+='+VOICE'; x_labels[21]+='+VOICE BIS';
            ax3.set_xticklabels( x_labels , fontsize=11 )
            ax3.tick_params(axis='x', which='both', length=0)
            ax3.xaxis.set_ticks_position('top')
            ax3.vlines(np.arange(13,94,16) , ymin,ymax*1.1, color='gray', alpha=0.6, zorder=20 , clip_on=False)
            ax3.legend(loc='upper left' , fontsize = 12).set_zorder(30)
        elif pm == 7 or pm == 14 : 
            x_labels = (np.arange(1,105,4)//4 - 2).astype('<U21')
            x_labels[0] = x_labels[2] = x_labels[-2] = ''; x_labels[1]=''; x_labels[-1]=''
            ax3.set_xticklabels( x_labels , fontsize=11 ); ax3.set_xlabel('Bar', fontsize=11)
            ax3.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=20)
        else : 
            ax3.set_xticks([])
            ax3.vlines(np.arange(13,94,16) , ymin,ymax, color='gray', alpha=0.6, zorder=20)
            

    # Save
    fig.tight_layout()
    # fig.savefig(output_dir + '/PMspecific-IMS1_' + bigtitle[effect] + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/PMspecific-IMS1_' + bigtitle[effect] + '.png', dpi=300, bbox_inches='tight'); plt.close() 
         
        
        
#%% 
##############################################################
#####   PLOT THE RESULTS OF THE ANOVA WITH CLUSTERS      #####
#####     3. GRAPH SHOWING THE PM-SPECIFIC ANOVAS        #####
#####   (with F-value imagesc, related to  fig S2)       #####
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
        absMax = np.amax(abs(F_obs))
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
    # fig.savefig(output_dir + '/PMspecific-IMS2_' + bigtitle[effect] + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/PMspecific-IMS2_' + bigtitle[effect] + '.png', dpi=300, bbox_inches='tight'); plt.close() 
    
        
#%% 
##############################################################
######   GEOMETRY OF IMS - ANALYSIS 1: 3D spatial maps  ######
######                (related to fig 3A)               ######
##############################################################

# Output dir (to store results)
output_dir = os.path.normpath( os.getcwd() + "/results_main_Geometry" )
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)


NB_PM=15

# We know what PMs are associated to either partner-driven, musicr-driven, or hybrid IMS
PM_vis = np.array([0,2,4,10]); PM_mus = np.array([1,3,7,8,13]); PM_both=[9]; PM_rest = [5,6,11,12]

## COMPUTE SPATIAL MAP along X, Y, Z for each PM
# We compute the mean velocity of the PM along XYZ, for each dyad, each PM (averaged across trials)
dir_mov_tr = np.zeros( (NB_PM , NB_DYADS*2, 3,  NB_TRIALS) )
for pm in range(NB_PM):
    # Project PM back onto the high-dimensional 3D space
    eigenmov = np.outer(PM_scores_time[:,pm] , PM_weights_vect[pm,:]).T
    iStart=0;
    for i in range(NB_DYADS*2) :
        if i in [62,63]: nbTr = NB_TRIALS - 1;  dir_mov_tr[pm,i,:,-1] = np.nan
        else: nbTr = NB_TRIALS
        for tr in range(nbTr):
            songsTrial = song[:,i//2,tr]
            tStop = int( min(musParts_tFrames[songsTrial , -1]) * fps ) # take the end frame of the shortest song between two subjects    
            pos_resh = np.reshape( eigenmov[:,iStart:iStart+tStop] ,(NB_MARKERS , 3 , eigenmov[:,iStart:iStart+tStop].shape[1]) )
            
            # Compute velocity
            vel = np.gradient( pos_resh , 1/fps , axis=2 ) 
            # We take the Quantity of Motion (QoM), i.e., abs(velocity), as a proxy of where the body moved
            # For this trial: Average QoM across time, and across body parts, for X,Y,Z separately
            dir_mov_tr[pm,i,:,tr] = np.mean(np.mean(abs(vel),axis=-1),axis=0)        # this gives one QoM value for X, one for Y, one for Z
            
            iStart += tStop
            
# Average these 3D spatial maps across trials (so one XYZ value per dyad and PM)
dir_mov = np.nanmean(dir_mov_tr,axis=-1) 

# Standardize to reduce inter-individual variability
for i in range(NB_DYADS*2):
    dir_mov[:,i,:] /= dir_mov[:,i,:].std()

# SET COLORS/SIZES for scatterplot (each PM of each dyad is a small dot, the average across dyads is a big dot, each PM class is colored)
colors_vis=np.array([(255,207,0),(255,169,30),(255,140,0),(223,120,0)])/255
colors_mus=np.array([(164,251,166),(74,229,74),(48,203,0),(15,146,0),(0,98,3)])/255
colors_pm10=['royalblue']
colors=[colors_vis[0],colors_mus[0],colors_vis[1],colors_mus[1],colors_vis[2],colors_mus[2],colors_mus[3],colors_pm10[0],colors_vis[3],colors_mus[4]]
PMs_subset = np.array([0,1,2,3,4,7,8,9,10,13])
edgecolors=['tomato','forestgreen','tomato','forestgreen','tomato','forestgreen','forestgreen','royalblue','tomato','forestgreen']
sizes=[200,200,200,200,200,200,200,200,200,200]
sizes_small=[40,40,40,40,40,40,40,40,40,40]

## PREPARE FRAME/SCALING
PMs_interest = np.setxor1d(np.arange(15),np.array(PM_rest))
minX = np.min(dir_mov[PMs_interest,:,0]); minY = np.min(dir_mov[PMs_interest,:,1]); minZ = np.min(dir_mov[PMs_interest,:,2])
maxX = np.max(dir_mov[PMs_interest,:,0]); maxY = np.max(dir_mov[PMs_interest,:,1]); maxZ = np.max(dir_mov[PMs_interest,:,2])


## MAKE MAIN SCATTER PLOT 
fig = plt.figure(figsize=(20,10))
ax  = fig.add_subplot(111, projection='3d',computed_zorder=False)
ax.view_init(17, -115)

# 1. Plot spatial map of the PMs for every participants (small dots)
for pm in range(NB_PM):
    if pm in PMs_subset:
        pm_sub = np.where(PMs_subset == pm)[0][0]
        ax.scatter(dir_mov[pm,:,0],dir_mov[pm,:,1],dir_mov[pm,:,2],linewidth=1.5, s=sizes_small[pm_sub], marker='o', facecolor=colors[pm_sub],depthshade=True,zorder=90,rasterized=True)

# 2. Plot spatial map of the PMs averaged across participants (big dots)
dir_mov_mean = dir_mov.mean(axis=1)     # average across dyads, for visualization
for pm in range(NB_PM):
    if pm in PMs_subset:
        pm_sub = np.where(PMs_subset == pm)[0][0]
        ax.scatter(dir_mov_mean[pm,0],dir_mov_mean[pm,1],dir_mov_mean[pm,2], s=sizes[pm_sub], linewidth=1.75, marker='o',facecolor=to_rgba('white', 0.9),edgecolor=colors[pm_sub],depthshade=False,zorder=100,rasterized=True)

# 3. Adjust plot
ax.zaxis.set_rotate_label(False)
ax.set_xlabel('Mediolateral',fontsize=14,labelpad=7); ax.set_ylabel('Anteroposterior',fontsize=14,labelpad=7); ax.set_zlabel('Vertical',fontsize=14,labelpad=7,rotation=90);  
ax.set_xlim(minX,5.2); ax.set_ylim(minY,7); ax.set_zlim(minZ,1.4);
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

#### SAME BUT IN SMALLER INSET #####
# Inset scatter plot of larger view (including some outliers that make the axes too large)
rect = [0.7, 0.7, 0.3, 0.3]
ax_inset1 = fig.add_axes(rect, anchor='NW', projection='3d',computed_zorder=False)
ax_inset1.view_init(17, -107)

# 1. Plot spatial map of the PMs for every participants (small dots)
for pm in range(15):
    if pm in PMs_subset:
        pm_sub = np.where(PMs_subset == pm)[0][0]
        ax_inset1.scatter(dir_mov[pm,:,0],dir_mov[pm,:,1],dir_mov[pm,:,2], s=1, marker='o', facecolor=colors[pm_sub],depthshade=True,zorder=90,rasterized=True)

# 2. Plot spatial map of the PMs averaged across participants (big dots)
for pm in range(15):
    if pm in PMs_subset:
        pm_sub = np.where(PMs_subset == pm)[0][0]
        ax_inset1.scatter(dir_mov_mean[pm,0],dir_mov_mean[pm,1],dir_mov_mean[pm,2], s=15, linewidth=0.7, marker='o',facecolor=to_rgba('white', 0.9),edgecolor=colors[pm_sub],depthshade=False,zorder=100,rasterized=True)
ax_inset1.set_xticks(np.arange(0,maxX,2)); ax_inset1.set_yticks(np.arange(0,maxY,2)); ax_inset1.set_zticks(np.arange(0,maxZ,1))

# 3. Adjust plot
ax_inset1.set_xlim(minX,maxX); ax_inset1.set_ylim(minY,maxY); ax_inset1.set_zlim(minZ,2.7);
ax_inset1.w_xaxis.pane.fill = False
ax_inset1.w_yaxis.pane.fill = False
ax_inset1.w_zaxis.pane.fill = False
ax_inset1.patch.set_linewidth(1)
ax_inset1.patch.set_edgecolor('black')

# Save
fig.savefig(output_dir + '/Geometry1_PMs_XYZ.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry1_PMs_XYZ.png', dpi=300, bbox_inches='tight'); plt.close() 


    
#%% 
##############################################################
######    GEOMETRY OF IMS ANALYSIS 2: EIGENDIRECTIONS   ######
######             --> 2d PCA and classifier            ######
######            (related to figs 3B, 4 and S4)        ######
##############################################################

NB_PM = 15

# We know what PMs are associated to either partner-driven, music-driven, or hybrid IMS
PM_vis = np.array([0,2,4,10]); PM_mus = np.array([1,3,7,8,13]); PM_both=[9]; PM_rest = [5,6,11,12]

## COMPUTE SPATIAL MAP along X, Y, Z for each PM, BUT THIS TIME NOT AVERAGING ACROSS BODY PARTS!!
# We compute the mean velocity of the PM along XYZ and body parts, for each dyad, each PM (averaged across trials)
dir_mov_tr_markers = np.zeros( (NB_PM , NB_DYADS*2, NB_MARKERS, 3,  NB_TRIALS) )
PM_label = np.zeros( (NB_PM , NB_DYADS*2) ); 
for pm in range(NB_PM):
    # Project the PM back onto the high-dimensional 3D space
    eigenmov = np.outer(PM_scores_time[:,pm] , PM_weights_vect[pm,:]).T
    iStart=0; 
    for i in range(NB_DYADS*2) :
        if i in [62,63]: nbTr = NB_TRIALS - 1;  dir_mov_tr_markers[pm,i,:,:,-1] = np.nan
        else: nbTr = NB_TRIALS
        for tr in range(nbTr):
            songsTrial = song[:,i//2,tr]
            tStop = int( min(musParts_tFrames[songsTrial , -1]) * fps ) # take the end frame of the shortest song between two subjects    
            pos_resh = np.reshape( eigenmov[:,iStart:iStart+tStop] ,(NB_MARKERS , 3 , eigenmov[:,iStart:iStart+tStop].shape[1]) )
           
            # Compute velocity
            vel = np.gradient( pos_resh , 1/fps , axis=2 )
            # We take the Quantity of Motion (QoM), i.e., abs(velocity), as a proxy of where the body moved
            # For this trial: Average velocity across time, BUT NOT ACROSS BODY PARTS
            dir_mov_tr_markers[pm,i,:,:,tr] = np.mean(abs(vel) , axis=-1)
            
            # Store whether the PM is associated to partner-driven, music-driven, or hybrid IMS, for subsequent analyses
            if pm in PM_vis: PM_label[pm,i] = 0
            elif pm in PM_mus: PM_label[pm,i] = 1
            elif pm == 9 : PM_label[pm,i] = 2
            else: PM_label[pm,i] = -1
            
            iStart += tStop

# Average these 3D body-part spatial maps across trials (so one 66-dimensional vector per dyad and PM)
dir_mov_markers = np.nanmean(dir_mov_tr_markers,axis=-1)

# Standardize to reduce inter-individual variability
for i in range(NB_DYADS*2):
    dir_mov_markers[:,i,:,:] /= dir_mov_markers[:,i,:,:].std()
dir_mov_markers = np.reshape( dir_mov_markers , (NB_PM*NB_DYADS*2 , NB_MARKERS*3 ))
PM_label = np.reshape( PM_label , (NB_PM * NB_DYADS * 2) )


##### EXTRACT EIGENDIRECTIONS ######
## SET THE MATRIX "D" FOR PCA 
# Combine spatial maps of PMs of interest (of either partner-driven, music-driven, or hybrid IMS)
# (66 data-points (22 body parts * 3 axes) x 700 features (10 PMs * 70 participants))
D = dir_mov_markers[(PM_label==0) | (PM_label==1) | (PM_label==2),:].T

# Demean the features
D_demean = D - np.mean(D,axis=0)[np.newaxis,:]

## APPLY PCA
U, S, V = np.linalg.svd(D_demean, full_matrices=False)
eigenval_eigendir = S**2                                                             # eigenvalues
eigendir_var_explained = np.cumsum(eigenval_eigendir) / np.sum(eigenval_eigendir)    # variance explained
eigendir_weights = U*S
eigendir_scores = V

## SET SIGN convention for visualizing EDs: flip them so that the biggest activiations are positiive
sflip=np.zeros((66))
for pc in range(66):
    sflip[pc]=np.sign(max(eigendir_scores[pc,:].min(), eigendir_scores[pc,:].max(), key=abs))
eigendir_weights *= sflip[np.newaxis,:]
eigendir_scores *= sflip[:,np.newaxis]

## RE-ORDER THE EIGENDIRECTION BODY-PART WEIGHTS in Xhead,Xshoulder,... Y... Z.... (for plots)
idx_x = [i for i, e in enumerate(range(NB_MARKERS*3)) if e%3==0]
idx_y = [i for i, e in enumerate(range(NB_MARKERS*3)) if e%3==1]
idx_z = [i for i, e in enumerate(range(NB_MARKERS*3)) if e%3==2]
marker_idx = np.concatenate((idx_x,idx_y,idx_z))
eigendir_weights = eigendir_weights[marker_idx,:]

## PLOT MATRIX "D" OF SPATIAL MAPS
fig=plt.figure()
plt.imshow(D_demean[marker_idx,:],extent=[0,100,0,1], aspect=100,rasterized=True)
# plt.savefig(output_dir + '/Geometry2_matrixD.pdf', dpi=600, bbox_inches='tight');
plt.savefig(output_dir + '/Geometry2_matrixD.png', dpi=300, bbox_inches='tight'); plt.close() 

## PLOT VARIANCE EXPLAINED by the first 20 eigendirections
fig = plt.figure()
plt.bar(np.arange(20),eigendir_var_explained[:20]*100,facecolor='w',edgecolor='k'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Geometry2_eigendir_varexplained.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry2_eigendir_varexplained.png', dpi=300, bbox_inches='tight'); plt.close() 


##### CLASSIFICATION ######
print('LOOO Cross-validation...')
## SET RIDGE LOGISTIC REGRESSION 'one-versus-rest' to test how eigendirections can predict the PM class
y_train = PM_label[(PM_label==0) | (PM_label==1) | (PM_label==2)]       # Labels (PM class)
Cs = np.hstack((10**np.arange(-3.0,4.0)))                               # C parameter to optimize/validate
NB_ed_CV = 10; NB_C_CV = len(Cs)                                        # Number of eigendirections and C values to optimize/cross validate
score_cv_ed = np.zeros((NB_ed_CV , NB_C_CV))
score_cv_ed_err = np.zeros((NB_ed_CV , NB_C_CV))
for ed in range(1,NB_ed_CV+1):                                          # Find optimal number of eigendirection, and C value; through cross validation
    print('Eigendirection : ' + str(ed))
    for i in range(NB_C_CV):
        print('C param: ' + str(Cs[i]))
        weights_mean, intercept_mean, conf_mean, proba_mean, fold_scores, class_model = class_logReg(eigendir_scores[:ed,:].T, y_train, NB_DYADS*2, ['Visual','Music','Visual+Music'], C=Cs[i],multiclass='ovr')
        score_cv_ed[ed-1,i] = fold_scores.mean()
        score_cv_ed_err[ed-1,i]= fold_scores.std()/sqrt(len(fold_scores))
 
## FIND OPTIMAL C parameter and eigendirection number
maxAcc = np.max(score_cv_ed)                            # Find max accuracy
idx_min = np.argmin(np.where(score_cv_ed==maxAcc)[0])   # Find it for the smallest eigendirection value (ie, it reaches the max accuracy the fastest)
ed_opt = np.where(score_cv_ed==maxAcc)[0][idx_min] + 1  # Optimal eigendirection number
c_opt = Cs[np.where(score_cv_ed==maxAcc)[1][idx_min]]   # Optimal C value   


## TEST CLASSIFIER ON SINGLE EIGENDIRECTIONS (related to fig 4B)
print('Test accuracy of classifier for the different eigendirections...')
# Compute classification accuracy
score_ed_alone = np.zeros((NB_ed_CV)); score_ed_alone_err = np.zeros((NB_ed_CV))
for ed in range(1,NB_ed_CV+1):
    weights_mean, intercept_mean, conf_mean, proba_mean, fold_scores, class_model = class_logReg(eigendir_scores[ed-1:ed,:].T, y_train, NB_DYADS*2, ['Visual','Music','Visual+Music'], C=c_opt,multiclass='ovr')

    score_ed_alone[ed-1] = fold_scores.mean()
    score_ed_alone_err[ed-1] = fold_scores.std()/sqrt(len(fold_scores))

# Plot classification accuracy
fig = plt.figure()
plt.bar(np.arange(NB_ed_CV),score_ed_alone[:NB_ed_CV]*100,edgecolor='k',facecolor='w', capsize=2)
plt.axhline(33,c='gray',linestyle='--'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Geometry2_eigendir_classifindiv.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry2_eigendir_classifindiv.png', dpi=300, bbox_inches='tight'); plt.close() 

## SAME BUT NOW classification accuracy when using the 4 selective EDs (based on their activation + their 4 biggest classif accuracy) (related to fig 4B)
# Classif accuracy of the selective EDs together
selEDs=np.array([0,1,4,5])
weights_mean_selEDs, intercept_mean_selEDs, conf_mean_selEDs, proba_mean_selEDs, fold_scores_selEDs, class_model_selEDs = class_logReg(eigendir_scores[selEDs,:].T, y_train, NB_DYADS*2, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_selEDs = fold_scores_selEDs.mean()
score_selEDs_err = fold_scores_selEDs.std()/sqrt(len(fold_scores_selEDs))

# Classification accuracy of the other EDs together
restEDs=np.setxor1d(np.array([0,1,4,5]),np.arange(ed_opt))
weights_mean_restEDs, intercept_mean_restEDs, conf_mean_restEDs, proba_mean_restEDs, fold_scores_restEDs, class_model_restEDs = class_logReg(eigendir_scores[restEDs,:].T, y_train, NB_DYADS*2, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_restEDs = fold_scores_restEDs.mean()
score_restEDs_err = fold_scores_restEDs.std()/sqrt(len(fold_scores_restEDs))

# Classification accuracy of the full optimal model
weights_mean_opt, intercept_mean_opt, conf_mean_opt, proba_mean_opt, fold_scores_opt, class_model_opt = class_logReg(eigendir_scores[:ed_opt,:].T, y_train, NB_DYADS*2, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_opt = fold_scores_opt.mean()
score_opt_err = fold_scores_opt.std()/sqrt(len(fold_scores_opt))

# Plot classification accuracy for the full optimal, selEDs and restEDs models
fig = plt.figure()
plt.bar(np.arange(3),np.hstack((score_opt,score_selEDs,score_restEDs))*100,capsize=2,facecolor='w',edgecolor='k')
plt.axhline(33,c='gray',linestyle='--'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Geometry2_eigendir_classiftogehter.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry2_eigendir_classiftogehter.png', dpi=300, bbox_inches='tight'); plt.close() 

# Plot confusion matrix of the selective model
fig = plt.figure()
sns.heatmap(conf_mean_selEDs*100, vmin=0, vmax=100, cmap=sns.color_palette("Blues", as_cmap=True), cbar=True,annot=True, cbar_kws={"orientation": "horizontal"}, square = True)
# fig.savefig(output_dir + '/Geometry2_eigendir_classifconf-selective.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry2_eigendir_classifconf-selective.png', dpi=300, bbox_inches='tight'); plt.close() 


#%% 
##############################################################
######    GEOMETRY OF IMS ANALYSIS 2: EIGENDIRECTIONS   ######
######            PLOT 1: eigendirection scores         ######
######           (related to fig 3B and fig S4)         ######
##############################################################

# Keep only the first 6 EDs
NB_ED = 6

## PLOT EIGENDIRECTION SCORES for the different PMs and participants
# as matrix/imshow
absMax = np.amax(abs(eigendir_scores[:6,:])); absMin = np.amin(abs(eigendir_scores[:6,:]))
fig=plt.figure()
plt.imshow(eigendir_scores[:6,:],extent=[0,100,0,1], aspect=50,rasterized=True)
# plt.savefig(output_dir + '/EDscores_imshow.pdf', dpi=600, bbox_inches='tight');
plt.savefig(output_dir + '/EDscores_imshow.png', dpi=300, bbox_inches='tight'); plt.close() 

# as bars, ranked by eigendirection score, and colored by PM class
pm_num = np.repeat(np.arange(10),70)
colors_vis=np.array([(255,207,0),(255,169,30),(255,140,0),(223,120,0)])/255
colors_mus=np.array([(164,251,166),(74,229,74),(48,203,0),(15,146,0),(0,98,3)])/255
colors_pm10=['royalblue']
colors=[colors_vis[0],colors_mus[0],colors_vis[1],colors_mus[1],colors_vis[2],colors_mus[2],colors_mus[3],colors_pm10[0],colors_vis[3],colors_mus[4]]

for ed in range(NB_ED):
    fig = plt.figure(figsize=(40,20))
    ind_sort = np.argsort(eigendir_scores[ed,:])[::-1]
    label_PM = pm_num[ind_sort].astype(int)
    bars = plt.bar(np.arange(700),eigendir_scores[ed,ind_sort],width=0.7,rasterized=True)
    i=0
    for item in bars:
        item.set_color(colors[label_PM[i]])
        i+=1
    # fig.savefig(output_dir + '/Geometry2_eigendir_scores' + str(ed+1) + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/Geometry2_eigendir_scores' + str(ed+1) + '.png', dpi=300, bbox_inches='tight'); plt.close() 


## PLOT EIGENDIRECTION SCORES for the different PMs (BUT AVERAGED ACROSS PARTICIPANTS)
eigendir_scores_meanSubj = np.zeros((66,10))
for pm in range(10):
    eigendir_scores_meanSubj[:,pm] = eigendir_scores[:,pm*70:(pm+1)*70].mean(axis=1)

pm_num_meanSubj=np.arange(10)    
for pc in range(NB_ED):
    fig = plt.figure(figsize=(20,20))
    ind_sort = np.argsort(eigendir_scores_meanSubj[pc,:])[::-1]
    label_PM = pm_num_meanSubj[ind_sort].astype(int)
    bars = plt.bar(np.arange(10),eigendir_scores_meanSubj[pc,ind_sort],width = 0.3)
    i=0
    for item in bars:
        item.set_color(colors[label_PM[i]])
        i+=1
    # fig.savefig(output_dir + '/Geometry2_eigendir_scores_meanSubj' + str(pc+1) + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/Geometry2_eigendir_scores_meanSubj' + str(pc+1) + '.png', dpi=300, bbox_inches='tight');  plt.close() 


## SAVE A PLOT WITH ONLY 1 BAR PER COLOR, just to know the colorcode legend
fig = plt.figure(figsize=(40,20))
idx = np.arange(0,700,70)
bars = plt.bar(np.arange(10),np.ones(10),width = 0.3)
i=0
for item in bars:
    item.set_color(colors[i])
    i+=1
# fig.savefig(output_dir + '/Geometry2_eigendir_scores_legend.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Geometry2_eigendir_scores_legend.png', dpi=300, bbox_inches='tight'); plt.close() 
    
    
    
#%% 
##############################################################
######    GEOMETRY OF IMS ANALYSIS 2: EIGENDIRECTIONS   ######
######           PLOT 2: eigendirection weights         ######
######           (related to fig 3b and fig S4)         ######
##############################################################

label_mod = ['Partner-driven','Music-driven','Both (PM 10)']
label_dim = ['Mediolateral','Anteroposterior','Vertical']
NB_ED_plot = 6                                               # Number of eigendirections to plot

# Prepare custom colormap from Gray to Pink (merging Gray-to-Red and Pink-to-Green)
import matplotlib.colors as colors
from typing import List, Tuple
div_cmaps: List[List[str]] = []
for cmap_name in ['PiYG_r','RdGy']:
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.5, b=1),
        cmap(np.linspace(0.5, 1)))
    
    cmap_list = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    div_cmaps.append(cmap_list)
div_cmaps[1] = list(reversed(div_cmaps[1]))
cmap_nodata_list = div_cmaps[1] + div_cmaps[0]
new_cmap = colors.ListedColormap(cmap_nodata_list)


## PLOT EIGENDIRECTION WEIGHTS AS MATRIX (IMSHOW)
absMax = np.amax(abs(eigendir_weights[:,:NB_ED_plot])); absMin = np.amin(abs(eigendir_weights[:,:NB_ED_plot]))
fig=plt.figure()
plt.imshow(eigendir_weights[:,:NB_ED_plot],extent=[0,100,0,1], aspect=200,vmin=-absMax,vmax=absMax,cmap=new_cmap,rasterized=True)
# plt.savefig(output_dir + '/Geometry2_eigendir_weights_matrix.pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_dir + '/Geometry2_eigendir_weights_matrix.png', dpi=300, bbox_inches='tight'); plt.close()



## PLOT EIGENDIRECTION WEIGHTS AS SKELETON/STICK FIGURE
# 1. Build skeleton for stick figure 
# taking the average posture across participants
skeleton = pmean_tr[:,:,0].mean(axis=0).reshape((-1,1))
skeleton = np.reshape( skeleton ,(NB_MARKERS , 3 ) )
data = skeleton.copy()

# 2. centering it on the pelvis body part
Ncenter = 13     # pelvis marker
Ox = data[Ncenter,0];   Oy = data[Ncenter,1];   Oz = data[Ncenter,2]
maxX = np.abs(data[:,0]-Ox).max(); maxY = np.abs(data[:,1]-Oy).max(); maxZ = np.abs(data[:,2]-Oz).max()
Kx = 2.2; Ky = 2; Kz = 0.6;             # Adjust scaling XYZ to your convenience for the plot

# 3. Actually do the skeleton plot
fig = plt.figure(figsize=(20,24))
iPlot=1
for ed in range(NB_ED_plot):
    # Create one figure/file every 3 eigendirections (so for 6 eigendirections,it will generate 2 files)
    if ed%3==0: 
        plt.subplots_adjust(wspace=-0.4)
        plt.subplots_adjust(hspace=-0.25)
        fig = plt.figure(figsize=(20,24)); iPlot=1
    # Prepare colormap and its scaling from -absMax to absMax
    absMax = np.amax(abs(eigendir_weights[:,ed])); absMin = np.amin(abs(eigendir_weights[:,ed]))
    weights_map=linspace(-absMax,absMax,100)
    norm = plt.Normalize(-absMax, absMax)
    
    # Plot one skeleton for each ed AND EACH SPATIAL AXIS
    for dim in range(3):
        ax = fig.add_subplot(3,3,iPlot,projection='3d')
        ax.view_init(20,124)
        ax.grid(False)
        
        # Scatterplot body parts, colored depending on their eigendirection weight value
        s=ax.scatter(xs=data[:,0], ys=data[:,1], zs=data[:,2], alpha=1, marker='o', s=130, 
                     c=eigendir_weights[dim*NB_MARKERS+np.arange(NB_MARKERS),ed], edgecolor='black',cmap=new_cmap,vmin=-absMax,vmax=absMax)
        ax.set_xlim(Ox-Kx*maxX,Ox+Kx*maxX); ax.set_ylim(Oy-Ky*maxY,Oy+Ky*maxY); ax.set_zlim(Oz-Kz*maxZ,Oz+Kz*maxZ)
        
        # Plot of the segments linking each body part
        for l in liaisons :
            c1 = l[0];   c2 = l[1]
            ax.plot([data[c1,0], data[c2,0]], [data[c1,1], data[c2,1]], [data[c1,2], data[c2,2]], '-', lw=1, c='black')
            ax.set_axis_off()
            ax.patch.set_visible(False)

        # Uncomment if you need colorbar
        # if dim==2: fig.colorbar(s, ax=ax)
            
        iPlot+=1
            
plt.subplots_adjust(wspace=-0.4)
plt.subplots_adjust(hspace=-0.25)
# plt.savefig(output_dir + '/Geometry2_eigendir_weights2.pdf', dpi=600, bbox_inches='tight');
plt.savefig(output_dir + '/Geometry2_eigendir_weights2.png', dpi=300, bbox_inches='tight'); plt.close() 
# plt.savefig(output_dir + '/Geometry2_eigendir_weights1.pdf', dpi=600, bbox_inches='tight');
plt.savefig(output_dir + '/Geometry2_eigendir_weights1.png', dpi=300, bbox_inches='tight'); plt.close() 



#%% 
##############################################################
######    TEMPORAL ORGANIZATION OF IMS: EIGENPERIODS    ######
######        2d PCA + classifier on periodicity        ######
######     Code is same as above, but on periodicity    ######
######         (related to fig 4 and fig S4)            ######
##############################################################

# Output dir (to store results)
output_dir = os.path.normpath( os.getcwd() + "/results_main_Temporal" )
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)

NB_PM = 15

# We know what PMs are associated to either partner-driven, music-driven, or hybrid IMS
PM_vis = np.array([0,2,4,10]); PM_mus = np.array([1,3,7,8,13]); PM_both=[9]; PM_rest = [5,6,11,12]


## COMPUTE IMS PERIODICITY MAP of each PM, i.e. the time-averaged XWT
periodicity = np.transpose( IMS_formatJASP.mean(axis=1).mean(axis=-1) , [1,2,0] )
periodicity = periodicity[:NB_PM,:,:]
PM_label = np.zeros( (NB_PM , NB_DYADS) ); 
for pm in range(NB_PM):
    for i in range(NB_DYADS) :
        # Store whether the PM is associated to partner-driven, music-driven, or hybrid IMS, for subsequent analyses
        if pm in PM_vis: PM_label[pm,i] = 0
        elif pm in PM_mus: PM_label[pm,i] = 1
        elif pm == 9 : PM_label[pm,i] = 2
        else: PM_label[pm,i] = -1

# Standardize to reduce inter-individual variability
for i in range(NB_DYADS):
    periodicity[:,i,:] /= periodicity[:,i,:].std()
periodicity = np.reshape( periodicity, (NB_PM*NB_DYADS , NB_FREQ ))
PM_label = np.reshape( PM_label , (NB_PM * NB_DYADS) )


##### EXTRACT EIGENPERIODS ######
## SET THE MATRIX "Dp" FOR PCA 
# Combine periodicity maps of PMs of interest (of either partner-driven, music-driven, or hybrid IMS)
# (83 data-points (83 periods) x 350 features (10 PMs * 35 dyads))
Dp = periodicity[(PM_label==0) | (PM_label==1) | (PM_label==2),:].T

# Demean the features
Dp_demean = Dp - np.mean(Dp,axis=0)[np.newaxis,:]

## APPLY PCA
U, S, V = np.linalg.svd(Dp_demean, full_matrices=False)
eigenval_eigenper = S**2                                                             # eigenvalues
eigenper_var_explained = np.cumsum(eigenval_eigenper) / np.sum(eigenval_eigenper)    # variance explained
eigenper_weights = U*S
eigenper_scores = V

## SET SIGN convention for visualizing EPs: flip them so that the biggest activiations are positiive
sflip=np.zeros((NB_FREQ))
for pc in range(NB_FREQ):
    sflip[pc]=np.sign(max(eigenper_scores[pc,:].min(), eigenper_scores[pc,:].max(), key=abs))
eigenper_weights *= sflip[np.newaxis,:]
eigenper_scores *= sflip[:,np.newaxis]

## PLOT VARIANCE EXPLAINED by the first 20 eigenperiods
fig = plt.figure()
plt.bar(np.arange(20),eigenper_var_explained[:20]*100,facecolor='w',edgecolor='k'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Temporal_eigenper_varexplained.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Temporal_eigenper_varexplained.png', dpi=300, bbox_inches='tight'); plt.close() 


##### CLASSIFICATION ######
print('LOOO Cross-validation...')
## SET RIDGE LOGISTIC REGRESSION 'one-versus-rest' to test how eigenperiods can predict the PM class
y_train = PM_label[(PM_label==0) | (PM_label==1) | (PM_label==2)]       # Labels (PM class)
Cs = np.hstack((10**np.arange(-3.0,4.0)))                               # C parameter to optimize/validate
NB_ep_CV = 10; NB_C_CV = len(Cs)                                        # Number of eigenperiods and C values to optimize/cross validate
score_cv_ep = np.zeros((NB_ep_CV , NB_C_CV))
score_cv_ep_err = np.zeros((NB_ep_CV , NB_C_CV))
for ep in range(1,NB_ep_CV+1):
    print('Eigenperiod : ' + str(ep))
    for i in range(NB_C_CV):
        print('C param: ' + str(Cs[i]))
        weights_mean, intercept_mean, conf_mean, proba_mean, fold_scores, class_model = class_logReg(eigenper_scores[:ep,:].T, y_train, NB_DYADS, ['Visual','Music','Visual+Music'], C=Cs[i],multiclass='ovr')
        score_cv_ep[ep-1,i] = fold_scores.mean()
        score_cv_ep_err[ep-1,i]= fold_scores.std()/sqrt(len(fold_scores))
 
## FIND OPTIMAL C parameter and eigenperiod number
maxAcc = np.max(score_cv_ep)                            # Find max accuracy
idx_min = np.argmin(np.where(score_cv_ep==maxAcc)[0])   # Find it for the smallest eigenperiod value (ie, it reaches the max accuracy the fastest)
ep_opt = np.where(score_cv_ep==maxAcc)[0][idx_min] + 1  # Optimal eigenperiod number
c_opt = Cs[np.where(score_cv_ep==maxAcc)[1][idx_min]]   # Optimal C value  


## TEST CLASSIFIER ON SINGLE EIGENPERIODS
print('Test accuracy of classifier for the different eigenperiods...')
# Compute classification accuracy
score_ep_alone = np.zeros((NB_ep_CV)); score_ep_alone_err = np.zeros((NB_ep_CV))
for ep in range(1,NB_ep_CV+1):
    weights_mean, intercept_mean, conf_mean, proba_mean, fold_scores, class_model = class_logReg(eigenper_scores[ep-1:ep,:].T, y_train, NB_DYADS, ['Visual','Music','Visual+Music'], C=c_opt,multiclass='ovr')

    score_ep_alone[ep-1] = fold_scores.mean()
    score_ep_alone_err[ep-1] = fold_scores.std()/sqrt(len(fold_scores))

# Plot classification accuracy
fig = plt.figure()
plt.bar(np.arange(NB_ep_CV),score_ep_alone[:NB_ep_CV]*100,edgecolor='k',facecolor='w', capsize=2)
plt.axhline(33,c='gray',linestyle='--'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Temporal_eigenper_classifindiv.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Temporal_eigenper_classifindiv.png', dpi=300, bbox_inches='tight'); plt.close() 

## SAME BUT NOW classification accuracy when using the 2 selective EPs (based on their activation + the 2 biggest classif accuracy) 
# Classif accuracy of the selective EPs together
selEPs=np.array([0,3])
weights_mean_selEPs, intercept_mean_selEPs, conf_mean_selEPs, proba_mean_selEPs, fold_scores_selEPs, class_model_selEPs = class_logReg(eigenper_scores[selEPs,:].T, y_train, NB_DYADS, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_selEPs = fold_scores_selEPs.mean()
score_selEPs_err = fold_scores_selEPs.std()/sqrt(len(fold_scores_selEPs))

# Classification accuracy of the other EPs together
restEPs=np.setxor1d(np.array([0,3]),np.arange(ep_opt))
weights_mean_restEPs, intercept_mean_restEPs, conf_mean_restEPs, proba_mean_restEPs, fold_scores_restEPs, class_model_restEPs = class_logReg(eigenper_scores[restEPs,:].T, y_train, NB_DYADS, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_restEPs = fold_scores_restEPs.mean()
score_restEPs_err = fold_scores_restEPs.std()/sqrt(len(fold_scores_restEPs))

# Classification accuracy of the full optimal model
weights_mean_opt, intercept_mean_opt, conf_mean_opt, proba_mean_opt, fold_scores_opt, class_model_opt = class_logReg(eigenper_scores[:ep_opt,:].T, y_train, NB_DYADS, ['Visual','Music','Visual+Music'], C=c_opt, multiclass='ovr')
score_opt = fold_scores_opt.mean()
score_opt_err = fold_scores_opt.std()/sqrt(len(fold_scores_opt))

# Plot classification accuracy for the full optimal, selEPs and restEPs models
fig = plt.figure()
plt.bar(np.arange(3),np.hstack((score_opt,score_selEPs,score_restEPs))*100,capsize=2,facecolor='w',edgecolor='k')
plt.axhline(33,c='gray',linestyle='--'); plt.ylim((0,105))
# fig.savefig(output_dir + '/Temporal_eigenper_classiftogehter.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Temporal_eigenper_classiftogehter.png', dpi=300, bbox_inches='tight'); plt.close()

# Plot confusion matrix of the selective model
fig = plt.figure()
sns.heatmap(conf_mean_selEPs*100, vmin=0, vmax=100, cmap=sns.color_palette("Blues", as_cmap=True), cbar=True,annot=True, cbar_kws={"orientation": "horizontal"}, square = True)
# fig.savefig(output_dir + '/Temporal_eigenper_classifconf-selective.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Temporal_eigenper_classifconf-selective.png', dpi=300, bbox_inches='tight'); plt.close() 



#%% 
##############################################################
######    TEMPORAL ORGANIZATION OF IMS: EIGENPERIODS    ######
######             PLOT 1: eigenperiod scores           ######
######             (related to fig 4 and fig S4)        ######
##############################################################

# Keep only the first 5 EPs
NB_EP = 5

## PLOT EIGENPERIOD SCORES for the different PMs and dyads
# as bars, ranked by eigenperiod score, and colored by PM class
pm_num = np.repeat(np.arange(10),35)
colors_vis=np.array([(255,207,0),(255,169,30),(255,140,0),(223,120,0)])/255
colors_mus=np.array([(164,251,166),(74,229,74),(48,203,0),(15,146,0),(0,98,3)])/255
colors_pm10=['royalblue']
colors=[colors_vis[0],colors_mus[0],colors_vis[1],colors_mus[1],colors_vis[2],colors_mus[2],colors_mus[3],colors_pm10[0],colors_vis[3],colors_mus[4]]

for ep in range(NB_EP):
    fig = plt.figure(figsize=(40,20))
    ind_sort = np.argsort(eigenper_scores[ep,:])[::-1]
    label_PM = pm_num[ind_sort].astype(int)
    bars = plt.bar(np.arange(350),eigenper_scores[ep,ind_sort],width=0.85,rasterized=True)
    plt.ylim((-0.25,0.3))
    i=0
    for item in bars:
        item.set_color(colors[label_PM[i]])
        i+=1
    # fig.savefig(output_dir + '/Temporal_eigenper_scores' + str(ep+1) + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/Temporal_eigenper_scores' + str(ep+1) + '.png', dpi=300, bbox_inches='tight'); plt.close() 


## PLOT EIGENPERIOD SCORES for the different PMs (BUT AVERAGED ACROSS DYADS)
eigenper_scores_meanSubj = np.zeros((NB_FREQ,10))
for pm in range(10):
    eigenper_scores_meanSubj[:,pm] = eigenper_scores[:,pm*35:(pm+1)*35].mean(axis=1)

pm_num_meanSubj=np.arange(10)    
for ep in range(NB_EP):
    fig = plt.figure(figsize=(20,20))
    ind_sort = np.argsort(eigenper_scores_meanSubj[ep,:])[::-1]
    label_PM = pm_num_meanSubj[ind_sort].astype(int)
    bars = plt.bar(np.arange(10),eigenper_scores_meanSubj[ep,ind_sort],width = 0.3)
    plt.ylim((-0.1,0.15))
    i=0
    for item in bars:
        item.set_color(colors[label_PM[i]])
        i+=1
    # fig.savefig(output_dir + '/Temporal_eigenper_scores_meanSubj_' + str(ep+1) + '.pdf', dpi=600, bbox_inches='tight');
    fig.savefig(output_dir + '/Temporal_eigenper_scores_meanSubj_' + str(ep+1) + '.png', dpi=300, bbox_inches='tight');  plt.close() 
        
    

## SAVE A PLOT WITH ONLY 1 BAR PER COLOR, just to know the colorcode legend
fig = plt.figure(figsize=(40,20))
idx = np.arange(0,350,35)
bars = plt.bar(np.arange(10),np.ones(10),width = 0.3)
i=0
for item in bars:
    item.set_color(colors[i])
    i+=1
# fig.savefig(output_dir + '/Temporal_eigenper_scores_legend.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/Temporal_eigenper_scores_legend.png', dpi=300, bbox_inches='tight'); plt.close() 
    
    
    
#%% 
##############################################################
######    TEMPORAL ORGANIZATION OF IMS: EIGENPERIODS    ######
######            PLOT 2: eigenperiod weights           ######
######           (related to fig 4 and fig S4)          ######
##############################################################
    
label_mod = ['Partner-driven','Music-driven','Both (PM 10)']
label_dim = ['Mediolateral','Anteroposterior','Vertical']
NB_EP_plot = 5                                               # Number of eigenperiods to plot

## PLOT EIGENPERIOD WEIGHTS AS FREQUENCY SPECTRUM/MATRIX (IMSHOW)
# 1. Prepare custom colormap from Gray to Pink (merging Gray-to-Red and Pink-to-Green)
import matplotlib.colors as colors
from typing import List, Tuple
div_cmaps: List[List[str]] = []
for cmap_name in ['PiYG_r','RdGy']:
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.5, b=1),
        cmap(np.linspace(0.5, 1)))
    
    cmap_list = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    div_cmaps.append(cmap_list)
div_cmaps[1] = list(reversed(div_cmaps[1]))
cmap_nodata_list = div_cmaps[1] + div_cmaps[0]
new_cmap = colors.ListedColormap(cmap_nodata_list)

# 2. Actually do the plot
fig = plt.figure(figsize=(40,1))
iPlot=1
for ep in range(NB_EP_plot):
    ax = fig.add_subplot(1,6,iPlot)
    absMax = np.amax(abs(eigenper_weights[:,ep])); absMin = np.amin(abs(eigenper_weights[:,ep]))
    
    per_plot = np.array([ eigenper_weights[:,ep] , eigenper_weights[:,ep] ])
    p = ax.pcolormesh(np.log2(freqs_per_beat),np.arange(2),per_plot,vmin=-absMax,vmax=absMax,cmap=new_cmap,rasterized=True)
    
    ax.set_ylim((-0.5,0.5)); ax.set_yticks([]); 
    ax.set_xticks( np.array([2,1,0,-1,-2,-3,-4]) ) 
    ax.set_xticklabels( ['0.25','0.5','1','2','4','8','16'] , fontsize = 12)
    ax.set_xlabel('Period = beat *' , fontsize = 15)
    ax.set_xlim(np.log2([freqs_per_beat.min(),freqs_per_beat.max()]))
    ax.invert_xaxis()
    
    # Uncomment if you need colorbar
    # fig.colorbar(p, ax=ax)
    
    iPlot += 1

# plt.savefig(output_dir + '/Temporal_eigenper_weights.pdf', dpi=600, bbox_inches='tight');
plt.savefig(output_dir + '/Temporal_eigenper_weights.png', dpi=300, bbox_inches='tight'); plt.close() 