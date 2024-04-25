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
############       AND COMPARE LINEAR (PCA)       ############
############     VS NON-LINEAR (AUTOENCODER)      ############
############         REDUCTION TECHNIQUES         ############
##############################################################

print('------------------------')
print('EXTRACTING PRINCIPAL MOVEMENTS...')
print('------------------------')


## PREPARE DATA
# Combine data into a matrix usable for PCA and Auto-Encoder (the 66 channels by time; concatenated over trials and participants)
pos_mat=data_subj[0,:,:]
for i in range(1,NB_DYADS*2):
    pos_mat= np.hstack((pos_mat,data_subj[i,:,:][:,np.where(~np.isnan(data_subj[i,0,:]))[0]]))
pos_mat = pos_mat.T
# del data_subj


## DEFINE NON LINEAR AUTO-ENCODER
import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredLogarithmicError

class Autoencoder_nonlin:
  def __init__(self, latent_dim=1):
    # create the encoder
    input_img = keras.Input(66) 
    encoded = layers.Dense(66, activation='linear')(input_img)
    encoded = layers.Dense(66, activation='tanh')(encoded)
    encoded = layers.Dense(latent_dim, activation='linear')(encoded)
    self.encoder = Model(inputs=[input_img], outputs=[encoded])

    # create the decoder
    decoded = layers.Dense(66, activation='tanh')(encoded)
    decoded = layers.Dense(66, activation='linear')(decoded)
    self.decoder = Model(inputs=[encoded], outputs=[decoded])

    # combine the encoder and decoder into a full autoencoder
    input_img = keras.Input(66) 
    z = self.encoder(input_img) # push observations into latent space
    o = self.decoder(z) # project from latent space to feature space
    self.model = Model(inputs=[input_img], outputs=[o])
    self.model.compile(loss='mse', optimizer='adam')



## APPLY AUTO-ENCODER (OPTIMIZATION AND TRAINING)
df = pd.DataFrame(pos_mat)
history_nonlin=[]; vaf_autoencoder_nonlin = []; autoencoders_nonlin=[]
for latent_dim in range(1,21):
    print("LATENT DIM " + str(latent_dim))
    
    autoencoder = Autoencoder_nonlin(latent_dim=latent_dim)
    autoencoders_nonlin.append(autoencoder)
    
    history_nonlin.append( 
            autoencoder.model.fit(
            df, 
            df, 
            epochs=10000, 
            batch_size=2048,
            validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(patience=5)]
        )
    )
    
    decoded_imgs = autoencoder.model.predict(pos_mat)
    vaf_autoencoder = (1-(np.sum(np.var(pos_mat-decoded_imgs,0)))/np.sum(np.var(pos_mat,0)))*100

    # VAF (%)
    vaf_autoencoder_nonlin.append(
        vaf_autoencoder
    )
   
    
## APPLY LINEAR PCA
# Apply PCA using Singular Value Decomposition
U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
eigenval_PM=S**2
common_nrj = np.cumsum(eigenval_PM) / np.sum(eigenval_PM);     nbEigen = [i for (i, val) in enumerate(common_nrj) if val>0.95][0];
indiv_var = eigenval_PM / np.sum(eigenval_PM)
common_PC_scores = (U*S)
common_eigen_vects = V

## COMPARE VARIANCE ACCOUNTED FOR (VAF) BY LINEAR PCA AND NON LINEAR AE
# compute vaf
vaf_autoencoder_nonlin_a = np.array(vaf_autoencoder_nonlin)
vaf_pca = common_nrj[:20]*100

# plot
df = pd.DataFrame(np.hstack((vaf_pca.reshape((-1,1)),vaf_autoencoder_nonlin_a.reshape((-1,1)))), columns=['PCA', 'nonlinAE'])
fig, ax = plt.subplots(figsize=(20,10))
ax.axhline(95,linestyle='--',color='gray',alpha=0.8,zorder=1)
df.plot.bar(ax=ax,color={"PCA": "C0", "nonlinAE": "C2"},alpha=0.9,zorder=10)
plt.xticks(np.arange(20)); ax.set_xticklabels(np.arange(1,21),rotation='horizontal')
plt.xlabel('Dimensions added'); plt.ylabel('Variance explained (%)')
fig.savefig(output_dir + '/PMs_explained-LIN-NONLIN.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/PMs_explained-LIN-NONLIN.png', dpi=600, bbox_inches='tight'); plt.close() 


