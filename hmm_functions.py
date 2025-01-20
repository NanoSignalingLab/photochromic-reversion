import pandas as pd
#import polars as ps

import numpy as np

from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import math

from tqdm import tqdm
from matplotlib.path import Path
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import metrics
from itertools import chain
from math import nan

def compute_msd(trajectory):
        totalsize=len(trajectory)
        msd=[]
        for i in range(totalsize-1):
            j=i+1
            msd.append(np.sum((trajectory[0:-j]-trajectory[j::])**2)/float(totalsize-j)) # Distance that a particle moves for each time point divided by time
        msd=np.array(msd)
        rmsd = np.sqrt(msd)
        return msd, rmsd 



def logD_from_mean_MSD(MSDs, dt):
        mean_msd = 0
        logD = 0

        mean_track=np.mean(MSDs[0:3])
        if mean_track!=0:
            mean_msd = mean_track
        else:
            mean_msd = 0.000000001
    
        logD = math.log10(mean_track/(dt*4)) # 2*2dimnesions* time
        return mean_msd, logD




def load_all(folderpath1):
    count=0
    onlyfiles = [f for f in listdir(folderpath1) if isfile(join(folderpath1, f))]
    for i in onlyfiles:
        if i.endswith(".csv"):
            path=os.path.join(folderpath1, i)
            print(path)
            df=pd.read_csv(path)
            if count==0:
                tracks=df
            else:
                df["tid"] = df["tid"] + len(tracks["tid"].unique())
            
                tracks = pd.concat([tracks, df], ignore_index=True, axis=0)
            count+=1
    return tracks


def preprocess_track(tracks, track_id, window_size, dt):
    one_track_xy = tracks[tracks['tid'] == track_id]
    truth = tracks[tracks['tid'] == track_id]["GT"]
    x, y = np.array(one_track_xy["pos_x"]), np.array(one_track_xy["pos_y"])

    m=np.column_stack((x,y))
    sliding_msds = []
    logDs = []
    for i in range(len(m) - window_size + 1):
        sliced_m = m[i: i + window_size]
      
        msds, rmsd = compute_msd(sliced_m)
        _, logD = logD_from_mean_MSD(msds, dt)
     
        sliding_msds.append(np.mean(msds))
        logDs.append(logD)
     
       

    steps = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

    #cut off the last part so its all the same length as msd sequence
    seq_len = len(sliding_msds)
    steps = steps[:seq_len]
    truth = truth[:seq_len]
    logDs = logDs[:seq_len]


    return sliding_msds, steps, logDs, truth

def preprocess_tracks_for_main(tracks, track_id, window_size, dt):

    one_track_xy = tracks[tracks['tid'] == track_id]
    x, y = np.array(one_track_xy["pos_x"]), np.array(one_track_xy["pos_y"])

    m=np.column_stack((x,y))
    sliding_msds = []
    logDs = []
    for i in range(len(m) - window_size + 1):
        sliced_m = m[i: i + window_size]
      
        msds, rmsd = compute_msd(sliced_m)
        _, logD = logD_from_mean_MSD(msds, dt)
     
        sliding_msds.append(np.mean(msds))
        logDs.append(logD)
     
       

    steps = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

    #cut off the last part so its all the same length as msd sequence
    

    seq_len = len(sliding_msds)
    steps = steps[:seq_len]
    logDs = logDs[:seq_len]




    return sliding_msds, steps, logDs





def preprocess_wrapper(window_size, tracks, dt):

    preprocessed_tracks = []
    truths = []
    lengths = []

    for i in tracks["tid"].unique():

        sliding_msds, steps, logD, truth = preprocess_track(tracks,i, window_size, dt)

        track_features = [sliding_msds, steps, logD]
        preprocessed_tracks.append(track_features)
        truths.append(truth)
        lengths.append(len(sliding_msds))

    preprocessed_tracks = np.array(preprocessed_tracks)

    print("tracks preprocessed",preprocessed_tracks.shape)

    # Step 1: Separate the two feature vectors
    msd = preprocessed_tracks[:, 0, :]
    steplength = preprocessed_tracks[:, 1, :]
    logD = preprocessed_tracks[:, 2, :]

    # Step 2: Scale each feature vector independently
    scaler = StandardScaler()

    scaled_msd = scaler.fit_transform(msd)
    scaled_steplength = scaler.fit_transform(steplength)
    scaled_logD = scaler.fit_transform(logD)

    scaled_data = np.stack((scaled_msd, scaled_steplength, scaled_logD), axis=1)

    print(scaled_data.shape)

    return preprocessed_tracks, scaled_data, lengths, truths



def preprocess_wrapper_for_main(window_size, tracks, dt):

    preprocessed_tracks = []
    lengths = []

    for i in tracks["tid"].unique():

        sliding_msds, steps, logD= preprocess_tracks_for_main(tracks,i, window_size, dt)
        #sliding_msds_for_df=sliding_msds.append([1, 1, 1, 1, 1, 1, 1, 1, 1])
        #steps_for_df=steps.append([1, 1, 1, 1, 1, 1, 1, 1, 1])
       # logD_for_df=logD.append([1, 1, 1, 1, 1, 1, 1, 1, 1])

        track_features = [sliding_msds, steps, logD]
        preprocessed_tracks.append(track_features)
        lengths.append(len(sliding_msds))
       #track_features_for_df=[sliding_msds_for_df, steps_for_df, logD_for_df]
    print(preprocessed_tracks[0])

    preprocessed_tracks = np.array(preprocessed_tracks)

    print("tracks preprocessed",preprocessed_tracks.shape)

    # Step 1: Separate the two feature vectors
    msd = preprocessed_tracks[:, 0, :]
    steplength = preprocessed_tracks[:, 1, :]
    logD = preprocessed_tracks[:, 2, :]

    # Step 2: Scale each feature vector independently
    scaler = StandardScaler()

    scaled_msd = scaler.fit_transform(msd)
    scaled_steplength = scaler.fit_transform(steplength)
    scaled_logD = scaler.fit_transform(logD)

    scaled_data = np.stack((scaled_msd, scaled_steplength, scaled_logD), axis=1)

    print(scaled_data.shape)
    return preprocessed_tracks, scaled_data, lengths



def calculate_precision(predicted_states, truths):
    predicted_states_flat= list(chain.from_iterable(predicted_states))
    #print(predicted_states_flat)

    df_predict=pd.DataFrame(predicted_states_flat, columns=["states" ])
    #print(df_predict)
    
    

    df_predict["states"]= df_predict["states"].replace(2,1)
    #df_predict["states"]= df_predict["states"].replace(2,0)
    df_predict["states"]= df_predict["states"].replace(3,1)


    arry_predict=df_predict["states"]
    
    arry_truths= list(chain.from_iterable(truths))
    #print(arry_predict)
    #print(truths)

    precision, recall, fbeta, support=metrics.precision_recall_fscore_support(arry_truths, arry_predict, pos_label=0)
    try:
        precision_confined=precision[0]
    except IndexError:
        precision_confined=nan
    try:
        precision_unconfined=precision[1]
    except IndexError:
        precision_unconfined=nan
    try:
        recall_confined=recall[0]
    except IndexError:
        recall_confined=nan
    try:
        recall_unconfined=recall[1]
    except IndexError:
        recall_unconfined=nan
    try:
        fbeta_confined=fbeta[0]
    except IndexError:
        fbeta_confined=nan
    try:
        fbeta_unconfined=fbeta[1]
    except IndexError:
        fbeta_unconfined=nan
    try:
        support_confined=support[0]
    except IndexError:
        support_confined=nan
    try:
        support_unconfined=support[0]
    except IndexError:
        support_unconfined=nan
    print("precision_confined",precision_confined )
    print("precision_unconfined",precision_unconfined )
    print("recall_confined",recall_confined)
    print("recall_unconfined",recall_unconfined)

    return precision_confined, precision_unconfined, recall_confined, recall_unconfined


def run_model(model, tracks, window_size, dt):
    preprocessed_tracks, scaled_data, lengths=preprocess_wrapper_for_main(window_size, tracks, dt)

    print("heere2",scaled_data.shape)
    concat_data = np.concatenate(scaled_data, axis = 1)
    concat_data = concat_data.T
    lengths = np.array(lengths)

    predicted_states = []
    predicted_states_for_df = []


    # Keep track of where each sequence starts and ends in the concatenated array
    start_idx = 0

    # Iterate over each track by using the lengths array
    for length in lengths:
        # Extract the specific sequence from concatenated_series using start_idx and length
        sequence = concat_data[start_idx:start_idx + length]
        
        # Reshape the sequence to 2D (needed by the HMM model)
        sequence_reshaped = sequence
        
        # Predict the hidden states for this sequence
        states = model.predict(sequence_reshaped)
        
        # Append the predicted states for this sequence to the list
        predicted_states.append(states)
        arry_fill=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        states_df=np.hstack((states,arry_fill))
        #predicted_states_for_df.append(states)
        #predicted_states_for_df.append([1, 1, 1, 1, 1, 1, 1, 1, 1])
        #predicted_states_for_df_flat=list(chain.from_iterable(predicted_states))
        predicted_states_for_df.append(states_df)



        #np.reshape(arry_frame, (traj.shape[0], 1)
        # Move to the start index of the next sequence
        start_idx += length
    predicted_states = np.array(predicted_states)
    #predicted_states.shape
    #print("states for df", predicted_states_for_df)

    #p1, p2, r1, r2=calculate_precision(predicted_states, truths)

    #predicted_states_flat= list(chain.from_iterable(predicted_states))
    #print(predicted_states_flat)
    #df_predict=pd.DataFrame(predicted_states_flat, columns=["states" ])
    #print(df_predict)
    #df_predict["states"]= df_predict["states"].replace(2,1)
    #df_predict["states"]= df_predict["states"].replace(2,0)
    #df_predict["states"]= df_predict["states"].replace(3,1)
    #arry_predict=df_predict["states"]


    #return p1, p2, r1, r2
    return  predicted_states_for_df


