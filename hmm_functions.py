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





def preprocess_track(tracks, track_id, window_size, dt):
    one_track_xy = tracks[tracks['tid'] == track_id]
    #truth = tracks[tracks['tid'] == track_id]["GT"]
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
    #truth = truth[:seq_len]
    logDs = logDs[:seq_len]


    return sliding_msds, steps, logDs

def scale_data(window_size, tracks, dt):

    preprocessed_tracks = []
    lengths = []

    for i in tracks["tid"].unique():

        sliding_msds, steps, logD = preprocess_track(tracks,i, window_size, dt)

        track_features = [sliding_msds, steps, logD]
        preprocessed_tracks.append(track_features)
        lengths.append(len(sliding_msds) + (window_size-1))

    preprocessed_tracks = np.array(preprocessed_tracks)

    # Step 1: Separate the feature vectors
    msd = [row[0] for row in preprocessed_tracks]
    steplength = [row[1] for row in preprocessed_tracks]
    logD = [row[2] for row in preprocessed_tracks]

    scaled_lists = []

    for one_list in [msd, steplength, logD]:

        # Determine the maximum length of the sublists
        max_length = max(len(row) for row in one_list)

        # Transpose the data while padding missing values with np.nan
        transposed = [
            [row[i] if i < len(row) else np.nan for row in one_list]
            for i in range(max_length)
        ]

        # Scale each feature independently
        scaler = StandardScaler()
        scaled_transposed = [
            scaler.fit_transform(np.array(feature).reshape(-1, 1)).flatten()
            for feature in transposed
        ]

        # Reconstruct the original structure, ignoring padded values
        scaled_list_of_lists = [
            [scaled_transposed[i][j] for i in range(len(scaled_transposed)) if not np.isnan(transposed[i][j])]
            for j in range(len(msd))
        ]

        #pad values at the end due to steplength to make feature vectors as long as original tracks
        for list in scaled_list_of_lists:
            list.append([0] * (window_size - 1))

        scaled_lists.append(scaled_list_of_lists)

        # Step 3: Combine the scaled features back
        scaled_data = list(zip(scaled_lists[0], scaled_lists[1], scaled_lists[2]))

    return preprocessed_tracks, scaled_data, lengths

def run_model(model, tracks, window_size, dt):
    preprocessed_tracks, scaled_data, lengths = scale_data(window_size, tracks, dt)

    print("heere2",scaled_data.shape)
    concat_data = np.concatenate(scaled_data, axis = 1)
    concat_data = concat_data.T
    lengths = np.array(lengths)

    predicted_states = []

    # Keep track of where each sequence starts and ends in the concatenated array
    start_idx = 0

    # Iterate over each track by using the lengths array
    for length in lengths:
        # Extract the specific sequence from concatenated_series using start_idx and length
        sequence = concat_data[start_idx:start_idx + length]
        
        # Predict the hidden states for this sequence
        states = model.predict(sequence)
        
        # Append the predicted states for this sequence to the list
        predicted_states.append(states)

        # Move to the start index of the next sequence
        start_idx += length

    predicted_states = np.array(predicted_states)

    return  predicted_states


