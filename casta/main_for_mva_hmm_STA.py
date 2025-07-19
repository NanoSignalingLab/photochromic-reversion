# %%

from casta.RandomWalkSims import (
    Gen_normal_diff,
    Gen_directed_diff,
    Get_params,
    Gen_confined_diff,
    Gen_anomalous_diff,
)

import matplotlib.pyplot as plt
import matplotlib
from my_Fingerprint_feat_gen import ThirdAppender, GetStatesWrapper #, GetMSDWrapper
from MLGeneral import ML, histogram
import pickle
import os
from pomegranate import *
from functools import partial
import numpy as np
# import multiprocess as mp
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from functools import reduce
import operator 
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from itertools import chain
import math
from scipy.stats import gaussian_kde
from sklearn.preprocessing import normalize
import shapely
from shapely.geometry import LineString, Point
from shapely import intersection
import itertools
from statistics import mean 
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import os
from os import listdir
from os.path import isfile, join
import warnings
import andi_datasets
from andi_datasets.models_phenom import models_phenom
from sklearn import metrics
from math import nan
from casta.hmm_functions import run_model


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    #################################
   
    import stochastic
    #stochastic.random.seed(3)
    #np.random.seed(7)

    
    ##################################
    # if we have csv with tracks: for 1 file only:
    
    def wrapper_one_file(f1, f2, min_track_length, dt, plotting_flag): # works I guess
        image_path_lys=f1.split("csv")
        image_path=image_path_lys[0] +"svg"
        tracks_input, deep_df1, traces, lys_x, lys_y, msd_df = load_file(f1, min_track_length) # execute this function to load the files
        mean_msd_df=msd_mean_track(msd_df, dt)
        train_result, states, lys_states = run_traces_wrapper(traces, dt)
        deep_df2=computing_distance_wrapper(deep_df1)
        deep_df3=calculate_angles_wrapper(deep_df2)
        deep_df4=calculate_KDE_wrapper(lys_x, lys_y, deep_df3)
        deep_df5=calculate_intersections_wrapper(lys_x, lys_y, deep_df4)
        deep_df6=fingerprints_states_wrapper(lys_states, deep_df5)
        grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
        deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short)
        plotting_final_image(deep_df_short2,lys_points2, image_path)
        make_fingerprint_file(f2, train_result, deep_df_short2, dt, mean_msd_df) # run function to make excel with all parameters

    ####################################
    # if we have multiple csv with tracks in a folder, makes one results excel per file: 

    def wrapper_multiple_files(folderpath1, min_track_length, dt, plotting_flag):
        onlyfiles = [f for f in listdir(folderpath1) if isfile(join(folderpath1, f))]
        for i in onlyfiles:
            
            if i.endswith(".csv"):
                path=os.path.join(folderpath1, i)
                print(path)
                image_path_lys=path.split("csv")
                image_path=image_path_lys[0] +"svg"
                tracks_input, deep_df1, traces, lys_x, lys_y, msd_df = load_file(path, min_track_length) # execute this function to load the files
                mean_msd_df=msd_mean_track(msd_df, dt)
                train_result, states, lys_states = run_traces_wrapper(traces, dt)
                deep_df2=computing_distance_wrapper(deep_df1)
                deep_df3=calculate_angles_wrapper(deep_df2)
                deep_df4=calculate_KDE_wrapper(lys_x, lys_y, deep_df3)
                deep_df5=calculate_intersections_wrapper(lys_x, lys_y, deep_df4)
                deep_df6=fingerprints_states_wrapper(lys_states, deep_df5)
                grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
                deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short)
                plotting_final_image(deep_df_short2,lys_points2, image_path)
                make_fingerprint_file(path, train_result, deep_df_short2, dt, mean_msd_df) # run function to make excel with all parameters



    ##########################################
    # if we simulate tracks with ANDI and read in only values for grid paramters for simulation:

    def read_in_values_and_execute(f1,min_track_length, dt, plotting_flag, plotting_saving_nice_image_flag,tracks_saving_flag ):
        df_values=pd.read_csv(f1)
        image_path_lys=f1.split("csv")
        image_path=image_path_lys[0]
    
        df_values= df_values.iloc[: , 1:]
        
        for index, row in df_values.iterrows():
            print("running simulation nr: ", index)
            trajectories, labels =make_simulation(row['compartements'], row['radius'], row["DS1"], row["alphas"], row["trans"])
            sim_tracks=make_dataset_csv(trajectories, labels)
            deep_df1, traces, lys_x, lys_y, msd_df= make_deep_df(sim_tracks, min_track_length)
            mean_msd_df=msd_mean_track(msd_df, dt)
            train_result, states, lys_states = run_traces_wrapper(traces, dt)
            deep_df2=computing_distance_wrapper(deep_df1)
            deep_df3=calculate_angles_wrapper(deep_df2)
            deep_df4=calculate_KDE_wrapper(lys_x, lys_y, deep_df3)
            deep_df5=calculate_intersections_wrapper(lys_x, lys_y, deep_df4)
            deep_df6=fingerprints_states_wrapper(lys_states, deep_df5)
            grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
            deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short)
            sim_tracks_2=make_GT_consecutive(sim_tracks)
            list_accuracy=calculate_accuracy(sim_tracks_2, deep_df_short2, mean_msd_df)
            if plotting_saving_nice_image_flag==1:
                image_path1=image_path+str(index)+".tiff"
                plot_GT_and_finger(sim_tracks_2, deep_df_short2, image_path1)
            if tracks_saving_flag==1:
                path_out_simulated_tracks_lys=f1.split(".csv")
                path_out_simualted_tracks=path_out_simulated_tracks_lys[0]+"_simulated_tracks_"+str(index)+"_.csv"
                sim_tracks_2.to_csv(path_out_simualted_tracks)
                
            if index==0:
                list_final_accuracy=[list_accuracy]
            else:
                list_final_accuracy.append(list_accuracy)
            
        
        df_final_accuracy=pd.DataFrame(list_final_accuracy, columns=["percent_both_confined", "percent_both_unconfined","percent_correct","percent_correct_confined","percent_correct_unconfined","percent_sim_confined","percent_sim_unconfined", "precision_confined",  "precision_unconfined","recall_confined", "recall_unconfined",  "fbeta_confined","fbeta_confined","fbeta_confined", "support_confined", "support_unconfined", "logD_mean_diff", "logD_mean_cluster_diff", "mean_clusters_per_track", "total_time_in_cluster_per_track", "mean_time_in_clusters_per_track", "mean_clustered_points" ])
  
        df_final_parameters_out=pd.concat([df_values,df_final_accuracy],axis=1)
        path_out_accuracy_lys=f1.split(".csv")
        path_out_accuracy=path_out_accuracy_lys[0] +"_sim_accuracy_results.xlsx"
        writer = pd.ExcelWriter(path_out_accuracy , engine='xlsxwriter')
        df_final_parameters_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()



        return df_final_parameters_out

    #############################################
    # for our own HMM for reading inmultiple csv files with real tracks, make one result excel per file

    def calculate_spatial_transient_wrapper(folderpath1, min_track_length, dt, plotting_flag):
        onlyfiles = [f for f in listdir(folderpath1) if isfile(join(folderpath1, f))]
        for i in onlyfiles:
            
            if i.endswith(".csv"):
                path=os.path.join(folderpath1, i)
                print(path)
                image_path_lys=path.split("csv")
                image_path=image_path_lys[0] +"svg"
                image_path_tiff=image_path_lys[0] +"tiff"

                tracks_input, deep_df1, traces, lys_x, lys_y, msd_df = load_file(path, min_track_length) # execute this function to load the files
                mean_msd_df=msd_mean_track(msd_df, dt)

                deep_df2= run_traces_wrapper(deep_df1, dt)
                deep_df3=computing_distance_wrapper(deep_df2)
                deep_df4=calculate_angles_wrapper(deep_df3)
                deep_df5=calculate_KDE_wrapper(lys_x, lys_y, deep_df4)
                deep_df6=calculate_intersections_wrapper(lys_x, lys_y, deep_df5)

                grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df1, lys_begin_end_big2, lys_points_big_only_middle2=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
                deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_begin_end_big2, lys_points_big_only_middle2)

                #plotting_final_image(deep_df_short2,lys_points2, image_path)
                mean_msd_df2=caluclate_diffusion_non_STA_tracks(deep_df_short2,mean_msd_df1 )

                plotting_final_image2(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_path_tiff)
                make_results_file(path, deep_df_short2, dt,mean_msd_df2 ) # run function to make excel with all parameters

          



    ############################################
    # for our own HMM if we simulate groundtruth and test it on it

    def calulate_hmm_precison_with_simulating_tracks( f1,min_track_length, dt, plotting_flag, plotting_saving_nice_image_flag,tracks_saving_flag ):

        df_values=pd.read_csv(f1)
        image_path_lys=f1.split("csv")
        image_path=image_path_lys[0]
    
        df_values= df_values.iloc[: , 1:]
        
        for index, row in df_values.iterrows():
            print("running simulation nr: ", index)
            trajectories, labels =make_simulation(row['compartements'], row['radius'], row["DS1"], row["alphas"], row["trans"])
            sim_tracks=make_dataset_csv(trajectories, labels)
            deep_df1, traces, lys_x, lys_y, msd_df= make_deep_df(sim_tracks, min_track_length)
            mean_msd_df=msd_mean_track(msd_df, dt)
            deep_df2= run_traces_wrapper(deep_df1, dt)
          
            deep_df3=computing_distance_wrapper(deep_df2)
            deep_df4=calculate_angles_wrapper(deep_df3)
            deep_df5=calculate_KDE_wrapper(lys_x, lys_y, deep_df4)
            deep_df6=calculate_intersections_wrapper(lys_x, lys_y, deep_df5)
            grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
            deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short)
            
            
            sim_tracks_2=make_GT_consecutive(sim_tracks)
            list_accuracy=calculate_accuracy(sim_tracks_2, deep_df_short2, mean_msd_df)
        
            if plotting_saving_nice_image_flag==1:
                image_path1=image_path+str(index)+".tiff"
                plot_GT_and_finger(sim_tracks_2, deep_df_short2, image_path1)
            if tracks_saving_flag==1:
                path_out_simulated_tracks_lys=f1.split(".csv")
                path_out_simualted_tracks=path_out_simulated_tracks_lys[0]+"_simulated_tracks_"+str(index)+"_.csv"
                sim_tracks_2.to_csv(path_out_simualted_tracks)
                
            if index==0:
                list_final_accuracy=[list_accuracy]
            else:
                list_final_accuracy.append(list_accuracy)
            
        
        df_final_accuracy=pd.DataFrame(list_final_accuracy, columns=["percent_both_confined", "percent_both_unconfined","percent_correct","percent_correct_confined","percent_correct_unconfined","percent_sim_confined","percent_sim_unconfined", "precision_confined",  "precision_unconfined","recall_confined", "recall_unconfined",  "fbeta_confined","fbeta_confined","fbeta_confined", "support_confined", "support_unconfined", "logD_mean_diff", "logD_mean_cluster_diff", "mean_clusters_per_track", "total_time_in_cluster_per_track", "mean_time_in_clusters_per_track", "mean_clustered_points" ])
  
        df_final_parameters_out=pd.concat([df_values,df_final_accuracy],axis=1)
        path_out_accuracy_lys=f1.split(".csv")
        path_out_accuracy=path_out_accuracy_lys[0] +"_sim_accuracy_results.xlsx"
        writer = pd.ExcelWriter(path_out_accuracy , engine='xlsxwriter')
        df_final_parameters_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()


    #############################################
    ### mmmkae function that takes in previosuly generated tracks (with GT)and calculates accuracy:

    def calculating_HMM_accuracy_from_tracks(folderpath1, min_track_length, dt, plotting_flag):
        onlyfiles = [f for f in listdir(folderpath1) if isfile(join(folderpath1, f))]
        index=0
        names_list=[]
        for i in onlyfiles:
            
            if i.endswith(".csv"):
                path=os.path.join(folderpath1, i)
                print(path)
                names_path_lys=path.split("\\")
                names=names_path_lys[-1]
                #print(names)
                names_list.append(names)


                #image_path=image_path_lys[0] +"svg"
                tracks_input, deep_df1, traces, lys_x, lys_y, msd_df = load_file(path, min_track_length) # execute this function to load the files
                #print(tracks_input)
                mean_msd_df=msd_mean_track(msd_df, dt)

                deep_df2= run_traces_wrapper(deep_df1, dt)
                deep_df3=computing_distance_wrapper(deep_df2)
                deep_df4=calculate_angles_wrapper(deep_df3)
                deep_df5=calculate_KDE_wrapper(lys_x, lys_y, deep_df4)
                deep_df6=calculate_intersections_wrapper(lys_x, lys_y, deep_df5)

                grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
                deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short)
                list_accuracy=calculate_accuracy(tracks_input, deep_df_short2, mean_msd_df)

                if index==0:
                    list_final_accuracy=[list_accuracy]
                    index=1
                else:
                    list_final_accuracy.append(list_accuracy)
                
            
        df_final_accuracy=pd.DataFrame(list_final_accuracy, columns=["percent_both_confined", "percent_both_unconfined","percent_correct","percent_correct_confined","percent_correct_unconfined","percent_sim_confined","percent_sim_unconfined", "precision_confined",  "precision_unconfined","recall_confined", "recall_unconfined",  "fbeta_confined","fbeta_confined","fbeta_confined", "support_confined", "support_unconfined", "logD_mean_diff", "logD_mean_cluster_diff", "mean_clusters_per_track", "total_time_in_cluster_per_track", "mean_time_in_clusters_per_track", "mean_clustered_points" ])
        df_final_accuracy["track_name"]=names_list
        print(df_final_accuracy)
        path_out_accuracy=folderpath1+"\\tracks_accuracy_results.xlsx"
        #path_out_accuracy=path_out_accuracy_lys[0] +"_sim_accuracy_results.xlsx"
        writer = pd.ExcelWriter(path_out_accuracy , engine='xlsxwriter')
        df_final_accuracy.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()



    #############################################

    def make_simulation(number_compartments, radius_compartments, DS1, alphas_value, trans_value):
        N=500
        T=200
        D=0.001
        DS2=1
        L = 1.5*128 # enalrge field of fiew to avoid boundy effects
        compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments, 
                                                                            r = radius_compartments,
                                                                            L = L)                                

        trajs_model5, labels_model5 = models_phenom().confinement(N = N,
                                                            L = L,
                                                            Ds = [DS1*D, DS2*D],
                                                            comp_center = compartments_center,
                                                            r = radius_compartments,
                                                            trans = trans_value, # Boundary transmittance
                                                            T=T,
                                                            alphas=[1, alphas_value])
        return trajs_model5, labels_model5
    ################################################
    def make_dataset_csv(traj, labels):
        for i in range (0, traj.shape[1]):
                arry_temp=traj[:, i]
                arry_ID=[i]*traj.shape[0] # track ID!
                arry_time=[i*0.1 for i in range(traj.shape[0])] # time steps!
                arry_frame=[i+1 for i in range(traj.shape[0])] # frame
                arry_ID=np.reshape(arry_ID, (traj.shape[0], 1)) 
                arry_time=np.reshape(arry_time, (traj.shape[0], 1))
                arry_frame=np.reshape(arry_frame, (traj.shape[0], 1))
            
                arry_lables=labels[:, i]
            
                arry_temp=np.hstack((arry_temp,arry_lables))
                arry_temp=np.hstack((arry_temp, arry_ID))
                arry_temp=np.hstack((arry_temp, arry_time))
                arry_temp=np.hstack((arry_temp, arry_frame))
            
                if i ==0:
                    arry_final=arry_temp
                else:
                    arry_final=np.vstack((arry_final, arry_temp))
                
                tracks_df=pd.DataFrame(arry_final, columns=["POSITION_X", "POSITION_Y", "pm1", "DIFFUSION","pm2", "TRACK_ID", "POSITION_T", "FRAME"])

        
        return tracks_df


    ############################################
  
    ############################################
    # function to directly load the cleaned trackmate files:

    def load_file(path2, min_track_length):
        df=pd.read_csv(path2)
        deep_df, list_traces, lys_x, lys_y, msd_df= make_deep_df(df, min_track_length)
        return df, deep_df, list_traces, lys_x, lys_y, msd_df

    def make_deep_df(df, min_track_length):
        grouped= df.sort_values(["FRAME"]).groupby("TRACK_ID")
        count2=0
        deep_all=[]
        list_traces=[]
        lys_x=[]
        lys_y=[]

        for i in grouped["TRACK_ID"].unique():
            s= grouped.get_group(i[0])
            
            if s.shape[0]>min_track_length: # parameter to set threshold of minimun length of track duration (eg. 25 time points)
                count2+=1
                pos_x=list(s["POSITION_X"])
                pos_y= list(s["POSITION_Y"])
                pos_t=list(s["POSITION_T"])
                tid=list(s["TRACK_ID"])
                lys_x.append(pos_x)
                lys_y.append(pos_y)
                m= np.column_stack(( pos_x, pos_y ))
                msd, rmsd = compute_msd(m)
                frames= list(s["FRAME"])
                n= np.column_stack((msd,(frames[1:]),tid[1:]))

                if(count2== 1):
                    msd_all = n
                else:
                    msd_all = np.vstack((msd_all, n))

                msd_df=pd.DataFrame(msd_all, columns=["msd", "frame", "track_id"])

                list_traces.append(m)
                m2=np.column_stack(( tid, pos_x, pos_y, pos_t)) 

                if(count2== 1):
                    deep_all = m2
                else:
                
                    deep_all = np.vstack((deep_all, m2))
        deep_all_df=pd.DataFrame(deep_all, columns=["tid", "pos_x", "pos_y", "pos_t"])

        return deep_all_df, list_traces, lys_x, lys_y, msd_df
    #############################################
    # function for MSD and diffusion:

    def compute_msd(trajectory):
        totalsize=len(trajectory)
        msd=[]
        for i in range(totalsize-1):
            j=i+1
            msd.append(np.sum((trajectory[0:-j]-trajectory[j::])**2)/float(totalsize-j)) # Distance that a particle moves for each time point divided by time
        msd=np.array(msd)
        rmsd = np.sqrt(msd)
        return msd, rmsd 
    
    ##############################################
    # function for logDs:

    def logD_from_mean_MSD(MSDs):

        mean_msd = 0
        logD = 0

        mean_track=mean(MSDs[0:3])
        if mean_track!=0:
            mean_msd = mean_track
        else:
            mean_msd = 0.000000001
    
        logD = math.log10(mean_track/(dt*4)) # 2*2dimnesions* time
        return mean_msd, logD

    def msd_mean_track(msd_df, dt):
        group2= msd_df.groupby("track_id")
        lys=[]
        lys2=[]
        for i in group2["track_id"].unique():

            s= group2.get_group(i[0])
            
            full_track=list(s["msd"])
            mean_msd, logD = logD_from_mean_MSD(full_track)
            lys.append(mean_msd)
            lys2.append(logD)

        track_means_df = pd.DataFrame(np.column_stack([lys, lys2]), columns=["msd", "logD"])
        
        return track_means_df

    ################################################
    # function loading HMM model:

    def run_traces_wrapper(deep_df, dt): 

        with open("model_4.pkl", "rb") as file: 
            model = pickle.load(file)
        print("loading HMM model")
        window_size=10

        predicted_states_for_df=run_model(model, deep_df,window_size, dt)
        #print("this is predicted states output:" ,predicted_states_for_df)

        #predicted_states_for_df= run_model(model, deep_df,  window_size, dt)

        predicted_states_flat= list(chain.from_iterable(predicted_states_for_df))
      
        deep_df["hmm_states"]=predicted_states_flat
        #print(deep_df)
        deep_df["hmm_states"]= deep_df["hmm_states"].replace(2,1)
        #deep_df["hmm_states"]= deep_df["hmm_states"].replace(2,0)
        deep_df["hmm_states"]= deep_df["hmm_states"].replace(3,1)

        #print("heere234", deep_df)


        return deep_df

    ##################################################
    # function for consecutive features:
        
    def consecutive(col, seg_len, threshold, deep_df): # col= string of cl indf, seg_len=segment length of consecutive, threshold number
        grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")
        lys_final=[]
        for i in grouped_plot["tid"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            c3=0
            seg1=seg_len-1
            seg2=seg_len+1
            
            while c3<len(s["pos_x"]): 
                if c3>=len(s["pos_x"])-seg_len: 
                        lys_six.append([1]*1) 
                else:
                        
                    if sum(s[col][c3:c3+seg2])<threshold: 
                        lys_six.append([0]*1)
                    elif sum(s[col][c3:c3+seg2])>=threshold and sum(s[col][c3:c3+seg_len])<threshold: 
                        lys_six.append([0]*seg_len) 
                        c3+=seg1 
                    else:
                        lys_six.append([1]*1)
                c3+=1
            lys_six_flat=list(chain.from_iterable(lys_six))
            lys_final.append(lys_six_flat)
            c3=0

        lys_final_flat=list(chain.from_iterable(lys_final))
        return lys_final_flat
    
    ################# end function for consecutive features
    
    ################# Calculate distance and add to dataframe
    def computing_distance_wrapper(deep_df):
    
        print("Computing distance")

        distance = []
        distance_flag = []
        threshold_dist = 0.09
        for i in range(len(deep_df["pos_x"])-1):
            x1, y1 = deep_df["pos_x"][i], deep_df["pos_y"][i]
            x2, y2 = deep_df["pos_x"][i+1], deep_df["pos_y"][i+1]

            p1 = [x1, y1]
            p2 = [x2, y2]

            dis = math.dist(p1, p2)
            distance.append(dis)
            if dis < threshold_dist:
                distance_flag.append(0)
            else:
                distance_flag.append(1)

        distance.append(0)
        distance_flag.append(0)
        deep_df["distance"] = distance
        deep_df["distance_flag"] = distance_flag
        

    ################## End distance calculation

    ################## Find consecutive short distances (4 in this case)
    
        tresh_l = 9
        c2=0
        dist_final=[]
        grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")

        for i in grouped_plot["tid"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            c3=0
            while c3<len(s["pos_x"]): 

                if c3>=len(s["pos_x"])-tresh_l:
                    lys_six.append([1]*1) 
                else:
                    if sum(s["distance_flag"][c3:c3+tresh_l+1])==0:
                        lys_six.append([0]*1)
                    elif sum(s["distance_flag"][c3:c3+tresh_l+1])!=0 and sum(s["distance_flag"][c3:c3+tresh_l])==0:
                        lys_six.append([0]*tresh_l)
                        c2+=tresh_l-1
                        c3+=tresh_l-1
                    else:
                        lys_six.append([1]*1)
                c2+=1
                c3+=1
            lys_six_flat=list(chain.from_iterable(lys_six))
            dist_final.append(lys_six_flat)
            c2+=1
            c3=0
        
        dist_final_flat=list(chain.from_iterable(dist_final))
        deep_df["dist_cont"]=dist_final_flat

        return deep_df

    ################### end distance

    ################### calulcate angles:

    def angle3pt(a, b, c):
        ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang
    

    def calculate_angles_wrapper(deep_df):
        n=deep_df["pos_t"]
        x=deep_df["pos_x"]
        y=deep_df["pos_y"]

        lys_angles=[]
        for i in range (len(x)-2):
            a=(x[i], y[i])
            b=(x[i+1], y[i+1])
            c=(x[i+2], y[i+2])
            angle1 = 180 - angle3pt(a, b, c)
            angle=180-abs(angle1)

            lys_angles.append(angle)

        lys_angles.append(0)
        lys_angles.append(0)
        deep_df["angles"]=lys_angles

        ### make consecutive angles:
        print("Computing angles")

        angle_cont_lys=consecutive("angles", 10, 600, deep_df)
        
        deep_df["angle_cont"]=angle_cont_lys
        deep_df['angles_cont_level'] = pd.cut(deep_df["angle_cont"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df['angles_cont_level'] = deep_df['angles_cont_level'].astype(str)
        #final_pal_only_0=dict(zero='#fde624' ,  one= '#380282') # zero=yellow=high angles

        return deep_df

    ###################### end anlges calc

    ###################### function to calculate KDE:
    def make_KDE_per_track(lys_x, lys_y):
        lys_z=[]
        lys_z_norm=[]
        for i in range(len(lys_x)):    
            x=lys_x[i]
            y=lys_y[i]
            xy= np.vstack([x,y])

            z = gaussian_kde(xy)(xy)
            lys_z.append(z)

            normz=normalize([z])
            lys_z_norm.append(normz[0])

        out = np.concatenate(lys_z).ravel().tolist()
        out2 = np.concatenate(lys_z_norm).ravel().tolist()
        return out, out2
        
    #################### end function

    #################### function for KDE:

    def calculate_KDE_wrapper(lys_x, lys_y, deep_df):
        print("Computing KDE")
        out, out2 =make_KDE_per_track(lys_x, lys_y)

        
        deep_df["KDE"]=out
        deep_df['KDE_level']=pd.qcut(deep_df["KDE"], 9,labels=["zero" , "one", "two", "three", "four", "five", "six", "seven", "eight"])
        deep_df['KDE_values']=pd.qcut(deep_df["KDE"], 9,labels=False)
        deep_df['KDE_level'] = deep_df['KDE_level'].astype(str)
        #final_pal=dict(zero= '#380282',one= '#440053',two= '#404388', three= '#2a788e', four= '#21a784', five= '#78d151', six= '#fde624', seven="#ff9933", eight="#ff3300")

        # invert KDE values: for consistency, low values = good
        lys_invert=[]
        for i in deep_df["KDE_values"]:
            KDE_invert=8-i
            lys_invert.append(KDE_invert)
        deep_df["KDE_invert"]=lys_invert

        ######################### find consecutive KDE:
        KDE_cont_lys=consecutive("KDE_invert", 10, 13, deep_df)

        deep_df["KDE_cont"]=KDE_cont_lys
        deep_df["KDE_cont_level"] = pd.cut(deep_df["KDE_cont"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df["KDE_cont_level"] = deep_df["KDE_cont_level"].astype(str)
        return deep_df

        ########################## KDE done

    ######################### function to calculate intersections:
    # check line intersection: for consistency: 0 = intersection, 1 = not
    def calc_intersections(lys_x, lys_y):

        lys_x=list(chain.from_iterable(lys_x))
        lys_y=list(chain.from_iterable(lys_y))

        ### interection 1: between line 1 and line 4
        intersect1=[]
        count=3
        intersect1.append([1]*4)
        for i in range (len(lys_x)-4):
            line1 = LineString([(lys_x[i], lys_y[i]), (lys_x[i+1], lys_y[i+1])])
            line2 = LineString([(lys_x[count], lys_y[count]), (lys_x[count+1], lys_y[count+1])])
            interp1=intersection(line1, line2)
            count+=1
            x1, x2, x3, x4=interp1.bounds
            x1=str(x1)
            if x1=="nan":
                intersect1.append([1])
            else:
                intersect1.append([0])

        inter_flat1=list(chain.from_iterable(intersect1))

        ### interection 2: between line 1 and line 5
        intersect2=[]
        count=4 
        intersect2.append([1]*5)
        for i in range (len(lys_x)-5):
            line1 = LineString([(lys_x[i], lys_y[i]), (lys_x[i+1], lys_y[i+1])])
            line2 = LineString([(lys_x[count], lys_y[count]), (lys_x[count+1], lys_y[count+1])])
            interp1=intersection(line1, line2)
            count+=1
            x1, x2, x3, x4=interp1.bounds
            x1=str(x1)
            if x1=="nan":
                intersect2.append([1])
            else:
                intersect2.append([0])
            
        inter_flat2=list(chain.from_iterable(intersect2))

        ### interection 3: between line 1 and line 6
        intersect3=[]
        count=5
        intersect3.append([1]*6)
        for i in range (len(lys_x)-6):
            line1 = LineString([(lys_x[i], lys_y[i]), (lys_x[i+1], lys_y[i+1])])
            line2 = LineString([(lys_x[count], lys_y[count]), (lys_x[count+1], lys_y[count+1])])
            interp1=intersection(line1, line2)
            count+=1
            x1, x2, x3, x4=interp1.bounds
            x1=str(x1)
            if x1=="nan":
                intersect3.append([1])
            else:
                intersect3.append([0])

        inter_flat3=list(chain.from_iterable(intersect3))

        ### interection 4: between line 1 and line 7
        intersect4=[]
        count=6
        intersect4.append([1]*7)
        for i in range (len(lys_x)-7):
            line1 = LineString([(lys_x[i], lys_y[i]), (lys_x[i+1], lys_y[i+1])])
            line2 = LineString([(lys_x[count], lys_y[count]), (lys_x[count+1], lys_y[count+1])])
            interp1=intersection(line1, line2)
            count+=1
            x1, x2, x3, x4=interp1.bounds
            x1=str(x1)
            if x1=="nan":
                intersect4.append([1])
            else:
                intersect4.append([0])

        inter_flat4=list(chain.from_iterable(intersect4))

        return inter_flat1, inter_flat2, inter_flat3, inter_flat4
        
    ########################## end intersection function
        
    def calculate_intersections_wrapper(lys_x, lys_y, deep_df):
        print("Computing intersections")

        inter_flat1, inter_flat2, inter_flat3, inter_flat4=calc_intersections(lys_x, lys_y)

        ## add all intersections:
        deep_df["intersect1"]=inter_flat1
        deep_df["intersect2"]=inter_flat2
        deep_df["intersect3"]=inter_flat3
        deep_df["intersect4"]=inter_flat4

        ## put all intersections together:
        lys_all=[]
        for i in range(len(deep_df["pos_x"])):
            if deep_df["intersect1"][i]==0 or deep_df["intersect2"][i]==0 or deep_df["intersect3"][i]==0 or deep_df["intersect4"][i]==0:
                lys_all.append(0)
            else:
                lys_all.append(1)

        deep_df["all_intersect"]=lys_all

        ######################### find consecutive intersections:

        intersect_cont=consecutive("all_intersect", 10, 6, deep_df)
        deep_df["intersect_cont"]=intersect_cont
        return deep_df

        ########################### end intersections
    

    ########################## get fingertprint states:

    def fingerprints_states_wrapper(lys_states, deep_df):

        for i in range(len(lys_states)): 
            lys_states[i].append(1)
        flat_lys=reduce(operator.concat, lys_states)
        deep_df["fingerprint_state"]=flat_lys

        ## all fingerprint states
        deep_df['state_level'] = pd.cut(deep_df["fingerprint_state"], [-1.0, 0.0, 1.0, 2.0,  3.0], labels=["zero" , "one", "two", "three"], include_lowest=True, ordered= False)
        deep_df['state_level'] = deep_df['state_level'].astype(str)
        
        ## change state 1 to state zero: 
        deep_df.loc[deep_df['fingerprint_state'] == 1, 'fingerprint_state'] = 0 

        ########## find consecutive zero state fingerprints:
        grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")
        c2=0
        lys_final=[]
        for i in grouped_plot["tid"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            c3=0
            while c3<len(s["pos_x"]): 

                if c3>=len(s["pos_x"])-11:
                    lys_six.append([1]*1) 
                else:
                    if sum(s["fingerprint_state"][c3:c3+12])==0:
                        lys_six.append([0]*1)
                    elif sum(s["fingerprint_state"][c3:c3+12])!=0 and sum(s["fingerprint_state"][c3:c3+11])==0:
                        lys_six.append([0]*11)
                        c2+=10
                        c3+=10
                    else:
                        lys_six.append([1]*1)
                c2+=1
                c3+=1
            lys_six_flat=list(chain.from_iterable(lys_six))
            lys_final.append(lys_six_flat)
            c2+=1
            c3=0

        lys_final_flat=list(chain.from_iterable(lys_final))
        deep_df["state_0_cont"]=lys_final_flat
        deep_df['state_0_cont_level'] = pd.cut(deep_df["state_0_cont"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df['state_0_cont_level'] = deep_df['state_0_cont_level'].astype(str)

        return deep_df
    
    
    ########################################### end fingerprint states wrapper

    ############## plot all features togheter (plus convex hull):
    def plotting_all_features_and_caculate_hull(deep_df, mean_msd_df, plotting_flag): # add ture =1or false =0 for plotting yes or no
        print("plotting all features")
        #print("heere is deepdf",deep_df)


        deep_df_short=deep_df[["angle_cont", "hmm_states","dist_cont" ,"intersect_cont" , "KDE_cont"]]
        deep_df_short["sum_rows"] = deep_df_short.sum(axis=1)
    
        deep_df_short["row_sums_level"] = pd.cut(deep_df_short["sum_rows"], [0, 1,2, 3, 4,5 ,6], labels=["zero" , "one", "two", "three", "four", "five"], include_lowest=True, ordered= False)
        final_pal=dict(zero= "#ff3300",one= '#fde624',two= '#78d151', three= "#2a788e", four="#404388" , five="#440053") #all colors 

        deep_df_short["pos_x"]=deep_df["pos_x"]
        deep_df_short["pos_y"]=deep_df["pos_y"]
        deep_df_short["pos_t"]=deep_df["pos_t"]
        deep_df_short["tid"]=deep_df["tid"]

        linecollection = []
        colors = []
        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        c2=0
        
        c2=0
        if plotting_flag==1:
            for i in grouped_plot["tid"].unique():
                s= grouped_plot.get_group(i[0])
                for i in range (len(s["pos_x"])-1):

                    line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
                    color = final_pal[deep_df_short["row_sums_level"][c2]]
                    linecollection.append(line)
                    colors.append(color)

                    c2+=1
                c2+=1

            lc = LineCollection(linecollection, color=colors, lw=2) # was 1
            
            fig = plt.figure()
            ax = fig.add_subplot()
            sns.set(style="ticks", context="talk")
        
            plt.gca().add_collection(lc)
            plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.01) #was 0.001
    
        
        ########################## calculate convex hull:
        # get red and green points: = where 5, 4 or 3 criteria agree for spatial arrest
        
        lys_points2=[] 
        #lys_starting_end_points2=[]
        #lys_intermediate2=[]
        lys_start_end_cluster2=[]

        
        c2=0
        for j in grouped_plot["tid"].unique():
            flag=0
        
            s= grouped_plot.get_group(j[0])

            ############################################################
            ### add julien counter for st ain beginning and end here:

            lys_points=[]
           
            lys_start_end_cluster=[]
            for i in range (len(s["pos_x"])-1):
            
                if s["sum_rows"][c2]==0 or s["sum_rows"][c2]==1 or s["sum_rows"][c2]==2:
                    pos_x=s["pos_x"][c2]
                    pos_y=s["pos_y"][c2]
                    m= np.column_stack(( pos_x, pos_y))
                   
                                
                    if flag==0:
                        pos_all=m

                        flag+=1
                        if i==0:
                           
                            lys_test1=[]
                            lys_test1.append("B") # clsuter in beginning of track
                      
                        else: 
                            lys_test1=[]
                            lys_test1.append("BC") # just begginning of clsuter


                    else:
                        
                        if i == len(s["pos_x"])-2:
                            pos_all = np.vstack((pos_all,m))
                        
                            lys_points.append(pos_all)
                            flag = 0

                            #lys_starting_end_points.append(["E"]) # clsuter in end of track
                            lys_test1.append("E")
                          

                        else:
                            pos_all = np.vstack((pos_all,m))
                            #lys_starting_end_points.append(["M"]) # middle of clsuter and cluster in tehn  middle
                         
                            lys_test1.append("M")




                else:
                    if flag!=0:
                        lys_points.append(pos_all)
                        lys_test1.append("CE")
                        #lys_test2.append(lys_test1)
                        lys_start_end_cluster.append(lys_test1)

                        #lys_starting_end_points.append(["IDK"]) # end of clsuter 
                   


                    
                    flag=0
                c2+=1
            
            lys_points2.append(lys_points)
         
            lys_start_end_cluster2.append(lys_start_end_cluster)
            
            
                
            c2+=1
        #print("heere2",lys_start_end_cluster2)
        #print(lys_points2)
        #print(lys_start_end_cluster2[2], len(lys_start_end_cluster2[2]))
        #print(lys_points2[2], len(lys_points2[2]))

        ## insert logD calucaltion of non-clustered tracks someweherre here:


        
        ######################### plot points together with above lines
        lys_area2=[]
        lys_perimeter2=[]
        lys_hull2 = []
        lys_points_big2=[]
        lys_logD_cluster2=[]
        lys_msd_cluster2=[]

        ####
        lys_begin_end_big2=[]
        lys_points_big_only_middle2=[]
        lys_msd_cluster_middle2=[]
        lys_logD_cluster_middle2=[]
        
        for j in range (len(lys_points2)):
            lys_area=[]
            lys_perimeter=[]
            lys_hull=[]
            lys_points_big=[]
            lys_logD_cluster=[]
            lys_msd_cluster=[]
            lys_msd_cluster_middle=[]
            lys_logD_cluster_middle=[]

            #### add clsuter begin end points here as well:
            lys_begin_end_big=[]
            lys_points_big_only_middle=[]

            
            for i in range(len(lys_points2[j])):
                
                points=lys_points2[j][i] 
                if len(points)>5:
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    if ratio<105:
                        lys_points_big.append(points)

                        ##################
                        if lys_start_end_cluster2[j][i][0]!="B":
                            lys_begin_end_big.append(lys_start_end_cluster2[j][i])
                            lys_points_big_only_middle.append(points)
                        ##################

                     
                        if len(points)>5:
                            msd, rmsd = compute_msd(points)
                            mean_msd, logD = logD_from_mean_MSD(msd)
                
                            lys_msd_cluster.append(mean_msd)
                            lys_logD_cluster.append(logD)

                        ####################
                            if lys_start_end_cluster2[j][i][0]!="B":
                                msd_middle, rmsd_middle = compute_msd(points)
                                mean_msd_middle, logD_middle = logD_from_mean_MSD(msd_middle)

                                lys_msd_cluster_middle.append(mean_msd_middle)
                                lys_logD_cluster_middle.append(logD_middle)
                



                        lys_hull.append(hull)
                        lys_area.append(hull.volume) 
                        lys_perimeter.append(hull.area) 
                        if plotting_flag==1:
                            for simplex in hull.simplices:
                                plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.5, color="red")

                            plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=0.5, color="black") #was 1
             

            lys_area2.append(lys_area)
            lys_perimeter2.append(lys_perimeter)
            lys_hull2.append(lys_hull)
            lys_points_big2.append(lys_points_big)
            lys_begin_end_big2.append(lys_begin_end_big)
            lys_points_big_only_middle2.append(lys_points_big_only_middle)
          
          
            if len(lys_points_big)>0:
                msd_mean = mean(lys_msd_cluster)
                logD_mean=mean(lys_logD_cluster)
            else:
                msd_mean=0
                logD_mean=0
            lys_msd_cluster2.append(msd_mean)
            lys_logD_cluster2.append(logD_mean)

            ##################
            if len(lys_points_big_only_middle)>0:
                msd_mean = mean(lys_msd_cluster_middle)
                logD_mean=mean(lys_logD_cluster_middle)
            else:
                msd_mean=0
                logD_mean=0
            lys_msd_cluster_middle2.append(msd_mean)
            lys_logD_cluster_middle2.append(logD_mean)


     

        mean_msd_df["cluster_msd"] = lys_msd_cluster2
        mean_msd_df["cluster_logD"]=lys_logD_cluster2

        mean_msd_df["cluster_msd_middle"] = lys_msd_cluster_middle2
        mean_msd_df["cluster_logD_middle"]=lys_logD_cluster_middle2
        
        

        if plotting_flag==1:
            plt.axis('equal') 
            plt.show()
        #print(mean_msd_df)
        return grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df, lys_begin_end_big2, lys_points_big_only_middle2

    ################################################################### end plotting plus convex hull1
   

        
    def convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_begin_end_big2, lys_points_big_only_middle2):
        print("calculating points in hull")
        ################# adding all the points that are additionally in the bounding area as cluster points
        
        lys_the_last=[]
        lys_area_last=[]
        lys_the_last_middle=[]
        lys_area_last_middle=[]

        #print(len(lys_points_big2))
        #print(len(lys_points_big_only_middle2))
        
        c2=0
        
        for trackn in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(trackn[0])
            
            lys_x=list(s["pos_x"])
            lys_y=list(s["pos_y"])
            sum_rows_temp = list(s["sum_rows"])

            for i in range(len(lys_x)):
                interm_lys=[]
                interm_lys_middle=[]
                
                for j in range(len(lys_points_big2[c2])): 
           

                    points=lys_points_big2[c2][j]
                   
                   
                    if [lys_x[i], lys_y[i]] in points:
                        interm_lys.append(0)
                        area=lys_area2[c2][j]
                    
                if len(interm_lys)>0:
                    lys_the_last.append(0)
                    lys_area_last.append(area)
                    
                else:
                
                    lys_the_last.append(1)
                    lys_area_last.append(0)
                  
                    

                for j in range(len(lys_points_big_only_middle2[c2])): 
                    points_middle=lys_points_big_only_middle2[c2][j]

                    
                    if [lys_x[i], lys_y[i]] in points_middle: ## for only clsuters in middle
                        interm_lys_middle.append(0)
                        area_middle=lys_area2[c2][j]
                
                if len(interm_lys_middle)>0:
                        lys_the_last_middle.append(0)
                        lys_area_last_middle.append(area_middle)
                else:
                    lys_the_last_middle.append(1)
                    lys_area_last_middle.append(0)
                

            c2+=1
        c2+=1
    
        deep_df_short["in_hull"]=lys_the_last
        deep_df_short["area"]=lys_area_last
        deep_df_short['in_hull_level'] = pd.cut(deep_df_short["in_hull"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df_short['in_hull_level'] = deep_df_short['in_hull_level'].astype(str)

        deep_df_short["in_hull_middle"]=lys_the_last_middle
        deep_df_short["area_middle"]=lys_area_last_middle
        deep_df_short['in_hull_level_middle'] = pd.cut(deep_df_short["in_hull_middle"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df_short['in_hull_level_middle'] = deep_df_short['in_hull_level_middle'].astype(str)
    
        return deep_df_short
    

    ## here insert function for log D of only non-clsutered:

    def caluclate_diffusion_non_STA_tracks(deep_df_short, mean_msd_df):
        print(mean_msd_df)


        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        lys_logD_no_STA=[]
        for trackn in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(trackn[0])
            pos_x=s["pos_x"]
            pos_y=s["pos_y"]
            if sum(s["in_hull"])==len(s["in_hull"]): # only get trackcs without any clsuter: all are 1=no clsuter
                   
                m= np.column_stack(( pos_x, pos_y))
                msd, rmsd = compute_msd(m)
                mean_msd, logD = logD_from_mean_MSD(msd)
                #lys_logD_no_STA.append([logD]*len(pos_x))
                lys_logD_no_STA.append(logD)


            
            else:
                #lys_logD_no_STA.append([0]*len(pos_x))
                lys_logD_no_STA.append(0)
        
        #lys_logD_no_STA_flat=list(chain.from_iterable(lys_logD_no_STA))
        #deep_df_short["mean_logD_without_STA"]=lys_logD_no_STA_flat
        mean_msd_df["mean_logD_without_STA"]=lys_logD_no_STA
        #print(mean_msd_df)

        return mean_msd_df




    




    ################################################
    ### plotting hull and tracks togehter: as arrested vs not spatially arrested
    def plotting_final_image(deep_df_short,lys_points2, image_path):
        final_pal=dict(zero= "#06fcde" , one= "#808080")
        linecollection = []
        colors = []

        fig = plt.figure()
        ax = fig.add_subplot()

        sns.set(style="ticks", context="talk")

        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        c2=0
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])

        
            for i in range (len(s["pos_x"])-1):

                line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
                color = final_pal[deep_df_short["in_hull_level"][c2]]
                linecollection.append(line)
                colors.append(color)

                c2+=1
            c2+=1

        lc = LineCollection(linecollection, color=colors, lw=1)

        
        plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.001)
        plt.gca().add_collection(lc)

        for j in range (len(lys_points2)):
            for i in range(len(lys_points2[j])):
                points=lys_points2[j][i] 
                if len(points)>3: ##this was 3 why?
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    if ratio<105:
                    
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=1) 

                        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="#008080")
                        #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                        
       
       
        plt.axis('equal') 
        plt.savefig(str(image_path), format="svg") # uncomment this to save nice svg
        plt.show()

        #plt.axis('equal') 
        #plt.savefig(str(image_path), dpi=3500,format="tiff") # uncomment this to save nice tiff
        #plt.show()

        ###insert new plotting function: 

    def plotting_final_image2(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_path_tiff):
        final_pal=dict(zero= "#06fcde" , one= "#808080")
        linecollection = []
        colors = []

        fig = plt.figure()
        ax = fig.add_subplot()

        sns.set(style="ticks", context="talk")

        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        c2=0
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])

        
            for i in range (len(s["pos_x"])-1):

                line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
                color = final_pal[deep_df_short["in_hull_level"][c2]]
                linecollection.append(line)
                colors.append(color)

                c2+=1
            c2+=1

        lc = LineCollection(linecollection, color=colors, lw=0.1) #was1

        
        plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.001, alpha=0)
        plt.gca().add_collection(lc)


        for j in range (len(lys_points_big2)):
            for i in range(len(lys_points_big2[j])):
                points=lys_points_big2[j][i] 
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.1 ) #was 1

                    #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="#008080")
                        #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                        
       
        for j in range (len(lys_points_big_only_middle2)):
                    for i in range(len(lys_points_big_only_middle2[j])):
                        points=lys_points_big_only_middle2[j][i] 
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.1, color="#008080") #was1

                            #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="red")
                                #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                                


        plt.axis('equal') 
        plt.savefig(str(image_path), format="svg") # uncomment this to save nice svg
        plt.show()

        #plt.axis('equal') 
        #plt.savefig(str(image_path_tiff), dpi=3500,format="tiff") # uncomment this to save nice tiff
        #plt.show()
      


     
    ########################### function to make a nice excel fiel with all the parameters per track:

    def make_fingerprint_file(f2, train_result, deep_df_short, dt, mean_msd_df): 
        lys_string=f2.split("\\")
        outpath1=lys_string[:-1]
        outpath2='\\'.join(outpath1)
        name=lys_string[-1].split(".csv")[0]
        outpath3=outpath2+"\\"+name
        print("saving results file in:", outpath3 )


        # adding hull area and number of points in clusters
        lys_nr_of_clusters=[]
        lys_time_in_clusters=[]
        lys_nr_of_unclustered=[]
        lys_mean_area=[]
        lys_sum_clusters=[]
        lys_time_per_cluster=[]
        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])

        

            clusters=s['in_hull'].value_counts()
            areas=s["area"].value_counts()
            lys_interm_area=[]

            
            for i in areas.keys():
                lys_interm_area.append(i)
            lys_interm_area.sort()

            


            if len(clusters)>1:
                # if track contains points both in clusters and not in clusters, assign each type
                lys_nr_of_clusters.append(clusters[0])
                lys_nr_of_unclustered.append(clusters[1])
                lys_time_in_clusters.append(dt*clusters[0])
                lys_mean_area.append(mean(lys_interm_area[1:]))
                lys_sum_clusters.append(len(lys_interm_area[1:]))
                lys_time_per_cluster.append(dt*clusters[0]/len(lys_interm_area[1:]))

                

            else:
                # if track only has one type of point, the "clusters[i]" object has only one entry, either 0 (points in clusters) or 1 (points not in clusters)
                ind=clusters.index[0]
                arry=clusters.array
                lys_mean_area.append(0)

                if ind==1:
                    # no cluster 
                    lys_nr_of_clusters.append(0)
                    lys_nr_of_unclustered.append(arry[0])
                    lys_time_in_clusters.append(dt*0)
                    lys_time_per_cluster.append(0)
                    lys_sum_clusters.append(0)
                else:
                    # all points of track are cluster points
                    lys_nr_of_clusters.append(arry[0])
                    lys_nr_of_unclustered.append(0)
                    lys_time_in_clusters.append(dt*arry[0])
                    lys_time_per_cluster.append(dt*arry[0])
                    lys_sum_clusters.append(1)
            
        # adding the fingerprint outputs:
        counter=0
        for i in train_result:
            counter+=1
            if(counter== 1):
                new_finger1=(np.array([i]))
                
            else:
                new_finger1=np.vstack((new_finger1, i))
        
        fingerprints_df_out_1=pd.DataFrame(new_finger1, columns=["alpha", "beta", "pval", "efficiency", "fractaldim", "gaussianity", "kurtosis", "msd_ratio", "trappedness", "t0", "t1", "t2", "t3","lifetime", "length_of_track",  "mean_steplength","msd" ])
        
        fingerprints_df_out=fingerprints_df_out_1[["alpha", "beta", "pval", "efficiency", "fractaldim", "gaussianity", "kurtosis", "msd_ratio", "trappedness","length_of_track",  "mean_steplength","msd"]]

        fingerprints_df_out["nr_of_spatially_arrested_points_per_track"]=lys_nr_of_clusters
        fingerprints_df_out["nr_of_non-arrested_points_per_track"]=lys_nr_of_unclustered
        fingerprints_df_out["tot_time_of_spatial_arrest_per_track"]=lys_time_in_clusters
        fingerprints_df_out["mean_area_spatial_arrest_events"]=lys_mean_area
        fingerprints_df_out["nr_of_spatial_arrest_events_per_track"]=lys_sum_clusters
        fingerprints_df_out["average_duration_of_spatial_arrest_events_per_track"]=lys_time_per_cluster
        fingerprints_df_out["logD_whole_track"]=mean_msd_df["logD"]
        fingerprints_df_out["MSD_cluster"]=mean_msd_df["cluster_msd"]
        fingerprints_df_out["logD_cluster"]=mean_msd_df["cluster_logD"]


        outpath4=outpath3+"_fingerprint_results"+".xlsx"
        writer = pd.ExcelWriter(outpath4 , engine='xlsxwriter')
        fingerprints_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()

        return fingerprints_df_out
    
    ############################### end function 
    ###### mkae fingerprint for new HMM:
    def make_results_file(f2, deep_df_short, dt, mean_msd_df):
        #print("this is deepdfshort",deep_df_short)

        lys_string=f2.split("\\")
        outpath1=lys_string[:-1]
        outpath2='\\'.join(outpath1)
        name=lys_string[-1].split(".csv")[0]
        outpath3=outpath2+"\\"+name
        print("saving results file in:", outpath3 )

        # adding hull area and number of points in clusters
        lys_nr_of_clusters=[]
        lys_time_in_clusters=[]
        lys_nr_of_unclustered=[]
        lys_mean_area=[]
        lys_sum_clusters=[]
        lys_time_per_cluster=[]

        lys_nr_of_clusters_middle=[]
        lys_nr_of_unclustered_middle=[]
        lys_time_in_clusters_middle=[]
        lys_mean_area_middle=[]
        lys_sum_clusters_middle=[]
        lys_time_per_cluster_middle=[]




        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        for i in grouped_plot["tid"].unique():

            s= grouped_plot.get_group(i[0])
            clusters=s['in_hull'].value_counts()
            areas=s["area"].value_counts()
            lys_interm_area=[]

            for i in areas.keys():
                lys_interm_area.append(i)
            lys_interm_area.sort()

       
            if len(clusters)>1:
                # if track contains points both in clusters and not in clusters, assign each type
                lys_nr_of_clusters.append(clusters[0])
                lys_nr_of_unclustered.append(clusters[1])
                lys_time_in_clusters.append(dt*clusters[0])
                lys_mean_area.append(mean(lys_interm_area[1:]))
                lys_sum_clusters.append(len(lys_interm_area[1:]))
                lys_time_per_cluster.append(dt*clusters[0]/len(lys_interm_area[1:]))

     

            else:
                # if track only has one type of point, the "clusters[i]" object has only one entry, either 0 (points in clusters) or 1 (points not in clusters)
                ind=clusters.index[0]
                arry=clusters.array
                lys_mean_area.append(0)  ## why did I do this?

                if ind==1:
                    # no cluster 
                    lys_nr_of_clusters.append(0)
                    lys_nr_of_unclustered.append(arry[0])
                    lys_time_in_clusters.append(dt*0)
                    lys_time_per_cluster.append(0)
                    lys_sum_clusters.append(0)


                else:
                    # all points of track are cluster points
                    lys_nr_of_clusters.append(arry[0])
                    lys_nr_of_unclustered.append(0)
                    lys_time_in_clusters.append(dt*arry[0])
                    lys_time_per_cluster.append(dt*arry[0])
                    lys_sum_clusters.append(1)
                
            ##try separate loop:
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])
            clusters_middle=s['in_hull_middle'].value_counts()
            areas_middle=s["area_middle"].value_counts()
            lys_interm_area_middle=[]

            for i in areas_middle.keys():
                lys_interm_area_middle.append(i)
            lys_interm_area_middle.sort()

            if len(clusters_middle)>1:
                lys_nr_of_clusters_middle.append(clusters_middle[0])
                lys_nr_of_unclustered_middle.append(clusters_middle[1])
                lys_time_in_clusters_middle.append(dt*clusters_middle[0])
                lys_mean_area_middle.append(mean(lys_interm_area_middle[1:]))
                lys_sum_clusters_middle.append(len(lys_interm_area_middle[1:]))
                lys_time_per_cluster_middle.append(dt*clusters_middle[0]/len(lys_interm_area_middle[1:]))
            
            else:
                ind=clusters_middle.index[0]
                arry=clusters_middle.array
                lys_mean_area_middle.append(0) 
                if ind==1:
                    lys_nr_of_clusters_middle.append(0)
                    lys_nr_of_unclustered_middle.append(arry[0])
                    lys_time_in_clusters_middle.append(dt*0)
                    lys_time_per_cluster_middle.append(0)
                    lys_sum_clusters_middle.append(0)
                
                else:
                
                    lys_nr_of_clusters_middle.append(0)
                    lys_nr_of_unclustered_middle.append(arry[0])
                    lys_time_in_clusters_middle.append(dt*0)
                    lys_time_per_cluster_middle.append(0)
                    lys_sum_clusters_middle.append(1)



                
       # print(lys_nr_of_clusters)
        #print(lys_nr_of_clusters_middle)

        ## below all the fully resolved ones: (only if cluster was in teh middle)
        casta_df_out=pd.DataFrame(lys_nr_of_clusters_middle, columns=["nr_of_STA_points_per_track"])
        casta_df_out["nr_of_non-STA_points_per_trck"]=lys_nr_of_unclustered_middle
        casta_df_out["tot_time_of_STA_per_track"]=lys_time_in_clusters_middle
        casta_df_out["mean_area_of_STA"]=lys_mean_area_middle
        casta_df_out["nr_of_STA_events_per_track"]=lys_sum_clusters_middle
        casta_df_out["average_duration_of_STA_events_per_track"]=lys_time_per_cluster_middle
        casta_df_out["MSD_STA"]=mean_msd_df["cluster_msd_middle"]
        casta_df_out["logD_STA"]=mean_msd_df["cluster_logD_middle"]
        casta_df_out["logD_whole_track"]=mean_msd_df["logD"]

        casta_df_out["logD_tracks_without_STA"]=mean_msd_df["mean_logD_without_STA"]



        # below including everything: also clusters in beginning and end
        casta_df_out["nr_of_SA_points_per_track"]=lys_nr_of_clusters
        casta_df_out["nr_of_non-SA_points_per_track"]=lys_nr_of_unclustered
        casta_df_out["tot_time_of_SA_per_track"]=lys_time_in_clusters
        casta_df_out["mean_area_of_SA"]=lys_mean_area
        casta_df_out["nr_of_SA_events_per_track"]=lys_sum_clusters
        casta_df_out["average_duration_of_SA_events_per_track"]=lys_time_per_cluster
        casta_df_out["MSD_SA"]=mean_msd_df["cluster_msd"]
        casta_df_out["logD_SA"]=mean_msd_df["cluster_logD"]




        outpath4=outpath3+"_CASTA_results"+".xlsx"
        writer = pd.ExcelWriter(outpath4 , engine='xlsxwriter')
        casta_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()

        return casta_df_out
        



        
    ############################### Function for consecutive GT:
    ##  curate sim da: all the things where pm is confined but for less than5 points -> make unconfiend!

    def make_GT_consecutive(sim_tracks_test):
        sim_tracks = sim_tracks_test
        sim_tracks["pm2"]= sim_tracks["pm2"].replace(1,0)
        sim_tracks["pm2"]= sim_tracks["pm2"].replace(2,1)
        
        grouped_plot= sim_tracks.sort_values(["POSITION_T"]).groupby("TRACK_ID")
        c2=0
        lys_final=[]
        for i in grouped_plot["TRACK_ID"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            #print(s)
            c3=0
            while c3<len(s["POSITION_X"]): 

                if c3>=len(s["POSITION_X"])-5: #was 11
                    if sum(s["pm2"][c3:])==0:
                        lys_six.append([0]*1) 
                    else:
                        lys_six.append([1]*1) 
                        
                    
                else:
                    if sum(s["pm2"][c3:c3+6])==0: # was +12
                    
                        lys_six.append([0]*1)
                    elif sum(s["pm2"][c3:c3+6])!=0 and sum(s["pm2"][c3:c3+5])==0: # was +12 adn +11
                        lys_six.append([0]*5) #was *11
                        c2+=4 # was 10
                        c3+=4 #was10
                    else:
                        lys_six.append([1]*1)
                c2+=1
                c3+=1
            lys_six_flat=list(chain.from_iterable(lys_six))
            lys_final.append(lys_six_flat)
            #print("this lys_six", lys_six_flat)
            #print(lys_final)
            c2+=1
            c3=0
        lys_final_flat=list(chain.from_iterable(lys_final))
        
        sim_tracks["GT"]=lys_final_flat
        return sim_tracks
    
    
    ##################################### 
    # function to calculate accuracy:

    def calculate_accuracy(sim_tracks2, finger_tracks, mean_msd_df):
        print(sim_tracks2)
        print(finger_tracks)
        #print("calculate accuracy",sim_tracks2["pm2"])
        #print("arry_predict", arry_predict)
      
        arry_sim=sim_tracks2["GT"] # if 0= confined, 1= not
        arry_finger=finger_tracks["in_hull"] # if 0= confined, 1=not
       
        both_confined=0
        both_unconfined=0
        sim_confined=0
        sim_unconfined=0
        sim_total_confined=0
        sim_total_unconfined=0
        
        
        for i in range(0,len(arry_sim)):
            if arry_sim[i]==0: # confined
                if arry_finger[i]==0:
                    both_confined+=1
                    sim_total_confined+=1

                else:
                    sim_confined+=1
                    sim_total_confined+=1

            else: # not confined
                if arry_finger[i]==1:
                    both_unconfined+=1
                    sim_total_unconfined+=1

                else:
                    sim_unconfined+=1
                    sim_total_unconfined+=1

        # both_confined = True postive
        # both_uncofined= True negative
        # sim_confined= False negative
        # sim_unconfined= false postivie

        percent_both_confined=(both_confined/len(arry_sim))*100
        percent_both_unconfined=(both_unconfined/len(arry_sim))*100
        percent_correct=((both_confined+both_unconfined)/len(arry_sim))*100 # this is officially accuracy
       
        precision, recall, fbeta, support=metrics.precision_recall_fscore_support(arry_sim, arry_finger, pos_label=0)
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
        

        if sim_total_confined!=0:
            percent_correct_confined=(both_confined/sim_total_confined)*100
            #precision=(both_confined/(both_confined+sim_unconfined))*100 # this is precision officially
            #recall=(both_confined/(both_confined+sim_confined))*100 # this is officailly recall

        else:
            percent_correct_confined=100

        if sim_total_unconfined!=0:
            percent_correct_unconfined=(both_unconfined/sim_total_unconfined)*100
        else:
            percent_correct_unconfined=100

        percent_sim_confined=(sim_confined/len(arry_sim))*100
        percent_sim_unconfined=(sim_unconfined/len(arry_sim))*100

       
        ########### calculate mean cluster per track:
        
        lys_total_clusters=[]
        lys_total_clusters_interm=[]
        
        c2=0
        grouped_plot= sim_tracks2.sort_values(["FRAME"]).groupby("TRACK_ID")

        for i in grouped_plot["TRACK_ID"].unique():
            change_counter=0
            beginn_counter=0
            value_counter=0
            total_clusters=0
           
            s= grouped_plot.get_group(i[0])

            for j in range(len(s["GT"])):
                if value_counter==0:
                    if s["GT"][c2]==0:
                        beginn_counter+=1
               
                else:
                    if s["GT"][c2]==0:
                        if s["GT"][c2-1]!=0:
                            change_counter+=1
                c2+=1
                value_counter+=1

            if beginn_counter!=0:
                total_clusters+=1
            total_clusters+=change_counter
            
            lys_total_clusters_interm.append(total_clusters)
        
        lys_total_clusters.append(lys_total_clusters_interm)
        #print("lystotal clsuters", lys_total_clusters)
        
        ############### end
        # calucalte mean number of points in clusters, mean time in clusters 

        count3=0
        lys_nr_of_cluster_points=[]
        lys_time_in_clusters=[]
        lys_nr_of_unclustered_points=[]
        lys_time_per_cluster=[]
        lys_all_points_in_clusters=[]
        logD_means_sim = []
        for i in grouped_plot["TRACK_ID"].unique():
            
            #print("here is i", i, count3)
            #print(lys_total_clusters[0][count3])
            s= grouped_plot.get_group(i[0])

            logD_mean = mean(np.log10(s["DIFFUSION"]))+1 #add +1 since conversion from pixel back to microns
            logD_means_sim.append(logD_mean)

            clusters=s['GT'].value_counts() # number of total clustered/ unclustered points
            #print("this clsuters0", clusters)
            
            #print("this is clsuters",clusters)
            if len(clusters)>1:
                # if track contains points both in clusters and not in clusters, assign each type
                lys_nr_of_cluster_points.append(clusters[0])
                lys_nr_of_unclustered_points.append(clusters[1])
                lys_time_in_clusters.append(dt*clusters[0])
                #lys_sum_clusters.append(len(lys_interm_area[1:]))
                lys_time_per_cluster.append(dt*clusters[0]/lys_total_clusters[0][count3])
                lys_all_points_in_clusters.append(clusters[0]/lys_total_clusters[0][count3])

            else:
                # if track only has one type of point, the "clusters[i]" object has only one entry, either 0 (points in clusters) or 1 (points not in clusters)
                ind=clusters.index[0]
                arry=clusters.array

                if ind==1:
                    # no cluster 
                    lys_nr_of_cluster_points.append(0)
                    lys_nr_of_unclustered_points.append(arry[0])
                    lys_time_in_clusters.append(dt*0)
                    lys_time_per_cluster.append(0)
                    #lys_sum_clusters.append(0)
                else:
                    # all points of track are cluster points
                    lys_nr_of_cluster_points.append(arry[0])
                    lys_nr_of_unclustered_points.append(0)
                    lys_time_in_clusters.append(dt*arry[0])
                    lys_time_per_cluster.append(dt*arry[0])
                    #lys_sum_clusters.append(1)
                    lys_all_points_in_clusters.append(clusters[0]/lys_total_clusters[0][count3])
                    
        
            count3+=1

        mean_total_clusters_per_track=mean(lys_total_clusters[0])
        total_time_in_clusters=mean(lys_time_in_clusters)
        mean_time_per_cluster_per_track=mean(lys_time_per_cluster)
        if len(lys_all_points_in_clusters)>0:
            mean_clustered_points=mean(lys_all_points_in_clusters)
        else:
            mean_clustered_points=0
        #print("points in clsuter", lys_all_points_in_clusters)
        print("mean",mean_clustered_points)



        ###########################

        #print("unique values",set(logD_means_sim))
        sim_cluster_logD=np.log10(min(sim_tracks2["DIFFUSION"]))+1 # add 1 logD for conversion back to microns
        logD_means_finger=list(mean_msd_df["logD"])
        logD_mean_cluster_finger=list(mean_msd_df["cluster_logD"])
        logD_difference=[]
        logD_cluster_difference=[]
        #print("logD sim",logD_means_sim)
        #print("log_D sim_clsuter",sim_cluster_logD )
        #print("logD_finger_tracks",logD_means_finger )
        #print("logD_finger_clsuter", logD_mean_cluster_finger)
        for i in range(len(logD_means_sim)):
            logD_difference.append(abs(logD_means_finger[i]-logD_means_sim[i]))
            if logD_mean_cluster_finger[i]!=0:
                logD_cluster_difference.append(abs(logD_mean_cluster_finger[i]-sim_cluster_logD))

        #for i in range(len(logD_means_sim)):
            #logD_difference.append(abs(logD_means_finger[i]-logD_means_sim[i]))
            #logD_cluster_difference.append(abs(logD_mean_cluster_finger[i]-sim_cluster_logD))
        logD_mean_diff=mean(logD_difference)
        logD_mean_cluster_diff=mean(logD_cluster_difference)
        #print("differnce", logD_cluster_difference)

        print("percent_both_confined", percent_both_confined)
        print("percent_both_unconfined", percent_both_unconfined)
        print("percent_correct", percent_correct)
        print("percent_correct_confined", percent_correct_confined)
        print("percent_correct_unconfined", percent_correct_unconfined)

        print("percent_only_sim_confined", percent_sim_confined)
        print("percent_only_sim_unconfined", percent_sim_unconfined)

        print("precision_confined",precision_confined)
        print("precision_unconfiend",precision_unconfined)
        print("recall_confined",recall_confined)
        print("recall_unconfined", recall_unconfined)
        print("fbeta_confined",fbeta_confined) # weighted mean of precision and recall: 1 good, 0 = bad
        print("fbeta_unconfiend",fbeta_unconfined) 
        print("support_confined",support_confined)
        print("support_unconfined", support_unconfined)
        #print("log D difference", logD_mean_diff)
        #print("cluster logD differnce", logD_mean_cluster_diff)
        print("mean total clusters per_track", mean_total_clusters_per_track)
        #print("total time in clusters per track", total_time_in_clusters)
        #print("mean time per_cluster",mean_time_per_cluster_per_track )

       
        list_accuracy=[percent_both_confined, percent_both_unconfined, percent_correct, percent_correct_confined, percent_correct_unconfined,percent_sim_confined, percent_sim_unconfined,
                      precision_confined,  precision_unconfined,recall_confined, recall_unconfined,  fbeta_confined,fbeta_confined,fbeta_confined, support_confined, support_unconfined, logD_mean_diff, logD_mean_cluster_diff, mean_total_clusters_per_track, total_time_in_clusters, mean_time_per_cluster_per_track,mean_clustered_points ]

        return list_accuracy

    #####################################################
    ## function for plotting for GT and finger here:
    def plot_GT_and_finger(sim_tracks2, finger_tracks, image_path1):
       
        arry_sim=sim_tracks2["GT"] # if 0= confined, 1= not
        arry_finger=finger_tracks["in_hull"] #

        finger_tracks["compairison_level"]=[0]*len(arry_sim)

        ## make: "zero"= both confined, "one"= both unconfined, "two" = sim_confined, "three"= finger confined

        for i in range(0,len(arry_sim)):
                if arry_sim[i]==0: # confined
                    if arry_finger[i]==0:
                    
                        finger_tracks["compairison_level"][i]="zero"
                        #both_confined+=1
                        #sim_total_confined+=1

                    else:
                        finger_tracks["compairison_level"][i]="two"
                        #sim_confined+=1
                        #sim_total_confined+=1

                else: # not confined
                    

                    if arry_finger[i]==1:
                        finger_tracks["compairison_level"][i]="one"
                        #both_unconfined+=1
                        #sim_total_unconfined+=1

                    else:
                        finger_tracks["compairison_level"][i]="three"

                        #sim_unconfined+=1
                        #sim_total_unconfined+=1

        final_pal=dict(zero= "#78d151",one= '#2a788e',two= '#ff3300', three= "#fde624") #all colors 
        # zero= both confined = green= #78d151
        # one= both unconfined= blue= #2a788e
        # two= sim_only_confined= red= 
        # three= finger_onyl_confiend= #fde624

        linecollection = []
        colors = []

        fig = plt.figure()
        ax = fig.add_subplot()

        sns.set(style="ticks", context="talk")

        grouped_plot= finger_tracks.sort_values(["pos_t"]).groupby("tid")
        c2=0
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])

        
            for i in range (len(s["pos_x"])-1):

                line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
                color = final_pal[finger_tracks["compairison_level"][c2]]
                linecollection.append(line)
                colors.append(color)

                c2+=1
            c2+=1

        lc = LineCollection(linecollection, color=colors, lw=1)

        
        plt.scatter(finger_tracks["pos_x"], finger_tracks["pos_y"], s=0.001)
        plt.gca().add_collection(lc)
        plt.axis('equal') 
        plt.savefig(str(image_path1), format="tiff") # uncomment this to save nice svg

        plt.show()

    ############################################
    # run final wrapper functions:
    ############################################
    ## for one file with spt-tracks:

    # f1= input path to file to be analyzed (see example below)
    # f2= outpath where images should be stored (eg. here stored at same location as input file)
    # min_track_length=25 # parameter to set threshold of minimun length of track duration (eg. 25 time points)
    # dt = 0.05  # frame rate in seconds (eg. 50 milliseconds)
    # plotting_flag =0 (0= no plotting, 1=plotting)
    # function:
    # wrapper_one_file(f1, f2, min_track_length, dt, plotting_flag) 

    # example:
    #f1=r"C:\Users\miche\Desktop\Test_deepSPT\tracks to check time in t0\Long_tracks_cell6-3_1476.csv"
    #f1=r"C:\Users\miche\Desktop\simualted tracks\datasets_folder\confinement_0_tracks.csv"
    #f2=f1
    #dt=0.05
    #plotting_flag=0
    #min_track_length=25

    #wrapper_one_file(f1, f2, min_track_length, dt, plotting_flag) 
    ############################################
    ## for folder with multiple real tracks:

    # folderpath1 = path to folder, min_track_length=25, dt=0.05, plotting_flag(0=no plotting, 1=plotting)
    # function:
    # wrapper_multiple_files(folderpath1, min_track_length, dt, plotting_flag) 
    
    # example:
    #dt=0.1
    #plotting_flag=0
    #min_track_length=25
    #folderpath1=r"Z:\labs\Lab_Gronnier\Michelle\TIRFM\7.8.24_At_BAK1_mut\D122A_BL\cluster_diff_plant1"
    #folderpath1=r"X:\labs\Lab_Gronnier\Michelle\TIRFM\17.10.24_At_MADS_mut\3451-3\cleaned"
    #folderpath1=r"X:\labs\Lab_Gronnier\Michelle\simulated_tracks\DC_MSS_fingperprint\test\track"

    #folderpath1=r"X:\labs\Lab_Gronnier\Michelle\TIRFM\HUAN\26.11.24_huan\3802_padbon_FLS2\cleaned"


    #wrapper_multiple_files(folderpath1, min_track_length, dt, plotting_flag) 

    ############################################
    ## for simulating tracks based on parameters stored in a file:

    # f1= input path to values for sim (a .csv file)
    # dt=0.1 
    # plotting_flag=0
    # min_track_length=25
    # plotting_saving_nice_image_flag=0 (0=no saving, 1=saving)
    # tracks_saving_flag=0 (0= no saving, 1=saving tracks.csv)
    # function:
    # read_in_values_and_execute(f1,min_track_length, dt, plotting_flag, plotting_saving_nice_image_flag, tracks_saving_flag)
    
    # example:

    #plotting_flag=0
    #dt=0.1
    #min_track_length=25
   # plotting_saving_nice_image_flag=0
    #tracks_saving_flag=0
    
    #f1=r"Z:\labs\Lab_Gronnier\Michelle\simulated_tracks\test_values5.csv"
    #f1=r"C:\Users\miche\Desktop\simualted tracks\plots\plot_values_D0.001_for_mean_clusters_plot.csv"
    #f1=r"X:\labs\Lab_Gronnier\Michelle\simulated_tracks\DC_MSS_fingperprint\simulation_parameters_for_Sven\Sven_values_D0.001_N500_T200_test.csv"
    #f1=r"X:\labs\Lab_Gronnier\Michelle\simulated_tracks\HMM_model\tracks_16.1.25_test_D0.01\D0.01_N500_T200_for_philip_test.csv"
    #read_in_values_and_execute(f1,min_track_length, dt, plotting_flag, plotting_saving_nice_image_flag, tracks_saving_flag)


#### for our own hmm to evaulate while simualting tracks:
    #f1=r"X:\labs\Lab_Gronnier\Michelle\simulated_tracks\HMM_model\test_model4\sim_values6.1_D0.001_N500_T200_6.12.24.csv"
    #f1=r"Z:\Research\Members\Michelle\simulated_tracks\sim_values_for_accruracy_new_HMM\sim_values_25.3.25_D0.001_N500_T200\sim_values1.1_D0.001_N500_T200_25.3.25.csv"
    #calulate_hmm_precison_with_simulating_tracks( f1,min_track_length, dt, plotting_flag, plotting_saving_nice_image_flag,tracks_saving_flag )



### for files oin a folder with real tracak for our own hmm:
### working on implementaiton of julines STAs only in the middle

    plotting_flag=0
    dt=0.05
    min_track_length=25
    plotting_saving_nice_image_flag=0
    tracks_saving_flag=0
    

    #folderpath1=r"C:\Users\miche\Desktop\simualted tracks\test_real_tracks"
    folderpath1=r"Z:\Research\Members\Michelle\TIRFM\24.09.10_At_FLS2_MADs\pub10_FLS2_3248-4\cleaned_casta\t"


    calculate_spatial_transient_wrapper(folderpath1, min_track_length, dt, plotting_flag)


    ### for accuracy based on previosuly generated tracks in a folder:

    #folderpath1=r"X:\labs\Lab_Gronnier\Michelle\simulated_tracks\DC_MSS_fingperprint\DC_MSS_1\DC_MSS_sim2_D0.015_N500_T200"
    #calculating_HMM_accuracy_from_tracks(folderpath1, min_track_length, dt, plotting_flag)





















    




            
        














                              
        











      
        


                




        


       










    