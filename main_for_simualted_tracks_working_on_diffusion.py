# %%

from RandomWalkSims import (
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
from os.path import isfile, join
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
import warnings
import andi_datasets
from andi_datasets.models_phenom import models_phenom


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    #################################
    # global variables:

    min_track_length=25 # parameter to set threshold of minimun length of track duration (eg. 25 time points)
    dt = 0.05  # frame rate in seconds (eg. 50 milliseconds)
    # f1= input path to file to be analyzed (see example below)
    # f2= path where images should be stored (eg. here stored at same location as input file)

    ##################################
    # if we have csv with tracks: for 1 file only:
    f1=r"C:\Users\miche\Desktop\Test_deepSPT\tracks to check time in t0\Long_tracks_cell6-3_1476.csv"
    f1=r"C:\Users\miche\Desktop\simualted tracks\datasets_folder\confinement_0_tracks.csv"
    f2=f1
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





    ##################################
    # if we dont: need to simulate tracks with ANDI:
    # read in only values for grid paramters for simualtion:



    ####################
    def read_in_values_and_execute(f1,min_track_length, dt, plotting_flag ):
        df_values=pd.read_csv(f1)
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
            list_accuracy=calculate_accuracy(sim_tracks, deep_df_short2, mean_msd_df)

            if index==0:
                list_final_accuracy=[list_accuracy]
            else:
                list_final_accuracy.append(list_accuracy)
            
        
        df_final_accuracy=pd.DataFrame(list_final_accuracy, columns=["percent_both_confined", "percent_both_unconfined","percent_correct","percent_correct_confined","percent_correct_unconfined","percent_sim_confined","percent_sim_unconfined", "precision_confined",  "precision_unconfined","recall_confined", "recall_unconfined",  "fbeta_confined","fbeta_confined","fbeta_confined", "support_confined", "support_unconfined", "logD_mean_diff", "logD_mean_cluster_diff" ])
  
        df_final_parameters_out=pd.concat([df_values,df_final_accuracy],axis=1)
        path_out_accuracy_lys=f1.split(".csv")
        path_out_accuracy=path_out_accuracy_lys[0] +"_sim_accuracy_results.xlsx"
        writer = pd.ExcelWriter(path_out_accuracy , engine='xlsxwriter')
        df_final_parameters_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()




        return df_final_parameters_out
    ##########################
    def make_simulation(number_compartments, radius_compartments, DS1, alphas_value, trans_value):
        N=100
        T=100
        D=0.001
        DS2=1
        L = 1.5*128 #enalrge field ov fiew to avoid boundy effects
        compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments, 
                                                                            r = radius_compartments,
                                                                            L = L)                                

        trajs_model5, labels_model5 = models_phenom().confinement(N = N,
                                                            L = L,
                                                            Ds = [DS1*D, DS2*D],
                                                            comp_center = compartments_center,
                                                            r = radius_compartments,
                                                            trans = trans_value, # Boundary transmittance
                                                            alphas=[1, alphas_value])
        return trajs_model5, labels_model5
    ################################
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


    ##################################
    

    #"""Compute fingerprints"""
    #if not os.path.isfile("X_fingerprints.npy"): 
      #  import pickle
        #print("Generating fingerprints")
      

    ###################### function to directly load the cleaned trackmate files:
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
                ### insert diffusion here
                msd, rmsd = compute_msd(m)
                frames= list(s["FRAME"])
                n= np.column_stack((msd,(frames[1:]),tid[1:]))

                if(count2== 1):
                    msd_all = n
                else:
                    msd_all = np.vstack((msd_all, n))

                msd_df=pd.DataFrame(msd_all, columns=["msd", "frame", "track_id"])


                ## till here
                list_traces.append(m)
                m2=np.column_stack(( tid, pos_x, pos_y, pos_t)) 

                if(count2== 1):
                    deep_all = m2
                else:
                
                    deep_all = np.vstack((deep_all, m2))
        deep_all_df=pd.DataFrame(deep_all, columns=["tid", "pos_x", "pos_y", "pos_t"])

        return deep_all_df, list_traces, lys_x, lys_y, msd_df
    ############################# end function loading
    ### new function for MSD and diffusion:

    def compute_msd(trajectory):
        totalsize=len(trajectory)
        msd=[]
        for i in range(totalsize-1):
            j=i+1
            msd.append(np.sum((trajectory[0:-j]-trajectory[j::])**2)/float(totalsize-j)) # Distance that a particle moves for each time point divided by time
        msd=np.array(msd)
        rmsd = np.sqrt(msd)
        return msd, rmsd 
    
    # def make_msd_df(df):
    #     count2=0 # added
    #     grouped= df.sort_values(["FRAME"]).groupby("TRACK_ID")
    #     msd_all=[]
    #     for i in grouped["TRACK_ID"].unique():

    #         s= grouped.get_group(i[0])
    #         if s.shape[0]>5: #added
    #             count2+=1 #added
    #             frames= list(s["FRAME"])
    #             tid=list(s["TRACK_ID"])
    #             #print(frames)

    #             col= s[["POSITION_X", "POSITION_Y"]]
    #             t= col.to_numpy()
    #             msd, rmsd = compute_msd(t)
    #             m= np.column_stack((msd,(frames[1:]),tid[1:]))
    #             #print(m)
    #             if(count2== 1):
    #                 msd_all = m
    #             else:
    #                 msd_all = np.vstack((msd_all, m))

    #         msd_df=pd.DataFrame(msd_all, columns=["msd", "frame", "track_id"])
    # # print(msd_df)
    #     return msd_df

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







   
        ############################ run the model:

        # if not os.path.isfile("HMMjson"):
        #     steplength = []
        #     for t in traces:
               
        #         x, y = t[:, 0], t[:, 1]
        #         steplength.append(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
        #     print("fitting HMM")
        #     model = HiddenMarkovModel.from_samples(
        #         NormalDistribution, n_components=4, X=steplength, n_jobs=3, verbose=True
        #     )
            
        #     print(model)
        #     model.bake()
        #     print("Saving HMM model")

        #     s = model.to_json()
        #     f = open("HMMjson", "w")
        #     f.write(s)
        #     f.close()
        # else:
        #     print("loading HMM model")
        #     s = "HMMjson"
        #     file = open(s, "r")
        #     json_s = ""
        #     for line in file:
        #         json_s += line
        #     model = HiddenMarkovModel.from_json(json_s)
        #     print(model)

    #### changed:
    def run_traces_wrapper(traces, dt): 
        print("loading HMM model")
        s = "HMMjson"
        file = open(s, "r")
        json_s = ""
        for line in file:
            json_s += line
        model = HiddenMarkovModel.from_json(json_s)
        print(model)

        d = []
        for t in traces: 
            x, y = t[:, 0], t[:, 1]
            SL = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2) * 10 # factor to scale step length (eg. 10)
            d.append((x, y, SL, dt))
        
        print("Computing fingerprints")
        print(f"Running {len(traces)} traces")
    

        train_result = []
        lys_states=[]
        for t in tqdm(d): # t = one track, make states per one step for plotting
            
            train_result.append(ThirdAppender(t, model=model)) 
            states = GetStatesWrapper(t, model)
            lys_states.append(states)
        return train_result, states, lys_states


        #### till here
        ##################################################

        ############# function for consecutive features:
        
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

        ######### make consecutive angles:
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
        
        ######################################### end function
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
        ##### check line intersection: for consistency: 0 = intersection, 1 = not
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
        
        ################################################## end intersection function
        
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
    
    
        ###########################################
        ############## plot all features togheter (plus convex hull):
    def plotting_all_features_and_caculate_hull(deep_df, mean_msd_df, plotting_flag): # add ture =1or false =0 for plotting yes or no
        print("plotting all features")


        deep_df_short=deep_df[["angle_cont", "state_0_cont","dist_cont" ,"intersect_cont" , "KDE_cont"]]
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

            lc = LineCollection(linecollection, color=colors, lw=1)
            
            fig = plt.figure()
            ax = fig.add_subplot()
            sns.set(style="ticks", context="talk")
        
            plt.gca().add_collection(lc)
            plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.001)
    
        
        ########################## calculate convex hull:
        # get red and green points: = where 5, 4 or 3 criteria agree for spatial arrest
        
        lys_points2=[] 
        
        c2=0
        for j in grouped_plot["tid"].unique():
            flag=0
        
            s= grouped_plot.get_group(j[0])
            lys_points=[]
            for i in range (len(s["pos_x"])-1):
            
                if s["sum_rows"][c2]==0 or s["sum_rows"][c2]==1 or s["sum_rows"][c2]==2:
                    pos_x=s["pos_x"][c2]
                    pos_y=s["pos_y"][c2]
                    m= np.column_stack(( pos_x, pos_y))
                                
                    if flag==0:
                        pos_all=m
                        flag+=1
                    else:
                        
                        if i == len(s["pos_x"])-2:
                            pos_all = np.vstack((pos_all,m))
                            lys_points.append(pos_all)
                            flag = 0
                        else:
                            pos_all = np.vstack((pos_all,m))
                else:
                    if flag!=0:
                        lys_points.append(pos_all)
                    
                    flag=0
                c2+=1
            
            lys_points2.append(lys_points)
                
            c2+=1
        
        ######################### plot points togehter with above lines
        lys_area2=[]
        lys_perimeter2=[]
        lys_hull2 = []
        lys_points_big2=[]
        lys_logD_cluster2=[]
        lys_msd_cluster2=[]
        for j in range (len(lys_points2)):
            lys_area=[]
            lys_perimeter=[]
            lys_hull=[]
            lys_points_big=[]
            lys_logD_cluster=[]
            lys_msd_cluster=[]
            for i in range(len(lys_points2[j])):
                points=lys_points2[j][i] 
                if len(points)>5:
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    if ratio<105:
                        lys_points_big.append(points)
                        ###added here:
                        if len(points)>5:
                            msd, rmsd = compute_msd(points)
                            mean_msd, logD = logD_from_mean_MSD(msd)
                
                            lys_msd_cluster.append(mean_msd)
                            lys_logD_cluster.append(logD)


                        lys_hull.append(hull)
                        lys_area.append(hull.volume) 
                        lys_perimeter.append(hull.area) 
                        if plotting_flag==1:
                            for simplex in hull.simplices:
                                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

                            plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1)
            lys_area2.append(lys_area)
            lys_perimeter2.append(lys_perimeter)
            lys_hull2.append(lys_hull)
            lys_points_big2.append(lys_points_big)
            print("lys_points len",len(lys_points_big))
            if len(lys_points_big)>0:
                print("lys_msd_clsuter",lys_msd_cluster)
                msd_mean = mean(lys_msd_cluster)
                logD_mean=mean(lys_logD_cluster)
            else:
                msd_mean=0
                logD_mean=0
            lys_msd_cluster2.append(msd_mean)
            lys_logD_cluster2.append(logD_mean)


            #try:
             #   msd_mean = mean(lys_msd_cluster)
            #except:
            #    print('error')
            #    print(lys_msd_cluster)
            #lys_msd_cluster2.append()
            #lys_logD_cluster2.append(mean(lys_logD_cluster))

        mean_msd_df["cluster_msd"] = lys_msd_cluster2
        mean_msd_df["cluster_logD"]=lys_logD_cluster2

        if plotting_flag==1:
            plt.axis('equal') 
            plt.show()
        return grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df

        ###################################################################
        
    def convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short):
        print("calculating points in hull")
        ################# adding all the points that are additionally in the bounding area as cluster points
        
        lys_the_last=[]
        lys_area_last=[]
        
        c2=0
        
        for trackn in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(trackn[0])
            
            lys_x=list(s["pos_x"])
            lys_y=list(s["pos_y"])
            sum_rows_temp = list(s["sum_rows"])

            for i in range(len(lys_x)):
                interm_lys=[]
                
                for j in range(len(lys_points_big2[c2])): 

                    points=lys_points_big2[c2][j]
                
                    hull=lys_hull2[c2][j]
                    hull_path = Path( points[hull.vertices] )
                    
                    if hull_path.contains_point((lys_x[i], lys_y[i]))==True: 

                        interm_lys.append(0)
                        area=lys_area2[c2][j]
                
                if len(interm_lys)>0:
                    lys_the_last.append(0)
                    lys_area_last.append(area)
                else:
                
                    lys_the_last.append(1)
                    lys_area_last.append(0)
            c2+=1
        c2+=1
    
        deep_df_short["in_hull"]=lys_the_last
        deep_df_short["area"]=lys_area_last
        deep_df_short['in_hull_level'] = pd.cut(deep_df_short["in_hull"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df_short['in_hull_level'] = deep_df_short['in_hull_level'].astype(str)
    
    
        return deep_df_short

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
                if len(points)>3:
                    
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



        #print(fingerprints_df_out)

    
        outpath4=outpath3+"_fingerprint_results"+".xlsx"
        writer = pd.ExcelWriter(outpath4 , engine='xlsxwriter')
        fingerprints_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
        writer.close()

        #### for simulated stuff want the whole excel:
        #outpath5=outpath3+"_fingerprint_tracks"+".xlsx"
       # writer = pd.ExcelWriter(outpath5 , engine='xlsxwriter')
        #print(outpath4, outpath5)
        #deep_df_short.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
       # writer.close()

        return fingerprints_df_out
    
    ############################### end function 
        
    

        ################################
        ## function calcualting accuracy:

        ###  TODO curate sim da: all the things where pm is confine dbit for less than5 points -> make unconfiend!!!!!





        #####################################

    def calculate_accuracy(sim_tracks, finger_tracks, mean_msd_df):
        arry_sim=sim_tracks["pm2"] # if 1= confined, 2= not
        arry_finger=finger_tracks["in_hull"] # if 0= confined, 1=not
        both_confined=0
        both_unconfined=0
        sim_confined=0
        sim_unconfined=0
        sim_total_confined=0
        sim_total_unconfined=0
        sim_cluster_logD=np.log10(min(sim_tracks["DIFFUSION"]))
        
        for i in range(0,len(arry_sim)):
            if arry_sim[i]==1.0: # confined
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
       
        from sklearn import metrics
        arry_finger[arry_finger==1]=2
        arry_finger[arry_finger==0]=1
        arry_sim_int=arry_sim.astype(int)
        print("heere",arry_finger)
        print("sim", arry_sim_int)

        precision, recall, fbeta, support=metrics.precision_recall_fscore_support(arry_sim_int, arry_finger)
        precision_confined=precision[0]
        precision_unconfined=precision[1]
        recall_confined=recall[0]
        recall_unconfined=recall[1]
        fbeta_confined=fbeta[0]
        fbeta_confined=fbeta[1]
        support_confined=support[0]
        support_unconfined=support[0]


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

        logD_means_sim = []
        grouped_plot= sim_tracks.sort_values(["FRAME"]).groupby("TRACK_ID")
        for i in grouped_plot["TRACK_ID"].unique():
            s= grouped_plot.get_group(i[0])

            logD_mean = mean(np.log10(s["DIFFUSION"]))
            logD_means_sim.append(logD_mean)

        print("unique values",set(logD_means_sim))
        logD_means_finger=list(mean_msd_df["logD"])
        logD_mean_cluster_finger=list(mean_msd_df["cluster_logD"])
        logD_difference=[]
        logD_cluster_difference=[]

        for i in range(len(logD_means_sim)):
            logD_difference.append(abs(logD_means_finger[i]-logD_means_sim[i]))
            logD_cluster_difference.append(abs(logD_mean_cluster_finger[i]-sim_cluster_logD))
        logD_mean_diff=mean(logD_difference)
        logD_mean_cluster_diff=mean(logD_cluster_difference)


        



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
        print("fbeta_unconfiend",fbeta_confined) 
        print("support_confined",support_confined)
        print("support_unconfined", support_unconfined)
        print("log D differnce", logD_mean_diff)
        print("cluster logD differnce", logD_mean_cluster_diff)

       
        list_accuracy=[percent_both_confined, percent_both_unconfined, percent_correct, percent_correct_confined, percent_correct_unconfined,percent_sim_confined, percent_sim_unconfined,
                      precision_confined,  precision_unconfined,recall_confined, recall_unconfined,  fbeta_confined,fbeta_confined,fbeta_confined, support_confined, support_unconfined, logD_mean_diff, logD_mean_cluster_diff ]

        return list_accuracy

        #calculate_accuracy(tracks_input, deep_df_short)

    ########################
    # run final wrapper functions:
    ## for one file woth tracks:
    # f1= input, f2=outputpath, (can be the same), min_track_length=25, dt=0.05, plotting_flag(0=no plotting, 1=plotting)
    # wrapper_one_file(f1, f2, min_track_length, dt, plotting_flag) 
    ## for simulating tracks based on parameters stored in a file:
    # read_in_values_and_execute(f1,min_track_length, dt, plotting_flag )
    # f1= input path to values for sim, min_track_length=25, dt=0.05, plotting_flag(0=no plotting, 1= plotting)
    plotting_flag=0
    dt=0.1
    #wrapper_one_file(f1, f2, min_track_length, dt,plotting_flag ) 
    f1=r"C:\Users\miche\Desktop\simualted tracks\datasets_folder\test_values - Kopie.csv"
    
    read_in_values_and_execute(f1,min_track_length, dt, plotting_flag )





















    




            
        














                              
        











      
        


                




        


       










    