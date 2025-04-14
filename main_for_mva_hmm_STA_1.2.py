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
#from pomegranate import *
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
from hmm_functions import run_model


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    #################################
   
    import stochastic
    #stochastic.random.seed(3)
    #np.random.seed(7)

    ##########################################
   

    #############################################
    # for our own HMM for reading inmultiple csv files with real tracks, make one result excel per file

    def calculate_spatial_transient_wrapper(folderpath1, min_track_length, dt, plotting_flag,image_saving_flag ):
        onlyfiles = [f for f in listdir(folderpath1) if isfile(join(folderpath1, f))]
        for i in onlyfiles:
            
            if i.endswith(".csv"):
                path=os.path.join(folderpath1, i)
                print(path)
                image_path_lys=path.split("csv")
                if image_saving_flag=="svg":
                    image_path=image_path_lys[0] +"svg"
                else:
                    image_path=image_path_lys[0] +"tiff"

                tracks_input, deep_df1, traces, lys_x, lys_y, msd_df = load_file(path, min_track_length) # execute this function to load the files
                mean_msd_df=msd_mean_track(msd_df, dt)

                deep_df2= run_traces_wrapper(deep_df1, dt)
                deep_df3=computing_distance_wrapper(deep_df2)
                deep_df4=calculate_angles_wrapper(deep_df3)
                deep_df5=calculate_KDE_wrapper(lys_x, lys_y, deep_df4)
                deep_df6=calculate_intersections_wrapper(lys_x, lys_y, deep_df5)

                grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df1, lys_begin_end_big2, lys_points_big_only_middle2=plotting_all_features_and_caculate_hull(deep_df6, mean_msd_df, plotting_flag)
                deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_begin_end_big2, lys_points_big_only_middle2)

                mean_msd_df2=caluclate_diffusion_non_STA_tracks(deep_df_short2,mean_msd_df1 )

                plotting_final_image2(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_saving_flag)
                make_results_file(path, deep_df_short2, dt,mean_msd_df2 ) # run function to make excel with all parameters



    #############################################

  
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
                            lys_start_end_cluster.append(lys_test1)
                          
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
        #print(mean_msd_df)


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

        return mean_msd_df


    ################################################
    ###insert new plotting function: 

    def plotting_final_image2(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_saving_flag):
        final_pal=dict(zero= "#06fcde" , one= "#808080")
        linecollection = []
        colors = []
        if image_saving_flag=="tiff":
            lw1=0.1
            s1=0.001
        else:
            lw1=1
            s1=0.1
        

        #fig = plt.figure() # was this before
        fig, ax = plt.subplots(1)
        #ax = fig.add_subplot()

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

        lc = LineCollection(linecollection, color=colors, lw=lw1)

        
        plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=s1, alpha=0)
        plt.gca().add_collection(lc)


        for j in range (len(lys_points_big2)):
            for i in range(len(lys_points_big2[j])):
                points=lys_points_big2[j][i] 
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=lw1, color="green") # all SA

                    #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="#008080")
                        #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                        
       
        for j in range (len(lys_points_big_only_middle2)):
                    for i in range(len(lys_points_big_only_middle2[j])):
                        points=lys_points_big_only_middle2[j][i] 
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=lw1, color="red") # only middle STA

                            #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="red")
                                #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                                


        if image_saving_flag=="svg":
            plt.axis('equal') 
            plt.savefig(str(image_path), format="svg") # 
            plt.show()
        else:
            #plt.axis('equal') # was this before
            ax.axis("equal")
            #axes = plt.gca()
            xmin, xmax=ax.get_xlim()
            ymin, ymax=ax.get_ylim()
            print(xmin, xmax)
            print(ymin, ymax)


                      # draw vertical line from (70,100) to (70, 250)$
            plt.plot([xmax-2, xmax-1], [ymin+1, ymin+1], 'k-', lw=1)

            plt.savefig(str(image_path), dpi=1500,format="tiff") # was 3500
            plt.show()
      
  
    ###### make reuslts file for CASTA new HMM:
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
        casta_df_out["nr_of_non-STA_points_per_track"]=lys_nr_of_unclustered_middle
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
        




##################################################

### for files in a folder with real tracak for our own hmm:
### working on implementaiton of julines STAs only in the middle

    plotting_flag=0
    dt=0.05
    min_track_length=25
    plotting_saving_nice_image_flag=0
    image_saving_flag="svg"
    image_saving_flag="tiff"



    

    #folderpath1=r"C:\Users\miche\Desktop\simualted tracks\test_real_tracks"
<<<<<<< HEAD
    folderpath1=r"C:\Users\miche\Desktop\simualted tracks\tracks_problem"
=======
    folderpath1=r"C:\Users\Philip\Desktop\tracks"
>>>>>>> f990cad14961a36ba8a72f2e1b8ac678df8cb3be


    calculate_spatial_transient_wrapper(folderpath1, min_track_length, dt, plotting_flag, image_saving_flag)






















    




            
        














                              
        











      
        


                




        


       










    