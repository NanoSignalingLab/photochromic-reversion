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
warnings.filterwarnings('ignore')

       




if __name__ == '__main__':
    
    #################################
    # global variables:

    min_track_length=25 # parameter to set threshold of minimun length of track duration (eg. 25 time points)
    dt = 0.05  # frame rate in seconds (eg. 50 milliseconds)
    # f1= input path to file to be analyzed (see example below)
    # f2= path where images should be stored (eg. here stored at same location as input file)

    ##################################
    
    f1=r"C:\Users\miche\Desktop\Test_deepSPT\cleaned_trackmate_1473_5_488.csv"
    f2=f1
    
    ##################################
    image_path_lys=f1.split("csv")
    image_path=image_path_lys[0] +"svg"

    """Compute fingerprints"""
    if not os.path.isfile("X_fingerprints.npy"): 
        import pickle

        print("Generating fingerprints")
      

        ###################### function to directly load the cleaned trackmate files:
        def load_file(path2, min_track_length):
            df=pd.read_csv(path2)
            deep_df, list_traces, lys_x, lys_y= make_deep_df(df, min_track_length)
            return deep_df, list_traces, lys_x, lys_y

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
                    list_traces.append(m)
                    m2=np.column_stack(( tid, pos_x, pos_y, pos_t)) 

                    if(count2== 1):
                        deep_all = m2
                    else:
                    
                        deep_all = np.vstack((deep_all, m2))
            deep_all_df=pd.DataFrame(deep_all, columns=["tid", "pos_x", "pos_y", "pos_t"])

            return deep_all_df, list_traces, lys_x, lys_y
        ############################# end function loading

        deep_df, traces, lys_x, lys_y = load_file(f1, min_track_length) # execute this function to load the files

        ############################ run the model:

        if not os.path.isfile("HMMjson"):
            steplength = []
            for t in traces:
               
                x, y = t[:, 0], t[:, 1]
                steplength.append(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
            print("fitting HMM")
            model = HiddenMarkovModel.from_samples(
                NormalDistribution, n_components=4, X=steplength, n_jobs=3, verbose=True
            )
            
            print(model)
            model.bake()
            print("Saving HMM model")

            s = model.to_json()
            f = open("HMMjson", "w")
            f.write(s)
            f.close()
        else:
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
        lys_msd=[]
        for t in tqdm(d): # t = one track, make states per one step for plotting
            
            train_result.append(ThirdAppender(t, model=model)) 
            states = GetStatesWrapper(t, model)
            lys_states.append(states)

        
        ##################################################

        ############# function for consecutive features:
        
        def consecutive(col, seg_len, threshold): # col= string of cl indf, seg_len=segment length of consecutive, threshold number
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

        ################### end distance

        ################### calulcate angles:

        def angle3pt(a, b, c):
            ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
            return ang + 360 if ang < 0 else ang
        
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

        angle_cont_lys=consecutive("angles", 10, 600)
        deep_df["angle_cont"]=angle_cont_lys
        deep_df['angles_cont_level'] = pd.cut(deep_df["angle_cont"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df['angles_cont_level'] = deep_df['angles_cont_level'].astype(str)
        #final_pal_only_0=dict(zero='#fde624' ,  one= '#380282') # zero=yellow=high angles

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
        KDE_cont_lys=consecutive("KDE_invert", 10, 13)

        deep_df["KDE_cont"]=KDE_cont_lys
        deep_df["KDE_cont_level"] = pd.cut(deep_df["KDE_cont"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df["KDE_cont_level"] = deep_df["KDE_cont_level"].astype(str)

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

        intersect_cont=consecutive("all_intersect", 10, 6)
        deep_df["intersect_cont"]=intersect_cont

        ########################### end intersections
   

        ########################## get fingertprint states:
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
       
       
        ###########################################
        ############## plot all features togheter (plus convex hull):
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
        for j in range (len(lys_points2)):
            lys_area=[]
            lys_perimeter=[]
            lys_hull=[]
            lys_points_big=[]
            for i in range(len(lys_points2[j])):
                points=lys_points2[j][i] 
                if len(points)>3:
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    if ratio<105:
                        lys_points_big.append(points)
                        
                        lys_hull.append(hull)
                        lys_area.append(hull.volume) 
                        lys_perimeter.append(hull.area) 
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

                        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1)
            lys_area2.append(lys_area)
            lys_perimeter2.append(lys_perimeter)
            lys_hull2.append(lys_hull)
            lys_points_big2.append(lys_points_big)

        plt.axis('equal') 
        plt.show()

        ###################################################################
        print("calculating points in hull")
        ################# adding all the points that are additionally in the bounding area as cluster points
        
        lys_the_last=[]
        lys_area_last=[]
        lys_hull_path=[]
        
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
                    lys_hull_path.append(hull_path)
                    
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
       
        
        








        ################################################
        ### plotting hull and tracks togehter: as arrested vs not spatially arrested
        
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
        #plt.savefig(str(image_path), format="svg") # uncomment this to save nice svg
        plt.show()
     
        ########################### function to make a nice excel fiel with all the parameters per track:

        def make_fingerprint_file(f2, train_result): 
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


            outpath4=outpath3+"_fingerprint_results"+".xlsx"
            writer = pd.ExcelWriter(outpath4 , engine='xlsxwriter')
            fingerprints_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
            writer.close()
            return fingerprints_df_out
        
        ############################### end function 
        
        #make_fingerprint_file(f2, train_result) # run function to make excel with all parameters

        ################################









    




            
        














                              
        











      
        


                




        


       










    