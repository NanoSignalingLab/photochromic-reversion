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

       




if __name__ == '__main__':
    # decide if you want a temporal resolution for relevant D estimation
    dt = 0.05  # s ##changed from 1/30 to 0.05
    
    
    f1=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\tracks to check time in t0\Long_tracks_cell7_1476.csv"
    f1=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\tracks to check time in t0\cleaned_trackmate_1476-1.csv"
    f1=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\tracks to check time in t0\cleaned_trackmate_1476-1_488.csv"

    f1=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\ROP6_cleaned\cleaned_trackmate_1476-3_IAA_488.csv" # problem
    f2=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\ROP6_cleaned\cleaned_trackmate_1476-3_IAA_488.csv" #problem

    f1=r"C:\Users\bcgvm01\Documents\Deep_SPT\tracks\ROP6_cleaned\cleaned_trackmate_1475_7_25w488.csv"
    f1=r"D:\TIRFM\Michelle von arx\1.2.24\atbak1_flg22_root\trackmate\cleaned_setting1\cleaned_trackmate_p1_004.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_1_488.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_4_488.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_5_488.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_6_488.csv" #weird one
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_7_488.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_8_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-3_IAA_488.csv" #problem
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-4_IAA_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-5_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-6_IAA_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-7_IAA_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-9_IAA_488.csv"
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-10_IAA_488.csv"

    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_3_488.csv" #one with time0.05
    #tested:
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-10_IAA_488.csv" #one weird cluster triangle
    f1=r"D:\Data for fingerprinting\ROP6_cleaned\cleaned_trackmate_1476-9_IAA_488.csv"
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\c_1942-3_488.csv"
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\c_1942-5_488.csv"
    f1=r"D:\Data for fingerprinting\1475_cleaned\cleaned_trackmate_1475_1_488.csv" # one cluster that is hulled but not higlighted, wtf if change min len to 75, its higlightd
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\cleaned_trackmate_1473_5_488.csv"
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\cleaned_trackmate_1474_3_1uW488.csv"
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\cleaned_trackmate_1476-1_488.csv"
    f1=r"D:\Data for fingerprinting\tracks to check time in t0\cleaned_trackmate_1477_1_25w488.csv"
    f1=r"D:\Data for fingerprinting\1477_cleaned\cleaned_trackmate_1477_2_488.csv"
####### data:
    f1=r"D:\Data for fingerprinting\1474_cleaned\cleaned_trackmate_1474_36_488.csv" #has one weird hull!!
    f1=r"X:\labs\Lab_Gronnier\Manuscripts\2024_LT_single molecule\Data\Data for fingerprinting\Data_fingerprint_new\ROP6_cleaned\cleaned_trackmate_1476-10_IAA_488.csv"
    f1=r"D:\TIRFM\Michelle von arx\1.2.24\atbak1_flg22_root\trackmate\cleaned_setting2\cleaned_trackmate_p2_001.csv" # weird cluster!
    f1=r"D:\TIRFM\Michelle von arx\1.2.24\atbak1_flg22_root\trackmate\cleaned_setting2\cleaned_trackmate_p2_009.csv" # weird cluser!
    f1=r"D:\TIRFM\Michelle von arx\1.2.24\atbak1_mock_root\trackmate\cleaned_setting2\cleaned_trackmate_p2_008.csv"
    f1=r"D:\Data for fingerprinting\1474_cleaned\cleaned_trackmate_1474_36_488.csv" #has one weird hull!!
  
    f1=r"X:\labs\Lab_Gronnier\Manuscripts\2024_LT_single molecule\Data\Data for fingerprinting\Data_fingerprint_new\1942_cleaned\cleaned_trackmate_1942_31_488.csv"
    f2=f1





    f2=f1
    image_path_lys=f1.split("csv")
    #print(image_path_lys[0], "here")
    image_path=image_path_lys[0] +"svg"



    """Compute fingerprints"""
    if not os.path.isfile("X_fingerprints.npy"): 
        import pickle

        print("Generating fingerprints")
      

        ###################### function to directly load the cleaned trackmate files:
        def load_file(path2):
            df=pd.read_csv(path2)
            deep_df, list_traces, lys_x, lys_y= make_deep_df(df)
            return deep_df, list_traces, lys_x, lys_y

        def make_deep_df(df):

            grouped= df.sort_values(["FRAME"]).groupby("TRACK_ID")
            count2=0
            deep_all=[]
            list_traces=[]
            lys_x=[]
            lys_y=[]

            for i in grouped["TRACK_ID"].unique():
                 # removedcount2+=1
                s= grouped.get_group(i[0])
                #print(s)
                if s.shape[0]>110: # can set how long the minimum tracks should be 75
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

        deep_df, traces, lys_x, lys_y = load_file(f1) # execute this to load the files

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
        for t in traces: # 
            x, y = t[:, 0], t[:, 1]
            SL = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2) * 10 # We added this to scale our tracks

            d.append((x, y, SL, dt))
          

       
        print("Computing fingerprints")
        print(f"Running {len(traces)} traces")
       

        train_result = []
        lys_states=[] # I added
        lys_msd=[]
        for t in tqdm(d): # t=is one track, make states per one step for plotting
            
            train_result.append(ThirdAppender(t, model=model)) 
            states = GetStatesWrapper(t, model)
            lys_states.append(states)

            #msd_ = GetMSDWrapper(t)
            #print(len(msd_))
            #lys_msd.append(msd_)
        
        #np.save("X_fingerprints", train_result) # uncomment this befor usig it, if we want to use their pre-computed  x
        
        ##################################################

        ############# function for consecutive ones:
        
        def consecutive(col, seg_len, threshold): #col=stringof cl indf, seg_len=segmentlengh of consecutive, threshold number
            grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")
            lys_final=[]
            for i in grouped_plot["tid"].unique():
                lys_six=[]
                s= grouped_plot.get_group(i[0])
                c3=0
                seg1=seg_len-1
                seg2=seg_len+1
                
                while c3<len(s["pos_x"]): # changeed from c2 to c3
                    if c3>=len(s["pos_x"])-seg_len: # for 4 consecuteive put -3, changed from c2, cahnge to 6, for11:=-11
                            lys_six.append([1]*1) # for simplicity
                    else:
                            
                        if sum(s[col][c3:c3+seg2])<threshold: # for 11:= +12
                            lys_six.append([0]*1)
                        elif sum(s[col][c3:c3+seg2])>=threshold and sum(s[col][c3:c3+seg_len])<threshold: # for11:=+12, +11
                            lys_six.append([0]*seg_len) # for11:=11
                            c3+=seg1 #for11:=10
                        else:
                            lys_six.append([1]*1)
                    c3+=1
                lys_six_flat=list(chain.from_iterable(lys_six))
                lys_final.append(lys_six_flat)
                c3=0

            lys_final_flat=list(chain.from_iterable(lys_final))
            return lys_final_flat
        
        ################################################ end function
        
        ################## MSD calc, wasn't really useful as a criterion
        # msd_diff = []
        # msd_diff_col = []
        # for i in lys_msd:
        #     diff = np.diff(i)
        #     diff = np.append(diff, 0)
        #     msd_diff.append(diff)

        #     norm_diff = ((diff-np.min(diff))/(np.max(diff)-np.min(diff)))
        #     #norm_diff = np.append(norm_diff, 0)

        #     msd_diff_col.append(norm_diff)
        
        # flat_msd=np.concatenate(lys_msd)
        # flat_msd_diff=np.concatenate(msd_diff)
        # flat_msd_diff_col=np.concatenate(msd_diff_col)
        ################# End of MSD stuff


        ################# Calculate distance and add to df
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

        ################## End distance calc

        ################# Find consecutive short distances (4 in this case)
        tresh_l = 9
        c2=0
        dist_final=[]
        grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")

        for i in grouped_plot["tid"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            c3=0
            while c3<len(s["pos_x"]): # changeed from c2 to c3

                if c3>=len(s["pos_x"])-tresh_l:
                    lys_six.append([1]*1) # for simplicity
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

        ################## calulcate angles:

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
        final_pal_only_0=dict(zero='#fde624' ,  one= '#380282') # zero=yellow=high angles

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

        #s2= sns.lineplot(data=deep_df, x=deep_df["pos_x"], y=deep_df["pos_y"], hue=deep_df["tid"], sort=False)
        #plt.show()
        #plt.close()


        deep_df["KDE"]=out
        deep_df['KDE_level']=pd.qcut(deep_df["KDE"], 9,labels=["zero" , "one", "two", "three", "four", "five", "six", "seven", "eight"])
        deep_df['KDE_values']=pd.qcut(deep_df["KDE"], 9,labels=False)
        deep_df['KDE_level'] = deep_df['KDE_level'].astype(str)
        final_pal=dict(zero= '#380282',one= '#440053',two= '#404388', three= '#2a788e', four= '#21a784', five= '#78d151', six= '#fde624', seven="#ff9933", eight="#ff3300")

        ## make Juliens KDE plot:
        #linecollection = []
        #colors = []
        #grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")
       # c2=0
        #for i in grouped_plot["tid"].unique():
        #    s= grouped_plot.get_group(i[0])
        #    for i in range (len(s["pos_x"])-1):

         #       line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
         #       color = final_pal[deep_df["KDE_level"][c2]]
         #       linecollection.append(line)
         #       colors.append(color)

         #       c2+=1
         #   c2+=1

        #lc = LineCollection(linecollection, color=colors, lw=1)
        
        #rcParams["figure.figsize"]= 10, 10
        #sns.set(style="ticks", context="talk")
        #plt.style.use("dark_background")
        #plt.gca().add_collection(lc)
        #plt.scatter(deep_df["pos_x"], deep_df["pos_y"], s=0.001)
        #plt.show()
        #plt.close()

        ### or:
        #rcParams["figure.figsize"]= 10, 10
        #plt.style.use("dark_background")
        #sns.kdeplot(data=deep_df, x="pos_x", y="pos_y",fill=True, thresh=0, levels=100, cmap="magma",) # cmap="mako"
       # s2= sns.lineplot(data=deep_df, x=deep_df["pos_x"], y=deep_df["pos_y"], hue=deep_df["tid"], sort=False)
        #plt.axis('square')

        #plt.show()
        #plt.close()





        
        # invert KDEvalues: for consistency, low values = good
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
        
        ################################################## end function
        print("Computing intersections")

        inter_flat1, inter_flat2, inter_flat3, inter_flat4=calc_intersections(lys_x, lys_y)

        ### add all intersections:
        deep_df["intersect1"]=inter_flat1
        deep_df["intersect2"]=inter_flat2
        deep_df["intersect3"]=inter_flat3
        deep_df["intersect4"]=inter_flat4

        ### put all intersections together:
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
        for i in range(len(lys_states)): ## need to add one value cause states is always one shorter than points!
            lys_states[i].append(1)

        flat_lys=reduce(operator.concat, lys_states)
        deep_df["fingerprint_state"]=flat_lys

        ### all fingerprint states
        deep_df['state_level'] = pd.cut(deep_df["fingerprint_state"], [-1.0, 0.0, 1.0, 2.0,  3.0], labels=["zero" , "one", "two", "three"], include_lowest=True, ordered= False)
        deep_df['state_level'] = deep_df['state_level'].astype(str)
        
        ### change state 1 to state zero: 
        deep_df.loc[deep_df['fingerprint_state'] == 1, 'fingerprint_state'] = 0 # change all others to zero first 

        ########## find consecutive 0 fingerprints
        grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")
        c2=0
        lys_final=[]
        for i in grouped_plot["tid"].unique():
            lys_six=[]
            s= grouped_plot.get_group(i[0])
            c3=0
            while c3<len(s["pos_x"]): # changed from c2 to c3

                if c3>=len(s["pos_x"])-11:
                    lys_six.append([1]*1) # for simplicity
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
        #pd.options.display.max_rows=5000
       
        
        ###########################################
        ############## plot all features togheter:
        print("plotting all features")


        deep_df_short=deep_df[["angle_cont", "state_0_cont","dist_cont" ,"intersect_cont" , "KDE_cont"]]
        deep_df_short["sum_rows"] = deep_df_short.sum(axis=1)
        #print(deep_df_short)
        deep_df_short["row_sums_level"] = pd.cut(deep_df_short["sum_rows"], [0, 1,2, 3, 4,5 ,6], labels=["zero" , "one", "two", "three", "four", "five"], include_lowest=True, ordered= False)
        #final_pal=dict(zero= '#380282',one= '#440053',two= '#404388', three= '#2a788e', four= '#21a784', five= '#78d151', six= '#fde624', seven="#ff9933", eight="#ff3300")
        final_pal=dict(zero= "#ff3300",one= '#fde624',two= '#78d151', three= "#2a788e", four="#404388" , five="#440053") #all colors 
        #final_pal=dict(zero= "#06fcde",one= "#06fcde",two= "#06fcde", three= "#808080", four="#808080" , five="#808080") #all colors 

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
        
        rcParams["figure.figsize"]= 11.7, 11.7#8.27
        sns.set(style="ticks", context="talk")
        #plt.style.use("dark_background")
        plt.gca().add_collection(lc)
        plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.001)
        #plt.show() #use below to plot togheter with hull

        ########################## calculate convex hull:
        ## get red and green points: = where 5, 4 or 3 criteria say its cluster
        
        lys_points2=[] #lys of arrys of points where criteria met
        
        c2=0
        for j in grouped_plot["tid"].unique():
            flag=0
           # print(j)
            s= grouped_plot.get_group(j[0])
            lys_points=[]
            for i in range (len(s["pos_x"])-1):
                #print("i: ", i) 
                if s["sum_rows"][c2]==0 or s["sum_rows"][c2]==1 or s["sum_rows"][c2]==2:
                    pos_x=s["pos_x"][c2]
                    pos_y=s["pos_y"][c2]
                    m= np.column_stack(( pos_x, pos_y))
                                  
                    if flag==0:
                        #print("cluster start")
                        pos_all=m
                        flag+=1
                    else:
                        
                        if i == len(s["pos_x"])-2:
                            #print("LAAAAAAAAAAAAAST POOOOOOOOOOOOOOOOOOOOS CLUUUUUUUUUSTEEEEEEER")
                            pos_all = np.vstack((pos_all,m))
                            lys_points.append(pos_all)
                            flag = 0
                        else:
                            pos_all = np.vstack((pos_all,m))
                else:
                    if flag!=0:
                        #print("LAAAAAAAAAAAAAST POOOOOOOOOOOOOOOOOOOOS")
                        lys_points.append(pos_all)
                        print(len(lys_points))
                    flag=0
                c2+=1
                #flag=0
            lys_points2.append(lys_points)
           # print(lys_points, "lys_points")
                
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
                points=lys_points2[j][i] # points of one cluster
                if len(points)>3:
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    #print(j, ratio)
                    if ratio<105:
                        #print("yes", j, ratio)
                        #print(points)

                        lys_points_big.append(points)
                        
                        lys_hull.append(hull)
                        lys_area.append(hull.volume) #is area
                        lys_perimeter.append(hull.area) # is perimeter
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

                        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1)
            lys_area2.append(lys_area)
            lys_perimeter2.append(lys_perimeter)
            lys_hull2.append(lys_hull)
            lys_points_big2.append(lys_points_big)
        #print(len(lys_hull2))         
        plt.show()
        #print(lys_points_big2[21], "lyspointsbig")


        #print(len(lys_points_big2))
        ###################################################################
        print("calculating points in hull")
        ################# adding all the points that are additionally in the bounding area as cluster points
        
        lys_the_last=[]
        lys_area_last=[]
        
        c2=0
        
        for trackn in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(trackn[0])
            #print("heerelens",len(s["pos_x"]),"c2: ", c2)
            #print(s["pos_x"])
            lys_x=list(s["pos_x"])#new
            lys_y=list(s["pos_y"])#new
           # print(trackn)
            sum_rows_temp = list(s["sum_rows"])

            #for i in range(len(s["pos_x"])): # i= index of point in one track, before
            for i in range(len(lys_x)): #new
                interm_lys=[]
                
                for j in range(len(lys_points_big2[c2])): # j=index of point in hull of cluster c2

                    points=lys_points_big2[c2][j]
                    #print("points_here", points)
                    #print("c2_points", )

                    #print( "lenpoints", len(points))
                   
                    hull=lys_hull2[c2][j]
                    hull_path = Path( points[hull.vertices] )
                    
                    #if hull_path.contains_point((s["pos_x"][i], s["pos_y"][i]))==True: #old
                    if hull_path.contains_point((lys_x[i], lys_y[i]))==True: #new

                        interm_lys.append(0)
                        area=lys_area2[c2][j]
                    # else:
                    #     for d in points[hull.vertices]:
                    #         print(d)
                    #         if (lys_x[i], lys_y[i])==(d[0], d[1]):
                    #             interm_lys.append(0)
                    #             area=lys_area2[c2][j]


                    #if hull_path.contains_point((27.125, 19.15))==True:
                       # print("HUUUUULL")

                   # if hull_path.contains_point((19.45, 21.97))==True:
                        #print("HUUUUULL 2222222222222")

                    #if hull_path.contains_point((19.45, 21.97))==True:
                        #print("HUUUUULL 2222222222222")
                
                #print(len(interm_lys))
                if len(interm_lys)>0:
                    lys_the_last.append(0)
                    lys_area_last.append(area)
                else:
                   # if (sum_rows_temp[i] < 3):
                       # print("BUUUUUUUG", sum_rows_temp[i])
                    lys_the_last.append(1)
                    lys_area_last.append(0)
            c2+=1
        c2+=1
        #print(lys_the_last, "the last")

        deep_df_short["in_hull"]=lys_the_last
        deep_df_short["area"]=lys_area_last
        deep_df_short['in_hull_level'] = pd.cut(deep_df_short["in_hull"], [-1.0, 0.0, 1.0], labels=["zero" , "one"], include_lowest=True, ordered= False)
        deep_df_short['in_hull_level'] = deep_df_short['in_hull_level'].astype(str)
        #print(deep_df_short['tid'])
       
        
              

        ################################################
        ### plotting hull and tracks togehter: as clustered vs unclustered
        
        #final_pal=dict(zero= "#fde624", one="#440053")
        final_pal=dict(zero= "#06fcde" , one= "#808080")
        linecollection = []
        colors = []

        #rcParams["figure.figsize"]= 11.7,  11.7#8.27
        fig = plt.figure()
        ax = fig.add_subplot()

        sns.set(style="ticks", context="talk")

        grouped_plot= deep_df_short.sort_values(["pos_t"]).groupby("tid")
        c2=0
        for i in grouped_plot["tid"].unique():
            s= grouped_plot.get_group(i[0])

            #plt.text(s["pos_x"][c2], s["pos_y"][c2],"#%d" %i, ha="center")

            #print(s, "this is s")
            for i in range (len(s["pos_x"])-1):

                line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
                color = final_pal[deep_df_short["in_hull_level"][c2]]
                linecollection.append(line)
                colors.append(color)

                c2+=1
            c2+=1

        lc = LineCollection(linecollection, color=colors, lw=1)

        
        #plt.style.use("dark_background")
        plt.scatter(deep_df_short["pos_x"], deep_df_short["pos_y"], s=0.001)
        plt.gca().add_collection(lc)

        #new version:
        for j in range (len(lys_points2)):
            for i in range(len(lys_points2[j])):
                points=lys_points2[j][i] # points of one cluster
                if len(points)>3:
                    
                    hull = ConvexHull(points)

                    ratio=hull.area/hull.volume
                    if ratio<105:
                        #print("yes", j)
                        

                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=1) #uncommented before

                        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=1, color="#008080")
                        #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull
                        
                
                        
     
           
        plt.axis('equal') #makes equal axis
        ##plt.savefig(str(image_path), format="svg") #tosave nice svg

        plt.show()
        #print(deep_df_short)

        #####################################################################
        ######
        # extract reelvant parameters of train result:
        print(train_result)
        print(len(train_result))


        ########################### function to make a nice excel fiel with all the parameters per track:

        def make_fingerprint_file(f2, train_result): 
            lys_string=f2.split("\\")
            outpath1=lys_string[:-1]
            outpath2='\\'.join(outpath1)
            name=lys_string[-1].split(".csv")[0]
            outpath3=outpath2+"\\"+name
            print(outpath3)

            ##### adding hull area and number of points in clusters
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
                #print(clusters)

                for i in areas.keys():
                    lys_interm_area.append(i)
                lys_interm_area.sort()

                if len(clusters)>1:
                    #if track contains points both in clusters and not in clusters, assign each type
                    lys_nr_of_clusters.append(clusters[0])
                    lys_nr_of_unclustered.append(clusters[1])
                    lys_time_in_clusters.append(dt*clusters[0])
                    lys_mean_area.append(mean(lys_interm_area[1:]))
                    lys_sum_clusters.append(len(lys_interm_area[1:]))
                    lys_time_per_cluster.append(dt*clusters[0]/len(lys_interm_area[1:]))

                else:
                    #if track only has one type of point, the "clusters[i]" object has only one entry, either 0 (points in clusters) or 1 (points not in clusters)
                    ind=clusters.index[0]
                    arry=clusters.array
                    lys_mean_area.append(0)

                    if ind==1:
                        #no cluster 
                        lys_nr_of_clusters.append(0)
                        lys_nr_of_unclustered.append(arry[0])
                        lys_time_in_clusters.append(dt*0)
                        lys_time_per_cluster.append(0)
                        lys_sum_clusters.append(0)
                    else:
                        #print(arry)
                        #all points of track are cluster points
                        lys_nr_of_clusters.append(arry[0])
                        lys_nr_of_unclustered.append(0)
                        lys_time_in_clusters.append(dt*arry[0])
                        lys_time_per_cluster.append(dt*arry[0])
                        lys_sum_clusters.append(1)
               
            ############## adding the fingerprint outputs:
            counter=0
            for i in train_result:
                counter+=1
                if(counter== 1):
                    new_finger1=(np.array([i]))
                    
                else:
                    new_finger1=np.vstack((new_finger1, i))
            print(new_finger1)    
          
            print(new_finger1[0], "zero")
            fingerprints_df_out_1=pd.DataFrame(new_finger1, columns=["alpha", "beta", "pval", "efficiency", "fractaldim", "gaussianity", "kurtosis", "msd_ratio", "trappedness", "t0", "t1", "t2", "t3","lifetime", "length_of_track",  "mean_steplength","msd" ])
            print(fingerprints_df_out_1)
            fingerprints_df_out=fingerprints_df_out_1[["alpha", "beta", "pval", "efficiency", "fractaldim", "gaussianity", "kurtosis", "msd_ratio", "trappedness","length_of_track",  "mean_steplength","msd"]]
            print(fingerprints_df_out)
            fingerprints_df_out["nr_of_spatially_arrested_points_per_track"]=lys_nr_of_clusters
            fingerprints_df_out["nr_of_non-arrested_points_per_track"]=lys_nr_of_unclustered
            fingerprints_df_out["tot_time_of_spatial_arrest_per_track"]=lys_time_in_clusters
            fingerprints_df_out["mean_area_spatial_arrest_events"]=lys_mean_area
            fingerprints_df_out["nr_of_spatial_arrest_events_per_track"]=lys_sum_clusters
            fingerprints_df_out["average_duration_of_spatial_arrest_events_per_track"]=lys_time_per_cluster
            print(fingerprints_df_out[["nr_of_non-arrested_points_per_track", "nr_of_spatially_arrested_points_per_track", "length_of_track"]])


            #print(lys_time_per_cluster)
######below uncomment again once done
            #outpath4=outpath3+"_fingerprint_results"+".xlsx"
            #writer = pd.ExcelWriter(outpath4 , engine='xlsxwriter')
            #fingerprints_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
            #writer.close()
            return fingerprints_df_out
        
        ############################### end function 
        
        make_fingerprint_file(f2, train_result) # run this to mkae excel

        ################################









    




            
        














                              
        











      
        


                




        


       










    