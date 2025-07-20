import pickle
import math
import pandas as pd
from itertools import chain
from shapely.geometry import LineString
from shapely import intersection

from casta.hmm_functions import run_model
from casta.utils import angle3pt, consecutive, make_KDE_per_track

def run_hmm(deep_df, dt): 

    with open("casta/data/model_4.pkl", "rb") as file: 
        model = pickle.load(file)
    print("loading HMM model")
    window_size=10

    predicted_states_for_df=run_model(model, deep_df,window_size, dt)
    predicted_states_flat= list(chain.from_iterable(predicted_states_for_df))
    
    deep_df["hmm_states"]=predicted_states_flat
    deep_df["hmm_states"]= deep_df["hmm_states"].replace(2,1)
    deep_df["hmm_states"]= deep_df["hmm_states"].replace(3,1)

    return deep_df

def calc_distance(deep_df):

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

def calc_angles(deep_df):
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

def calc_KDE(lys_x, lys_y, deep_df):
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

def find_intersections(lys_x, lys_y):

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
    
def calc_intersections(lys_x, lys_y, deep_df):
    print("Computing intersections")

    inter_flat1, inter_flat2, inter_flat3, inter_flat4=find_intersections(lys_x, lys_y)

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
