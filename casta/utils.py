import math
import pandas as pd
import numpy as np
from itertools import chain
from scipy.stats import gaussian_kde
from statistics import mean 

from sklearn.preprocessing import normalize

def load_file(path, min_track_length):
    df=pd.read_csv(path)
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

def angle3pt(a, b, c):
    ang = math.degrees(
    math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

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
        mean_msd, logD = logD_from_mean_MSD(full_track, dt)
        lys.append(mean_msd)
        lys2.append(logD)

    track_means_df = pd.DataFrame(np.column_stack([lys, lys2]), columns=["msd", "logD"])
    
    return track_means_df

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