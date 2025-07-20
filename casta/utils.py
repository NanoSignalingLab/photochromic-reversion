import math
import pandas as pd
import numpy as np
from importlib import resources
from itertools import chain
from scipy.stats import gaussian_kde

from sklearn.preprocessing import normalize

def load_example_data():
    with resources.files('casta.data').joinpath('example_track.csv').open('r') as f:
            df = pd.read_csv(f)

    with resources.path('casta.data', '') as path:
            path = str(path)

    return df, path

def angle3pt(a, b, c):
    ang = math.degrees(
    math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

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