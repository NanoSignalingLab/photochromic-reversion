# plot trajectories for 3 colours only!:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean
import os,glob
from os import listdir
from os.path import isfile, join
from matplotlib import rcParams
from functools import reduce
import operator 


f1=r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\BAK1_flg22\trackmate\BAK1_flg22_allspots_p1_003.csv"
f1=r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\BAK1_mock\trackmate\BAK1_mock_allspots_p2_007.csv"
f1=r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\FLS_mock\trackmate\FLS2_mock_allspots_p2_001.csv"
f1=r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\FLS_mock\trackmate\FLS2_mock_allspots_p2_004.csv"
f1=r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\FLS2_flg22\trackmate\FLS2_flg22_allspots_p1_001.csv"


df = pd.read_csv(f1)

grouped= df.sort_values(["FRAME"]).groupby("TRACK_ID")
vel_all=[]
xy_all=[]
x_all=[]
y_all=[]

for i in grouped["TRACK_ID"].unique():

    s= grouped.get_group(i[0])
    if s.shape[0]>5 and s.shape[0]<30000: #added
            
   # print(s)
        frames= list(s["FRAME"])
        tid=list(s["TRACK_ID"])
        x=s["POSITION_X"]
        y=s["POSITION_Y"]
        xy=np.column_stack((x,y))

        lys=s[[ 'POSITION_X', 'POSITION_Y']].values.tolist()

        xy_all.append(lys)


        lag1=[]
        vel=[]
        time=0.05
        for i in range (len (xy)-1):
            l1 = math.dist([xy[i][0], xy[i][1]], [xy[i+1][0], xy[i+1][1]])
            lag1.append(l1)
            v=l1/time
            vel.append(v)
        
    
        vel_all.append([mean(vel)]*len(xy))
    
        

# for categorized velocity: under progress
out = reduce(operator.concat, vel_all)
out2 = reduce(operator.concat, xy_all)

df_final = pd.DataFrame(out2, columns=['POS_X', "POS_Y"])
df_final["mean_vel"]=out




#sns.set(style="ticks", context="talk")

#rcParams["figure.figsize"]= 11.7, 8.27

plt.style.use("dark_background")
plt.plot([1, 2], [1, 1], color="white") # 1 um scale bar


# for 4 colours:
df_final['vel_level'] = pd.cut(df_final["mean_vel"], [0.0, 2.0,  4.0,  8.0], labels=["zero_one_two" , "three_four_five", "six_seven_eight"], include_lowest=True, ordered= False)
df_final['vel_level'] = df_final['vel_level'].astype(str)


print(df_final['vel_level'].value_counts())
print(df_final['mean_vel'])
print(df_final['vel_level'])


final_pal=dict(zero_one_two="#6495ED", three_four_five="#E0FFFF",six_seven_eight="#DC143C")
#fig = plt.figure(figsize=(5,5), dpi=400)

plt.axis('equal') 
s1= sns.lineplot(data=df_final, x=df_final["POS_X"], y=df_final["POS_Y"], units= "mean_vel", hue= df_final["vel_level"], hue_order = ["zero_one_two" , "three_four_five", "six_seven_eight"],  estimator=None, lw=0.2, palette=final_pal, sort=False)
plt.savefig(r"C:\Users\miche\Desktop\presi_surf_BAK1\FLS2_flg22.tiff", dpi=2400)

plt.show()


#s1= sns.lineplot(data=df_final, x=df_final["POS_X"], y=df_final["POS_Y"], units= "mean_vel", hue= df_final["vel_level"], hue_order = ["zero_one_two" , "three_four_five", "six_seven_eight"],  estimator=None, lw=0.25, palette=dict(zero_one_two="#E2E512", three_four_five="#248290",six_seven_eight="#6A5ACD"),legend="full")
# for 4 colours:
df_final['vel_level'] = pd.cut(df_final["mean_vel"], [0.0, 1.0,  3.0,  8.0], labels=["zero_one_two" , "three_four_five", "six_seven_eight"], include_lowest=True, ordered= False)
df_final['vel_level'] = df_final['vel_level'].astype(str)
plt.axis('equal') 

s1= sns.lineplot(data=df_final, x=df_final["POS_X"], y=df_final["POS_Y"], units= "mean_vel", hue= df_final["vel_level"], hue_order = ["zero_one_two" , "three_four_five", "six_seven_eight"],  estimator=None, lw=0.5, palette=final_pal, sort=False)
plt.show()



###########################################
#To save as svg or as TIFF:
#plt.savefig(r"C:\Users\miche\Desktop\TIRFM\python_data\5.5.23_NB_mEOS3.2\FLS2_flg22\trackmate\FLS2_flg22_allspots_p1_003.svg", format="svg")
#plt.savefig(r"C:\Users\bcgvm01\Desktop\MVA_TIRFM\5.5.23_mEOS3.2_NB\5.5.23_NB_mEOS3.2\BAK1_mock\trackmate\BAK1_mock_allspots_p2_001.tiff", dpi=400)

