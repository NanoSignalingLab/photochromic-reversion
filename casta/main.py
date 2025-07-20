import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull
from statistics import mean 
from pathlib import Path

import os
from os import listdir
from os.path import isfile, join

from casta.features import run_hmm, calc_distance, calc_angles, calc_KDE, calc_intersections
from casta.plot import plotting_final_image
from casta.results import make_results_file
from casta.utils import *

import warnings
warnings.filterwarnings('ignore')

def calculate_sta(dir: str,
                  out_dir: str = None, 
                  min_track_length: int = 25, 
                  dt: float = 0.05, 
                  plot: bool = False,
                  image_format: str = "svg"):
    
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    for i in onlyfiles:
        if i.endswith(".csv"):
            path=os.path.join(dir, i)
            csv_name = Path(i)
            base_name = csv_name.stem  # Gets filename without extension

            extension = "svg" if image_format == "svg" else "tiff"
            image_path = Path(out_dir) / f"{base_name}.{extension}"

            tracks_input, df, traces, lys_x, lys_y, msd_df = load_file(path, min_track_length) # execute this function to load the files
            mean_msd_df=msd_mean_track(msd_df, dt)

            df=run_hmm(df, dt)
            df=calc_distance(df)
            df=calc_angles(df)
            df=calc_KDE(lys_x, lys_y, df)
            df=calc_intersections(lys_x, lys_y, df)

            grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_points2, mean_msd_df1, lys_begin_end_big2, lys_points_big_only_middle2=plotting_all_features_and_calculate_hull(df, mean_msd_df, plot, dt)
            deep_df_short2=convex_hull_wrapper(grouped_plot,lys_area2, lys_perimeter2, lys_hull2, lys_points_big2, deep_df_short, lys_begin_end_big2, lys_points_big_only_middle2)

            mean_msd_df2=calculate_diffusion_non_STA_tracks(deep_df_short2,mean_msd_df1 )

            plotting_final_image(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_format)
            make_results_file(path, out_dir, deep_df_short2, dt,mean_msd_df2) # run function to make excel with all parameters

############## plot all features togheter (plus convex hull):
def plotting_all_features_and_calculate_hull(deep_df, mean_msd_df, plotting_flag, dt): # add ture =1or false =0 for plotting yes or no
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
    if plotting_flag:
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
                        mean_msd, logD = logD_from_mean_MSD(msd, dt)
            
                        lys_msd_cluster.append(mean_msd)
                        lys_logD_cluster.append(logD)

                    ####################
                        if lys_start_end_cluster2[j][i][0]!="B":
                            msd_middle, rmsd_middle = compute_msd(points)
                            mean_msd_middle, logD_middle = logD_from_mean_MSD(msd_middle, dt)

                            lys_msd_cluster_middle.append(mean_msd_middle)
                            lys_logD_cluster_middle.append(logD_middle)
            



                    lys_hull.append(hull)
                    lys_area.append(hull.volume) 
                    lys_perimeter.append(hull.area) 
                    if plotting_flag:
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
    
    

    if plotting_flag:
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

def calculate_diffusion_non_STA_tracks(deep_df_short, mean_msd_df):
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
