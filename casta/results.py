import pandas as pd
from pathlib import Path
from statistics import mean

def make_results_file(f2, out_dir, deep_df_short, dt, mean_msd_df):
    #print("this is deepdfshort",deep_df_short)

    f2_path = Path(f2)
    name = f2_path.stem  # Gets filename without extension

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

    if out_dir is not None:
        outpath = Path(out_dir) / f"{name}_CASTA_results.xlsx"
    else:
        outpath = f2_path.parent / f"{name}_CASTA_results.xlsx"
    print("Saving results to: ", outpath)
    writer = pd.ExcelWriter(outpath, engine='xlsxwriter')
    casta_df_out.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
    writer.close()

    return casta_df_out