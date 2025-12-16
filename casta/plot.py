import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import ConvexHull

def plotting_final_image(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_format):
    final_pal=dict(zero= "#06fcde" , one= "#808080")
    linecollection = []
    colors = []
    if image_format=="tiff":
        lw1=1
        s1=0.001
    else:
        lw1=1
        s1=0.1

    fig = plt.figure() # was this before
    #fig, ax = plt.subplots(1) #for tif?
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)

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
                            


    if image_format=="svg":
        plt.axis('equal') #before
        plt.savefig(str(image_path), format="svg") # 
        plt.show()
    else:
        #plt.axis('equal') # was this before
        ax.axis("equal")
        xmin, xmax=ax.get_xlim()
        ymin, ymax=ax.get_ylim()
        # draw vertical line from (70,100) to (70, 250)$
        #plt.plot([xmax-2, xmax-1], [ymin+1, ymin+1], 'k-', lw=1)

        plt.savefig(str(image_path), dpi=1500,format="tiff") # was 3500
        plt.show()

def plot_original_tracks(deep_df):
    cmap = plt.get_cmap("viridis")  # Or "viridis_r" for reversed
    sns.set(style="ticks", context="talk")

    # Get unique track IDs and normalize to colormap range
    unique_ids = deep_df["tid"].unique()
    norm = plt.Normalize(vmin=min(unique_ids), vmax=max(unique_ids))

    line_segments = []
    colors = []

    for tid in unique_ids:
        track = deep_df[deep_df["tid"] == tid].sort_values("pos_t")
        points = track[["pos_x", "pos_y"]].values

        # Make segments from consecutive points
        for i in range(len(points) - 1):
            segment = [points[i], points[i + 1]]
            line_segments.append(segment)
            colors.append(cmap(norm(tid)))  # Assign color based on tid

    lc = LineCollection(line_segments, colors=colors, linewidths=1.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    plt.scatter(deep_df["pos_x"], deep_df["pos_y"], s=0.1, alpha=0.3, color="black")  # Optional: background points
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Single Particle Tracks")
    plt.show()
    plt.close()

################## plot feature values onto line segments ##############
def plot_values_on_track(deep_df, value, image_path):

    if value == "KDE_invert":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.set(style="ticks", context="talk")

    linecollection = []
    colors = []
    grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")

    custom_colors = ['#380282', '#440053', '#404388', '#2a788e', '#21a784', '#78d151', '#fde624', '#ff9933', '#ff3300']
    custom_colors_short = ['#404688', '#c7e020']

    custom_cmap = LinearSegmentedColormap.from_list("custom_continuous", custom_colors, N=256)
    custom_cmap_r = custom_cmap.reversed()

    custom_short = LinearSegmentedColormap.from_list("custom_short", custom_colors_short, N=256)
    custom_short_r = custom_short.reversed()

    if value != "in_hull_level":
        norm = Normalize(vmin=deep_df[value].min(), vmax=deep_df[value].max())
    cmap = plt.get_cmap("viridis_r")

    h1 = (deep_df["pos_x"].max() - deep_df["pos_x"].min())*0.05
    h2 = (deep_df["pos_y"].max() - deep_df["pos_y"].min())*0.05
    xlim = (deep_df["pos_x"].min()-h1, deep_df["pos_x"].max()+h1)
    ylim = (deep_df["pos_y"].min()-h2, deep_df["pos_y"].max()+h2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    
    c2=0
    final_pal=dict(zero="#c7e020", one="#404688")
    
    for i in grouped_plot["tid"].unique():
        s= grouped_plot.get_group(i[0])
        
        for i in range (len(s["pos_x"])-1):
            line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
            dist_val = deep_df[value].iloc[c2]

            if value == "angles":
                color = "#D3D3D3"
            elif value == "pos_x": #just used as a placeholder column name to get empty track picture
                color = "#000000" 
            elif value == "all_intersect":
                color = custom_short_r(norm(dist_val))
            elif value == "in_hull_level":
                color = final_pal[deep_df["in_hull_level"][c2]]
            else:
                color = cmap(norm(dist_val))

            linecollection.append(line)
            colors.append(color)
            c2+=1
        c2+=1

    lc = LineCollection(linecollection, color=colors, lw=1) # was 1

    plt.gca().add_collection(lc)

    if value == "KDE_invert":
            sns.kdeplot(data=s, x="pos_x", y="pos_y",fill=True, thresh=0, levels=100, cmap="mako",alpha=1, ax=ax)

    if value == "angles":
        deep_df["shifted_val"] = deep_df[value].shift(1)
        df_sorted = deep_df.sort_values(by="shifted_val", ascending = False)
        plt.scatter(
            df_sorted["pos_x"],
            df_sorted["pos_y"],
            c=df_sorted["shifted_val"],
            cmap=cmap,
            norm=norm,
            s=2,
            zorder=10
        )
    else:
        plt.scatter(deep_df["pos_x"], deep_df["pos_y"], s=0.01, alpha=0) #was 0.00

    plt.savefig(str(image_path), dpi=1500,format="tiff")
    plt.show()


def plot_values_on_track_hull(deep_df, value, lys_points_middle, image_path):
    if value == "KDE_invert":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.set(style="ticks", context="talk")

    linecollection = []
    colors = []
    grouped_plot= deep_df.sort_values(["pos_t"]).groupby("tid")

    custom_colors = ['#380282', '#440053', '#404388', '#2a788e', '#21a784', '#78d151', '#fde624', '#ff9933', '#ff3300']
    custom_colors_short = ['#404688', '#c7e020']

    custom_cmap = LinearSegmentedColormap.from_list("custom_continuous", custom_colors, N=256)
    custom_cmap_r = custom_cmap.reversed()

    custom_short = LinearSegmentedColormap.from_list("custom_short", custom_colors_short, N=256)
    custom_short_r = custom_short.reversed()

    if value != "in_hull_level":
        norm = Normalize(vmin=deep_df[value].min(), vmax=deep_df[value].max())
    cmap = plt.get_cmap("viridis_r")

    h1 = (deep_df["pos_x"].max() - deep_df["pos_x"].min())*0.05
    h2 = (deep_df["pos_y"].max() - deep_df["pos_y"].min())*0.05
    xlim = (deep_df["pos_x"].min()-h1, deep_df["pos_x"].max()+h1)
    ylim = (deep_df["pos_y"].min()-h2, deep_df["pos_y"].max()+h2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)

    
    c2=0
    final_pal=dict(zero="#06fcde", one="#808080")
    
    for i in grouped_plot["tid"].unique():
        s= grouped_plot.get_group(i[0])
        
        for i in range (len(s["pos_x"])-1):
            line = [(s["pos_x"][c2], s["pos_y"][c2]), (s["pos_x"][c2+1], s["pos_y"][c2+1])]
            dist_val = deep_df[value].iloc[c2]

            if value == "angles":
                color = "#D3D3D3"
            elif value == "pos_x": #just used as a placeholder column name to get empty track picture
                color = "#000000" 
            elif value == "all_intersect":
                color = custom_short_r(norm(dist_val))
            elif value == "in_hull_level":
                color = final_pal[deep_df["in_hull_level"][c2]]
            else:
                color = cmap(norm(dist_val))

            linecollection.append(line)
            colors.append(color)
            c2+=1
        c2+=1

    lc = LineCollection(linecollection, color=colors, lw=1) # was 1

    plt.gca().add_collection(lc)

    if value == "KDE_invert":
            sns.kdeplot(data=s, x="pos_x", y="pos_y",fill=True, thresh=0, levels=100, cmap="mako",alpha=1, ax=ax)

    if value == "angles":
        deep_df["shifted_val"] = deep_df[value].shift(1)
        df_sorted = deep_df.sort_values(by="shifted_val", ascending = False)
        plt.scatter(
            df_sorted["pos_x"],
            df_sorted["pos_y"],
            c=df_sorted["shifted_val"],
            cmap=cmap,
            norm=norm,
            s=2,
            zorder=10
        )
    else:
        plt.scatter(deep_df["pos_x"], deep_df["pos_y"], s=0.01, alpha=0) #was 0.00

    for j in range (len(lys_points_middle)):
                    for i in range(len(lys_points_middle[j])):
                        points=lys_points_middle[j][i] 
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=1, color="red") #,color="#c7e020") 
                            #plt.text(points[0][0], points[0][1],"#%d" %j, ha="center") # uncomment this to label the hull

    plt.savefig(str(image_path), dpi=1500,format="tiff")
    plt.show()