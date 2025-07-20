import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

def plotting_final_image(deep_df_short, lys_points_big2, lys_points_big_only_middle2, image_path, image_format):
    final_pal=dict(zero= "#06fcde" , one= "#808080")
    linecollection = []
    colors = []
    if image_format=="tiff":
        lw1=0.1
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
        #square_inches = 5  # You can choose another value
        #fig.set_size_inches(square_inches, square_inches)
        #ax.set_aspect('equal', adjustable='box')
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