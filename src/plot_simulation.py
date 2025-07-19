import andi_datasets
from andi_datasets.datasets_phenom import datasets_phenom
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_trajectories import plot_trajs
import stochastic
import numpy as np
import matplotlib.pyplot as plt


number_compartments = 1000
radius_compartments = 0.8
N=10
T=100
D=0.01

L = 1.5*128 #enalrge field ov fiew to avoid boundy effects
#trans:0.9 82%
#trans:0.5 82%


compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments, 
                                                                      r = radius_compartments,
                                                                      L = L # size of the environment
                                                                      )

#fig, ax = plt.subplots(figsize = (4,4))
#for c in compartments_center:
    #circle = plt.Circle((c[0], c[1]), radius_compartments, facecolor = 'None', edgecolor = 'C1', zorder = 10)
    #ax.add_patch(circle) 
#plt.setp(ax, xlim = (0, L), ylim = (0, L))

dict_model5 = {'model': 'confinement', 
               'L': L,
               "Ds":[ 10*D, D],
               'trans': 0.15,
              # 'Ds': [5*D, 1*D],
                'comp_center': compartments_center,
                'r': radius_compartments, 
               "alphas": [1, 0.1]
              
              }

dict_all = [dict_model5]

trajs, labels = datasets_phenom().create_dataset(N_model = N, # number of trajectories per model
                                                 T = T,
                                                 dics = dict_all,
                                                 save = True, path = 'datasets_folder/'
                                                )


plot_trajs(trajs, L, N, num_to_plot=3,

           comp_center = compartments_center,
           r_cercle = radius_compartments,
           #plot_labels = True, 
           # labels = labels, 
           #num_to_plot=1
           )
print("done")
plt.show()
