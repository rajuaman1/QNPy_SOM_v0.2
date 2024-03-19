# QNPy_SOM_v0.2
The Self Organizing Map module here is intended to serve as an addition to the QNPy package in order to cluster light curve data before modeling. Thus, it is recommended to see the [QNPy readme](https://github.com/kittytheastronaut/QNPy-002-Progress/blob/main/README.md) for a more detailed idea of the entire QNPy pipeline.

Introduction
============
Conditional Neural Processes (CNPs) excel at learning complex patterns in data with recurring gaps. However, application to larger datasets requires novel methods to prioritize efficiency and effectively capture subtle trends in the data. Self Organizing Maps (SOMs) provide both these advantages. SOMs provide an unsupervised clustering algorithm that can be trained quickly and include new data points without the need to train over every data point again. Thus, we present QNPy as an ensemble model of SOMs and CNPs.

SOMs comprise a network of nodes mapped onto a (usually) two-dimensional grid. Each node has an input weight associated with it. As the SOM trains on the input data, each input point is assigned a Best Matching Unit (BMU) where the node is at the minimum Euclidean distance from the input. Then, the BMU is updated to match the input data point (the amount that the node moves is dependent on the learning rate). Furthermore, each node can affect neighboring nodes via a neighborhood function (usually Gaussian). 

Once the training is ended, each input data point is assigned to a cluster depending on the final BMU. Thus at the end, each node provides a cluster. These can be the final cluster or the distance matrix (a matrix containing the distance of each node with each of the other nodes) of the SOM can be used to group different nodes into more hierarchical clusters. This is done by calculating gradients between the nodes until the lowest node is reached. (For more info, refer to [Hamel and Brown](https://homepage.cs.uri.edu/faculty/hamel/pubs/improved-umat-dmin11.pdf)).

In QNPy, we treat each light curve as a data point and the magnitudes are the features. Thus, the SOM can effectively parse topological differences in the light curves. These differences allow the CNP to train on similar light curves and effectively capture subtle differences in the modeled light curves. In addition, the clusters now allow for CNPs to be trained in parallel on smaller batches of data, which allows for a massive speed-up in the training time of the QNPy package.

The SOM is based on the [minisom package](https://github.com/JustGlowing/minisom) which uses NumPy packages to handle the input data. Thus, every input data point must have the same length. We handle this similarly with the CNP by padding all the light curves to the same length. We also scale the light curves to treat different magnitude ranges differently.

Thus, SOMs provide a useful companion to the CNPs to form an ensemble model with improved speed and accuracy in modeling large amounts of light curve data.

Requirements
============
See QNPy Requirements

Examples
========
Check out the `Example Notebooks` folder [here](https://github.com/rajuaman1/QNPy_SOM_v0.2/tree/main/Example%20Notebooks) for examples of the different SOM builds and visualizations that we have included in QNPy. The data format is the same as the other parts of QNPy which can be seen in the example light curves.

Folder Structure
================
The SOM module automatically creates folders for saving plots and saves your trained SOM. The only requirement for the file structure is to save light curves before the module and choose directories to save plots and models during the module's runtime. The light curves should be saved under a directory (can be named anything) with the filters saved as subfolders. Then, each light curve should be saved as a CSV file with the id as the file name. For example, if you have a light curve in the g filter with ID 10422 and you want to save it in a folder known as `Light_Curves`, it should be saved under the directory `Light_Curves/g/10422.csv`.

As with other QNPy light curves, your data must contain: `mjd` - MJD or time, `mag` - magnitude, and `magerr` - magnitude error.

Module and Functions
===========================
In the clustering module, we first load the light curves from the directory. This also creates the ids from the file names. Thus, it is recommended to have the same light curves saved across all the different bands. Then, we pad the light curves to make them all the same length. In QNPy, we have seen that we require at least 100 data points for accurate modeling. Thus, we recommend that the light curves be padded to at least 100 points (even if the longest curve is under 100 points, which can be controlled through a keyword in the padding function). Finally, we scale the light curves. We have provided many different scalers including minmax, standard and robust scalers. Our `default` scaler is an adapted version of a minmax scaler that scales all the data to the range [-2,2].

Then, a SOM is trained on the scaled data. The SOM contains different tunable hyperparameters to better adapt to different data sets. These hyperparameters can be tested with different metrics including quantization error, topographical error, silhouette score, davies-bouldin index, or calinski-harabasz score. The trained SOM can be saved as well.

The trained SOM is then used to assign the IDs to different clusters. Then they can be saved into different folders.

We also provide different plots for visualization of the data. These will be described in the plotting functions.

Described below are the functions used in the module:

**Main Functions**
```
Load_Light_Curves(folder,filters):
'''
Loads light curves from a specified folder

Parameters
----------
folder: str 
The folder where the light curves are stored

filters: list or str(if each filter is a single letter)
The filters that are to be loaded. Each filter should have a subfolder named after it

Returns
--------
light_curves: list of lists of dataframes
The list of light curves arranged by filter

ids: list
The ids of the light curves (Ensure that they are the same in all filters)
'''
```
```  
Pad_Light_Curves(light_curves,filters,minimum_length = 100):
'''
Pads the light curves with the mean value at the end of the curve

Parameters
----------
light_curves: list of lists of dataframes 
The light curves stored in a list. These lists are then stored in a bigger list

filters: list or str(if each filter is a single letter)
The filters to be used

minimum_length: int
The minimum length to pad to

Returns
--------
light_curves: list of lists
The new padded light curves
'''
```
```
scale_curves(light_curves,what_scaler = 'default',scale_times = True):
'''
Scaling the curves (from a single filter) from the choice of minmax, standard and robust. By default, it scales to a range of [-2,2]
Parameters
----------
light_curves: list of dataframes 
The light curves stored in a list.

what_scaler: string
The type of scaler to use. There are default (see above), standard scaler, min-max scaler and robust scalers available

scale_times: bool
Whether to scale the time axis as well (These are always scaled to the default scaler)

Returns
--------
scaled_curves: np.ndarray 
The scaled light curves

scaled_times:np.ndarray
The scaled time steps. It is an empty list if the keyword scale_times is False
'''
```
```
SOM_1D(scaled_curves,som_x = None,som_y = None,learning_rate = 0.1,sigma = 1.0,topology = 'rectangular',pca_init = True,\
                neighborhood_function='gaussian',train_mode = 'random',batch_size = 5,epochs = 50000,save_som = True,\
           model_save_path = './',random_seed = 21,stat = 'q',plot_frequency = 100,early_stopping_no = None):
'''
Training a SOM on ONE dimensional data (The magnitude of the light curves)
Parameters
----------
scaled_curves: list of dataframes 
The scaled light curves stored in a list.

som_x: int
The x size of the SOM. If None is given, make sure the som_y is None as well. Then, it chooses the recommended SOM 
size of sqrt(sqrt(length))

som_y: int
The y size of the SOM. If None is given, make sure the som_x is None as well. Then, it chooses the recommended SOM 
size of sqrt(sqrt(length))

learning_rate: float
How much the SOM learns from the new data that it sees

sigma: float
The effect each node has on its neighboring nodes

topology: 'rectangular' or 'hexagonal':
The topology of the SOM. Note that visualizations are mainly built for the rectangular at the moment.

pca_init: bool
Whether to initialize the SOM weights randomly or to initialize by PCA of the input data

neighborhood_function: str
Choose from 'gaussian','mexican hat','bubble', or 'triangle'. These affect the influence of a node on its neighbors

train_mode:'random' or 'all'
When chosen random, it chooses a random curve each epoch. When trained on all, it batches the data and trains on every
light curve for a certain number of epochs.

batch_size: int
How big the batch is for the 'all' train mode. The smaller the batch size, the finer the progress bar displayed

epochs: int
This is defined in two ways. If the train_mode is random, then it is the number of iterations that the SOM runs on.
If it is all, then it is the number of times that the SOM trains on each input datapoint. Note that the lr and sigma
decay in each epoch.

save_som: bool
Whether to save the trained SOM 

model_save_path:str
The file to save the SOM in

random_seed:int
The starting state of the random weights of the SOM. Use for reproducibility

stat: 'q','t', or 'qt'
Whether to record the quantization error, topographical error or both. Note that calculating them is expensive

plot_frequency: int
The number of epochs

early_stopping_no: int or None
The number of batches to process before stopping. Use None if you should train on all

Returns
--------
som_model:
The trained SOM that can be saved or used for analysis

q_error: list
The quantization errors recorded

t_error: list
The topographic errors recorded

indices_to_plot: list
The indices to plot for the quantization or/and topographic errors
'''
```
```
Assign_Cluster_Labels(som_model,scaled_curves,ids):
'''
Assigns Cluster labels to each of the curves, making a dataframe with their ids

Parameters
----------
som_model:  
The trained SOM

scaled_curves: np.ndarray
The scaled curves that were used to train the SOM

ids: list
The ids of the curves

Returns
--------
cluster_df: Dataframe
A map matching each of the cluster ids with the cluster they belong to
'''
```
```
Gradient_Cluster_Map(som,scaled_curves,ids,dimension = '1D',fill = 'mean',interpolation_kind = 'cubic',clusters = None,som_x = None,som_y = None):
'''
Translates the SOM nodes into larger clusters based on their gradients. Implementation of 
https://homepage.cs.uri.edu/faculty/hamel/pubs/improved-umat-dmin11.pdf

Parameters
----------
som_model:  
The trained SOM

scaled_curves: np.ndarray
The scaled curves used to train the SOM

ids: list
The ids of the curves

dimension: str
If 1D, does 1D clustering, else multivariate

fill: str
'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function

interpolation_kind: 
Any of the scipy.interp1d interpolation kinds. Recommended to use cubic

clusters:
The clusters that the ids are in (only for multi-variate)

som_x: int
The x-dimensions of the SOM

som_y: int
The y-dimensions of the SOM

Returns
--------
cluster_map:
The new clusters that the ids are in
'''
```
```
Cluster_Metrics(scaled_curves,cluster_map,metric = 'Silhoutte'):
'''
Measures metrics related to the clustering

Parameters
----------
scaled_curves: np.ndarray
The scaled curves used to train the SOM

cluster_map: pd.Dataframe
A map of each of the ids to the clusters

metric: str
The metric to be measured. It can be Silhoutte, DBI or CH. This is for silhoutte score, Davies-Bouldin index and calinski-harabasz score

Returns
--------
score:
The metric that is calculated
'''
```
```
save_chosen_cluster(chosen_cluster,filters,cluster_map,overwrite = True,save_path = './',source_path = './Light_Curves'):
'''
Saves the chosen cluster into a folder

Parameters
----------
chosen_cluster: int
The cluster to save

filters: str
The filters to save

cluster_map: pd.Dataframe
A map of each of the ids to the clusters

overwrite: bool
Whether to overwrite the current folder

save_path: str
The path to save to. This creates a folder for the cluster in that directory

source_path: str
The path that the light curves are saved in
'''
```
***Multi-Band Clustering Functions***
These functions are only used for multi-band clustering
```
multi_band_clustering(light_curves,ids,filter_names = 'ugriz',som_x = None,som_y = None,sigma = 1.0,learning_rate = 0.5,\
                          num_iterations = 2,batch_size = 5,early_stopping_no = None):
'''
Multiband light curve clustering

Parameters
----------
light_curves: 
The light curves to be used

ids: list,array
The ids of the quasars

filter_names: list or str(if the filters are one letter)
The filters that are used

som_x: int
The x size of the SOM. If None is given, make sure the som_y is None as well. Then, it chooses the recommended SOM 
size of sqrt(sqrt(length))

som_y: int
The y size of the SOM. If None is given, make sure the som_x is None as well. Then, it chooses the recommended SOM 
size of sqrt(sqrt(length))

sigma: float
The effect each node has on its neighboring nodes

learning_rate: float
How much the SOM learns from the new data that it sees

num_iterations: int
The number of iterations that the som is trained on each batch

batch_size: int
The size of each batch

early_stopping_no: int or None
The number of batches to process before stopping. Use None if you should train on all

Returns
--------
som:
The trained SOM

processed_light_curve:
The flat light curves used for the SOM

processed_mask:
The mask used for the SOM
'''
```
```
find_cluster_and_quantization_errors(som,data,masks):
'''
Finding the clusters and the quantization errors from the trained 2D SOM

Parameters
----------
som: 
The trained SOM

data: 
The processed light curves from the trained SOM

masks: 
The masks used from the trained SOM

Returns
--------
min_clusters:
The clusters for each of the data points

quantization_error:
The quantization error of each of the data points
'''
```
***Visualization Functions***
These functions are used for visualizations

```
Plot_Lc(Light_Curve,header = 'Light Curve',save_fig = False,filename = 'Figure',x_axis = 'mjd',return_fig = False):
'''
Plots light curves interactively. Adapted from https://github.com/DamirBogdan39/time-series-analysis/tree/main

Parameters
----------
Light_Curve: Dataframe 
The light curve to plot. Should be in a dataframe with mjd (or any x axis), mag and magerr

header: str
The header of the file

save_fig: bool
Whether to save the figure

filename: str
What to name the saved html file

x_axis: str
What to label the x axis 

return_fig: bool
Whether the figure is returned

Returns
--------
Figure:
The interactive plot of the light curve
'''
```
```
plot_training(training_metric_results,metric,plotting_frequency,indices_to_plot,figsize = (10,10),save_figs = True,fig_save_path = './'):
'''
Plots the metric given (quantization error or topographic error)

Parameters
----------
training_metric_results: list 
The result obtained from the SOM training

metric: str
Name of the metric

plotting_frequency: int
How much was the plotting frequency set during the SOM training

indices_to_plot: list
The indices to plot obtained from the SOM training

figsize: tuple
The size of the figure

save_figs: bool
Whether to save the figure or not

fig_save_path:str
Where to save the figure. Note that it creates a directory called Plots in the location given.

Returns
--------
Plot:
The plot of the metric
'''
```
```
Plot_SOM_Scaled_Average(som_model,scaled_curves,dba = True,figsize = (10,10),save_figs = True,figs_save_path = './',\
                           plot_weights = True,plot_avg = True,plot_background = True,one_fig = True,show_fig = True):
'''
Plotting the SOM Clusters with the average light curve and the SOM weights of each cluster. The average can be either simple mean
or using a dba averaging method (https://github.com/fpetitjean/DBA)

Parameters
----------
som_model:  
The trained SOM

scaled_curves: np.ndarray
The scaled curves that were the input for training

dba: bool
Whether to use Dynamic Barymetric Time Averaging

figsize: tuple
The size of the figure

save_figs: bool
Whether to save the figure or not

fig_save_path: str
Where to save the figure. Note that it creates a directory called Plots in the location given.

plot_avg: bool
Whether to plot the mean light curve of the cluster

plot_weights: bool
Whether to plot the SOM weight of the cluster

plot_background: bool
Whether to plot the light curves that make up the cluster

one_fig: bool
Whether to plot all the clusters into one figure or seperate figures

show_fig: bool
Whether to show each of the plots in the seperate figures case

Returns
--------
Plot:
The plots of the clusters
'''
```
```
SOM_Distance_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
'''
Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors

Parameters
----------
som_model:  
The trained SOM

cmap: str
The matplotlib based color scale to use for the plots

figsize: tuple
The size of the figure

save_figs: bool
Whether to save the figure or not

fig_save_path: str
Where to save the figure. Note that it creates a directory called Plots in the location given.

Returns
--------
Plot:
The heatmap plot
'''
```
```
SOM_Activation_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
'''
Plots a heatmap of the SOM Nodes. The brighter, the more light curves activate the SOM

Parameters
----------
som_model:  
The trained SOM

cmap: str
The matplotlib based color scale to use for the plots

figsize: tuple
The size of the figure

save_figs: bool
Whether to save the figure or not

fig_save_path: str
Where to save the figure. Note that it creates a directory called Plots in the location given.

Returns
--------
Plot:
The heatmap plot
'''
```
```
SOM_Clusters_Histogram(cluster_map,color = 'tab:blue',save_figs = True,figs_save_path = './',figsize = (5,5)):
'''
Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors

Parameters
----------
cluster_map:  
The dataframe with each id and the cluster that it belongs to

color: str
The color to plot the histogram

save_figs: bool
Whether to save the figure or not

fig_save_path: str
Where to save the figure. Note that it creates a directory called Plots in the location given.

figsize: tuple
The size of the figure

Returns
--------
Plot:
The Histogram of how many curves are in each cluster
'''
```
```
plotStarburstMap(som):
'''
Interactive plot of the distance map and gradients of the SOM

Parameters
----------
som:  
The trained SOM

Returns
--------
Plot of the distance map and gradients
'''
```
```
Averaging_Clusters(chosen_cluster,cluster_map,lcs,plot = True,dba = True):
'''
Creating a representation of the chosen cluster with the light curves and the average light curve

Parameters
----------
chosen_cluster: int 
The cluster of interest

cluster_map: pd.Dataframe
A map of each of the ids to the clusters

lcs: list of list of pd.Dataframes
The light curves (provide the input from just one filter)

plot: bool
Whether to plot or just return the average value 

dba: bool
Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves

Returns
--------
average_x:
The x_axis (i.e timesteps) of the average light curve

average_y:
The y_axis (i.e magnitudes) of the average light curve

x:
The timesteps of all the light curves concatenated into one array

y: 
The magnitudes of all the light curves concatenated into one array

len(x):
The length of all the light curves
'''
```
```
Plot_All_Clusters(cluster_map,lcs,color = 'tab:blue',dba = True,figsize = (10,10),save_figs = True,figs_save_path = './'):
'''
Plots all of the clusters on a magnitude plot with the average representation included

Parameters
----------
cluster_map: pd.Dataframe
A map of each of the ids to the clusters

lcs: list of list of pd.Dataframes
The light curves (provide the input from just one filter)

color: str
The color to plot the averaged curve in

dba: bool
Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves

figsize: tuple
The figure size

save_figs: bool
Whether to save the figure or not

figs_save_path: str
Where to save the figure. Note that it is saved under a directory called Plots in that directory.
'''
```
```
Cluster_Properties(cluster_map,selected_cluster,lcs,redshifts_map = None,plot = True,return_values = False,\
                       the_property = 'all',save_figs = True,figs_save_path = './'):
'''
Getting the selected property of a chosen cluster

Parameters
----------
cluster_map: pd.Dataframe
A map of each of the ids to the clusters

chosen_cluster: int 
The cluster of interest

lcs: list of list of pd.Dataframes
The light curves (provide the input from just one filter)

redshifts_map: pd.Dataframe
The redshift associated with each source id

plot: bool
Whether to plot or just return the average value 

return_values: bool
Whether to return the values for the property

the_property: str
The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all

save_figs: bool
Whether to save the figure or not

figs_save_path: str
Where to save the figure. Note that it is saved under a directory called Plots in that directory.

Returns
--------
return_list: 
The list of the property of interest
'''
```
```
Cluster_Properties_Comparison(cluster_map,lcs,redshifts_map,the_property = 'Fvar',color = '#1f77b4',\
                                  figsize = (15,15),save_figs = True,figs_save_path = './'):
'''
Plotting the property of interest for all the clusters onto one figure

Parameters
----------
cluster_map: pd.Dataframe
A map of each of the ids to the clusters

lcs: list of list of pd.Dataframes
The light curves (provide the input from just one filter)

redshifts_map: pd.Dataframe
The redshift associated with each source id

the_property: str
The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all

color: str
The color to make the histogram

figsize: tuple
The figure size

save_figs: bool
Whether to save the figure or not

figs_save_path: str
Where to save the figure. Note that it is saved under a directory called Plots in that directory.

Returns
--------
return_list: 
The list of the property of interest
'''
```
```
Structure_Function(cluster_map,selected_cluster,lcs,bins,save_figs = True,figs_save_path = './'):
'''
Create the structure function for a given cluster

Parameters
----------
cluster_map: pd.Dataframe
A map of each of the ids to the clusters

selected_cluster: int
The cluster of interest

lcs: list of list of pd.Dataframes
The light curves (provide the input from just one filter)

bins:int or list
The bins to use for the structure function

save_figs: bool
Whether to save the figure or not

figs_save_path: str
Where to save the figure. Note that it is saved under a directory called Plots in that directory.

Returns
--------
S+ and S- Plot: 
A plot of the S+ and S- functions for the cluster

Difference Plot:
The evolution of the normalized S+ - S- throughout the observation time of the cluster

S Plot:
The evolution of the (regular) structure function through the observation time of the cluster
'''
```
