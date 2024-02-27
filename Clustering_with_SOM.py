import pandas as pd
import numpy as np
from minisom import MiniSom
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import pickle
import math
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
import os
import matplotlib
import plotly.graph_objects as go
from glob import glob
from copy import deepcopy
import shutil

#Function to plot light curve adapted from Damir's notebook
def Plot_Lc(Light_Curve,header = 'Light Curve',save_fig = False,filename = 'Figure',x_axis = 'mjd',return_fig = False):
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
    fig = go.Figure()
    error_bars = go.Scatter(
         x=Light_Curve[x_axis],
         y=Light_Curve['mag'],
         error_y=dict(
             type='data',
             array=Light_Curve['magerr'],
             visible=True
         ),
         mode='markers',
         marker=dict(size=4),
         name='mag with error bars'
     )
    fig.add_trace(error_bars)
    if x_axis == 'time' or x_axis == 'mjd':
        fig.update_xaxes(title_text='MJD (Modified Julian Date)')
    elif x_axis == 'phase':
        fig.update_xaxes(title_text='Phase (No of Periods)')
    else:
        fig.update_xaxes(title_text = x_axis)
    fig.update_yaxes(title_text='Magnitude')
    fig.update_layout(
    yaxis = dict(autorange="reversed")
)
    fig.update_layout(title_text=header, showlegend=True)
    if save_fig:
        fig.write_html("{}.html".format(filename))
    fig.show()
    if return_fig:
        return fig

def Load_Light_Curves(folder,filters):
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
    light_curves = []
    for Filter in filters:
        get_id = True
        one_filter_curves = []
        ids = []
        filenames = glob(f'{folder}/{Filter}\*.csv')
        for file in tqdm(filenames,desc ='Loading {} curves'.format(Filter)):
            one_filter_curves.append(pd.read_csv(file))
            if get_id:
                ids.append(file[len(folder)+len(str(Filter))+2:-4])
        get_id = False
        light_curves.append(one_filter_curves)
    return light_curves,ids

def Pad_Light_Curves(light_curves,filters,minimum_length = 100):
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
    light_curves_copy = deepcopy(light_curves)
    #Getting the longest light curve
    longest = minimum_length
    for i,Filter in enumerate(filters):
        for light_curve in light_curves_copy[Filter]:
            if len(light_curve)>longest:
                longest = len(light_curve)
                longest_series = light_curve
    #Reindexing the curves less than the longest one
    for i,Filter in enumerate(filters):
        for j,light_curve in tqdm(enumerate(light_curves_copy[Filter]),desc = 'Padding Light Curves'):
            if len(light_curve) != longest:
                fill_number = longest - len(light_curve)
                new_rows = pd.DataFrame({'mjd':list(np.linspace(light_curve['mjd'].iloc[-1]+0.2,light_curve['mjd'].iloc[-1]+0.2*(fill_number+1),fill_number)),
                'mag':[light_curve['mag'].mean()]*fill_number,
                'magerr':[light_curve['magerr'].mean()]*fill_number})
                new_rows = pd.DataFrame(new_rows)
                light_curves_copy[i][j] = pd.concat((light_curve,new_rows))
    return light_curves_copy[:len(filters)]

def scale_curves(light_curves,what_scaler = 'default',scale_times = True):
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
    #
    scaler_dictionary = {'standard':StandardScaler(),'minmax':MinMaxScaler(),'robust':RobustScaler(),'default':MinMaxScaler(feature_range=(-2,2))}
    scaled_curves = []
    scaled_times = []
    #Scaling each light curve
    scaled_curves_one_filt =[]
    for i in tqdm(range(len(light_curves)),desc = 'Scaling Magnitudes'):
        mags_to_scale = pd.DataFrame(light_curves[i]['mag'])
        scaler = scaler_dictionary[what_scaler]
        scaled_curves.append(scaler.fit_transform(mags_to_scale))
        scaled_curves[i] = scaled_curves[i].reshape(len(mags_to_scale))
    #Scaling the times if selected
    if scale_times:
        for i in tqdm(range(len(light_curves)),desc = 'Scaling Times'):
            times_to_scale = pd.DataFrame(light_curves[i]['mjd'])
            scaler_time = scaler_dictionary['default']
            scaled_times.append(scaler_time.fit_transform(times_to_scale))
            scaled_times[i] = scaled_times[i].reshape(len(times_to_scale))
    return scaled_curves,scaled_times

def SOM_1D(scaled_curves,som_x = None,som_y = None,learning_rate = 0.1,sigma = 1.0,topology = 'rectangular',pca_init = True,\
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
    default_som_grid_length = math.ceil(math.sqrt(math.sqrt(len(scaled_curves))))
    if som_x is None and som_y is None:
        som_x = som_y = default_som_grid_length
    elif som_x is None or som_y is None:
        print('Please Provide both som_x and som_y or neither, going with the default values of the sqrt')
        som_x = som_y = default_som_grid_length
    som_model = MiniSom(som_x,som_y,len(scaled_curves[0]),learning_rate = learning_rate,sigma = sigma,\
                       topology = topology, neighborhood_function = neighborhood_function,random_seed=random_seed)
    if pca_init is True:
        som_model.pca_weights_init(scaled_curves)
    max_iter = epochs
    q_error = []
    t_error = []
    indices_to_plot = []
    if stat == 'both':
        stat = 'qt'
    if train_mode == 'random':
        if early_stopping_no is None:
            early_stopping_no = max_iter
        for i in tqdm(range(max_iter),desc = 'Evaluating SOM'):
            rand_i = np.random.randint(len(scaled_curves))
            som_model.update(scaled_curves[rand_i], som_model.winner(scaled_curves[rand_i]), i, max_iter)
            if (i % plot_frequency == 0 or i == len(scaled_curves)-1) and plot_training:
                indices_to_plot.append(i)
                if 'q' in stat:
                    q_error.append(som_model.quantization_error(scaled_curves))
                if 't' in stat:
                    t_error.append(som_model.topographic_error(scaled_curves))
            if i == early_stopping_no:
                break
    elif train_mode == 'all':
        count = 0
        if early_stopping_no is None:
            early_stopping_no = len(scaled_curves)+batch_size
        for i in tqdm(range(0,len(scaled_curves),batch_size),desc = 'Batch Training'):
            batch_data = scaled_curves[i:i+batch_size]
            for t in range(epochs):
                for idx,data_vector in enumerate(batch_data):
                    som_model.update(batch_data[idx], som_model.winner(batch_data[idx]), t,epochs)
                if (t % plot_frequency == 0 or t == len(scaled_curves)-1) and plot_training:
                    if 'q' in stat:
                        q_error.append(som_model.quantization_error(scaled_curves))
                    if 't' in stat:
                        t_error.append(som_model.topographic_error(scaled_curves))
                    indices_to_plot.append(count)
                count += 1
            if i>early_stopping_no+batch_size:
                break  
    if save_som:
        with open(model_save_path+'som_model.p', 'wb') as outfile:
            pickle.dump(som_model, outfile)
        print('Model Saved')
    return som_model, q_error,t_error, indices_to_plot

def plot_training(training_metric_results,metric,plotting_frequency,indices_to_plot,figsize = (10,10),save_figs = True,fig_save_path = './'):
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
    
    #Plots the metric given (quantization error or topographic error) 
    plt.figure(figsize = figsize)
    plt.plot(indices_to_plot,training_metric_results)
    plt.ylabel(metric)
    plt.xlabel('iteration index')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(fig_save_path+'Plots/Model_Training_'+metric+'.png')  

def Plot_SOM_Scaled_Average(som_model,scaled_curves,dba = True,figsize = (10,10),save_figs = True,figs_save_path = './',\
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
    som_x,som_y = som_model.distance_map().shape
    win_map = som_model.win_map(scaled_curves)
    total = len(win_map)
    cols = int(np.sqrt(len(win_map)))
    rows = total//cols
    if total % cols != 0:
        rows += 1
    if one_fig:
        fig, axs = plt.subplots(rows,cols,figsize = figsize,layout="constrained")
        fig.suptitle('Clusters')
        count = 0
        for x in tqdm(range(som_x),desc = 'Creating Plots'):
            for y in range(som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    no_obj_in_cluster = 0
                    for series in win_map[cluster]:
                        if plot_background:
                            if no_obj_in_cluster == 0:
                                axs.flat[count].plot(series,c="gray",alpha=0.5,label = 'Light Curves')
                            else:
                                axs.flat[count].plot(series,c="gray",alpha=0.5)
                        no_obj_in_cluster += 1
                if plot_weights:
                    weights = som_model.get_weights()[x][y]
                    axs.flat[count].plot(range(len(weights)),weights,c = 'red',label = 'SOM Representation')
                if plot_avg:
                    if dba is True:
                        axs.flat[count].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="blue",label = 'Average Curve')
                    else:
                        axs.flat[count].plot(np.mean(np.vstack(win_map[cluster]),axis=0),c="blue",label = 'Average Curve')
                axs.flat[count].set_title(f"Cluster {x*som_y+y+1}: {no_obj_in_cluster} curves")
                axs.flat[count].legend()
                count += 1
        if save_figs:
            if 'Plots' not in os.listdir():
                os.makedirs(figs_save_path+'Plots')
            plt.savefig(figs_save_path+'Plots/Scaled_Averaged_Clusters.png')
        plt.show()
    else:
        for x in tqdm(range(som_x),desc = 'Creating Plots'):
            for y in range(som_y):
                plt.figure(figsize = figsize)
                cluster = (x,y)
                if cluster in win_map.keys():
                    no_obj_in_cluster = 0
                    for series in win_map[cluster]:
                        if plot_background:
                            if no_obj_in_cluster == 0:
                                plt.plot(series,c="gray",alpha=0.5,label = 'Light Curves')
                            else:
                                plt.plot(series,c="gray",alpha=0.5)
                        no_obj_in_cluster += 1
                if plot_weights:
                    weights = som_model.get_weights()[x][y]
                    plt.plot(range(len(weights)),weights,c = 'red',label = 'SOM Representation')
                if plot_avg:
                    if dba is True:
                        plt.plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="blue",label = 'Average Curve')
                    else:
                        plt.plot(np.mean(np.vstack(win_map[cluster]),axis=0),c="blue",label = 'Average Curve')
                plt.title(f"Cluster {x*som_y+y+1}: {no_obj_in_cluster} curves")
                plt.xlabel('Cadence Counts')
                plt.ylabel('Scaled Magnitude')
                plt.legend()
                if save_figs:
                    if 'Plots' not in os.listdir():
                        os.makedirs(figs_save_path+'Plots')
                    if 'Scaled_Clusters' not in os.listdir(figs_save_path+'Plots'):
                        os.makedirs(figs_save_path+'Plots/Scaled_Clusters')
                    plt.savefig(figs_save_path+f'Plots/Scaled_Clusters/Cluster_{x*som_y+y+1}.png')
                if show_fig is False:
                    plt.close()

def SOM_Nodes_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
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
    plt.figure(figsize = figsize)
    plt.pcolor(som_model.distance_map().T, cmap=cmap,edgecolors='k')
    cbar = plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/SOM_Nodes_Map.png')
    plt.show()
    
def Assign_Cluster_Labels(som_model,scaled_curves,ids):
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
    cluster_map = []
    som_y = som_model.distance_map().shape[1]
    for idx in tqdm(range(len(scaled_curves)),desc = 'Creating Dataframe'):
        winner_node = som_model.winner(scaled_curves[idx])
        cluster_map.append((ids[idx],winner_node[0]*som_y+winner_node[1]+1))
    clusters_df=pd.DataFrame(cluster_map,columns=["ID","Cluster"])
    return clusters_df

def SOM_Clusters_Histogram(cluster_map,color,save_figs = True,figs_save_path = './',figsize = (5,5)):
    '''
    Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors
    
    Parameters
    ----------
    cluster_map:  
    The dataframe with each id and the cluster that it belongs to
    
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
    cluster_map.value_counts('Cluster').plot(kind = 'bar',color = color,figsize = figsize)
    plt.ylabel('No of quasars')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/Clusters_Histogram.png')
        
def findMin(x, y, umat):
    #Finds minimum node
    newxmin=max(0,x-1)
    newxmax=min(umat.shape[0],x+2)
    newymin=max(0,y-1)
    newymax=min(umat.shape[1],y+2)
    minx, miny = np.where(umat[newxmin:newxmax,newymin:newymax] == umat[newxmin:newxmax,newymin:newymax].min())
    return newxmin+minx[0], newymin+miny[0]

def findInternalNode(x, y, umat):
    #Finds node with internal minimum
    minx, miny = findMin(x,y,umat)
    if (minx == x and miny == y):
        cx = minx
        cy = miny
    else:
        cx,cy = findInternalNode(minx,miny,umat)
    return cx, cy
        
def Get_Gradient_Cluster(som):
    #Get the SOM Gradient Clusters
    cluster_centers = []
    cluster_pos  = []
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            cluster_centers.append(np.array([cx,cy]))
            cluster_pos.append(np.array([row,col]))
    return np.array(cluster_centers),np.array(cluster_pos)

def SOM_Nodes_to_Gradient_Centers(som,size_of_data):
    #Create a mapping from the som nodes to the gradient clusters
    gradient_centers = [int(cluster_center[0]*np.ceil(np.sqrt(np.sqrt(size_of_data)))+cluster_center[1]) for cluster_center in Get_Gradient_Cluster(som)[0]]
    actual_clusters = [int(cluster_center[0]*np.ceil(np.sqrt(np.sqrt(size_of_data)))+cluster_center[1]) for cluster_center in Get_Gradient_Cluster(som)[1]]
    clustering_map = pd.DataFrame({'Original Clusters':actual_clusters,'New Clusters':gradient_centers})
    return clustering_map

def Gradient_Cluster_Map(som,scaled_curves,ids,dimension = '1D',fill = 'mean',interpolation_kind = 'cubic',clusters = None,som_x = None,som_y = None):
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
    if dimension == '1D':
        cluster_centers,cluster_pos = Get_Gradient_Cluster(som)
    else:
        cluster_centers,cluster_pos = Get_Gradient_Cluster_2D(som,fill,interpolation_kind)
    cluster_numbers = np.arange(len(np.unique(cluster_centers,axis = 0)))
    unique_cluster_centers = np.unique(cluster_centers,axis = 0)
    cluster_numbers_map = []
    for i in range(len(scaled_curves)):
        if dimension == '1D':
            winner_node = som.winner(scaled_curves[i])
            winner_node = np.array(winner_node)
        else:
            winner_x = (clusters[i]-1)//som_y
            winner_y = (clusters[i]-1)%som_y
            winner_node = np.array([winner_x,winner_y])
        #Gets the central node where the winning cluster is in
        central_cluster = cluster_centers[np.where(np.isclose(cluster_pos,winner_node).sum(axis = 1) == 2)][0]
        cluster_number = cluster_numbers[np.where(np.isclose(unique_cluster_centers,central_cluster).sum(axis = 1) == 2)]
        cluster_numbers_map.append(cluster_number[0]+1)
    return pd.DataFrame({'ID':ids,'Cluster':cluster_numbers_map})

def matplotlib_cmap_to_plotly(cmap, entries):
    #Used for creating interactive plot
    h = 1.0/(entries-1)
    colorscale = []

    for k in range(entries):
        C = (np.array(cmap(k*h)[:3])*255)
        colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return colorscale

def plotStarburstMap(som):
    #Plots the Starburst Gradient Visualization of the clusters
    boner_rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    bone_r_cmap = matplotlib.colormaps.get_cmap('bone_r')

    bone_r = matplotlib_cmap_to_plotly(bone_r_cmap, 255)

    layout = go.Layout(title='Gradient Based Clustering')
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Heatmap(z=som.distance_map().T, colorscale=bone_r))
    shapes=[]
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            shape=go.layout.Shape(
                    type="line",
                    x0=row,
                    y0=col,
                    x1=cx,
                    y1=cy,
                    line=dict(
                        color="Black",
                        width=1
                    )
                )
            shapes=np.append(shapes, shape)

    fig.update_layout(shapes=shapes.tolist(), 
        width=500,
        height=500) 
    
    fig.show()
    
def outliers_detection(clusters_df,som,scaled_curves,ids,outlier_percentage = 0.2):
    '''
    Gives the percentage of the clusters that have high quanitization errors (defined by percentile) for each cluster
    
    Parameters
    ----------
    clusters_df:
    A map of each of the ids to the clusters
    
    som:  
    The trained SOM
    
    scaled_curves: np.ndarray
    The scaled curves used to train the SOM
    
    ids: list
    The ids of the curves
    
    outlier_percentage: float
    This top percentile that defines an outlier
 
    Returns
    --------
    Plots:
    Distribution of Outliers per cluster and distribution of quantization error
    '''
    #Detects outliers that aren't quantized well as a percentage of the clusters
    quantization_errors = np.linalg.norm(som.quantization(scaled_curves) - scaled_curves, axis=1)
    error_treshold = np.percentile(quantization_errors, 
                               100*(1-outliers_percentage)+5)
    outlier_ids = np.array(ids)[quantization_errors>error_treshold]
    outlier_cluster = []
    for i in range(len(clus.ID)):
        if str(clus.ID[i]) in outlier_ids:
            outlier_cluster.append(clus.Cluster[i])
    #Plot the number of outliers per cluster
    plt.figure()
    plt.hist(clus['Cluster'],bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'Total number of clusters',edgecolor = 'k')
    plt.hist(outlier_cluster,bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'outliers',edgecolor = 'k')
    plt.xlabel('Cluster')
    plt.ylabel('No of Quasars')
    plt.legend()
    #Plot the treshold for quantization error
    plt.figure()
    plt.hist(quantization_errors,edgecolor = 'k',label = f'Threshold = {outlier_percentage}')
    plt.axvline(error_treshold, color='k', linestyle='--')
    plt.legend()
    plt.xlabel('Quantization Error')
    plt.ylabel('No of Quasars')
    
def Cluster_Metrics(scaled_curves,cluster_map,metric = 'Silhoutte'):
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
    if metric == 'Silhoutte':
        score = silhouette_score(scaled_curves,cluster_map['Cluster'])
    elif metric == 'DBI':
        score = davies_bouldin_score(scaled_curves,cluster_map['Cluster'])
    elif metric == 'CH':
        score = calinski_harabasz_score(scaled_curves,cluster_map['Cluster'])
    else:
        score = 0
        print('Please use Silhoutte, DBI or CH')
    return score

def save_chosen_cluster(chosen_cluster,filters,cluster_map,overwrite = True,save_path = './',source_path = './Light_Curves'):
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
    #Save the light curves in the chosen cluster in a way they can be processed by QNPy
    #If folder isn't made, will create this folder to save the curves
    if overwrite:
        if f'Cluster_{chosen_cluster}' in os.listdir(save_path):
            shutil.rmtree(save_path+f'Cluster_{chosen_cluster}')
    os.makedirs(save_path+f'Cluster_{chosen_cluster}',exist_ok = True)
    chosen_ids = cluster_map['ID'][cluster_map['Cluster'] == chosen_cluster].to_numpy()
    for Filter in tqdm(filters,desc = 'Saving Filters'):
        os.makedirs(save_path+f'Cluster_{chosen_cluster}/'+Filter,exist_ok = True)
        for ID in chosen_ids:
            shutil.copyfile(source_path+'/'+Filter+f'/{ID}.csv', save_path+f'Cluster_{chosen_cluster}/{Filter}/{ID}.csv')
    print('Cluster Saved')
    
    
### 2D Clustering Functions

def scale_to_range(series, min_val=-2, max_val=2):
    #Used to scale a series
    min_series = series.min()
    max_series = series.max()
    return min_val + (max_val - min_val) * (series - min_series) / (max_series - min_series)

def masked_euclidean_distance(data1, data2, mask):
    "Calculate Euclidean distance, ignoring masked elements."
    return np.sqrt(np.ma.sum((np.ma.array(data1, mask=mask) - np.ma.array(data2, mask=mask)) ** 2))

def multi_band_processing(light_curves,ids,filter_names = 'ugriz',return_wide = False):
    '''
    Processes the light curves into a wide table
    
    Parameters
    ----------
    light_curves: 
    The light curves to be used
    
    ids: list,array
    The ids of the quasars
    
    filter_names: list or str(if the filters are one letter)
    The filters that are used
    
    return_wide: bool
    Whether to return the wide table or a flat table
    
    Returns
    --------
    light_curves_wide:
    The pivot table of the light curves with time steps
    
    light_curves_wide.isna():
    The mask used with the wide light curves
    
    OR 
    
    light_curves_flat:
    The flattened pivot table of the light curves
   
    mask_flat:
    The mask used
    '''
    #The preprocessing of tne light curves
    data = deepcopy(light_curves)
    #Adding ID and Filter Columns to the data
    for filter_name in range(len(filter_names)):
        for j in range(len(data[filter_name])):
            data[filter_name][j]['Filter'] = filter_name
            data[filter_name][j]['ID'] = ids[j]
    #Concatenating all the light curves together
    concatenated_by_filter = []
    for i in tqdm(range(len(filter_names)),desc = 'concat'):
        concat_filter = pd.concat(data[i])
        concatenated_by_filter.append(concat_filter)
    big_df = pd.concat(concatenated_by_filter)
    #Scaling the time and magnitude
    big_df['time_scaled'] = big_df.groupby(['ID', 'Filter'])['mjd'].transform(scale_to_range)
    big_df['mag_scaled'] = big_df.groupby(['ID', 'Filter'])['mag'].transform(scale_to_range)
    #Creating the mask from the densest bin (This isn't being used currently, but can be changed)
    densest_band = big_df.groupby('Filter').count().idxmax()['mjd']
    big_df['mask'] = big_df['Filter'] != densest_band
    # Pivot the DataFrame to wide format
    light_curves_wide = big_df.pivot_table(index='ID', columns=['time_scaled', 'Filter'], values='mag_scaled')
    # Create mask for missing data (This mask fills in all the missing data)
    mask = light_curves_wide.isna()
    #Flatten the pivot table
    light_curves_flat = light_curves_wide.values
    mask_flat = mask.values
    if return_wide:
        return light_curves_wide,light_curves_wide.isna()
    else:
        return light_curves_flat,mask_flat
    
def multi_band_clustering(light_curves,ids,filter_names = 'ugriz',som_x = None,som_y = None,sigma = 1.0,learning_rate = 0.5,\
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
    #First, process the input data for clustering
    processed_light_curves, processed_mask = multi_band_processing(light_curves,ids,filter_names)
    print('Processed')
    default_som_grid_length = math.ceil(math.sqrt(math.sqrt(len(processed_light_curves))))
    print(default_som_grid_length)
    #Now, initialize the SOM
    if som_x is None and som_y is None:
        som_x = som_y = default_som_grid_length
    elif som_x is None or som_y is None:
        print('Please Provide both som_x and som_y or neither, going with the default values of the sqrt')
        som_x = som_y = default_som_grid_length
    som = MiniSom(som_x, som_y, processed_light_curves.shape[1], sigma, learning_rate)
    som.random_weights_init(processed_light_curves)
    failed = 0
    #Training the SOM
    for i in tqdm(range(0, len(processed_light_curves), batch_size),desc = 'Batch Training'):
        batch_data = processed_light_curves[i:i + batch_size]
        batch_mask = processed_mask[i:i + batch_size]
        if early_stopping_no is None:
            early_stopping_no = len(processed_light_curves)+batch_size
        for t in range(num_iterations):
            for idx, data_vector in enumerate(batch_data):
                data_mask = batch_mask[idx]
                bmu_index = None
                min_distance = float('inf')
                iteration_weights = som.get_weights()
                # Find BMU considering masked data
                for x in range(som_x):
                    for y in range(som_y):
                        w = iteration_weights[x, y]
                        distance = masked_euclidean_distance(data_vector, w, data_mask)
                        if distance < min_distance:
                            min_distance = distance
                            bmu_index = (x, y)
                # Update SOM weights
                try:
                    som.update(data_vector, bmu_index, t, num_iterations)
                except:
                    failed += 1
        if i == early_stopping_no:
                break
    return som,processed_light_curves,processed_mask

def find_cluster_and_quantization_errors(som,data,masks):
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
    #Finding the BMU and the quantization error of each data point
    flat_weights = som.get_weights().reshape(-1,som.get_weights().shape[2])
    min_clusters = []
    quantization_error = []
    for i in tqdm(range(len(data))):
        data_point = data[i]
        mask = masks[i]
        distances = []
        for weight in flat_weights:
            distances.append(masked_euclidean_distance(data_point,weight,mask))
        min_clusters.append(np.argmin(distances)+1)
        quantization_error.append(np.min(distances))
    return min_clusters,quantization_error

def Get_Gradient_Cluster_2D(som,fill = 'mean',interpolation_kind = 'cubic'):
    '''
    Finding the gradient clusters from the 2D SOM
    
    Parameters
    ----------
    som: 
    The trained SOM
    
    fill: str
    'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function
    
    interpolation_kind: 
    Any of the scipy.interp1d interpolation kinds. Recommended to use cubic
    
    Returns
    --------
    cluster_centers:
    The cluster centers
    
    cluster_pos:
    The cluster positions
    '''
    new_som = deepcopy(som)
    for i in tqdm(range(len(new_som._weights))):
        for j in range(len(new_som._weights[i])):
            if fill == 'mean':
                new_som._weights[i][j] = np.nan_to_num(new_som._weights[i][j],nan = np.nanmean(new_som._weights[i][j]))
            elif fill == 'interpolate':
                array = new_som._weights[i][j]
                # Find indices of non-NaN values
                non_nan_indices = np.where(~np.isnan(array))[0]
                interpolator = interp1d(non_nan_indices, array[non_nan_indices], kind=interpolation_kind, fill_value='extrapolate')
                array_interpolated = interpolator(np.arange(len(array)))
                new_som._weights[i][j] = array_interpolated
    cluster_centers = []
    cluster_pos  = []
    for row in np.arange(new_som.distance_map().shape[0]):
        for col in np.arange(new_som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, new_som.distance_map().T)
            cluster_centers.append(np.array([cx,cy]))
            cluster_pos.append(np.array([row,col]))
    return np.array(cluster_centers),np.array(cluster_pos)