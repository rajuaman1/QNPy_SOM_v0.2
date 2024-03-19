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

 .. code-block:: python
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
  .. code-block:: python
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

  
