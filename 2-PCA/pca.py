# Import Block

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import and prepare data
babies_wrangled = pd.read_csv("babies_wrangled.csv")
babies_cv = babies_wrangled.loc[:, ["gestation", "race", "age", "wt"]]
babies_np = babies_cv.to_numpy()

def standardize(data):
    ## Data must be passed as np array
    
    # Standardizing our variables:
    mean_vec = np.mean(data, axis = 0)
    sd_vec = np.std(data, axis = 0)

    data_std = data.copy()

    for i in range(data.shape[1]):
        data_std[:, i] = (data[:, i]- mean_vec[i]*np.ones(data.shape[0]))/sd_vec[i]
        
    return data_std

def unstandardize(projected_data, original_data):
    ## Data must be passed as np array
    
    # Unstandardizing our variables:
    mean_vec = np.mean(original_data, axis = 0)
    sd_vec = np.std(original_data, axis = 0)
    
    data_undo = projected_data.copy()
    
    for i in range(projected_data.shape[1]):
        data_undo[:,i] = projected_data[:,i]*sd_vec[i]+mean_vec[i]
    
    return data_undo

def dimension_reduction_PCA(data, desired_dimension, plot_std):
    ## Note: data should be passed as np array, desired_dimension should be an int, 
    ## and plot_std should be True or False
    
    # standardize data
    data_std = standardize(data)
    
    # Step one: Set up PCA by defining the number of components 
    pca_alg = PCA(n_components = desired_dimension)

    # Step two: Fit the algorithm to the standardized data
    pfit = pca_alg.fit(data_std)
    
    # Step three: Transform the data. 
    # (This step is what actually projects the data onto the components)
    data_in_nd = pca_alg.transform(data_std)
    
    if plot_std == True:
        data_in_nd = unstandardize(data_in_nd, data)
        
    if desired_dimension == 1:
        plt.scatter(data_in_nd,np.zeros(data_in_nd.shape[0]))
    
    # Adds regression line
    elif desired_dimension == 2:
        plt.scatter(data_in_nd[:, 0], data_in_nd[:, 1])
        
        lm = linear_model.LinearRegression()
        mod = lm.fit(data_in_nd[:, 0].reshape(-1, 1), data_in_nd[:, 1])
        print("m is", mod.coef_[0], "and b is", mod.intercept_)
        
        plt.plot(data_in_nd[:, 0], mod.predict(data_in_nd[:, 0].reshape(-1, 1)), color = 'purple')
    
    elif desired_dimension == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Create the SCATTER() plot 
        ax.scatter(data_in_nd[:,0], data_in_nd[:,1], data_in_nd[:,2]);
    
    total_variation = np.sum(pfit.explained_variance_ratio_)
    
    return total_variation
    