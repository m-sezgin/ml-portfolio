import pandas as pd
import numpy as np
from sklearn import linear_model

def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)

def data_wrangle(dataset_file, lst):
    
    # Import data with pandas
    data = pd.read_csv(dataset_file)
    
    for column in lst:
        # tip retrieved from https://cmdlinetips.com/2018/01/
        # how-to-get-unique-values-from-a-column-in-pandas-data-frame/
        unique_values = data[column].unique()
        
        numeric_val = 0
        col_names = []
        
        for value in unique_values:
            
            # For clarity
            print(value , "in column" , column , "is now" , numeric_val)
            
            data = data.replace(value, numeric_val)
            numeric_val += 1
            
            # Extract the data's column names from the observation rows
            col_names = data.loc[data[column] == value].columns
            
    if lst == []:
        col_names = data.columns
            
    return np.array(col_names), data.to_numpy()

def kfold_CV(data, col_names, inputs, output, k):
    # Shuffle data so cross-val error is not coincidentally zero
    np.random.shuffle(data)
    
    ### Note: does not always split data evenly depending on number of rows and value of k
    if data.shape[0]%k != 0:
        maximum = data.shape[0]%k
    else:
        maximum = -1
    
    # If inputs or output is given as a single variable string convert to list
    if type(inputs) != list:
        inputs = [inputs]
        
    if type(output) != list:
        output = [output]
    
    # Convert columns to np array if they are passed as a list for np.argwhere to function properly   
    if type(col_names) == list:
        col_names = np.array(col_names)
    
    input_col_inds = []
    output_col_inds = []
    
    # Find numerical column index for input and output variables using list of ordered columns
    for i in inputs:
        input_col_inds.append(np.argwhere(col_names == str(i))[0][0])
    for p in output: 
        output_col_inds.append(np.argwhere(col_names == str(p))[0][0])
        
    num_rows = data.shape[0]
    
    start = 0
    end = num_rows//k
    count = 0
    test_errors = []

    # Loop over all folds in the data set, letting each act as the test set
    for fold in range(k):
        
        # Split data into train and test
        test_data = data[start:end, :]
        train_indices = list(set(range(num_rows)).difference(list(set(range(start, end)))))
        train_data = data[train_indices, :]
        
        # Create and train a model
        lm = linear_model.LinearRegression()
        mod = lm.fit(train_data[:, input_col_inds], train_data[:, output_col_inds])
    
        # Compute the testing error and add it to the list of testing errors
        test_preds = mod.predict(test_data[:, input_col_inds])
        test_error = compute_mse(test_preds, test_data[:, output_col_inds])
        test_errors.append(test_error)
                             
        start = start + num_rows//k
        end = start + num_rows//k
        count += 1
        
        if count <= maximum:
            start += 1
            end += 1
        
    # Compute the cross-val error
    return np.mean(test_errors)