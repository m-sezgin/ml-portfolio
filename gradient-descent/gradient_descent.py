# Import libraries
from sklearn import linear_model
from numpy import linalg as LA

import numpy as np
import pandas as pd

def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)

def compute_m_partial(in_vals, truth_vec, predict_vec):
    return -2*np.mean(in_vals*(truth_vec-predict_vec))

def compute_b_partial(truth_vec, predict_vec):
    return -2*np.mean(truth_vec-predict_vec)

def adjust_L(current_L, grad_step_num):
    # We can adjust this function however we want
    new_L = current_L * .95
    return new_L

def gradient_descent(n_steps, var1, var2, learning_rate, gradient_tolerance):
    # Initialize the max number of steps you wish to take
    max_steps = n_steps

    # Initialize starting parameters
    m = 3
    b = 0

    # Set a tolerance for the smallest you will allow 
    # the length of the gradient to be before stopping 
    grad_tol = gradient_tolerance

    # Create empty lists to store values for m, b, and the associated MSE
    outm = []
    outb = []
    outdm = []
    outdb = []
    outmse = []

    # Create an iterative process (ie. a loop) that will take N_STEPS
    for stp in range(max_steps):
        # For the current values of m and b: 
    
        # 1. Compute the MSE
        preds = m*var1 + b
        errormse = compute_mse(var2,preds)
    
        # 2. Store m, b, and the associated MSE in the output lists:
        outm.append(m)
        outb.append(b)
        outmse.append(errormse)
    
        # Update m and b by:
    
        # 1. Computing the gradient
        d_m = compute_m_partial(var1,var2,preds)
        d_b = compute_b_partial(var2, preds)
        
        outdm.append(d_m)
        outdb.append(d_b)
    
        # Compute the length of the gradient
        #norm_grad = np.linalg.norm()
        norm_grad = LA.norm([d_m, d_b])
    
        # If the length of the gradient is small enough, stop iterating
        if norm_grad < grad_tol:
            break
        
        # Update the values for m and b
        m = m - (learning_rate*d_m)
        b = b - (learning_rate*d_b)
    
        #Update learning rate
        #learning_rate = adjust_L(learning_rate, stp)
        
    return [outm[-1],outb[-1]], outm, outb, outdm, outdb

def stochastic_gd(n_steps, data, learning_rate, gradient_tolerance):
    ## This implementation takes a numpy multi-dimensional array with explanatory variable as
    ## column 1 and response variable as column 2 for data shuffling purposes

    # Initialize starting parameters
    m = 0
    b = 0

    # Create empty lists to store values for m, b, and the associated MSE
    outm = []
    outb = []
    outmse = []

    # Create an iterative process (ie. a loop) that will take N_STEPS
    for stp in range(n_steps):
        
        # shuffle data
        np.random.shuffle(data)
        
        # For the current values of m and b: 
    
        # 1. Compute the MSE
        rand_ind = np.random.choice(data.shape[0], size=1)
        preds = m*data[:, 0][rand_ind] + b
        errormse = compute_mse(data[:, 1][rand_ind],preds)
    
        # 2. Store m, b, and the associated MSE in the output lists:
        outm.append(m)
        outb.append(b)
        outmse.append(errormse)
    
        # Update m and b by:
        
        # 1. Computing the gradient
        d_m = compute_m_partial(data[:, 0][rand_ind], data[:, 1][rand_ind],preds)
        d_b = compute_b_partial(data[:, 1][rand_ind], preds)
    
        
        # Update the values for m and b
        m = m - (learning_rate*d_m)
        b = b - (learning_rate*d_b)
    
        # Update learning rate
        learning_rate = adjust_L(learning_rate, stp)
        
    return [outm[-1], outb[-1]], outm, outb

def interaction_gd(n_steps, var1, var2, response, learning_rate, gradient_tolerance):
    ## Only takes two explanatory variables

    # Initialize starting parameters
    m1 = 0
    m2 = 0
    m3 = 0
    b = 0

    # Set a tolerance for the smallest you will allow 
    # the length of the gradient to be before stopping 
    grad_tol = gradient_tolerance

    # Create empty lists to store values for m, b, and the associated MSE
    outm1 = []
    outm2 = []
    outm3 = []
    outb = []
    outmse = []

    # Create an iterative process (ie. a loop) that will take N_STEPS
    for stp in range(n_steps):
        # For the current values of m and b: 
    
        # 1. Compute the MSE
        preds = m1*var1 + m2*var2 + m3*var1*var2 + b
        errormse = compute_mse(response,preds)
    
        # 2. Store m, b, and the associated MSE in the output lists:
        outm1.append(m1)
        outm2.append(m2)
        outm3.append(m3)
        outb.append(b)
        outmse.append(errormse)
    
        # Update m and b by:
    
        # 1. Computing the gradient
        d_m1 = compute_m_partial(var1,response,preds)
        d_m2 = compute_m_partial(var2,response,preds)
        d_m3 = compute_m_partial(np.multiply(var1, var2),response,preds)
        d_b = compute_b_partial(response, preds)
    
        # Compute the length of the gradient
        #norm_grad = LA.norm([d_m1, d_b1])
    
        # If the length of the gradient is small enough, stop iterating
        #if norm_grad < grad_tol:
            #break
        
        # Update the values for m and b
        m1 = m1 - (learning_rate*d_m1)
        m2 = m2 - (learning_rate*d_m2)
        m3 = m3 - (learning_rate*d_m3)
        b = b - (learning_rate*d_b)
    
        # Update learning rate
        #L = adjust_L(L, stp)
        
    return [outm1[-1], outm2[-1], outm3[-1], outb[-1]], outm1, outm2, outm3, outb