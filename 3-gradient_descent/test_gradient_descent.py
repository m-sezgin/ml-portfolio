# Import block
from sklearn import linear_model

import numpy as np
import pandas as pd

import gradient_descent

# Prepare the data
babies_wrangled = pd.read_csv("babies_wrangled.csv")

# For regular gradient descent
gestation = babies_wrangled[["gestation"]].to_numpy()
weight = babies_wrangled[["wt"]].to_numpy()

# For stochastic gradient descent
babies_gw = babies_wrangled.loc[:, ["gestation", "wt"]].to_numpy()

# For interaction model gradient descent
babies_interaction = babies_wrangled.copy()
babies_interaction["age*smoke"] = babies_interaction["smoke"]*babies_interaction["smoke"]
babies_int_np = babies_interaction.loc[:, ["age", "smoke", "age*smoke", "wt"]].to_numpy()

def test_gd_size():
    out = gradient_descent.gradient_descent(50, gestation, weight, .00001, .001)
    assert len(out) == 5
    assert len(out[0]) == 2
    for i in range(1, 5):
        assert len(out[i]) == 50


def test_gd_type():
    out = gradient_descent.gradient_descent(50, gestation, weight, .00001, .001)
    assert type(out) == tuple
    
    for i in range(0, 5):
        assert type(out[1]) == list
    
def test_stochastic_size():
    out = gradient_descent.stochastic_gd(50, babies_gw, .00001, .001)
    assert len(out) == 3
    assert len(out[0]) == 2
    assert len(out[1]) == 50
    assert len(out[2]) == 50
    
    
def test_stochastic_type():
    out = gradient_descent.stochastic_gd(50, babies_gw, .00001, .001)
    assert type(out) == tuple
  
    for i in range(0, 5):
        assert type(out[1]) == list
    
def test_interaction_size():
    out = gradient_descent.interaction_gd(50, babies_int_np[:, 0], 
    babies_int_np[:, 1], babies_int_np[:, 3], .00000001, .001)
    assert len(out) == 5
    assert len(out[0]) == 4
    for i in range(1, 5):
        assert len(out[i]) == 50
    
    
def test_interaction_type():
    out = gradient_descent.interaction_gd(50, babies_int_np[:, 0], 
    babies_int_np[:, 1], babies_int_np[:, 3], .00000001, .001)
    assert type(out) == tuple
  
    for i in range(0, 5):
        assert type(out[1]) == list