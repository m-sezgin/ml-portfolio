import pandas as pd
import numpy as np
import pca

babies_wrangled = pd.read_csv("babies_wrangled.csv")
babies_cv = babies_wrangled.loc[:, ["gestation", "race", "age", "wt"]]
babies_np = babies_cv.to_numpy()

def test_standardize_size():
	out = pca.standardize(babies_np)
	assert out.shape == (babies_np.shape[0], babies_np.shape[1])
    
def test_standardize_type():
	out = pca.standardize(babies_np)
	assert type(out) == np.ndarray
    
def test_unstandardize_size():
	babies_std = pca.standardize(babies_np)
	out = pca.unstandardize(babies_std, babies_np)
	assert out.shape == (babies_np.shape[0], babies_np.shape[1])
    
def test_unstandardize_type():
	babies_std = pca.standardize(babies_np)
	out = pca.unstandardize(babies_std, babies_np)
	assert type(out) == np.ndarray
    
    
def test_dim_red_pca_type():
	out = pca.dimension_reduction_PCA(babies_np, 1, False)
	assert type(out) == np.float64
 