import pytest
import pandas as pd
import numpy as np
import cross_val

hw_data = pd.read_csv("hw5data.csv")

# Test data wrangle function

def test_data_wrangle_len():
	out = cross_val.data_wrangle("hw5data.csv",['job'])
	assert len(out) == 2

def test_data_wrangle_type():
	out = cross_val.data_wrangle("hw5data.csv",['job'])
	assert type(out) == type((3,2))

def test_data_wrangle_data():
	out = cross_val.data_wrangle("hw5data.csv",['job'])[1]
	assert np.sum(np.isnan(out)) == 0

# Test cross-val function

def test_kfold_CV_type():
	cols = ['neuroticism', 'performance', 'job', 'salary']
	data_np = cross_val.data_wrangle("hw5data.csv",['job'])[1]
	CV = cross_val.kfold_CV(data_np, cols, ['job'], 'salary',12)
	assert isinstance(CV, float) 

def test_kfold_CV_shape():
	cols = ['neuroticism', 'performance', 'job', 'salary']
	data_np = cross_val.data_wrangle("hw5data.csv",['job'])[1]
	CV = cross_val.kfold_CV(data_np, cols, ['job'], 'salary',12)

	expected = 1
	assert len([CV]) == expected

