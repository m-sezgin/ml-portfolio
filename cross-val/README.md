# Linear Regression and Cross-Validation

Welcome to the first component of this portfolio! In this component, 
we will take a look at how cross-validation can help us select the appropriate
variables for a linear regression model.

In order to explain cross-val, I will use an example dataset from my introductory
statistics final group project, which is a maternal health and infant outcome dataset 
called `babies.csv`. This data set contains information regarding the behavioral habits,
education level, and physical characteristics of mothers, in addition to 
the birth weight of their babies (as an outcome variable for infant health) 
and information about the fathers of these babies.

Though this data set has many variables, we will only be using a few
for the sake of conciseness and clarity. The variables we will be using
are the length of the gestation period (`gestation`), the mother's 
race (`race`), the mother's smoking habits (`smoke`), the mother's age (`age`),
and the infant's birth weight (`wt`). 

The code and explanation of concepts for this assignment are
all located in a notebook called `cross_val.ipynb`. 
There are unit tests for this piece that check that all aspects of the
code work as intended. In order to run these unit tests,
I have placed the functions from the cross-val notebook in the 
file `cross_val.py`. The file `test_cross_val.py` contains the tests for 
the functions. If you see these files, you can ignore them.

## Wrangling the Data

In this section we will cover data wrangling necessary to perform cross-validation.
The example data we're using already has its categorical variables stored as numbers
instead of strings, so although we need to convert it to a numpy array, we 
won't have to modify the data itself. But if this is not the case with your 
data and you would like to perform cross-validation on it, see the function
outlined in `cross_val.ipynb`.

## Implementing k-fold Cross-Validation

In this section, I explain the theory and motivations behind cross-val.
To illustrate cross-val in action, I made a function `kfold_CV`, which can be found
in the cross_val notebook. The inputs for this function are the chosen dataset as a numpy
array (with only numerical data), a list of the columns of the data set in order (so 
the function can fetch the input and output variable columns using numerical indexing),
a list of the input variables (as strings), a list of the output variables (as strings),
and the number of folds we want to divide our data into for cross-val, k.

Using these inputs, `kfold_CV` is able to select the input columns
and output columns from the data set, divide the data into k folds,
and then iteratively choose one fold to be the test set for k rounds of
training and testing. The linear regression model is trained on 100-(100/k)% of the 
data each iteration, and is then asked to predict the values of output variable
based on the input variable values in the test set. The error is then computed from 
the true and predicted values for each iteration, with the final cross-val error
being the mean of all these errors.

## Using 10-fold Cross-Validation

In this section, we will use a concrete value of k to provide a tangible example of 
cross-val. Using the k-fold cross-val function above with k = 10, we will select the
appropriate combination of variables to best predict `wt` with 10-fold cross-validation. 
We can do this by testing the following combinations of our input variables, 
which are (`gestation`, `race`, `age`, `smoke`):

* All possible one-variable inputs (4 choose 1)
* All possible two-variable combinations of our inputs (4 choose 2)
* All possible three-variable combinations of our inputs (4 choose 3)
* All possible four-variable combinations of our inputs (4 choose 4)

## Determining the full model

Once we have determined what variables our linear regression model 
should include, we'll use all of the data available to us to find
the coefficients for our model. By training our model on all of the data,
instead of just 90% of it like above, we will be able to more accurately tune 
our model.
