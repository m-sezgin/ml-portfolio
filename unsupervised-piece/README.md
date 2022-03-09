# PCA and Linear Regression

Welcome to the second component of this portfolio! This piece is best understood after reading/interacting with the first notebook on cross-validation, so be sure to check that notebook out before reading through this one.

## How Does PCA work and Why Should We Use It?

PCA or "Principal Component Analysis" is an algorithm that is helpful for reducing the dimensionality of our data. But why do we care about dimension reducing our data? Recall from the first notebook that the ultimate variable combination we chose was three explanatory variables and one response variable, making the data set four-dimensional. Because we don't know how to visualize or plot lines for four-dimensional data, it can be helpful to get an n-dimensional approximation of the data for visualization/plotting purposes (n must be a number in the range 1-3). 

Though I go into more detail about how PCA (using SVD) actually works in the notebook for this piece, briefly, PCA works by finding lines that maximize and capture the most spread within the data, so we can retain the most information about the data when we project it into lower dimensions. There are lines called components that contain varying amounts of information about the spread of the data. To understand this better, suppose we have four-dimensional data, and principal component 1 for this data captures 80% of the variation in the original data set, and component 2 captures 10% of additional variation. The two-dimensional approximation of our data using these components will then capture 90% of the variation in the original data, making it a pretty good approximation of the original data set.


## PCA for Regression

In this section, we cover how to use sklearn's PCA implementation to dimension-reduce the babies data set from earlier, and we cover how to plot our projected data. This section also covers how to choose the appropriate dimension for a dataset by looking at the percent of variation explained by principal components. Then we fit a trendline to the dimension-reduced data, and discuss the results, and what they mean in context of the results from the notebook on cross-validation.

## Conclusion

This notebook finishes off with a function that you can use for your own dataset that will reduce the dimensionality of high-dimensional data for visualization purposes. This means that this function will only plot data that has been reduced to either 1, 2, or 3 dimensions. This function will also return the percent of variation explained by the number of principal components that have been chosen.
