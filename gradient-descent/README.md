# Gradient Descent and Linear Regression

In this notebook, we'll take a look at gradient descent and discuss how it can be useful for finding the slope and intercept coefficients for lines of best fit.

## Intuition for Gradient Descent

In this section, we will briefly discuss mean-squared error and how it can reflect the goodness of fit of a line with a certain slope and intercept coefficient. There is also a quick overview of grid search, and I show how we can ultimately use a grid of m and b values and their corresponding mean-squared error values to visualize a three-dimensional surface for gradient descent.

## "Walking Down a Mountain"

In this section, I introduce a powerful and intuitive metaphor for gradient descent in which we think of gradient descent similar to walking down a mountain. If we start at the top of a mountain, and we are trying to find the lowest point in the valley below, how should we proceed to find the lowest point most efficiently? This section covers step size, direction of greatest change, and changing step size (changing learning rate).

## Coding Gradient Descent From Scratch

In this section, we'll take a look at what a from-scratch implementation of gradient descent looks like. Given the values for the max number of steps, the data to be used, the learning rate, and the gradient tolerance, this function performs standard gradient descent to move towards the values of m and b that produce the lowest mean-squared error value. Then we will plot how this function moves towards the optimal values of m and b (which are found by a separate function), and assess the efficacy of the algorithm on the particular dataset we are using. 


## Other Types of Gradient Descent

Now that we've seen vanilla gradient descent in the above section, what other types of gradient descent are out there? I introduce stochastic and mini-batch gradient descent, with a from-scratch implementation of stochastic gradient descent to more concretely identify how the algorithm works (mini-batch is not included for the sake of brevity)


## How Does This Implementation Stack Up?

Next, we compare sklearn's implementation of stochastic gradient descent to the above from-scratch implementation of stochastic gradient descent. Is one better at finding the optimal values of m and b? How do the runtimes of the two implementations compare? Is one better memory-usage wise? These are questions that are explored and discussed in this section.


## Using Gradient Descent For Interaction Models

We will then see gradient descent in a new application, gradient descent for interaction models. There is a from-scratch implementation of gradient descent for interaction models in this section, as well as a discussion of the performance of the algorithm on the particular data set used.

## Conclusion

Finally, we discuss how effective overall these gradient descent algorithms were for the particular data set used and where there is room for improvement with these algorithms.
