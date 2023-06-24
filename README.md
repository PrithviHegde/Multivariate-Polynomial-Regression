# Multivariate-Polynomial-Regression
This project consists of 3 Parts, A, B and C.


Part A: Prior and Posterior Distributions:

Given the alpha and beta parameters of a beta distribution s, the prior and posterior of s were plotted, and after a survey to gather additional information about s, 
the beta distribution of s was updated. The likelihood of s also was documented.


Part B: Multivariate Polynomial Regression and Regularization:

A polynomial regression model of degrees 1-9 was created to estimate aquatic toxicity via the LC50 value, based on 2 molecular descriptors, MLOGP and GATS1i. 
The best fit polynomial model was then determined, based on testing accuracy.
L1 and L2 regularization was then implemented, taking the hyperparameter q as 0.5, 1, 2 and 4.
Surface plots of the predicted polynomial were also created, using matPlotLib.


Part C: Visualising Regularization:

Using the dataset of Part B, the contours of the Sum of Squares Error function, subject to |𝑤1|^q + |𝑤1|^q <= eta was plotted, with respect to q = 0.5, 1, 2 and 4.
The points of intersection of the contour and the constraint region, where the global minima occurs, was also documented.
The findings and plots are documented in 1C_Graphs.

