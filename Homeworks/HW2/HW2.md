# Linear Regression Homework

## Problem 1: Linear Regression

Convert to Linear System of Equations for solution



## Problem 2: Evidence

### A. Marginal Likeilihood

<u>Method 1</u>

Completing squares for posterior and then use normalization constants

See how the completing squares is done. 

The complete steps for completing square method is at PRML pg 167



## Gaussian with unknown variance

Inverse Gamma Distribution

PDF: $\text{IG}(x|\alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} (\frac{1}{x})^{\alpha+1}\exp(-\frac{\beta}{x})$

CDF: $Q(\alpha, \frac{\beta}{x}) = \frac{\Gamma(\alpha,\beta/x)}{\Gamma(\alpha)}$ Also called the *Regularized gamma function*

Mean: $\frac{\beta}{\alpha-1}$ for $\alpha > 1$

Mode: $\frac{\beta}{\alpha-1}$

Variance: $\frac{\beta^2}{(\alpha-1)^2(\alpha-2)}$ for $\alpha>2$



Gamma Distribution

PDF: $\text{Gam}(x|\alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}\exp(-\beta x)$



## Problem 3: Mixture of Conjugate priors

Prove a mixture of conjugate prior is indead a conjugate prior.

This means that to show the posterior possess the same form as the prior. 

Beta Distribution

$\text{Beta}(\alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\text{B}(\alpha, \beta)}$, where $\text{B}(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma{\beta}}{\Gamma(\alpha+\beta)}​$



## Problem 4: Optimal threshold on classification probability

After calculating the posterior expected loss, start weighting different outcomes against each other. 



## Problem 5: Bayes Factor

A useful identity: $\text{B}(x,y) = \frac{(x-1)!(y-1)!}{(x+y-1)!}$

The Bayes Factor hypothesis testing uses the marginal likelihood calculated from A, as it is a constant instead of a function. 



## Problem 6: Bahaviour of training set error with increasing sample size, multi-output regression and ridge regression. 

Working with numpy matrix

```python
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
```

There is a serious problem due to the high-condition number with the input matrix. I feel like it is probably easier to solve without inverse. 