# Fundamental Theorem of Linear Algebra  \(Ax=b\)
Let \(A\) be an \(m\times n\) matrix,  \(Ax=b\), \(A'y=c\).
Null(A) = \(\mathcal{N}(A)\) is the null space of \(A\), the set of all solutions to \(Ax=0\).
Column space of \(A\) is the set of all linear combinations of the columns of \(A\), denoted as \(\mathcal{R}(A)\).
dim(\(\mathcal{N}(A^T)\)) + dim(\(\mathcal{R}(A)\)) = \(n\) .
dim(\(\mathcal{N}(A)\)) + dim(\(\mathcal{R}(A^T)\)) = \(m\) .
## Existence:
\(b\) is in the column space of \(A\), denoted as \(\mathcal{R}(A)\), which means we can find a combination of the columns of \(A\) that equals \(b\).
## Uniqueness:
!!! info "from \(Ax=0\) to \(Ax=b\)"
    general solution = homogeneous solution + particular solution. 
    If the homogeneous solution is unique, then the general solution is unique.
    If \(Ax=0\) has only the trivial solution, then columns are linearly independent, \(A\) is full column rank.


# Assumptions, consequences(solutions and diagnosis)
- error structure: heteroscedasticity - standard errors are not valid - generalized least squares
- multicollinearity - coefficients not stable - lasso or orthogonalization - variance inflation factor (VIF)

!!! info "Heteroscedasticity"
    Heteroskedasticity affects the efficiency of OLS estimates and the validity of standard errors, but not necessarily their unbiasedness or consistency (unless it's a symptom of a misspecified model that also causes endogeneity)

!!! note "Instrumental Variables"
    to solve endogeneity, that is correlation between independent variables and error term, introduce an instrument variable that is correlated with the independent variable but not with the error term.

## Geometric interpretation
- Orthogonally project \(y\) onto the column space of \(X\) to get \(\hat{y}\), closest point to \(y\) in the column space of \(X\), thus residuals \(y-\hat{y}\) are orthogonal to the column space of \(X\), i.e. \((y-\hat{y})^T X = 0\). The normal equations: 
    \[
    X^T(y-\hat{y}) = 0.
    \]
Thus residual \(e_1 + e_2 + ... + e_n = 0\).
\(y_i = x_i^T\hat\beta + e_i\), explains why the hyperplane goes through \( (\bar x, \bar y) \).

## Estimation
\(\hat\beta = (X^TX)^{-1}X^Ty\), one-dim: \(\hat\beta = \frac{l_{xy}}{l_{xx}}\).
\(E[\hat\beta] = \beta\), \(Var(\hat\beta) = \sigma^2(X^TX)^{-1}\).

### Estimating \(\sigma^2\)
\(\hat\sigma^2 = \frac{1}{n-p-1} \sum_{i=1}^n e_i^2\).

## R-squared
\(R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST} = \frac{\sum_{i=1}^n (\hat y_i - \bar y)^2}{\sum_{i=1}^n (y_i - \bar y)^2}\).
\(R^2 = corr^2(y, \hat y) \)

!!! tip "linear correlation"
    \(r(X, aY+b) = r(X, Y)\) for some constants \(a\) and \(b\).

one-dim: \(R^2 = corr^2(X, y) = \frac{l_{xy}^2}{l_{xx}l_{yy}}\), because there's only one independent variable.

## Logistic regression
- for each sample i, \(y \in \{0, 1\}\), \(p = P(y=1|x)\), \(\frac{p}{1-p} = e^{x^T\beta}\).
- multinomial logistic regression: \(y \in \{0, 1, 2, ..., K\}\),  \(p_k = P(y=k|x)\), \(\frac{p_k}{p_0} = e^{x^T\beta_k}\), \(k=1,2,...,K\).

## Ridge regression
\(\hat\beta_{ridge} = (X^TX + \lambda I)^{-1}X^Ty\).

