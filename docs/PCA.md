centralized features \(X\), an \(n \times p\) matrix.

A transformation: \(XV = X[v_1, v_2, \ldots, v_k]\).
each component is orthogonal to the others, unit-length, and maximizes variance.

Indeed, \(v_i\) is an eigenvector of the covariance matrix \(\Sigma = X'X\).

### First Component:
max \(Var(Xv)\) subject to \(|v| = 1\).
Solution - 1(Lagrange):
\(\mathcal{L}(v) = v'X'Xv - \lambda(v'v - 1)\), take derivative w.r.t. \(v\): \(\mathcal{L}'(v) = 0\), get \(X'Xv = \lambda v\).

Solution - 2(Rayleigh Quotient):

!!! tip Rayleigh Quotient
    \(\frac{v'X'Xv}{v'v}\) is maximized when \(v\) is the eigenvector of \(X'X\) corresponding to the largest eigenvalue.
\(v = \argmax \frac{v'X'Xv}{v'v}\), subject to \(\|v\| = 1\). Diagonalize \(X'X = P\Lambda P'\), P is orthogonal, \(PP' = I\).

### Second Component:
 \( \max Var(Xu)\) subject to \(|u| = 1\) and \(u'v = 0\).
Lagrange: \(\mathcal{L}(u) = u'X'Xu - \lambda(u'u - 1) - \mu(u'v)\).

### Calculation Procedure:
Center the data, eigenvalue decomposition of \(X'X\), sort eigenvalues in descending order, and select the top \(k\) eigenvectors.