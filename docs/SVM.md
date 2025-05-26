Binary classification \(y_i \in \{-1, 1\}\).

There's a hyperplane \(f(x)=w^T x + b = 0\) that separates the data points, the distance from any point \(\bm x\) to the hyperplane: the projection to the normal vector \(\bm w\). \(\frac{\bm w'(\bm x - \bm x_0)}{\|\bm w\|}=\frac{f(x)}{\|w\|}\), where \(\bm x_0\) is a point on the hyperplane. 

!!! info "Normal of a hyperplane"
    The normal vector of the hyperplane \(w^T x + b = 0\) is \(\bm w\), because for any two points \(\bm x_1\) and \(\bm x_2\) on the hyperplane, \(\bm w^T (\bm x_1 - \bm x_2) = 0\).

### Problem Formulation
\[ 
    \max M \\ \text{s.t } y_i\frac{f(x_i)}{\|w\|} \ge M, \forall i 
\] 

constrain  \(\|w\| = 1/M \) to rewrite as a minimization problem 

\[
    \min \frac{1}{2}\|w\|^2 \\ \text{s.t } y_i(w^T x_i + b) \ge 1
\].

!!!info "KKT"
    \[ 
        \min f(x) \\ \text{s.t } g(x) \leq 0, h(x) = 0 
    \]
    is equivalent to \(\min f(x) + \lambda h(x) + \mu g(x), \mu \geq 0\). At the boundary \(g(x) = 0\), we use \(\mu > 0\) to prevent \(g(x)\) from going positive.

!!!info "Primal and Dual"
    After introducing Lagrange multipliers, we have the Lagrangian function \(L(w, \lambda, \mu)\) that contains the original parameters \(w\) and the new multipliers \(\lambda, \mu\). The primal optimization order: 
    \(P^* = \min_w \max_{\lambda, \mu \ge 0} L(w, \lambda, \mu)\), while the dual optimization order is the opposite: \(D^* = \max_{\lambda, \mu \ge 0} \min_w L(w, \lambda, \mu)\).

Dual of SVM:
Substitute \(w\) and \(b\) in the Lagrange from \(\nabla_w L = 0, \nabla_b L = 0\). We will get

\[
    \max_\alpha \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j.
\]

### Soft Margin

\[
    \min \frac{1}{2}\|w\|^2 + C\sum_i\xi_i\\
    \text{s.t } y_i(w^T x_i + b) \ge 1-\xi_i, \xi_i \ge 0.
\]

### Kernel
Assume hyperplane in the kernel feature space: \(f(n(x)) = w^T n(x) + b = 0\) could separate the data points, \(n(x) \) is the feature mapping.

replace \(x_i^T x_j\) with a kernel function \(K(x_i, x_j)\).
