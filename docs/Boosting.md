learn sequentially. Judge like Primary Court - Intermediate Court - Supreme Court.

Adaptive Boosting(classifier, \(y_i \in \{-1, 1\}\)):
Two - weighting schemes:
- \(\alpha_i\) for each weak learner. 
- \(w_i\) for each training sample.

!!! warning "Overfitting"

### Gradient Boosting
mimics gradient descent in function space, 
\(F_t(x) = F_{t-1}(x) + \nu h_t(x)\), \(\nu\) is the learning rate.

fit a weak learner to the negative gradient of the pseudo-residuals.
\(-\frac{\partial L(y, F_t(x))}{\partial F_t(x)} \)
