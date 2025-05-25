## Metrics
#### Binary classification
- **precision**: one shot hit! \(\frac{TP}{TP+FP}\)
- **recall**: fire eye! \(\frac{TP}{TP+FN}\)
- **AUCROC**: TPR -- FPR, horizontal axis is "I agree anyway". TPR = \(\frac{TP}{TP+FN}\), FPR = \(\frac{FP}{FP+TN}\).

## MSE decomposition
- **MSE**: \(E[(y - \hat y)^2] = E[(\hat y - E[\hat y])^2] + E[(E[\hat y] - y)^2] \)
Variance of estimator + bias^2 of estimator.
Another three parts version assumes $y = f(x) + \epsilon$.
- *Overfitting*: high variance(sensitive to changes), low bias. 
- *Underfitting*: low variance, high bias.

