Intuition: 
data X: (n, p), n samples, p features, 已经中心化了.
Consider a linear aggregate feature for each sample:
$Xv$, where $v\in\mathbb{R}^p$, satisfying $||v||_2=1$ and maximizing the variance of $Xv$.
k表示knowledge, 用(k-#)表示一个前置知识
(+ k-1 方向投影，内积的几何意义 $Xv$)
歪个楼:Geometrically, $Xv$ is the projection of X onto the line spanned by $v$.

(第一特征值)
推导: $Xv: (n, 1)$, 
$var(Xv)$ 是指synthetic feature的方差, $=v'X'Xv$.
Result: 这个v是协方差矩阵X'X的normed特征向量.
(+ k-3 symmetric matrix) n个正交的特征向量，谱分解 $AV = V\Lambda$, 所以 $A = V\Lambda V'$, $V$正交的特征向量矩阵.
(+ k-2 Rayleigh quotient for symmetric matrix) $\max_v \frac{v'X'Xv}{v'v}$
proofhint: $X'X = Q\Lambda Q'$, 令$u = Q'v$ 做替换.
proof2(Lagrange): $L(v) = v'X'Xv - \lambda (v'v - 1)$, $\frac{\partial L}{\partial v} = ?$

Rayleigh quotient的本质其实是正交展开视角，因为对称矩阵的特征向量组成正交基，所以任意向量可以用正交展开来表示, 考虑加权...

(第二主成分)
Lagrange: $L(v) = u'X'Xu - \lambda (u'u - 1) - \mu (u'v - 0)$, $\frac{\partial L}{\partial u} = ?$

...

实践：变化 $Z = XV$, $V = [v_1, v_2, \ldots, v_k]$.

重构误差视角 TBD.



