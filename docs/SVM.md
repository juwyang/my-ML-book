SVM:
2分类问题: 根据线性的hyperplane方程$w^Tx+b=0$来划分数据集.

两个contributions: soft margin, kernel trick.

hyperparameter in soft margin: cost(misclassification penalty) C.

(+ k-1 超平面的法线): p维空间的超平面 $w^Tx+b=0$; $w \in R^p$. 向量$w = (w_1, w_2, \ldots, w_p)$就是超平面的法线, 它与超平面上任意两点的连线垂直. $w(\bm{x_1}-\bm{x_2})=0$

怎么得到超平面和确定哪些是支持向量？
(+ k-2 点到超平面的距离): $\bm{x_0}$是超平面上一点，其他任意一点 $\bm x$, $\bm x-\bm{x_0}$与法向量$\bm w/\|\bm w\|$的内积就是投影距离. 

优化问题:找到最大化间隔的超平面.
$$\max_w \ M \\ 
s.t. \ y_i(w^Tx_i+b)/\|w\| \geq M$$

tour: 如果改成最小二乘损失怎么样，最大化总和垂直距离?

由于现在 $\|w\|$没限定，我们用1/M限制它, 把M消掉, 转化为min问题，应用KKT.

(+ k-3 inequality constraint KKT的形式，最优化条件和约束effectiveness)

(+ k-4 primal and dual)
现在的损失函数有系数和乘子$\alpha_i$ 两个调整: $L(w, b, \alpha_i)$, primal $\min_w L$, dual $\max_\alpha L$.

然后使用二次规划或者Sequential Minimal Optimization算法, 求解出$\alpha_i$, 其中那些$\alpha_i>0$的点就是支持向量, 利用最优化需要满足的导数为0的$\alpha, w$关系可以回代得到$w$, 超平面的方程. 在回代中可以发现只有那些$\alpha_i>0$的点才会影响到$w$, 也就是支持向量.
$w - \sum \alpha_i y_i x_i = 0$

带soft-margin, 目标函数:
$$\min_w \frac{1}{2}\|w\|^2 + C\sum_i \xi_i\\
\text{subject to } y_i(w^Tx_i+b) \geq 1-\xi_i$$
C是一个超参数.


Q: kernel trick是怎么应用到SVM的？
因为在dual form中，$x_i'x_j$作为整体出现, 因此可以将其替换为其他的核函数$K(x_i,x_j)$, 相应地分割超平面是在特征空间$\Phi(\bm x)$中的 $\bm w'\Phi(\bm x)+b=0$. 做预测的时候也只需要
核函数就行.

kernel: 已知$K(x,y)$是一个核函数, 相当于应用了其对应的feature map $\phi(x)$, 使得$K(x,y)=\phi(x)^T\phi(y)$. 也就是说你用一个和$\phi(x)$一样复杂的feature最后得到的结果就是这么多.

和logistic regression相比: 能应对non-linear的情况.




