Gradient Boosting 是generic idea, sequentially add weak learners.
两个配置: 学习器的类型和损失函数.
Decision rules: $$F_T(x)=\sum_{t=1}^T \alpha_t h_t(x)$$, $h_t$是weak learner, $\alpha_t$是学习器的权重.
有两个权重需要处理: decision rule中不同学习器的权重$\alpha_t$, 和每一轮训练中每个样本的权重$w_i^{t}$.
学习器权重$\alpha_t$，分类正确率越高的学习器权重越大.
训练样本权重$w_i^{t}$，上一轮被分错了它的权重就越大.

比如Adaptive Boosting, 处理{-1, +1}二分类问题, 使用的是exponential loss.
