# 线性代数基本定理 (Fundamental Theorem of Linear Algebra)

## 关于 $Ax=b$ 的解、零空间 (Null Space) 与列空间 (Column Space)

### 理解 $Ax=b$ 是否有解
方程 $Ax=b$ 是否有解，取决于向量 $b$ 是否在矩阵 $A$ 的列空间 (column space) 里。矩阵 $A$ 的列空间是一个经过原点的子空间（例如，在三维空间中可能是一个平面或一条直线）。
$A$ 的秩 (Rank)，记为 $R(A)$ 或 $\text{rank}(A)$，指的是 $A$ 的列空间的维度，也就是 $A$ 的列空间中线性无关 (linearly independent) 的列向量的最大数目。如果有 $r$ 个线性无关的列，那么剩下的 $n-r$ 个列（假设 $A$ 是 $m \times n$ 矩阵）对于张成列空间而言没有提供新的维度，它们可以由这 $r$ 个线性无关的列线性表示。

-   **列空间 (Column Space):** $\mathcal{R}(A)$ (也常用 $C(A)$)
-   **零空间 (Null Space):** $\mathcal{N}(A)$

列空间 $\mathcal{R}(A)$ 决定了方程 $Ax=b$ 是否有解（即 $b \in \mathcal{R}(A)$ 时有解）。
零空间 $\mathcal{N}(A)$ 决定了解的唯一性 (uniqueness)。如果解存在，其唯一性仅取决于矩阵 $A$ 的性质（具体来说是 $\mathcal{N}(A)$ 是否只包含零向量），与向量 $b$ 无关。

### 零空间 (Null Space)
零空间 $\mathcal{N}(A)$ 是所有使得 $Ax=0$ 的向量 $x$ 的集合。
零空间的维度是 $n-r$，其中 $n$ 是矩阵 $A$ 的列数（未知数的数量），$r$ 是矩阵 $A$ 的秩。
为什么是 $n-r$？因为 $Ax=0$ 代表一个齐次线性方程组。如果有 $n$ 个未知数，而矩阵 $A$ 的秩为 $r$，意味着有 $r$ 个有效（不冗余）的限制条件。因此，剩下 $n-r$ 个自由变量，这些自由变量的个数就是零空间的维度。
对于 $Ax=0$，$x$ 向量与 $A$ 的所有行向量都正交 (orthogonal)。这是因为 $A$ 的每一行与 $x$ 的点积都为0。
例如：
$\begin{bmatrix}1 & -1 & 0 \\0 & 1 & -1 \\1 & 0 & -1 \\\end{bmatrix}\begin{bmatrix}1 \\ 1\\ 1\end{bmatrix}=\begin{bmatrix}0 \\ 0\\ 0\end{bmatrix}$。
这里，$x = \begin{bmatrix}1 \\ 1\\ 1\end{bmatrix}$ 在矩阵 $A = \begin{bmatrix}1 & -1 & 0 \\0 & 1 & -1 \\1 & 0 & -1 \\\end{bmatrix}$ 的零空间中。

### 行空间 (Row Space)
行空间 $\mathcal{R}(A^T)$ 是由 $A$ 的行向量张成的子空间。行空间的维度等于列空间的维度，即等于矩阵的秩 $r$。
对于矩阵 $A_{m \times n}$ (即 $m$ 行 $n$ 列)：
-   零空间 $\mathcal{N}(A) \subset \mathbb{R}^n$ 中的向量与 $A$ 的行向量（即 $A^T$ 的列向量）正交。
-   维度基本关系（秩-零度定理的一部分）：$\text{dim}(\mathcal{N}(A)) + \text{dim}(\mathcal{R}(A^T)) = n$ (零空间的维度 + 行空间的维度 = 矩阵的列数)。

### 行的几何图像
从行的角度看，$Ax=b$ 的每一个方程 $A_{i \cdot} x = b_i$ (其中 $A_{i \cdot}$ 是 $A$ 的第 $i$ 行) 在 $\mathbb{R}^n$ 中定义了一个超平面 (hyperplane)（除非该行为零）。$Ax=b$ 的解（如果存在）就是这 $m$ 个超平面的交点。

## 总结 $A_{m \times n}x=b$ 的四个基本子空间关系
这四个基本子空间是：列空间 $\mathcal{R}(A)$，零空间 $\mathcal{N}(A)$，行空间 $\mathcal{R}(A^T)$ (即 $A^T$的列空间)，以及左零空间 $\mathcal{N}(A^T)$ (即 $A^T$的零空间)。
它们之间的维度关系如下：
1.  $\text{dim}(\mathcal{R}(A^T)) + \text{dim}(\mathcal{N}(A)) = n$
    (行空间的维度 + 零空间的维度 = 列数 $n$)
2.  $\text{dim}(\mathcal{R}(A)) + \text{dim}(\mathcal{N}(A^T)) = m$
    (列空间的维度 + 左零空间的维度 = 行数 $m$)

并且，行空间与零空间是 $\mathbb{R}^n$ 内的正交补子空间，列空间与左零空间是 $\mathbb{R}^m$ 内的正交补子空间。

## $x$ 的存在性 (Existence of solution for $Ax=b$)
解存在的条件：
1.  向量 $b$ 必须在矩阵 $A$ 的列空间中 ($b \in \mathcal{R}(A)$)。
2.  等价地，向量 $b$ 必须与 $A^T$ 的零空间（即左零空间 $\mathcal{N}(A^T)$）中的所有向量正交 ($b \perp \mathcal{N}(A^T)$)。
    理由：因为 $\mathcal{R}(A)$ 和 $\mathcal{N}(A^T)$ 是 $\mathbb{R}^m$ 中的正交补。如果 $b \in \mathcal{R}(A)$，那么它必须与 $\mathcal{N}(A^T)$ 中的所有向量正交。具体来说，如果 $y \in \mathcal{N}(A^T)$，则 $A^T y = 0$。若 $b \in \mathcal{R}(A)$，则存在 $x$ 使得 $Ax=b$。那么 $y^T b = y^T (Ax) = (y^T A) x = (A^T y)^T x = 0^T x = 0$。

## $x$ 的唯一性 (Uniqueness of solution for $Ax=b$)
如果 $Ax=b$ 有解，解唯一的条件：
1.  矩阵 $A$ 的零空间 $\mathcal{N}(A)$ 只包含零向量，即 $\mathcal{N}(A) = \{0\}$。
    因为如果 $x_p$ 是 $Ax=b$ 的一个特解 (particular solution)，则通解 (general solution) 是 $x_p + x_h$，其中 $x_h$ 是 $Ax=0$ 的任意解 (即 $x_h \in \mathcal{N}(A)$)。如果 $\mathcal{N}(A)=\{0\}$，则 $x_h$ 只能是零向量，所以解 $x_p$ 是唯一的。
2.  等价地，矩阵 $A$ 的列向量线性无关。这意味着秩 $r=n$ (列满秩)。
3.  等价地，矩阵 $A$ 的行空间 $\mathcal{R}(A^T)$ 必须是整个 $\mathbb{R}^n$ 空间 (因为 $\text{dim}(\mathcal{R}(A^T)) = r = n$)。这意味着有 $n$ 个有效的、线性无关的方程对应 $n$ 个未知数，没有自由变量。

## 特殊情况分析

-   **当对任何 $b \in \mathbb{R}^m$，$Ax=b$ 都有解时：**
    这说明 $A$ 的列空间是整个 $\mathbb{R}^m$，即 $\mathcal{R}(A) = \mathbb{R}^m$。因此，列空间的维度（即秩 $r$）是 $m$。所以 $\text{rank}(A)=m$ (行满秩)。这也意味着 $A$ 有 $m$ 个线性无关的行向量。
-   **当 $Ax=b$ 的解是唯一时 (如果存在)：**
    这说明 $Ax=0$ 只有零解，即 $\mathcal{N}(A) = \{0\}$。因此 $\text{dim}(\mathcal{N}(A)) = 0$。根据秩-零度定理，$\text{rank}(A) = n - \text{dim}(\mathcal{N}(A)) = n$。这意味着 $A$ 的 $n$ 个列向量线性无关 (列满秩)。
    由于行秩等于列秩，所以 $\text{rank}(A^T) = \text{rank}(A) = n$。

## 矩阵分解 (Factorization)

### 基于消元的分解 (Elimination) - LU 分解
$A = LU$
-   $L$ 是一个下三角矩阵 (Lower triangular)，其对角线元素通常为1。它记录了高斯消去过程中用于消去主元下方元素的乘数。
-   $U$ 是一个上三角矩阵 (Upper triangular)，是 $A$ 经过高斯消元后得到的行阶梯形式矩阵。
-   这种分解主要用于求解线性方程组 $Ax=b$，可以通过 $L(Ux)=b$ 分两步求解：先解 $Ly=b$ 得到 $y$，再解 $Ux=y$ 得到 $x$。
-   对于一个 $n \times n$ 矩阵，LU 分解的计算复杂度通常为 $O(n^3)$。

### 基于正交化的分解 (Orthogonalization) - QR 分解
$A_{m \times n} = Q_{m \times n}R_{n \times n}$ (假设 $m \ge n$ 且 $A$ 列满秩以便简化讨论 $R$ 的形式)
-   $Q$ 是一个 $m \times n$ 矩阵，其列向量 $q_1, q_2, \dots, q_n$ 构成一组标准正交基 (orthonormal basis) for $\mathcal{R}(A)$。即 $Q^T Q = I_n$ (单位矩阵)。
-   $R$ 是一个 $n \times n$ 的上三角矩阵，其对角线元素通常为正。
-   $Q$ 的列向量可以通过对 $A$ 的列向量 $a_1, a_2, \dots, a_n$ 进行 Gram-Schmidt 正交化方法得到。
    过程如下：
    1.  $u_1 = a_1$, $q_1 = u_1 / \|u_1\|$
    2.  $u_2 = a_2 - (a_2^T q_1)q_1$, $q_2 = u_2 / \|u_2\|$
    3.  $u_k = a_k - \sum_{i=1}^{k-1} (a_k^T q_i)q_i$, $q_k = u_k / \|u_k\|$
-   这个过程表明 $A$ 的每个列向量 $a_k$ 可以表示为 $Q$ 中前 $k$ 个列向量 $q_1, \dots, q_k$ 的线性组合 ($a_k = \sum_{i=1}^k r_{ik} q_i$)，这使得 $R$ 呈上三角形式 ($R = Q^T A$)。因为 $q_k$ (或 $u_k$) 的构造只取决于 $a_k$ 和它之前的标准正交向量 $q_1, \dots, q_{k-1}$。

## 最小二乘法 (Least Squares) 的几何理解

当 $Ax=b$ 无解时 (即 $b \notin \mathcal{R}(A)$)，我们转而寻找一个最优近似解 $\hat{x}$，使得 $A\hat{x}$ 尽可能接近 $b$。这等价于最小化误差向量 $e = b - A\hat{x}$ 的长度 (范数)。

### 将向量 $b$ 投影到由单向量 $a$ 张成的子空间上
目标：$\min_{\bar{x}} \| b - \bar{x}a \|$。我们希望找到 $a$ 的一个倍数 $p = \bar{x}a$ (即 $b$ 在 $a$ 上的投影)，使得 $p$ 与 $b$ 的距离最小。
几何上，当误差向量 $b-p$ 与向量 $a$ 正交时，距离最小。
$(b - \bar{x}a)^T a = 0 \implies b^T a - \bar{x} a^T a = 0 \implies \bar{x} = \frac{a^T b}{a^T a}$
所以，投影向量 $p = a \frac{a^T b}{a^T a} = \frac{aa^T}{a^T a} b$。
投影到向量 $a$ 张成的直线上的投影矩阵是 $P = \frac{aa^T}{a^T a}$。

### 将向量 $b$ 投影到 $A$ 的列空间 $\mathcal{R}(A)$ 上
目标：$\min_x \|Ax - b\|$
我们寻找 $A$ 的列空间 $\mathcal{R}(A)$ 中的一个向量 $p = A\hat{x}$，使得该向量与 $b$ 的距离最近。
这意味着误差向量 $e = b - A\hat{x}$ 必须垂直于 $A$ 的整个列空间 $\mathcal{R}(A)$。换句话说，误差向量 $e$ 与 $A$ 的任何列向量都正交。
这可以表示为：$A^T (b - A\hat{x}) = 0$。
这导出了著名的**正规方程 (Normal Equations)**:
$A^T A \hat{x} = A^T b$
如果 $A$ 的列线性无关（即 $A^T A$ 可逆），则最小二乘解为 $\hat{x} = (A^T A)^{-1} A^T b$。
$b$ 在 $A$ 的列空间上的投影是 $p = A\hat{x} = A(A^T A)^{-1} A^T b$。
因此，投影到 $A$ 的列空间的投影矩阵是 $P = A(A^T A)^{-1} A^T$。