# 🧮 Principal Component Analysis (PCA)

## 🎯 General Idea

**Principal Component Analysis (PCA)** is a **dimensionality reduction** technique that transforms a dataset into a new coordinate system where the axes (called *principal components*) represent directions of **maximum variance**.

The main goal is to find new features that:
- Capture as much of the original data’s variation as possible.
- Are **uncorrelated** (orthogonal to each other).
- Allow us to **compress** the data while retaining most of its information.

Intuitively, PCA finds the directions (vectors) in which the data varies the most, and projects the data onto these directions.

Mathematically, PCA solves for a projection vector $w$ such that the **variance of the projected data** $Xw$ is maximized:

$$
\max_{w} \ \text{Var}(Xw)
$$

subject to the constraint that $w$ has unit length:

$$
\|w\|^2 = 1
$$

The resulting $w$ is the **first principal component** — the direction of maximal variance.  
Subsequent components are chosen to be orthogonal to the previous ones and capture the next highest remaining variance.

---

## 🧩 Mathematical Derivation — From Variance Maximization to Eigenvector Problem

Let the dataset be represented by a matrix $X \in \mathbb{R}^{m \times n}$,  
where each of the $m$ rows is a sample and each of the $n$ columns is a feature.  
We assume $X$ is **mean-centered**, i.e., each column has zero mean.

---

### 1️⃣ Objective Function

We aim to find a direction $w$ (a unit vector) that maximizes the variance of the data projected onto $w$:

$$
\max_w \ \text{Var}(Xw)
$$

Since the data is mean-centered, the sample covariance matrix is:

$$
S = \frac{1}{m} X^\top X
$$

Thus, the variance of the projection $Xw$ is:

$$
\text{Var}(Xw) = w^\top S w
$$

So our optimization becomes:

$$
\max_w \ w^\top S w \quad \text{subject to} \quad \|w\|^2 = 1
$$

---

### 2️⃣ Lagrangian Formulation

We use a Lagrange multiplier $\lambda$ to enforce the unit norm constraint:

$$
\mathcal{L}(w, \lambda) = w^\top S w - \lambda (w^\top w - 1)
$$

Taking the derivative with respect to $w$ and setting it to zero:

$$
\frac{\partial \mathcal{L}}{\partial w} = 2S w - 2\lambda w = 0
$$

Simplifying:

$$
S w = \lambda w
$$

This gives us a set of **stationary points** — directions $w$ where the gradient is zero.  
However, a zero derivative doesn’t automatically mean a *maximum*. It could also be a *minimum* or a *saddle point*.  
So how do we know that this gives us a **maximum variance direction**?

---

✅ **In summary**
- Setting the derivative to zero finds *stationary points* (possible extrema).  
- Each stationary point corresponds to an eigenvector of $S$.  
- The eigenvector with the **largest eigenvalue** corresponds to the **maximum variance direction** —  
  hence, the **first principal component**.

---

### 3️⃣ Eigenvalue Interpretation

This is the **eigenvalue equation** for the covariance matrix $S$:

$$
S w = \lambda w
$$

- Each **eigenvector** $w_i$ represents a principal direction (or *principal component*).
- Its corresponding **eigenvalue** $\lambda_i$ represents the **variance of the data** when projected onto that direction.

---

#### 🧭 Why the eigenvalue equals the variance

Let’s look at what happens when we project the mean-centered data $X$ onto one eigenvector $w_i$:

$$
z_i = X w_i
$$

The variance of this projected data is:

$$
\text{Var}(z_i) = \frac{1}{m} z_i^\top z_i = \frac{1}{m} w_i^\top X^\top X w_i = w_i^\top S w_i
$$

But since $S w_i = \lambda_i w_i$, we get:

$$
w_i^\top S w_i = w_i^\top (\lambda_i w_i) = \lambda_i (w_i^\top w_i)
$$

Because eigenvectors are normalized to unit length ($w_i^\top w_i = 1$), we have:

$$
\text{Var}(z_i) = \lambda_i
$$

✅ Therefore, the **eigenvalue $\lambda_i$ directly equals the variance** of the data when projected onto its corresponding eigenvector.

---

#### 📈 Intuition

- The **covariance matrix** $S$ measures how features vary together.
- The **eigenvectors** find the directions in which this variation is independent (uncorrelated).
- The **eigenvalues** quantify *how much* variance exists along each of those directions.

Thus, directions with larger $\lambda_i$ capture stronger patterns and wider spread in the data —  
while smaller $\lambda_i$ directions correspond to weaker variations (noise or redundant information).

---

#### 🧮 Ordering by significance

Since $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n$:

- The **first principal component** ($w_1$) captures the **largest variance**.
- The **second** ($w_2$) captures the next largest, orthogonal to $w_1$.
- And so on — each subsequent component explains less variation.

That’s why in PCA, we often keep only the top $k$ components with the largest eigenvalues —  
they retain most of the data’s structure while reducing dimensionality.


---

✅ **Summary**
- PCA starts from a variance maximization problem.
- By introducing the Lagrange multiplier, it becomes an **eigenvalue problem** of the covariance matrix $S$.
- Solving for eigenvectors gives the projection directions, and the eigenvalues quantify their importance.

---

## ⚙️ Incremental Principal Component Analysis (Incremental PCA)

### 🎯 Motivation

Standard PCA requires computing and storing the full covariance matrix:

$$
S = \frac{1}{m} X^\top X
$$

and then performing an eigen-decomposition or Singular Value Decomposition (SVD).  
This becomes infeasible when:

- The dataset is **too large** to fit in memory.
- Data arrives **in batches or streams** (e.g., real-time updates).
- We want to **update** the PCA model without recomputing everything from scratch.

**Incremental PCA (IPCA)** solves this by processing the data in *mini-batches* —  
it updates the principal components **incrementally** as new data comes in.

---

### 🧩 Core Idea

Instead of computing PCA on the full dataset at once,  
IPCA iteratively updates the mean, covariance, and eigenvectors using partial data blocks.

Let the dataset be split into chunks:

$$
X = 
\begin{bmatrix}
X^{(1)} \\
X^{(2)} \\
\vdots \\
X^{(k)}
\end{bmatrix}
$$

At each iteration, IPCA:
1. Centers the incoming batch $ X^{(t)} $ using the *running mean*.
2. Projects it into the current principal subspace.
3. Updates the mean, covariance, and components to reflect the new batch.

---

### 🧮 Algorithm Intuition

Suppose after processing batch $ t-1 $, we have:

- Mean: $ \mu_{t-1} $
- Principal components: $ W_{t-1} $
- Singular values (or eigenvalues): $ \Sigma_{t-1} $

When we receive a new batch $ X^{(t)} $:

1. **Center** it:  
   $ \tilde{X}^{(t)} = X^{(t)} - \mu_{t-1} $

2. **Stack** old and new information:  
   Combine the projections of previous components and the new data:

   $$
   M = 
   \begin{bmatrix}
   \Sigma_{t-1} W_{t-1}^\top \\
   \tilde{X}^{(t)}
   \end{bmatrix}
   $$

3. **Run a small SVD** on $ M $:  
   $$
   M = U \Sigma_t V^\top
   $$
   and take the top $ k $ singular vectors $ V_k 4 as the new principal components $ W_t $.

4. **Update the mean** and continue to the next batch.

This way, each update only requires an SVD on a **small temporary matrix**,  
not the full dataset — drastically reducing memory and computation.

---

### ⚡ Why It Works

Incremental PCA relies on the fact that PCA can be expressed via SVD of the centered data matrix:

$$
X = U \Sigma V^\top
$$

Each batch contributes additional information to the span of $V$ (the principal directions).  
By updating $V$ iteratively through small SVDs, IPCA approximates what would be obtained by running full PCA on all data combined.

---

### 📈 Advantages

- Works with datasets **too large to fit into memory**.
- Efficient for **online learning** or **streaming data**.
- Updates principal components incrementally without reprocessing old data.

---

### ⚠️ Limitations

- It gives an **approximation** of true PCA (slightly less accurate than full SVD).
- Requires careful handling of data centering — mean drift can affect results.
- If data distribution changes drastically, old components may become obsolete.

---

### 🧠 In Practice (scikit-learn)

In scikit-learn, use:

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=k, batch_size=200)
ipca.fit(X)
```

or partial incrementally

```python
for batch in data_stream:
    ipca.partial_fit(batch)
```

---

## 🌐 Kernel PCA — Extending PCA to Nonlinear Spaces

### 🎯 Motivation

Standard PCA finds directions of maximum variance using **linear projections** of the original data.  
However, if the true structure of the data is **nonlinear** (e.g. spirals, concentric circles),  
no linear combination of features can separate or unfold it properly.

**Kernel PCA (KPCA)** solves this by:
- Mapping data into a **higher-dimensional feature space** where linear separation becomes possible.
- Performing PCA **in that new space** — *without ever computing the mapping explicitly.*

---

### 🧠 1️⃣ Implicit Mapping

Let each data point $x_i \in \mathbb{R}^n$ be mapped into a (possibly infinite-dimensional) feature space $\mathcal{F}$ via a nonlinear function $\phi$:

$$
\phi: \mathbb{R}^n \to \mathcal{F}, \quad x_i \mapsto \phi(x_i)
$$

We want to perform PCA on $\phi(X) = [\phi(x_1), \phi(x_2), \dots, \phi(x_m)]$.

The covariance matrix in that space would be:

$$
C = \frac{1}{m} \sum_{i=1}^m \phi(x_i) \phi(x_i)^\top
$$

But computing $\phi(x)$ directly is infeasible when $\mathcal{F}$ is huge or infinite.

---

### 🪄 2️⃣ The Kernel Trick

Instead of working with $\phi(x)$ directly, we use a **kernel function** that computes inner products in $\mathcal{F}$ *without explicitly mapping*:

$$
k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
$$

Common kernels:
- **Polynomial:** $k(x_i, x_j) = (x_i^\top x_j + c)^d$
- **RBF (Gaussian):** $k(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$
- **Sigmoid:** $k(x_i, x_j) = \tanh(\alpha x_i^\top x_j + c)$

This allows us to replace all dot products $\phi(x_i)^\top \phi(x_j)$ with kernel evaluations $k(x_i, x_j)$.

---

### 🔢 3️⃣ Derivation — PCA in Feature Space

We start with the eigenvalue problem in feature space:

$$
C v = \lambda v
$$

Substitute $C = \frac{1}{m} \sum_{i=1}^m \phi(x_i) \phi(x_i)^\top$:

$$
\frac{1}{m} \sum_{i=1}^m \phi(x_i) \langle \phi(x_i), v \rangle = \lambda v
$$

We can express $v$ as a linear combination of all training points in feature space:

$$
v = \sum_{j=1}^m \alpha_j \phi(x_j)
$$

Substitute this into the equation:

$$
\frac{1}{m} \sum_{i=1}^m \phi(x_i) \sum_{j=1}^m \alpha_j \langle \phi(x_i), \phi(x_j) \rangle = \lambda \sum_{j=1}^m \alpha_j \phi(x_j)
$$

Using the kernel function $k(x_i, x_j)$:

$$
\frac{1}{m} \sum_{i=1}^m \phi(x_i) \sum_{j=1}^m \alpha_j k(x_i, x_j) = \lambda \sum_{j=1}^m \alpha_j \phi(x_j)
$$

Now, take the inner product of both sides with $\phi(x_t)$:

$$
\frac{1}{m} \sum_{i=1}^m k(x_t, x_i) \sum_{j=1}^m \alpha_j k(x_i, x_j) = \lambda \sum_{j=1}^m \alpha_j k(x_t, x_j)
$$

This can be written compactly as the **kernel eigenvalue problem**:

$$
K \alpha = m \lambda \alpha
$$

where $K_{ij} = k(x_i, x_j)$ is the **kernel matrix**.

---

### 🧮 4️⃣ Normalization and Centering

Because PCA requires mean-centered data, we must also center $K$ in feature space:

$$
\tilde{K} = K - \mathbf{1}_m K - K \mathbf{1}_m + \mathbf{1}_m K \mathbf{1}_m
$$

where $\mathbf{1}_m$ is an $m \times m$ matrix with all entries $1/m$.

Then solve:

$$
\tilde{K} \alpha = m \lambda \alpha
$$

and normalize $\alpha$ so that the corresponding feature-space eigenvectors have unit length.

---

### 📊 5️⃣ Projecting New Points

To compute the projection of a new data point $x'$ onto the $k$-th principal component,  
we again use only kernels:

$$
y_k(x') = \sum_{i=1}^m \alpha_i^{(k)} \, k(x_i, x')
$$

No explicit mapping $\phi$ is ever computed — the entire algorithm depends only on **pairwise dot products** through the kernel function.

---

### ✅ Summary — Why It Works

| Step | Ordinary PCA | Kernel PCA |
|------|---------------|------------|
| Representation | Raw feature space $x$ | Implicit nonlinear feature map $\phi(x)$ |
| Covariance | $S = \frac{1}{m} X^\top X$ | $C = \frac{1}{m} \Phi^\top \Phi$ |
| Eigenproblem | $S w = \lambda w$ | $K \alpha = m \lambda \alpha$ |
| Core operation | Dot products $x_i^\top x_j$ | Kernel evaluations $k(x_i, x_j)$ |
| Computed quantities | Eigenvectors in $\mathbb{R}^n$ | Coefficients $\alpha$ in $\mathbb{R}^m$ |

By expressing everything in terms of **dot products**, kernel PCA can perform the same variance-maximizing projection  
in a **high-dimensional (even infinite)** space — uncovering nonlinear patterns that ordinary PCA cannot see.

---

## 🧩 Comparison — PCA vs t-SNE vs UMAP

| Aspect | **PCA** | **t-SNE** | **UMAP** |
|:--|:--|:--|:--|
| **Type** | Linear | Nonlinear (probabilistic) | Nonlinear (topological) |
| **Core Idea** | Find orthogonal axes that maximize variance | Match pairwise similarities (Gaussian → t-distribution) | Preserve manifold structure via fuzzy graphs |
| **Mathematical Form** | Eigen-decomposition of covariance matrix | KL-divergence minimization between pairwise probabilities | Cross-entropy minimization between fuzzy graphs |
| **Projection** | $y = XW$ (explicit linear mapping) | Directly optimizes low-dim coordinates | Directly optimizes low-dim coordinates |
| **Interpretability** | High — each axis is a linear combo of original features | None — nonlinear, axes have no clear meaning | None — nonlinear, focuses on topology |
| **Preserves** | Global variance structure | Local neighborhood structure | Local *and* some global manifold geometry |
| **Scalability** | Excellent — $O(nd^2)$ | Poor — $O(n^2)$ | Very good — $O(n \log n)$ (with NN-Descent) |
| **Stochasticity** | Deterministic | Stochastic (different runs vary) | Slightly stochastic (due to random init) |
| **Hyperparameters** | None (just #components) | Perplexity, learning rate | $n\_neighbors$, $min\_dist$, metric |
| **Output Dim** | Any | Usually 2D or 3D | Usually 2D or 3D |
| **Use Case** | Data compression, feature decorrelation | Visualization of clusters / manifolds | Visualization and general nonlinear embedding |
| **Strengths** | Simple, interpretable, fast | Excellent local cluster separation | Fast, preserves both local & global structure |
| **Weaknesses** | Linear only | Slow, poor global preservation | Harder to interpret, hyperparameter sensitive |

---

### 🧭 Intuitive Summary

- **PCA** → best when structure is mostly **linear** and interpretability matters.  
- **t-SNE** → great for visualizing **local clusters**, but loses global relations and scales poorly.  
- **UMAP** → combines both worlds: captures nonlinear structure, scales to large data, and preserves manifold shape better.

---

### 📈 Quick Heuristic for Choosing

| Goal | Recommended Method |
|------|---------------------|
| Feature reduction for regression/classification | **PCA** |
| Exploratory visualization (small dataset) | **t-SNE** |
| Large-scale visualization / manifold learning | **UMAP** |

---

✅ **Summary**

UMAP stands out as the modern default for nonlinear visualization:
- It captures **local** and **global** structures,
- Scales nearly linearly via **NN-Descent**, and
- Produces layouts that are often both interpretable and visually stable.

PCA remains the go-to for **interpretable linear compression**,  
while t-SNE remains useful when extreme **local separation** is desired despite its computational cost.
