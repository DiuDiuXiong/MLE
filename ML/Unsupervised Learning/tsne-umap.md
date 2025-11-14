# 🌈 Manifold Learning for Visualization: t-SNE & UMAP

High-dimensional data often lies on a **low-dimensional manifold** inside the large feature space. Manifold learning techniques try to **reveal that structure**.

t-SNE and UMAP are two of the most popular **non-linear dimensionality reduction** techniques, especially for **2D/3D data visualization**.

---

## 🎯 Goal

- Preserve **local structure**: points close in high-dim space should stay close
- Allow distortion of **global structure** when necessary (clusters may move apart)
- Reveal **clusters** and **hidden manifolds** when PCA/linear methods fail

---

## 🔍 When to use these methods?

| Method | Best for | What you see in plot |
|--------|----------|--------------------|
| **t-SNE** | Complex clusters, non-linear structure | Very clear separation between groups (even exaggerated) |
| **UMAP** | Preserving both local + some global structure | More continuous manifolds, cluster relationships |

---

## 🧠 Big Picture Concepts

### t-SNE
- Models **pairwise similarity** using probability distributions
- Optimizes the match between high-dim & low-dim similarities
- Very good at **cluster separation**
- Bad at global layout (cluster distances often meaningless)

### UMAP
- Based on **topology & nearest-neighbor graphs**
- Tries to preserve the **shape** and **continuity** of data manifold
- Faster, scalable, meaningful global geometry

---

## ⚠️ Why They Are **Not Interpretable** Like PCA

Both t-SNE and UMAP **directly optimize the embedded coordinates**:

- Let the 2D embedding be $Y = \{y_1, y_2, \dots, y_N\}$
- The algorithms **iteratively move each $y_i$** to minimize a cost function

There is **no linear mapping** like $y = W^\top x$  
→ So, unlike PCA:

- You **cannot explain** the embedding by feature contributions
- You **cannot project new points** without re-running (or approximating) the algorithm
- Axes have **no semantic meaning**

➡️ These are **visualization** tools, **not explainable** feature extractors.

---

## 🚫 Important Cautions

- Both are primarily **visualization-only**
- Axes have **no real meaning**
- Results can change between runs (stochastic algorithms)
- Sensitive to hyperparameters (especially t-SNE: perplexity, LR, iterations)

---

# 🔷 t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE focuses on preserving **local neighborhoods** when mapping high-dimensional data into 2D/3D space for visualization.

It works by matching **pairwise similarity distributions** between:

- High-dimensional points: $X = \{x_1, x_2, ..., x_N\}$
- Low-dimensional embedding: $Y = \{y_1, y_2, ..., y_N\}$

---

## 1️⃣ Similarity Distribution in High-Dim Space

For each point $x_i$, define a **conditional probability** that $x_j$ is a neighbor:

$$
p_{j|i} = 
\frac{
\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)
}{
\sum_{k \neq i}
\exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)
}
$$

Then symmetrize:

$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}
$$

📌 $\sigma_i$ is chosen **per point** by binary search so that **perplexity** matches a target value.

---

### 🔹 Perplexity

A measure controlling the “effective number of neighbors”, we tune this to back tune $\sigma_i$:

$$
\text{Perplexity}(P_i) = 2^{H(P_i)}
$$

Where entropy:

$$
H(P_i) = - \sum_j p_{j|i} \log_2 p_{j|i}
$$

➡️ Larger perplexity = more global structure  
➡️ Small perplexity = tighter neighborhood

---

## 2️⃣ Similarity Distribution in Low-Dim Space

Use a **Student-t distribution** (heavy-tailed) to avoid collapsed clusters:

$$
q_{ij} = 
\frac{
\left(1 + \|y_i - y_j\|^2\right)^{-1}
}{
\sum_{k \neq l}
\left(1 + \|y_k - y_l\|^2\right)^{-1}
}
$$

📌 Heavy tails prevent the **crowding problem**  
(points too close together in low-dimensional space)

---

## 3️⃣ Objective Function — KL Divergence

t-SNE minimizes the mismatch between the distributions:

$$
\mathcal{L} = KL(P \parallel Q)
= \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

---

## 4️⃣ Optimization (Gradient Descent)

Coordinates $y_i$ are updated iteratively:

$$
\frac{\partial \mathcal{L}}{\partial y_i} =
4 \sum_j (p_{ij} - q_{ij})
\frac{(y_i - y_j)}{1 + \|y_i - y_j\|^2}
$$

🚀 Special tricks t-SNE uses

- **Early exaggeration**: temporarily increase $p_{ij}$ → punch clusters apart
- **Momentum** to stabilize updates
- **Barnes–Hut / FFT approximations** to scale beyond $N^2$

---

## 🧩 Strengths & Weaknesses

| Strengths | Weaknesses |
|----------|------------|
| Excellent cluster separation | Global structure often meaningless |
| Captures non-linear manifolds | Slow for large datasets |
| Great for exploratory analysis | Hard to tune (perplexity sensitive) |
| Widely used in ML & biology | Not deterministic → different runs differ |

---

## ✔️ Best Use Cases

- Visualizing embeddings: word vectors, image features, neural activations
- High-dim datasets with **well-formed clusters**
- Data exploration to discover structure

➡️ **Not** suitable for feature engineering or downstream modeling.

---

# 🌐 Uniform Manifold Approximation and Projection (UMAP)

## 🎯 General Idea

**UMAP (Uniform Manifold Approximation and Projection)** is a nonlinear dimensionality reduction and visualization technique that preserves both **local structure** (like neighbors in t-SNE) and **global geometry** (like manifold shape).  

It is grounded in two theoretical ideas:
1. **Manifold assumption:** The data lies on a low-dimensional manifold embedded in a high-dimensional space.
2. **Fuzzy topology:** Both the high-dimensional manifold and its low-dimensional embedding can be represented as weighted graphs, which we make as *similar as possible*.

UMAP, like t-SNE, directly optimizes low-dimensional coordinates $(y_1, y_2, \dots, y_n)$ —  
there is **no projection matrix $W$**, meaning UMAP is *not directly interpretable or linear*.

---

## 🧩 1️⃣ Constructing the High-Dimensional Graph

UMAP first builds a **weighted graph** that approximates the data’s manifold structure.

For each data point $x_i$:
- Compute the distances to its $k$ nearest neighbors.
- Define a **local radius** $\rho_i$ (the distance to its nearest neighbor).
- Define a **smooth local scale parameter** $\sigma_i$ so that the probability of neighbor membership satisfies:

  $$
  \sum_{j} \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right) \approx \log_2(k)
  $$

Then, the **fuzzy membership** (connection strength) of $x_j$ in the neighborhood of $x_i$ is:

$$
\mu_{i|j} = \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right)
$$

This represents how strongly $x_i$ and $x_j$ are connected in the local manifold structure.

Finally, UMAP **symmetrizes** these local connections:

$$
\mu_{ij} = \mu_{i|j} + \mu_{j|i} - \mu_{i|j} \mu_{j|i}
$$

to get a global **weighted adjacency matrix** representing the high-dimensional manifold.

---

## ⚙️ 2️⃣ The Normalized Local Radius

The **normalized local radius** is a scaling step that adjusts each neighborhood’s distances so that local density differences do not dominate.  

- The raw Euclidean distances $d(x_i, x_j)$ can vary a lot across regions of different density.
- UMAP rescales each distance relative to the *local radius* $\rho_i$ and the *local scale* $\sigma_i$, so that all neighborhoods contribute equally to the manifold structure.

Formally, each distance is normalized as:

$$
d'_{ij} = \frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}
$$

This normalization ensures that:
- Densely packed areas don’t overwhelm sparse regions.
- The graph’s connectivity reflects the *relative* local structure, not the absolute distances.

Intuitively:
> UMAP adjusts every point’s neighborhood scale so that "one step" in dense regions is comparable to "one step" in sparse ones.

---

## 🧭 3️⃣ Constructing the Low-Dimensional Graph

Next, UMAP builds a similar fuzzy graph in the low-dimensional embedding space (e.g., 2D or 3D).

The **similarity** between embedded points $y_i$ and $y_j$ is modeled as:

$$
\nu_{ij} = \frac{1}{1 + a \|y_i - y_j\|^{2b}}
$$

where $(a, b)$ are hyperparameters chosen so that this curve approximates the high-dimensional fuzzy membership’s decay.

---

## 🔥 4️⃣ Optimization Objective

UMAP minimizes the **cross-entropy** between the two fuzzy sets — one in high-dimensional space ($\mu_{ij}$), one in low-dimensional space ($\nu_{ij}$):

$$
C = \sum_{i \ne j} \Big[ \mu_{ij} \log\frac{\mu_{ij}}{\nu_{ij}} + (1 - \mu_{ij}) \log\frac{1 - \mu_{ij}}{1 - \nu_{ij}} \Big]
$$

The optimization is done via **stochastic gradient descent (SGD)** directly on the coordinates $\{y_i\}$.

- Attractive forces pull connected points together.
- Repulsive forces push unconnected points apart.

---

## ⚡ 5️⃣ Computational Efficiency — Why UMAP Can Reach $O(n \log n)$

Although UMAP involves pairwise distances, it avoids the full $O(n^2)$ computation by combining two key ideas:

1. **Approximate Nearest Neighbor Search (NN-Descent)**  
   - Instead of comparing all pairs, UMAP uses an approximate neighbor algorithm (like NN-Descent) to find the $k$ nearest neighbors for each point.  
   - NN-Descent has an average complexity of **$O(n \log n)$** and is highly parallelizable.

2. **Sparse Graph Representation**  
   - Each point connects only to its $k$ nearest neighbors (sparse adjacency).  
   - Subsequent optimization operates only on these edges — about $O(kn)$ total, where $k$ is typically 10–50.

Together, these reduce both **graph construction** and **optimization** to approximately $O(n \log n)$ time and $O(kn)$ memory, making UMAP far faster than t-SNE ($O(n^2)$).

---

## ✅ Summary

| Concept | Meaning | Effect |
|----------|----------|--------|
| **Manifold assumption** | Data lies on a low-dimensional manifold | Enables nonlinear structure recovery |
| **Fuzzy topological graph** | Weighted edges represent neighborhood membership | Connects local density to global shape |
| **Normalized local radius** | Rescales each neighborhood by $\rho_i$, $\sigma_i$ | Balances dense and sparse regions |
| **Kernel-like optimization** | Minimizes cross-entropy between graphs | Aligns manifold geometry in low-dim space |
| **$O(n \log n)$ complexity** | Achieved via NN-Descent and sparsity | Scalable to millions of samples |

UMAP therefore provides a balance between **local structure preservation** (like t-SNE)  
and **computational scalability** and **global structure retention** —  
making it one of the fastest and most versatile nonlinear embedding methods used today.

---

## ⚙️ Nearest Neighbor Descent (NN-Descent)

### 🎯 General Idea

**NN-Descent** is a fast algorithm for building the **$k$-nearest neighbor graph** that underlies UMAP (and many other manifold methods).  

A naïve way to find each point’s $k$ nearest neighbors would require computing *all* pairwise distances —  
an **$O(n^2)$** operation.  
NN-Descent avoids that by using **an iterative approximation strategy** that converges quickly to the true neighborhood relationships.

In practice, it achieves **near-$O(n \log n)$** time complexity and **$O(kn)$** memory,  
while producing neighbor graphs that are 90–99% accurate compared to the exact one.

---

### 🧠 1️⃣ Core Intuition

The key assumption is simple:

> *“A neighbor of my neighbor is likely to also be my neighbor.”*

That means we can discover new neighbors for a point without checking all data points —  
just by exploring the neighbors of the neighbors it already has.

NN-Descent leverages this by maintaining and refining an approximate neighbor list for every point.

---

### 🔄 2️⃣ Algorithm Steps

1. **Initialization**  
   - Start with a random neighbor list for each point (a few random candidates).  
   - Store both the neighbor IDs and their distances.

2. **Iterative Neighbor Propagation**  
   - For each point $x_i$, look at its current neighbor set $N(x_i)$.
   - For each neighbor $x_j \in N(x_i)$, examine that neighbor’s own neighbors $N(x_j)$.
   - Compute distances between $x_i$ and the new candidates from $N(x_j)$.
   - If any candidate is closer than the current farthest neighbor, update $N(x_i)$.

   This step propagates good neighbor relationships through the graph, discovering better connections quickly.

3. **Pruning and Convergence**  
   - After several rounds of updates, most neighbor lists stop changing — meaning the graph has converged.
   - Typically only 3–5 passes are needed for convergence, regardless of dataset size.

---

### 🧩 3️⃣ Why It’s So Fast

- **Local exploration instead of global comparison**  
  Instead of comparing every pair of points ($O(n^2)$), NN-Descent compares only neighbors-of-neighbors.  
  Each iteration explores a small subset of potential pairs → drastically fewer distance computations.

- **Sparsity**  
  Each point stores only $k$ neighbors → total comparisons scale as $O(kn)$, not $O(n^2)$.

- **Randomization and heuristics**  
  Random restarts and sampling keep the search from getting stuck in local optima.

- **Parallelization**  
  Each point’s neighbor list update is independent, so the algorithm parallelizes easily across CPU cores.

Empirically, NN-Descent builds neighbor graphs roughly **100× faster** than brute-force methods  
while maintaining over 95% recall accuracy.

---

### 🧮 4️⃣ Complexity Summary

| Step | Naïve Search | NN-Descent |
|------|---------------|------------|
| Pairwise distance computation | $O(n^2)$ | $O(n \log n)$ (empirical) |
| Memory usage | $O(n^2)$ | $O(kn)$ |
| Accuracy | Exact | Approx. 95–99% |
| Scalability | Poor | Excellent (millions of points) |

---

### ✅ Why It Matters for UMAP

UMAP relies on a $k$-nearest neighbor graph to model the high-dimensional manifold.  
Without NN-Descent, building that graph would dominate runtime.

By using NN-Descent:
- Graph construction becomes almost linear in $n$.
- Memory usage stays manageable (since only $k$ edges per node are stored).
- The downstream optimization (embedding) starts with an already sparse, high-quality neighborhood graph.

Together, these make UMAP’s total complexity roughly **$O(n \log n)$**,  
allowing it to scale to millions of data points — a major advantage over t-SNE.
