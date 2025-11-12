# ⚔️ Support Vector Machine (SVM)

Support Vector Machines (SVMs) are **margin-based classifiers** that aim to find the **widest possible decision boundary** separating two classes.

---

## 🧭 1. Core Idea: The Maximum-Margin Hyperplane

For a binary classification problem with training data $\{(x_i, y_i)\}$ where $y_i \in \{-1, +1\}$,  
we want to find a separating **hyperplane**:

$$
w^\top x + b = 0
$$

- Points with $w^\top x + b > 0$ are classified as $+1$  
- Points with $w^\top x + b < 0$ are classified as $-1$

Among all possible hyperplanes, SVM chooses the one with the **maximum margin**, i.e., the **maximum distance between the hyperplane and the nearest points from both classes**.

These nearest points are called **support vectors**.

---

## 🧮 2. The Margin Definition

For a given hyperplane $(w, b)$, the **signed distance** from a point $x_i$ to that hyperplane is:

$$
\text{dist}(x_i, w, b) = \frac{w^\top x_i + b}{\|w\|}
$$

### Why this is the distance (geometric view)

A hyperplane can be written as:
$$
H = \{x \mid w^\top x + b = 0\}
$$
where the vector $w$ is **normal (perpendicular)** to the plane.

To measure how far a point $x_i$ is from $H$, we look at the **projection** of the vector from any point on $H$ to $x_i$ onto the direction of $w$.

1. Take any point $x_0$ that lies on the hyperplane ($w^\top x_0 + b = 0$).  
   Then the vector from $x_0$ to $x_i$ is $(x_i - x_0)$.

2. The projection of $(x_i - x_0)$ onto $w$ gives the perpendicular component:
   $$
   \text{proj}_w(x_i - x_0) = \frac{w^\top (x_i - x_0)}{\|w\|^2} w
   $$

3. The **length** of that projection — the perpendicular distance — is:
   $$
   \| \text{proj}_w(x_i - x_0) \| = \frac{|w^\top (x_i - x_0)|}{\|w\|}
   $$

4. Since $w^\top x_0 + b = 0$, we can rewrite the numerator:
   $$
   w^\top (x_i - x_0) = w^\top x_i + b
   $$

Thus:
$$
\text{dist}(x_i, w, b) = \frac{|w^\top x_i + b|}{\|w\|}
$$

The **sign** of $(w^\top x_i + b)$ indicates on which side of the hyperplane $x_i$ lies,  
so the **signed distance** (positive or negative depending on class side) is:
$$
\text{signed dist}(x_i, w, b) = \frac{w^\top x_i + b}{\|w\|}
$$

### Interpretation

- $w$ defines the **orientation** of the plane.  
- $b$ shifts the plane away from the origin.  
- Dividing by $\|w\|$ normalizes the effect of scaling $w$ — without it, changing $w$’s magnitude would artificially stretch or shrink the margin.

Hence, when we maximize the margin, we are effectively maximizing the minimum of $\frac{y_i(w^\top x_i + b)}{\|w\|}$ across all points.


---

## ⚙️ 3. Hard-Margin Optimization Problem

We want all points to be correctly classified **and** maximize the margin.

We can scale $(w, b)$ such that the closest points (support vectors) satisfy:

$$
y_i (w^\top x_i + b) = 1
$$

Then, the margin (distance to the closest point) is:

$$
\text{margin} = \frac{1}{\|w\|}
$$

So maximizing the margin is equivalent to minimizing $\|w\|^2$.

The **optimization problem** becomes:

$$
\min_{w, b} \ \frac{1}{2}\|w\|^2
\quad \text{subject to } y_i(w^\top x_i + b) \ge 1, \ \forall i
$$

---

## 🧩 4. Introducing Slack — Soft Margin SVM

In practice, perfect separation may be impossible.  
We introduce slack variables $\xi_i \ge 0$ to allow some violations:

$$
\min_{w, b, \xi} \ \frac{1}{2}\|w\|^2 + C \sum_i \xi_i
\quad \text{subject to } y_i(w^\top x_i + b) \ge 1 - \xi_i
$$

- $C$ controls the trade-off between a **large margin** and **low classification error**.  
- Larger $C$ → penalize misclassifications more strictly.

---

## 🧮 5. The Lagrangian Formulation (Primal → Dual)

We introduce Lagrange multipliers $\alpha_i \ge 0$ for the constraints:

$$
L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1]
$$

Take derivatives and set to zero (KKT conditions):

$$
\frac{\partial L}{\partial w} = 0 \ \Rightarrow \ w = \sum_i \alpha_i y_i x_i
$$
$$
\frac{\partial L}{\partial b} = 0 \ \Rightarrow \ \sum_i \alpha_i y_i = 0
$$

Substituting these conditions back into the Lagrangian eliminates $w$ and $b$ and gives the **dual optimization problem**:

$$
\max_{\alpha} \ \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^\top x_j)
\quad
\text{subject to } \sum_i \alpha_i y_i = 0,\ \alpha_i \ge 0
$$

---

### ⚙️ How SVM Is Actually Solved

#### 1. No Closed-Form Solution

Unlike linear regression (which has a normal-equation closed form),  
SVM involves **inequality constraints** and **non-differentiable hinge loss**,  
so there is *no analytical solution* for $(w,b)$.  
The optimal solution must satisfy the **Karush–Kuhn–Tucker (KKT)** conditions,  
which define a *quadratic optimization* problem.

---

#### 2. Quadratic Programming (QP) Solvers

The standard (dual) formulation is a **Quadratic Programming (QP)** problem:

$$
\begin{align}
\max_{\alpha}\ & \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^\top x_j)\\
\text{s.t. } & 0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0
\end{align}
$$

This can be solved using numerical QP techniques:

- **Interior-point methods** (general-purpose convex optimizers).  
- **Sequential Minimal Optimization (SMO):**  
  Decomposes the large QP into tiny 2-variable subproblems with closed-form solutions;  
  iteratively optimizes them until convergence.  
  Most practical SVM libraries (e.g. `libsvm`, `scikit-learn`) use SMO variants.

At convergence, we recover:
$$
w = \sum_i \alpha_i y_i x_i,\quad
b = y_k - w^\top x_k\quad(\text{for any support vector }k\text{ with }0 < \alpha_k < C)
$$

---

#### 3. Gradient-Based (Primal) Solvers

For large-scale or streaming data, solving the QP directly is expensive ($O(n^3)$).  
We can instead optimize the **primal** form directly with **Stochastic Gradient Descent (SGD)**:

##### Objective (hinge loss form)
$$
J(w,b) = \frac{1}{2}\|w\|^2 + C\sum_i \max(0,\,1 - y_i(w^\top x_i + b))
$$

##### Subgradient for one sample $(x_i, y_i)$
If $y_i(w^\top x_i + b) \ge 1$:
$$
\nabla_w = w,\quad \nabla_b = 0
$$
Else:
$$
\nabla_w = w - C\, y_i x_i,\quad \nabla_b = -C\, y_i
$$

##### SGD update
Given learning rate $\eta_t$:
$$
w \leftarrow w - \eta_t \nabla_w,\quad b \leftarrow b - \eta_t \nabla_b
$$

This approach underlies algorithms like **Pegasos**, which use simple mini-batch SGD with weight decay to handle the $\frac{1}{2}\|w\|^2$ term.

---

#### 4. Comparison: Dual vs Primal Solvers

| Aspect | Dual (QP / SMO) | Primal (SGD / Pegasos) |
|---------|-----------------|-------------------------|
| Optimization variable | $\alpha_i$ (one per sample) | $w,b$ directly |
| Suitable for | Small–medium datasets | Very large datasets |
| Convergence | Exact optimum | Approximate but scalable |
| Implementation | Requires specialized QP solver | Simple iterative gradient updates |
| Key idea | Optimize margins via support vectors | Minimize hinge loss directly |

---

### 💡 Practical Notes

- In practice, **linear SVMs** for high-dimensional data (like text) almost always use **primal SGD solvers** (e.g., `LinearSVC` or `SGDClassifier`).  
- For **nonlinear SVMs** (with kernels), solvers stay in the **dual** form, since the kernelized features make explicit $w$ infeasible to represent.  
- Both rely on convex optimization — guaranteeing a *global optimum* (no local minima).

---

## ⚙️ 6.1 Solving the Dual Problem — Interior-Point vs SMO

Once the dual form of SVM is derived, we must solve the **Quadratic Programming (QP)** problem:

$$
\begin{align}
\max_{\alpha}\ & W(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^\top x_j)\\
\text{s.t. } & 0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0
\end{align}
$$

This is a **convex optimization** problem — it has a single global maximum.  
Two main approaches exist for solving it efficiently.

---

### 🧩 1. Interior-Point Methods

Interior-point methods are **general-purpose convex optimization algorithms** used for large QP systems.

#### Intuition

They treat the constraints implicitly by adding **barrier terms** that penalize infeasible points.  
Instead of walking along the edges of the feasible region (like simplex),  
they take *smooth steps through the interior* until reaching the optimum.

The algorithm solves a **sequence of unconstrained problems**:
$$
\min_{\alpha}\ \frac{1}{2}\alpha^\top Q\alpha - 1^\top \alpha + \mu \sum_i [-\log(\alpha_i) - \log(C - \alpha_i)]
$$

where $Q_{ij} = y_i y_j (x_i^\top x_j)$ and $\mu$ is a small positive constant that shrinks toward 0 as the solution approaches the feasible boundary.

At each iteration, it uses **Newton’s method** to update $\alpha$:
$$
\alpha \leftarrow \alpha - H^{-1} \nabla f(\alpha)
$$

- $H$ is the Hessian of the current objective (second derivative matrix).
- $\nabla f(\alpha)$ is the gradient of the objective.
- As $\mu \to 0$, the barrier disappears and $\alpha$ approaches the true constrained optimum.

#### Pros & Cons

| | Pros | Cons |
|--|------|------|
| ✅ | Extremely accurate; converges in few steps | ❌ Requires large dense matrix inversions |
| ✅ | Theoretically elegant (used in convex optimization libraries) | ❌ $O(n^3)$ complexity — impractical for large datasets |

Interior-point methods are best for **small-to-medium datasets** where exact precision is needed.

---

### ⚡ 2. Sequential Minimal Optimization (SMO)

SMO was introduced by **John Platt (1998)** to make SVM optimization feasible for large datasets.

#### Key Idea

Instead of solving for all $\alpha_i$ simultaneously,  
SMO optimizes **two Lagrange multipliers at a time**, keeping all others fixed.

Why two?  
Because the constraint $\sum_i \alpha_i y_i = 0$ couples all variables —  
updating one $\alpha_i$ would violate it,  
but updating a *pair* $(\alpha_i, \alpha_j)$ can maintain the constraint.

#### Step-by-Step Outline

1. **Choose a pair of indices** $(i, j)$ to optimize — typically one violating KKT conditions most, and another chosen heuristically.  
2. **Compute the bounds** $L$ and $H$ for $\alpha_j$ to ensure $0 \le \alpha_j \le C$ and $\sum_i \alpha_i y_i = 0$:
   $$
   L = \max(0, \alpha_j - \alpha_i), \quad H = \min(C, C + \alpha_j - \alpha_i)
   $$
   (exact formulas depend on $y_i, y_j$).
3. **Compute $\eta$**, the second derivative along the optimization direction:
   $$
   \eta = 2(x_i^\top x_j) - (x_i^\top x_i) - (x_j^\top x_j)
   $$
   If $\eta \ge 0$, skip this pair (non-improving direction).
4. **Update $\alpha_j$** with a one-dimensional step:
   $$
   \alpha_j' = \alpha_j - \frac{y_j (E_i - E_j)}{\eta}
   $$
   where $E_i = f(x_i) - y_i$ is the prediction error.
5. **Clip $\alpha_j'$** to stay within $[L, H]$.
6. **Update $\alpha_i$** accordingly to maintain the constraint:
   $$
   \alpha_i' = \alpha_i + y_i y_j (\alpha_j - \alpha_j')
   $$
7. **Compute new bias term $b$** using either of the updated points:
   $$
   b_1 = b - E_i - y_i(\alpha_i' - \alpha_i)(x_i^\top x_i) - y_j(\alpha_j' - \alpha_j)(x_i^\top x_j)
   $$
   $$
   b_2 = b - E_j - y_i(\alpha_i' - \alpha_i)(x_i^\top x_j) - y_j(\alpha_j' - \alpha_j)(x_j^\top x_j)
   $$
   Then choose:
   $$
   b = 
   \begin{cases}
   b_1, & 0 < \alpha_i' < C \\
   b_2, & 0 < \alpha_j' < C \\
   \frac{b_1 + b_2}{2}, & \text{otherwise}
   \end{cases}
   $$

8. **Repeat** until all KKT conditions are satisfied (i.e., all $\alpha_i$ are optimal or at boundary).

---

#### Why SMO Is Efficient

- Each step updates only two variables — so no large matrix inversion.
- Each 2-variable subproblem has a **closed-form solution**.
- Requires only kernel evaluations $(x_i^\top x_j)$, which can be cached.
- Complexity is roughly **O(n²)** but scales well in practice due to sparsity and convergence heuristics.

#### Pros & Cons

| | Pros | Cons |
|--|------|------|
| ✅ | Scales to tens/hundreds of thousands of samples | ❌ Sequential; not fully parallelizable |
| ✅ | No external QP solver needed | ❌ Can be slow if many support vectors |
| ✅ | Simple closed-form 2D updates | ❌ Sensitive to choice of heuristics for pair selection |

---

### 💡 Practical Summary

| Solver | Works On | Complexity | Notes |
|---------|-----------|-------------|--------|
| **Interior-Point** | Generic convex QPs | $O(n^3)$ | Exact, but slow; used in theoretical/academic solvers |
| **SMO** | SVM-specific dual form | $O(n^2)$ (often much faster) | Dominates practical implementations (`libsvm`, `scikit-learn`) |
| **SGD / Pegasos** | Primal form | $O(n)$ per epoch | Used for large-scale linear SVMs (e.g., `LinearSVC`, `SGDClassifier`) |

---

**Summary:**
- **Interior-point methods**: accurate but expensive matrix-based optimization.  
- **SMO**: clever decomposition of SVM’s dual problem into many small 2-variable problems with closed-form solutions — fast, stable, and the basis of almost all real-world SVM solvers today.

---

## 🌌 Kernel Support Vector Machine (SVM)

Now that we have the dual form of the linear SVM, recall that the **only place where the data $x_i$ appear** is through **dot products** $(x_i^\top x_j)$.  
This is what allows us to apply the **kernel trick**.

---

### 🧭 1. Revisiting the Dual Objective

The dual problem for SVM is:

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^\top x_j)
\quad
\text{s.t. } 0 \le \alpha_i \le C,\quad \sum_i \alpha_i y_i = 0
$$

All data interactions appear **only through** the inner products $(x_i^\top x_j)$.

So we can replace every occurrence of $(x_i^\top x_j)$ by a **kernel function**:

$$
k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)
$$

where $\phi(\cdot)$ is an implicit (possibly infinite-dimensional) feature mapping.

This transforms the problem into:

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j k(x_i, x_j)
\quad
\text{s.t. } 0 \le \alpha_i \le C,\quad \sum_i \alpha_i y_i = 0
$$

This is the **kernelized SVM**.

---

### 🧩 2. Kernelized Decision Function

After solving for $\alpha_i$, predictions for a new data point $x$ are made using only the kernel:

$$
f(x) = \text{sign}\Big(\sum_i \alpha_i y_i k(x_i, x) + b\Big)
$$

- The model no longer needs to compute $w$ explicitly.
- The computation depends only on **support vectors** ($\alpha_i > 0$).
- The decision boundary can now be **nonlinear in the original input space**.

---

### ⚙️ 3. Geometric Interpretation

In the high-dimensional **feature space** $\mathcal{H}$ defined by $\phi(x)$:

- The SVM is still a *linear* classifier:
  $$
  f(x) = \text{sign}(w^\top \phi(x) + b)
  $$
- But when viewed in the **original input space**, this boundary becomes **nonlinear** — curved, warped, or even disjoint.

So the kernel trick lets SVM **act linearly in $\mathcal{H}$ but nonlinearly in $\mathbb{R}^n$**,  
without ever computing $\phi(x)$ explicitly.

---

### 🔧 4. Common Kernels

| Kernel | Formula | Effect |
|---------|----------|--------|
| **Linear** | $k(x_i, x_j) = x_i^\top x_j$ | Equivalent to ordinary linear SVM |
| **Polynomial** | $k(x_i, x_j) = (x_i^\top x_j + c)^d$ | Captures feature interactions up to degree $d$ |
| **RBF (Gaussian)** | $k(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ | Infinite-dimensional mapping; very flexible |
| **Sigmoid** | $k(x_i, x_j) = \tanh(\beta x_i^\top x_j + \theta)$ | Related to neural activation functions |
| **Custom kernels** | $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ | As long as $K$ is positive semidefinite |

---

### 💡 5. Key Takeaways

- The kernel trick replaces every $(x_i^\top x_j)$ with $k(x_i, x_j)$.
- SVM training and prediction remain mathematically identical — only the notion of “similarity” changes.
- Choosing the right kernel and hyperparameters $(C, \gamma, d, \dots)$ controls the shape and flexibility of the boundary.
- Because kernel matrices are $n \times n$, **kernel SVMs scale poorly** for very large datasets — motivating linear or approximate variants.

---

## ⚙️ Support Vector Regression (SVR)

Support Vector Regression (SVR) applies the same SVM principle — **maximizing the margin** — to regression tasks.  
Instead of finding a separating hyperplane between two classes, we find a **tube (margin band)** around the regression line that fits the data *as flat as possible*.

---

### 🧭 1. Intuition: The ε-Insensitive Tube

The goal of SVR is to find a function $f(x) = w^\top x + b$ that:

- Predicts $y_i$ with at most **ε deviation** (the width of the tube), and  
- Keeps $w$ as small as possible (a flat function = less variance).

We tolerate small errors within ±ε and penalize only points **outside** the tube.

---

### 🧩 2. Primal Formulation

We introduce slack variables $\xi_i, \xi_i^*$ for deviations above and below the ε-tube:

$$
\begin{align}
\min_{w, b, \xi_i, \xi_i^*} \quad &
\frac{1}{2}\|w\|^2 + C \sum_i (\xi_i + \xi_i^*)\\
\text{s.t.} \quad &
\begin{cases}
y_i - (w^\top x_i + b) \le \epsilon + \xi_i \\
(w^\top x_i + b) - y_i \le \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \ge 0
\end{cases}
\end{align}
$$

- The first two inequalities ensure all points lie **within or outside the ε-tube**.
- $\xi_i, \xi_i^*$ measure how far a point is **outside** the tube.
- $C$ controls how much penalty we give to those deviations.

---

### ⚙️ 3. Lagrangian Form (Primal → Dual)

We introduce Lagrange multipliers $\alpha_i, \alpha_i^*$ for the two constraints and $\mu_i, \mu_i^*$ for non-negativity:

$$
\begin{align}
L &= \frac{1}{2}\|w\|^2 + C\sum_i(\xi_i+\xi_i^*) - \sum_i \alpha_i(\epsilon + \xi_i - y_i + w^\top x_i + b) - \sum_i \alpha_i^*(\epsilon + \xi_i^* + y_i - w^\top x_i - b) \\
&\quad - \sum_i (\mu_i \xi_i + \mu_i^* \xi_i^*)
\end{align}
$$

KKT stationarity gives:

$$
\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_i(\alpha_i-\alpha_i^*)x_i
$$
$$
\frac{\partial L}{\partial b}=0 \Rightarrow \sum_i(\alpha_i-\alpha_i^*)=0
$$
$$
\frac{\partial L}{\partial \xi_i}=0 \Rightarrow \alpha_i = C-\mu_i,\quad 0\le\alpha_i\le C
$$
(similarly for $\alpha_i^*$)

---

### 🧮 4. Dual Form

Substitute back to eliminate $w, b, \xi_i, \xi_i^*$:

$$
\max_{\alpha_i, \alpha_i^*} 
\Big[
-\frac{1}{2}\sum_{i,j}(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)(x_i^\top x_j) - \epsilon\sum_i(\alpha_i+\alpha_i^*) + \sum_i y_i(\alpha_i-\alpha_i^*)
\Big]
$$

subject to:
$$
\sum_i(\alpha_i-\alpha_i^*)=0,\quad 0\le\alpha_i,\alpha_i^*\le C
$$

---

### 💻 5. Prediction Function

Once optimized:

$$
f(x) = \sum_i (\alpha_i - \alpha_i^*) (x_i^\top x) + b
$$

Only points **outside the ε-tube** have nonzero $(\alpha_i-\alpha_i^*)$,  
so they act as **support vectors**.

---

### 🌌 6. Kernelized SVR

The dual depends only on $(x_i^\top x_j)$,  
so we can replace them with a kernel $k(x_i, x_j)$:

$$
f(x) = \sum_i (\alpha_i - \alpha_i^*) k(x_i, x) + b
$$

This yields **ε-SVR with kernels**,  
allowing smooth nonlinear regression curves — similar in spirit to kernel SVM.

---

### 🧠 7. Summary

| Concept | Role |
|----------|------|
| $\epsilon$ | Defines “no-penalty” tube width |
| $C$ | Trade-off between flatness and tolerance violations |
| $\xi_i, \xi_i^*$ | Measure how far points exceed the tube |
| $\alpha_i, \alpha_i^*$ | Dual weights for upper/lower constraint |
| Support vectors | Points with errors $>\epsilon$ (nonzero duals) |
| Kernel | Enables nonlinear regression boundaries |

---

### 💡 Practical Notes

- Small $\epsilon$ → narrow tube → more support vectors → higher variance.  
- Large $\epsilon$ → wide tube → fewer support vectors → smoother model.  
- Popular implementations (`scikit-learn`, `libsvm`) use **ε-SVR** with **RBF kernel** by default.  
- Prediction still depends only on kernel evaluations with support vectors.

---

**Intuition Recap:**  
SVM classification finds a **wide margin between classes**,  
while SVR finds a **flat function with an ε-insensitive band** — both balancing model simplicity ($\|w\|^2$) with data fit.
