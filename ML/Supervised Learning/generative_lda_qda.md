# 📘 Linear Discriminant Analysis (LDA) & Quadratic Discriminant Analysis (QDA)

## 🧩 Before We Begin: What Is Covariance?

Covariance measures **how two features vary together**.

For two features $x_j$ and $x_k$:

- If they **increase and decrease together**, covariance is **positive**  
- If one increases when the other decreases, covariance is **negative**  
- If they are unrelated, covariance is close to **zero**

Mathematically, the covariance between feature $j$ and feature $k$ is:

$$
\text{Cov}(x_j, x_k)
= \mathbb{E}\big[(x_j - \mu_j)(x_k - \mu_k)\big].
$$

In practice (sample estimate):

$$
\hat{\text{Cov}}(x_j, x_k)
= \frac{1}{N}
\sum_{i=1}^{N}
(x_j^{(i)} - \mu_j)(x_k^{(i)} - \mu_k).
$$

---

## ✅ Covariance Matrix

For a feature vector $x \in \mathbb{R}^d$, the **covariance matrix** $\Sigma \in \mathbb{R}^{d \times d}$ collects all pairwise covariances:

$$
\Sigma_{jk}
= \text{Cov}(x_j, x_k).
$$

Properties:

- Diagonal elements: **variances**, e.g., $\Sigma_{jj} = \text{Var}(x_j)$
- Off-diagonal: **covariances**
- Symmetric: $\Sigma_{jk} = \Sigma_{kj}$
- Positive semi-definite

Interpretation:

- $\Sigma$ describes the **shape, orientation, and spread** of a Gaussian distribution  
- Determines whether LDA or QDA yields **linear** or **quadratic** boundaries  
  (LDA uses shared covariance, QDA uses class-specific covariance)

---

## 🟦 Linear Discriminant Analysis (LDA)

LDA is both a **classifier** and a **dimensionality reduction technique**.  
It is built on a **Gaussian generative model** with one key assumption:

> Each class has its own mean vector, but **all classes share the same covariance matrix**  
> $$x \mid y = k \sim \mathcal{N}(\mu_k,\ \Sigma).$$

This shared-covariance assumption makes the resulting decision boundaries **linear**, hence the name *Linear* Discriminant Analysis.

---

### ✅ 1. Training LDA: Estimate Means, Covariance, Priors

#### **1) Estimate class priors**
$$
p(y=k) = \frac{N_k}{N}.
$$

#### **2) Estimate class means**
For class \(k\):
$$
\mu_k = \frac{1}{N_k} \sum_{i: y^{(i)} = k} x^{(i)}.
$$

This is just the **sample average** of all points belonging to each class.

#### **3) Estimate the *shared* covariance**

LDA assumes all classes share one covariance matrix \(\Sigma\).  
The estimator is the **pooled covariance**:

$$
\Sigma = \frac{1}{N} 
\sum_{k=1}^{K}
\sum_{i: y^{(i)} = k}
(x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^\top.
$$

Important:
- $\Sigma$ is **not class-specific**
- This is what makes the final boundaries linear

Once these parameters are learned, there is **no iterative optimization** — everything is closed-form.

---

## ✅ 2. Using LDA for Classification

Starting from the Gaussian generative model:

$$
x \mid y=k \sim \mathcal{N}(\mu_k, \Sigma),
$$

the class-conditional log likelihood is:

$$
\log p(x \mid y=k)
= -\frac{1}{2}(x-\mu_k)^\top \Sigma^{-1}(x-\mu_k) - \frac{1}{2} \log|\Sigma|.
$$

Plug into Bayes rule:

$$
\log p(y=k \mid x) \propto
x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log p(y=k).
$$

This expression is **linear in $x$**.  
Thus the decision boundary between any two classes \(k, m\) is:

$$
w^\top x + b = 0.
$$

This confirms LDA → **linear classifier**.

---

### ✅ 3. LDA for Dimensionality Reduction

LDA can also find a low-dimensional projection that **maximizes class separability**.

#### ✅ Goal
Find a projection matrix $W$ (size $d \times r$) that maximizes:

$$
\frac{|W^\top S_b W|}{|W^\top S_w W|}
$$

where:
- $S_b$ = between-class scatter  
  $$
  S_b = \sum_{k=1}^K N_k (\mu_k - \mu)(\mu_k - \mu)^\top
  $$
- $S_w$ = within-class scatter  
  $$
  S_w = \sum_{k=1}^K 
  \sum_{i: y^{(i)}=k}
  (x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^\top
  $$

---

### ✅ From the Optimization Objective to the Eigenvalue Problem

LDA maximizes the ratio:

$$
J(w) = \frac{w^\top S_b w}{w^\top S_w w}.
$$

Because scaling $w$ does not change the ratio, we enforce the constraint:

$$
w^\top S_w w = 1.
$$

So the problem becomes:

$$
\max_w \quad w^\top S_b w \quad \text{s.t.} \quad w^\top S_w w = 1.
$$

Construct the Lagrangian (See other for explain):

$$
\mathcal{L}(w, \lambda)
= w^\top S_b w - \lambda (w^\top S_w w - 1).
$$

Take derivative w.r.t. $w$ and set to zero:

- $2S_b w - 2\lambda S_w w = 0$

Divide by 2:

$$
S_b w = \lambda S_w w.
$$

This is the **generalized eigenvalue equation**.  
Its eigenvectors are the LDA directions, and eigenvalues measure class separability along each direction.

Thus:

$$
W = \text{top eigenvectors of } S_w^{-1} S_b.
$$

The constraint in LDA is:
- “Make the length of w (measured with a certain matrix) equal to 1.”

This constraint does not affect the direction of w,
it only fixes its scale.

But the direction of w is what we care about — the projection vector.
The length is irrelevant because scaling w up or down does not change the actual classification or projection.

So:
- The derivative w.r.t. λ just says “make sure w has the correct length.”
- The derivative w.r.t. w gives the important equation that determines the direction of w.

---

### ✅ Result: The Rank of $S_b$ is Limited

A crucial property:

> **The rank of $S_b$ is at most $K - 1$**, where $K$ is the number of classes.

Reason:
- $S_b$ is built from vectors $(\mu_k - \mu)$  
- These $K$ vectors satisfy  
  $$
  \sum_{k=1}^K N_k(\mu_k - \mu) = 0
  $$  
  so only $K - 1$ of them can be linearly independent.

Therefore:

- $S_b$ has at most $K - 1$ nonzero eigenvalues  
- LDA can find at most $K - 1$ discriminant components

#### ✅ Output dimension = at most **K − 1**

Examples:
- 2 classes → 1D projection  
- 3 classes → 2D projection  
- 10 classes → 9D projection


---

### ✅ Summary

- LDA fits Gaussian class-conditional distributions with **shared covariance**  
- Training is **closed-form**: compute class means, shared covariance, class priors  
- Classification yields **linear boundaries**  
- As dimensional reduction, LDA gives ≤ K − 1 meaningful directions because  
  **the between-class scatter has rank ≤ K − 1**

---

## 🟥 Quadratic Discriminant Analysis (QDA)

Quadratic Discriminant Analysis is the more flexible sibling of LDA.  
It uses the **same Gaussian generative model idea**, but removes LDA’s strongest assumption.

---

### ✅ Key Assumption Difference

LDA assumes:

- All classes share **one covariance matrix**.

QDA assumes:

- Each class has **its own covariance matrix**.

In symbols:

- LDA: one shared $\Sigma$
- QDA: a separate $\Sigma_k$ for each class $k$

Because of this, QDA is more expressive and can fit curved boundaries.

---

### ✅ 1. QDA Model

QDA assumes:

$$
x \mid y=k \sim \mathcal{N}(\mu_k,\ \Sigma_k).
$$

So the likelihood for each class has **its own shape**, orientation, and spread.

---

### ✅ 2. Training QDA (Closed-Form)

Training QDA = computing sample statistics **per class**.

### **1) Class prior**
$$
p(y=k) = \frac{N_k}{N}.
$$

### **2) Class mean**
$$
\mu_k = \frac{1}{N_k} \sum_{i: y^{(i)}=k} x^{(i)}.
$$

### **3) Class-specific covariance**
For each class:

$$
\Sigma_k
= \frac{1}{N_k}
\sum_{i: y^{(i)}=k}
(x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^\top.
$$

Training QDA still has **no gradients** and no iterative optimization — all closed-form.

But there are more covariance matrices to estimate, so it requires more data per class.

---

## ✅ 3. Using QDA for Classification

Start with the Gaussian likelihood for class $k$:

$$
\log p(x \mid y=k)
= -\frac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k) - \frac{1}{2}\log|\Sigma_k|.
$$

Plug into Bayes rule:

$$
\log p(y=k \mid x)
= -\frac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k) - \frac{1}{2}\log|\Sigma_k| + \log p(y=k) + \text{constant}.
$$

Because $\Sigma_k^{-1}$ is **different** for each class, the expression in $x$ becomes **quadratic**, not linear.

Thus:

### ✅ QDA gives **quadratic decision boundaries**

This is the key contrast to LDA’s linear boundaries.

---

## ✅ 4. When QDA Works Well

- Each class has a noticeably **different covariance structure**  
  (e.g., one class is spherical, another elongated)
- There is **enough data** to reliably estimate a covariance matrix per class  
- Classes form curved or irregular shapes in feature space

---

## ✅ 5. When QDA Is Risky

- Small sample sizes: estimating one covariance per class is unstable  
- High-dimensional data: $\Sigma_k$ is $d \times d$, so QDA can overfit  
- Many classes + few samples = very poor covariance estimates

Thus LDA is more robust with limited data and higher dimensions.

---

## ✅ Summary of LDA vs QDA

| Aspect | LDA | QDA |
|--------|-----|------|
| Covariance | Shared $\Sigma$ | Class-specific $\Sigma_k$ |
| Boundary | Linear | Quadratic |
| Flexibility | Lower | Higher |
| Parameters | Fewer | Many more |
| Data requirement | Low | High |
| Risk of overfitting | Small | Large |
| Best use-case | High-dim, small-data | Rich data, curved boundaries |

---
