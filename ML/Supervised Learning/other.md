# Other
Other separate math that arise during notes.

## List
- Moore–Penrose Pseudoinverse (for Linear Regression)
- SVD & how it can solve ^

## 🧱 Moore–Penrose Pseudoinverse

The **Moore–Penrose pseudoinverse** of a matrix $ X $, denoted $ X^{+} $, is a **generalized inverse** that always exists — even when \( X \) is not square or not invertible.  
It is the unique matrix that satisfies the following four properties:

1. $ X X^{+} X = X $  
2. $ X^{+} X X^{+} = X^{+} $  
3. $ (X X^{+})^\top = X X^{+} $  
4. $ (X^{+} X)^\top = X^{+} X $

---

### 🧭 Interpretation

- $ X X^{+} $ acts as an **orthogonal projector** onto the **column space** of $ X $.  
- $ X^{+} X $ acts as an **orthogonal projector** onto the **row space** of $ X $.  
- These ensure that $ X^{+} $ behaves like the “best possible” inverse of $ X $ within those subspaces.

---

### 🎯 Why $ w^\star = X^{+} y $ solves least squares

When solving the least-squares problem  
$ \min_w \|Xw - y\|_2^2 $,  
the pseudoinverse provides the **unique** solution that minimizes the squared residual:

$$
w^\star = X^{+} y
$$

This ensures that:

- $ Xw^\star $ is the **orthogonal projection** of $ y $ onto the column space of $ X $:
  $$
  \hat{y} = X X^{+} y
  $$
- If multiple $ w $ produce the same projection (e.g., $ X $ has dependent columns),  
  the pseudoinverse selects the one with the **smallest Euclidean norm** $ \|w\|_2 $.

---

### 💡 Summary
$$
\boxed{
w^\star = X^{+}y
\quad\Longrightarrow\quad
\begin{cases}
Xw^\star \text{ is the closest point in } \operatorname{col}(X) \text{ to } y, \\[4pt]
w^\star \text{ has the minimum norm among all least-squares solutions.}
\end{cases}
}
$$

---

## ⚙️ Computing the Pseudoinverse with SVD
The idea is that if use can compute such X+, then we got the w. So the question is how we compute thet X+.

### 🔹 Step 1: Singular Value Decomposition

Any real matrix $ X \in \mathbb{R}^{m \times n} $ can be factorized as:

$$
X = U\,\Sigma\,V^\top
$$

where:
- $ U \in \mathbb{R}^{m \times m} $ and $ V \in \mathbb{R}^{n \times n} $ are **orthogonal matrices**  
  (their columns are orthonormal vectors, so $U^\top U = I$ and $V^\top V = I$).
- $ \Sigma \in \mathbb{R}^{m \times n} $ is a **diagonal (rectangular) matrix of singular values**:
  $$
  \Sigma =
  \begin{bmatrix}
  \sigma_1 & & & \\
  & \sigma_2 & & \\
  & & \ddots & \\
  & & & \sigma_r
  \end{bmatrix}
  $$
  with $ \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0 $,  
  and $ r = \text{rank}(X) $.

Geometrically:
- $V$ and $U$ are **rotations or reflections** (they preserve length and angles).  
- $V$ defines a new **input basis** — its columns are the **right singular vectors**. (Maybe in a weird angle but the fact that they are orthorganal matrix means each column is orthorgonal to others so its an axis and multiply that is rotation)  
- $U$ defines a new **output basis** — its columns are the **left singular vectors**.  
- $\Sigma$ simply **scales and possibly reduces** the data along its principal directions —  
it stretches some axes by their singular values $\sigma_i$ and collapses others (where $\sigma_i = 0$).

So, in plain words:
> $V^\top$ rotates input vectors into the axes where $X$ acts purely by stretching,  
> $\Sigma$ applies that stretching,  
> and $U$ rotates the stretched result into the output space.

---

### 🔹 Step 2: Understanding why this helps

The hard part of inversion is the “mixing” between input features, but SVD separates that out.

Start from:
$$
X w = U\,\Sigma\,V^\top w.
$$

Multiply both sides by $U^\top$ (which just rotates the coordinate system, without changing lengths):

$$
U^\top X w = \Sigma V^\top w.
$$

Now, let:
$$
x' = V^\top w, \quad y' = U^\top (Xw).
$$

This gives:
$$
\Sigma x' = y'.
$$

Here:
- $V^\top w$ represents the input vector $w$ **rotated** into the basis of the right singular vectors.  
- $U^\top Xw$ represents the output vector **rotated** into the basis of the left singular vectors.

Because $\Sigma$ is diagonal, $X$ now acts *independently* on each singular direction — no more mixing between features.

Hence, solving for $w$ becomes:
$$
x'_i = \frac{y'_i}{\sigma_i}, \quad \text{for each nonzero } \sigma_i.
$$

That’s what “dividing by singular values” really means — solving each 1D scaling relationship along independent axes.

---

### 🔹 Step 3: Constructing the pseudoinverse

We now define the pseudoinverse of $\Sigma$ as:

$$
\Sigma^+ =
\begin{bmatrix}
1/\sigma_1 & & & \\
& 1/\sigma_2 & & \\
& & \ddots & \\
& & & 1/\sigma_r
\end{bmatrix}^\top
$$

We take reciprocals only for **nonzero** singular values and leave zeros as zero.  
This avoids division by zero and keeps the transformation stable.

Then, returning to the original coordinate systems:

$$
w = V\,\Sigma^+\,U^\top y,
$$

so the pseudoinverse of $X$ is:

$$
\boxed{
X^+ = V\,\Sigma^+\,U^\top
}
$$

---

### 🔹 Step 4: Why this works

Let’s verify that this satisfies the Moore–Penrose pseudoinverse conditions:

1. $ X X^+ X = U\,\Sigma\,V^\top V\,\Sigma^+\,U^\top U\,\Sigma\,V^\top = U\,\Sigma\,\Sigma^+\,\Sigma\,V^\top = X $
2. $ X^+ X X^+ = V\,\Sigma^+\,U^\top U\,\Sigma\,V^\top V\,\Sigma^+\,U^\top = V\,\Sigma^+\,\Sigma\,\Sigma^+\,U^\top = X^+ $
3. Both $ X X^+ $ and $ X^+ X $ are symmetric because $ \Sigma\,\Sigma^+ $ and $ \Sigma^+\,\Sigma $ are symmetric.

Therefore $ X^+ = V\,\Sigma^+\,U^\top $ is the unique matrix that satisfies all pseudoinverse properties.

---

### 🔹 Step 5: Connecting to least squares

The pseudoinverse directly yields the **minimum-norm least-squares solution**:

$$
w^\star = X^+ y = V\,\Sigma^+\,U^\top y
$$

Interpretation:
- $U^\top y$ projects the target vector $y$ into the singular basis of $X$.  
- $\Sigma^+$ rescales each component by $1/\sigma_i$ — reversing $X$’s stretching.  
- $V$ rotates the result back into the original input feature space.

---

### 🧠 Geometric Intuition

| Component                 | Meaning |
|---------------------------|----------|
| $V^\top$                  | Rotates the input vector into singular directions |
| $\Sigma$                  | Stretches/compresses along those directions |
| $U$                       | Rotates the stretched result into output space |
| $\Sigma^+$                | Reverses scaling only along nonzero directions |
| $V$                       | Rotates back to original input coordinates |
| $X^+ = V\,\Sigma^+\,U^\top$ | Complete mapping back from output to input, safely and minimally |

---

### 💬 In short

SVD reveals how $X$ behaves geometrically:  
1. **Rotate input** ($V^\top$),  
2. **Stretch** ($\Sigma$),  
3. **Rotate output** ($U$).  

By reversing those steps and inverting only the meaningful stretches,  
the pseudoinverse $X^+$ becomes a **stable, minimal-norm generalized inverse** —  
it “undoes” $X$’s action as much as possible, even when $X$ isn’t invertible.
