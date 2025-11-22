# Linear Regression, L1 Regularization (Lasso), L2 Regularization (Ridge), and Elastic Net

## 🔹 Linear Regression (Ordinary Least Squares)

### 🧭 Idea
We assume a **linear relationship** between features and target and choose parameters to **minimize squared errors** (OLS).  
Geometrically, we project the target vector onto the column space of the design matrix to get the closest point in Euclidean distance.

---

### 📐 Model
Let there be $m$ samples and $n$ features. Using an **augmented** design matrix to absorb the intercept:

$$
\tilde{X} \in \mathbb{R}^{m \times (n+1)}, \quad
\tilde{X} = \begin{bmatrix} \mathbf{1} & X \end{bmatrix}, \quad
\tilde{w} = \begin{bmatrix} b \\ w \end{bmatrix}, \quad
\hat{y} = \tilde{X}\tilde{w}.
$$

(If you prefer handling the intercept separately, set $ \hat{y} = Xw + b \mathbf{1} $. The derivation is identical.)

---

### 📉 OLS Loss (Objective)
We minimize the sum of squared residuals:

$$
L(\tilde{w})
= \frac{1}{2m}\,\|\tilde{X}\tilde{w} - y\|_2^2
= \frac{1}{2m}(\tilde{X}\tilde{w} - y)^\top(\tilde{X}\tilde{w} - y).
$$

(The factor $ \tfrac{1}{2m} $ is for algebraic convenience; it doesn’t change the minimizer.)

---

### ✏️ Closed-Form Solution — Step by Step


**1) Expand the quadratic form**

$$
L(\tilde{w}) =
\frac{1}{2m}*
\Bigl[
\tilde{w}^\top \tilde{X}^\top \tilde{X}\tilde{w} - 2\,y^\top \tilde{X}\tilde{w} + y^\top y
\Bigr]
$$

**2) Take gradient w.r.t. $ \tilde{w} $**
Using $ \nabla_{\tilde{w}}\left(\tilde{w}^\top A \tilde{w}\right)= (A + A^\top)\tilde{w} $ and symmetry of $ \tilde{X}^\top \tilde{X} $:
$$
\nabla_{\tilde{w}} L(\tilde{w})
= \frac{1}{2m}\left(2 \tilde{X}^\top \tilde{X}\tilde{w} - 2 \tilde{X}^\top y\right)
= \frac{1}{m}\left(\tilde{X}^\top \tilde{X}\tilde{w} - \tilde{X}^\top y\right).
$$

**3) Set gradient to zero (normal equations)**
$$
\tilde{X}^\top \tilde{X}\,\tilde{w} = \tilde{X}^\top y.
$$

**4) Solve for $ \tilde{w} $**
If $ \tilde{X}^\top \tilde{X} $ is **invertible** (full column rank):
$$
\boxed{\;\tilde{w}^\star = \left(\tilde{X}^\top \tilde{X}\right)^{-1}\tilde{X}^\top y\;}
$$

If not invertible (or ill-conditioned), use the **Moore–Penrose pseudoinverse**:
$$
\boxed{\;\tilde{w}^\star = \tilde{X}^{+}\, y\;}
\quad\text{with}\quad
\tilde{X}^{+} = \left(\tilde{X}^\top \tilde{X}\right)^{+}\tilde{X}^\top.
$$
For more detail on this, check [other](./other.md).
---

### 🧠 Notes
- The solution is unique iff $ \tilde{X} $ has **full column rank**.  
- Centering/scaling features and including a column of ones (or centering \(y\)) are common preprocessing steps but do not change the derivation.  
- Geometric view: $ \hat{y} = \tilde{X}\tilde{w}^\star $ is the **orthogonal projection** of $y $ onto $ \operatorname{col}(\tilde{X}) $.

---

## 🚀 Solving OLS via Gradient Descent

While the **closed-form solution** gives an exact answer, in practice — especially when $n$ (features) is large — 
it’s often more efficient to solve OLS iteratively using **gradient descent**.

---

### 🔹 Objective Reminder

We minimize the same loss function:

$$
L(\tilde{w}) = \frac{1}{2m} \|\tilde{X}\tilde{w} - y\|_2^2
$$

---

### 🔹 Derive the Gradient

Taking derivative w.r.t. $\tilde{w}$ (as derived before):

$$
\nabla_{\tilde{w}} L(\tilde{w})
= \frac{1}{m} \tilde{X}^\top (\tilde{X}\tilde{w} - y)
$$

🟩 **Gradient Highlight**

$$
\boxed{
\nabla_{\tilde{w}} L(\tilde{w}) = \frac{1}{m}\tilde{X}^\top (\hat{y} - y)
}
\quad \text{where} \quad \hat{y} = \tilde{X}\tilde{w}.
$$

This is the direction in which the loss increases the fastest — so to minimize it, we move in the *opposite* direction.

---

### 🔹 Gradient Descent Update Rule

The parameter update at each iteration $t$ is:

$$
\tilde{w}^{(t+1)} = \tilde{w}^{(t)} - \eta\,\nabla_{\tilde{w}} L(\tilde{w}^{(t)})
$$

Substituting the gradient:

$$
\boxed{
\tilde{w}^{(t+1)} = \tilde{w}^{(t)} - \frac{\eta}{m}\,\tilde{X}^\top(\tilde{X}\tilde{w}^{(t)} - y)
}
$$

where:
- $\eta$ is the **learning rate** (step size),
- $m$ is the number of samples.

---

### 🔹 Matrix View (Shape Intuition)

To visualize how this works across all features and samples:

$$
\begin{bmatrix}
\leftarrow & x_1^\top & \rightarrow \\
\leftarrow & x_2^\top & \rightarrow \\
\vdots & \vdots & \vdots \\
\leftarrow & x_n^\top & \rightarrow
\end{bmatrix}
\;
\begin{bmatrix}
(\hat{y}_1 - y_1) \\
(\hat{y}_2 - y_2) \\
\vdots \\
(\hat{y}_m - y_m)
\end{bmatrix}
\;\Rightarrow\;
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\vdots \\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
$$


- The left matrix ($\tilde{X}^\top$) has **each feature** as a row.  
- The middle column vector is the **residual** $(\hat{y} - y)$.  
- The resulting vector is the **gradient** for each weight $w_j$ —  
  the correlation between feature $j$ and the residuals.

---

### 🔹 Intuition

- Each step of gradient descent **projects residual errors back through features** to adjust weights.  
- If a feature $x_j$ has a high correlation with residuals, its corresponding $w_j$ will change more.  
- Over iterations, the residuals shrink until convergence (when the gradient becomes zero).

---

### ⚙️ Practical Notes

- **Convergence**: guaranteed for convex OLS loss when $\eta$ is small enough.  
- **Batch vs. Stochastic GD**:
  - *Batch GD*: uses all $m$ samples each step (as above).  
  - *Stochastic GD*: uses one or mini-batches of samples for faster but noisier convergence.
- **Scaling**: standardizing features helps avoid slow convergence due to uneven feature magnitudes.

---

### ✅ Summary

| Concept | Formula / Intuition |
|----------|---------------------|
| Gradient | $\nabla_{\tilde{w}} L = \frac{1}{m}\tilde{X}^\top(\tilde{X}\tilde{w} - y)$ |
| Update Rule | $\tilde{w} \leftarrow \tilde{w} - \frac{\eta}{m}\tilde{X}^\top(\tilde{X}\tilde{w} - y)$ |
| Interpretation | Update each $w_j$ proportional to correlation between feature $x_j$ and residuals |
| Convergence Condition | Step size $\eta$ small enough so loss decreases each iteration |

---

Gradient descent thus gives an efficient numerical way to reach the same optimum as the closed-form OLS solution,  
especially valuable when $(\tilde{X}^\top\tilde{X})$ is too large to invert directly.

---

## 🔹 Ridge Regression (L2 Regularization)

**Idea.** Add an $L_2$ penalty on weights to shrink coefficients and stabilize the solution (especially when $X^\top X$ is ill-conditioned or $n \gg m$).

We use the non-augmented model $\hat{y} = Xw + b\mathbf{1}$ and **do not penalize the intercept** $b$.

---

### 📉 Loss
$$
L(w,b)=\frac{1}{2m}\,\|Xw + b\mathbf{1} - y\|_2^2 \;+\; \frac{\lambda}{2}\,\|w\|_2^2,
\qquad \lambda \ge 0.
$$

---

### ✏️ Gradients (Derivation)

**W.r.t. $w$:**
$$
\nabla_w L
= \frac{1}{m}X^\top(Xw + b\mathbf{1} - y) \;+\; \lambda w.
$$

**W.r.t. $b$:**
$$
\nabla_b L
= \frac{1}{m}\,\mathbf{1}^\top(Xw + b\mathbf{1} - y).
$$

🟩 **Gradient highlight**
$$
\boxed{\;\nabla_w L = \frac{1}{m}X^\top(\hat{y}-y) + \lambda w,\qquad
\nabla_b L = \frac{1}{m}\,\mathbf{1}^\top(\hat{y}-y)\;}
$$

Set to zero for optimality.

---

### 🧮 Normal Equations and Closed Form

From $\nabla_w L = 0$ and $\nabla_b L = 0$:

---

#### 1️⃣ Derivation — where the equations come from

We start from the ridge loss:

$$
L(w,b)=\frac{1}{2m}\|Xw+b\mathbf{1}-y\|_2^2+\frac{\lambda}{2}\|w\|_2^2
$$

Take partial derivatives and set them to zero.

- **Derivative w.r.t. $w$:**

$$
\nabla_w L=\frac{1}{m}X^\top(Xw+b\mathbf{1}-y)+\lambda w = 0
$$

Multiply both sides by $m$ and rearrange:

$$
X^\top Xw + m\lambda w + bX^\top\mathbf{1} = X^\top y
\tag{★}
$$

- **Derivative w.r.t. $b$:**

$$
\nabla_b L=\frac{1}{m}\mathbf{1}^\top(Xw+b\mathbf{1}-y)=0
$$

Multiply both sides by $m$:

$$
\mathbf{1}^\top Xw + bm = \mathbf{1}^\top y
\quad\Longrightarrow\quad
b = \overline{y}-\overline{x}^\top w
\tag{☆}
$$

where $\overline{x}$ and $\overline{y}$ are the columnwise means of $X$ and $y$.

Now plug $(☆)$ into $(★)$ to eliminate $b$ and simplify.

---

#### 2️⃣ Normal equations

Substituting $b$ gives:

$$
(X^\top X + m\lambda I)w = X^\top (y - b\mathbf{1})
$$

This is the **ridge normal equation** — the same as OLS, but with an extra regularization term $m\lambda I$ that stabilizes inversion.

---

#### 3️⃣ Closed-form solution

If data are centered (so $\overline{x}=0$ and $\overline{y}=0$),  
the intercept term disappears, giving the standard compact form:

$$
\boxed{\,w^\star = (X^\top X + m\lambda I)^{-1}X^\top y\,}
$$

and the bias follows as $b^\star=\overline{y}-\overline{x}^\top w^\star$
(which equals $\overline{y}$ when $X$ is centered).

---

✅ **Intuition**

| Term | Effect |
|------|---------|
| $X^\top X$ | captures data correlation (as in OLS) |
| $m\lambda I$ | adds diagonal “ridge” → improves conditioning, shrinks large weights |
| $(X^\top X + m\lambda I)^{-1}$ | ensures stable inversion even if $X^\top X$ is singular |

Hence ridge regression modifies the normal equations so that the inverse always exists and large weights are penalized, balancing bias and variance.

---

## 🔹 Lasso (L1 Regularization)

**Idea.** Add an $L_1$ penalty to encourage **sparsity** (many coefficients exactly zero):
$$
L(w,b)=\frac{1}{2m}\|Xw + b\mathbf{1}-y\|_2^2 + \lambda\|w\|_1,
\qquad \|w\|_1=\sum_{j=1}^n |w_j|,\ \ \lambda\ge 0.
$$
(As usual, we **do not** penalize the intercept $b$.)

---

### ❓ Why there’s **no closed-form** solution
- The OLS part is quadratic, but the $L_1$ term is **nondifferentiable at 0** and **not jointly quadratic** in $w$. 
- With correlated features, the variables **don’t decouple** nicely, so there’s no single matrix inverse like ridge.  
- We instead use **subgradients** and **proximal/coordinate** methods.

---

### 🧮 Subgradient (a.k.a. “gradient”) conditions

Smooth part gradient:
$$
\nabla_w\Big(\tfrac{1}{2m}\|Xw+b\mathbf{1}-y\|_2^2\Big)=\tfrac{1}{m}X^\top(Xw+b\mathbf{1}-y).
$$

Subgradient of $L_1$:
$$
\partial |w_j|=
\begin{cases}
\{\operatorname{sign}(w_j)\}, & w_j\neq 0,\\[4pt]
[-1,\,1], & w_j=0.
\end{cases}
$$

First-order optimality (KKT) for each $j$:
$$
0 \in \tfrac{1}{m}X^\top(Xw+b\mathbf{1}-y) + \lambda\,s,
\qquad s_j\in\partial|w_j|.
$$

Interpretation:
- If $w_j\neq 0$: $\tfrac{1}{m}x_j^\top(Xw+b\mathbf{1}-y) + \lambda\,\operatorname{sign}(w_j)=0$.
- If $w_j=0$: $\tfrac{1}{m}x_j^\top(Xw+b\mathbf{1}-y)\in[-\lambda,\lambda]$.

This **interval condition at zero** is the algebraic reason L1 can set coefficients **exactly to 0**.

---

### 🧱 Coordinate Descent: the “$a, b$” soft-threshold step

Hold all coordinates but $w_j$ fixed. Let the **partial residual** (excluding feature $j$) be
$$
r^{(j)}=y - b\mathbf{1} - \sum_{k\neq j} x_k w_k \;=\; y - (Xw + b\mathbf{1}) + x_j w_j.
$$

The 1D subproblem in $w_j$ is
$$
\min_{w_j}\ \frac{1}{2m}\|x_j w_j - r^{(j)}\|_2^2 + \lambda |w_j|
\ \equiv\
\min_{w_j}\ \frac{a_j}{2}\,w_j^2 \;-\; b_j\,w_j \;+\; \lambda |w_j|,
$$
where
$$
a_j=\frac{1}{m}\|x_j\|_2^2,\qquad
b_j=\frac{1}{m}\,x_j^\top r^{(j)}.
$$

Because of $|w_j|$, use **subgradient optimality**:
$$
0 \in \partial \phi(w_j) = a_j w_j - b_j + \lambda\,\partial|w_j|.
$$

Recall
$$
\partial|w_j| =
\begin{cases}
\{+1\}, & w_j>0,\\
[-1,\,1], & w_j=0,\\
\{-1\}, & w_j<0.
\end{cases}
$$

We solve by cases.

---

#### Case 1: $w_j>0$
Then $\partial|w_j|=\{+1\}$ and the optimality condition is
$$
0 = a_j w_j - b_j + \lambda \;\;\Longrightarrow\;\; w_j = \frac{b_j-\lambda}{a_j}.
$$
For this to be consistent with $w_j>0$ we need $b_j-\lambda>0 \iff b_j>\lambda$.

---

#### Case 2: $w_j<0$
Then $\partial|w_j|=\{-1\}$ and
$$
0 = a_j w_j - b_j - \lambda \;\;\Longrightarrow\;\; w_j = \frac{b_j+\lambda}{a_j}.
$$
For consistency with $w_j<0$ we need $b_j+\lambda<0 \iff b_j<-\lambda$.

---

#### Case 3: $w_j=0$
Then the subgradient condition requires
$$
0 \in -\,b_j + \lambda\,[-1,1]
\;\;\Longleftrightarrow\;\;
|b_j| \le \lambda.
$$
So $w_j=0$ is optimal when the correlation is small: $|b_j|\le \lambda$.

---

### ✅ Combine the cases (soft-threshold)

Putting the three cases together:
$$
w_j =
\begin{cases}
\dfrac{b_j-\lambda}{a_j}, & b_j>\lambda,\\[8pt]
0, & |b_j|\le \lambda,\\[8pt]
\dfrac{b_j+\lambda}{a_j}, & b_j<-\lambda,
\end{cases}
\qquad\Longleftrightarrow\qquad
w_j \;=\; \dfrac{\operatorname{sign}(b_j)\,\max\{|b_j|-\lambda,\,0\}}{a_j}.
$$

This is exactly the **soft-thresholding** rule:
- If the (scaled) correlation $b_j$ is within the band $[-\lambda,\lambda]$, the optimum is **cut to zero**.
- Otherwise, the optimum is the **unregularized solution** $b_j/a_j$ **shrunk toward 0** by $\lambda/a_j$.

---

🧠 **Interpretation**
- When $b_j > \lambda$: The correlation between feature $x_j$ and the residuals is strong and positive —  
  $w_j$ remains positive but is **shrunk** by $\lambda/a_j$.
- When $b_j < -\lambda$: The correlation is strong and negative — $w_j$ remains negative but is **moved toward zero** by $\lambda/a_j$.
- When $|b_j| \le \lambda$: The penalty fully cancels the correlation — the optimal $w_j = 0$.

Hence, **L1 regularization encourages sparsity**:  
features with weak correlation ($|b_j| \le \lambda$) are *zeroed out*,  
while strong ones ($|b_j| > \lambda$) are merely *softly reduced* toward zero.


---


**Rectangle view (what the $b_j$ correlation looks like):**
$$
\underbrace{
\begin{bmatrix}
\leftarrow & x_1^\top & \rightarrow
\end{bmatrix}}_{\text{feature }j}
\;
\underbrace{
\begin{bmatrix}
r^{(j)}_1\\ r^{(j)}_2\\ \vdots\\ r^{(j)}_m
\end{bmatrix}}_{\text{partial residual}}
\;\;\Rightarrow\;\;
b_j=\tfrac{1}{m}x_j^\top r^{(j)}.
$$

This is exactly the “$a, b$” form many interviews expect:
- **$a_j$** is the (scaled) feature energy.
- **$b_j$** is the (scaled) correlation between feature $j$ and the current residual.

---

### ⚙️ Why **soft-thresholding** (prox) instead of plain subgradient GD?

- **Plain subgradient descent** on $L_1$ near zero can **“bounce” around 0** because the subgradient set is $[-1,1]$ at 0; it nudges but does not decisively set weights to zero.
- The **proximal step** (soft-thresholding) is the **exact minimizer** of the sum “quadratic + $\lambda|w_j|$” in each coordinate, which:
  - performs a **discrete shrink-to-zero** when the correlation is small ($|b_j|\le\lambda$),
  - otherwise shrinks continuously, giving stable, monotone progress and true sparsity.

(Equivalently, full-vector proximal gradient—ISTA—does:  
$w^{t+1}=\operatorname{Soft}\!\big(w^t-\eta\tfrac{1}{m}X^\top(Xw^t+b\mathbf{1}-y),\ \eta\lambda\big)$.)

---

### 🧠 Summary

| Aspect | L1 (Lasso) behavior |
|---|---|
| Closed form (joint) | **No** (nondifferentiable, coupled by $X$) |
| Optimality condition | $0\in \tfrac{1}{m}X^\top(Xw+b\mathbf{1}-y)+\lambda\,\partial\|w\|_1$ |
| Coordinate update | $w_j \leftarrow \mathcal{S}(b_j,\lambda)/a_j$ with $a_j=\tfrac{1}{m}\|x_j\|_2^2$, $b_j=\tfrac{1}{m}x_j^\top r^{(j)}$ |
| Sparsity mechanism | **Soft-thresholding** sets $w_j=0$ when $|b_j|\le\lambda$ |
| Why prox/soft-threshold | Avoids oscillation at 0; gives exact sparse updates |

L1’s soft-thresholding is the mathematical reason it **“cuts down”** coefficients to **exact zero**, delivering sparsity that ridge (L2) cannot.

---

## 🔹 Elastic Net (L1 + L2 Regularization)

**Idea.** Combine L1’s **sparsity** with L2’s **stability/grouping**. We add both penalties (intercept $b$ is not penalized).
A common parameterization uses a single strength $\lambda\ge 0$ and a mixing $\alpha\in[0,1]$:
- $\alpha=1$ → Lasso
- $\alpha=0$ → Ridge
- $0<\alpha<1$ → Elastic Net

---

### 📉 Loss
$$
L(w,b)
=\frac{1}{2m}\,\|Xw + b\mathbf{1} - y\|_2^2
\;+\;\lambda\Big[(1-\alpha)\,\frac{\|w\|_2^2}{2}\;+\;\alpha\,\|w\|_1\Big].
$$

---

### 🧮 Gradients / Subgradients

Smooth part:
$$
\nabla_w\Big(\tfrac{1}{2m}\|Xw+b\mathbf{1}-y\|_2^2\Big)
= \tfrac{1}{m}X^\top(Xw+b\mathbf{1}-y),\qquad
\nabla_b L = \tfrac{1}{m}\,\mathbf{1}^\top(Xw+b\mathbf{1}-y).
$$

Regularization parts:
- L2: $\nabla_w\big(\lambda(1-\alpha)\tfrac{\|w\|_2^2}{2}\big)=\lambda(1-\alpha)\,w$.
- L1: $\lambda\alpha\,\partial\|w\|_1$ with
  $\partial|w_j|=
  \begin{cases}
  \{\operatorname{sign}(w_j)\}, & w_j\neq 0\\
  [-1,1], & w_j=0
  \end{cases}$.

**Putting together (KKT condition):**
$$
0 \in \tfrac{1}{m}X^\top(Xw+b\mathbf{1}-y) \;+\; \lambda(1-\alpha)\,w \;+\; \lambda\alpha\,\partial\|w\|_1.
$$

---

### 🧱 Coordinate Descent (same soft-thresholding pattern as L1)

Hold all coordinates except $w_j$ fixed and define the **partial residual**:
$$
r^{(j)} = y - b\mathbf{1} - \sum_{k\ne j} x_k w_k
= y - (Xw+b\mathbf{1}) + x_j w_j.
$$

As before, set
$$
a_j=\tfrac{1}{m}\|x_j\|_2^2,\qquad
b_j=\tfrac{1}{m}x_j^\top r^{(j)}.
$$

The 1D subproblem in $w_j$ becomes
$$
\min_{w_j}\ \frac{a_j}{2}w_j^2 - b_j w_j \;+\; \lambda(1-\alpha)\frac{w_j^2}{2} \;+\; \lambda\alpha|w_j|.
$$

Group the quadratic terms: $a_j \mapsto a_j + \lambda(1-\alpha)$.
This yields the **soft-threshold** update:

$$
\boxed{
\quad
w_j \;\leftarrow\; \frac{\mathcal{S}(b_j,\ \lambda\alpha)}{\,a_j + \lambda(1-\alpha)\,}
\quad}
\qquad
\text{where}\quad
\mathcal{S}(b_j,\tau)=\operatorname{sign}(b_j)\,\max\{|b_j|-\tau,0\}.
$$

Expanded piecewise form:
$$
w_j=
\begin{cases}
\dfrac{b_j-\lambda\alpha}{\,a_j+\lambda(1-\alpha)\,}, & b_j>\lambda\alpha,\\[8pt]
0, & |b_j|\le \lambda\alpha,\\[8pt]
\dfrac{b_j+\lambda\alpha}{\,a_j+\lambda(1-\alpha)\,}, & b_j<-\lambda\alpha.
\end{cases}
$$

**Why it works (intuition):**
- L1 ($\lambda\alpha$) still performs **soft-thresholding** → can set coefficients exactly to zero (sparsity).
- L2 ($\lambda(1-\alpha)$) **stabilizes** and **spreads** weight among correlated features (grouping effect) and improves numerical conditioning via the added diagonal in the denominator.

---

### 🔲 Matrix “Rectangle” view (gradient intuition)

Elastic Net gradient combines ridge-like shrinkage and lasso-like sparsity pressure:
$$
\underbrace{
\begin{bmatrix}
\leftarrow & x_1^\top & \rightarrow \\
\leftarrow & x_2^\top & \rightarrow \\
\vdots & \vdots & \vdots \\
\leftarrow & x_n^\top & \rightarrow
\end{bmatrix}
}_{X^\top}
\;
\underbrace{
\begin{bmatrix}
\hat{y}_1 - y_1\\
\hat{y}_2 - y_2\\
\vdots\\
\hat{y}_m - y_m
\end{bmatrix}
}_{(\hat y - y)}
\;+\;
\underbrace{
\begin{bmatrix}
\lambda(1-\alpha)w_1\\
\lambda(1-\alpha)w_2\\
\vdots\\
\lambda(1-\alpha)w_n
\end{bmatrix}
}_{\text{L2 shrink}}
\;+\;
\underbrace{\lambda\alpha\,\partial\|w\|_1}_{\text{L1 soft-threshold}}
\;\Rightarrow\;
\underbrace{
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots\\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
}_{\nabla_w L}
$$

---

### 🔲 Matrix “Rectangle” view (gradient intuition)

Elastic Net gradient combines ridge-like shrinkage and lasso-like sparsity pressure:
$$
\underbrace{
\begin{bmatrix}
\leftarrow & x_1^\top & \rightarrow \\
\leftarrow & x_2^\top & \rightarrow \\
\vdots & \vdots & \vdots \\
\leftarrow & x_n^\top & \rightarrow
\end{bmatrix}
}_{X^\top}
\;
\underbrace{
\begin{bmatrix}
\hat{y}_1 - y_1\\
\hat{y}_2 - y_2\\
\vdots\\
\hat{y}_m - y_m
\end{bmatrix}
}_{(\hat y - y)}
\;+\;
\underbrace{
\begin{bmatrix}
\lambda(1-\alpha)w_1\\
\lambda(1-\alpha)w_2\\
\vdots\\
\lambda(1-\alpha)w_n
\end{bmatrix}
}_{\text{L2 shrink}}
\;+\;
\underbrace{\lambda\alpha\,\partial\|w\|_1}_{\text{L1 soft-threshold}}
\;\Rightarrow\;
\underbrace{
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots\\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
}_{\nabla_w L}
$$

---

### 🧠 Summary

| Aspect | Elastic Net behavior |
|---|---|
| Loss | $\tfrac{1}{2m}\|Xw+b\mathbf{1}-y\|_2^2+\lambda\big((1-\alpha)\tfrac{\|w\|_2^2}{2}+\alpha\|w\|_1\big)$ |
| Sparsity | Yes, via L1 (soft-thresholding) |
| Stability | Yes, via L2 (denominator $a_j+\lambda(1-\alpha)$) |
| Coord. update | $w_j \leftarrow \dfrac{\mathcal{S}(b_j,\lambda\alpha)}{a_j+\lambda(1-\alpha)}$ |
| Intercept | Not penalized; update $b$ from mean residual as in ridge/lasso |
| Use when | You want **sparsity** but also **robustness to multicollinearity** and better conditioning |

---
## 🔸 Comparison: OLS vs Ridge vs Lasso vs Elastic Net

| Method | Penalty Term                                                                           | Key Property | Solution Form | Can Set w_j = 0? | Typical Use Case |
|:-------|:---------------------------------------------------------------------------------------|:------------|:--------------|:-----------------|:-----------------|
| **Ordinary Least Squares (OLS)** | *None*                                                                                 | Minimizes pure squared error | `w~ = (X^T X)^(-1) X^T y` | ❌ | Baseline model when n < m and features are well-conditioned |
| **Ridge (L2)** | `(λ/2) \|    \|w \| \|_2^2`                                                            | Shrinks all weights smoothly, stabilizes inversion | `w = (X^T X + m λ I)^(-1) X^T y` | ❌ | Multicollinearity or many small correlated features |
| **Lasso (L1)** | `λ                          \| \|w \| \|_1`                                            | Drives some weights exactly to zero (sparsity) | No closed form — uses coordinate descent with soft-thresholding | ✅ | Feature selection; high-dimensional, sparse problems |
| **Elastic Net (L1+L2)** | `λ[(1-α)                                    \|  \|w \| \|_2^2 / 2 + α \| \|w \| \|_1]` | Combines sparsity + grouping; balances L1 and L2 | `w_j ← S(b_j, λ α) / (a_j + λ (1-α))` | ✅ (but less aggressive than pure L1) | Correlated features, need both feature selection and stability |

---

### 🧩 How the Regularizers Affect the Gradient

From the general form:
$$
\nabla_w L = \frac{1}{m}X^\top(Xw - y) + \lambda(1-\alpha)w + \lambda\alpha\,\partial\|w\|_1,
$$

- **OLS:** Only the first term ($X^\top(Xw-y)$) — moves directly opposite the residual correlation.  
- **Ridge:** Adds $\lambda w$, which acts like *friction*, shrinking weights smoothly toward 0.  
- **Lasso:** Replaces the continuous $\lambda w$ with a *piecewise constant force* ($\lambda\operatorname{sign}(w)$),  
  which **stops movement completely when $|b_j|\le\lambda$** → some weights exactly 0.  
- **Elastic Net:** Combines both — has L2’s stabilizing term *and* L1’s threshold zone,  
  giving smoother shrinkage while still allowing sparsity.

---

### ⚖️ When to Use Which

| Scenario | Recommended Model | Why |
|-----------|-------------------|-----|
| Features are few, well-behaved, and not correlated | **OLS** | No need for regularization; unbiased and efficient |
| Features are many and correlated (multicollinearity) | **Ridge** | $X^\top X + \lambda I$ stays invertible, prevents large weight swings |
| You expect only a few features to matter | **Lasso** | L1 shrinks weak coefficients to **exactly 0**, performing feature selection |
| You have many correlated but also irrelevant features | **Elastic Net** | Keeps L1’s sparsity but L2 ensures correlated groups survive together |
| You want numerical stability and generalization | **Ridge** or **Elastic Net (low α)** | Less variance, better generalization on noisy data |
| You want interpretability / model compression | **Lasso** or **Elastic Net (high α)** | Sparse weights simplify the model and improve explainability |

---

### 🧠 Intuitive Summary (Linking Back to the Formula)

| Model | Gradient Behavior | Geometric Effect |
|:------|:------------------|:----------------|
| **OLS** | $X^\top(Xw-y)$ | Minimizes squared residuals — weight vector unconstrained |
| **Ridge (L2)** | $X^\top(Xw-y)+\lambda w$ | Penalizes large coefficients — circular constraint, smooth shrinkage |
| **Lasso (L1)** | $X^\top(Xw-y)+\lambda\,\text{sign}(w)$ | Sharp corners in constraint ($L_1$ ball) — corners → some weights hit zero |
| **Elastic Net** | $X^\top(Xw-y)+\lambda(1-\alpha)w+\lambda\alpha\,\text{sign}(w)$ | “Rounded diamond” constraint — can zero some weights but keeps correlated ones together |

In other words:
- **L2:** smooths but never kills.
- **L1:** kills but can be unstable.
- **Elastic Net:** *disciplined killer* — still sparse, but less extreme.

---

### 🧾 Quick Reference Formulas

| Method | Update Rule (coordinate descent form) |
|---------|--------------------------------------|
| **Ridge** | $w_j \leftarrow \frac{b_j}{a_j+\lambda}$ |
| **Lasso** | $w_j \leftarrow \frac{\mathcal{S}(b_j,\lambda)}{a_j}$ |
| **Elastic Net** | $w_j \leftarrow \frac{\mathcal{S}(b_j,\lambda\alpha)}{a_j+\lambda(1-\alpha)}$ |

where  
$\mathcal{S}(b_j,\lambda)=\operatorname{sign}(b_j)\max\{|b_j|-\lambda,0\}$  
and $a_j=\tfrac{1}{m}\|x_j\|_2^2,\;b_j=\tfrac{1}{m}x_j^\top r^{(j)}$.

