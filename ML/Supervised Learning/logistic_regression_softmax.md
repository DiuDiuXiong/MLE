# Logistic Regression & Softmax Regression

## 🧩 From Linear Scores to Probabilities

After computing the **linear combination of features**,

$$
z_i = w_i^T x + b_i,
$$

each \( z_i \) represents an *unnormalized score* for class \( i \).  
These scores can be any real numbers — positive or negative — but we need to map them into a **probability space**, where:

$$
P_i \ge 0, \quad \sum_i P_i = 1.
$$

To achieve this, we transform each score by comparing it to all others and normalizing:

$$
P_i = \frac{f(z_i)}{\sum_j f(z_j)}.
$$

This ensures that all \( P_i \) values are non-negative and sum to one.

---

## 💡 Why Use the Exponential Function

We choose $f(z)=e^{z}$ because:

- Positivity: $e^{z} > 0$ for all $z$, so probabilities are always valid.
- Monotonicity: Larger $z$ leads to larger $e^{z}$, preserving order of confidence.
- Smooth Growth: The exponential sharply amplifies differences between scores, helping clear separation between likely and unlikely classes.
- Mathematical Convenience: The derivative of $e^{z}$ is itself, which simplifies optimization during gradient-based training.

Thus, the normalized exponential form:

$$
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

is the natural bridge between *linear models* and *probabilistic outputs*.

--- 

## 🔹 Logistic Regression

Logistic Regression is the **binary** form of the normalized exponential idea.  
Instead of computing separate scores for every class, we only learn one **weight vector** for the *positive* class.

For a given input:

$$
z = w^T x + b
$$

We interpret this as the **score for the positive case**.  
The negative case is implicitly fixed — its score is treated as 0.  
Hence, the probability of the positive outcome is determined by comparing its score to that of the negative one:

$$
P(y=1|x) = \frac{e^{z}}{e^{z} + e^{0}} = \frac{e^{z}}{1 + e^{z}}
$$

Simplifying gives the **sigmoid (logistic) function**:

$$
P(y=1|x) = \frac{1}{1 + e^{-z}}
$$

Thus, logistic regression models the likelihood of the positive class using a **single linear score**,  
and the decision boundary lies where

$$
z = 0 \quad \text{i.e.,} \quad w^T x + b = 0
$$

---

## ⚙️ Training Logistic Regression

To use the model effectively, we must **train** it — that is, find the parameters \( w \) and \( b \) that best fit the data.  
The usage part (how to compute scores and the decision boundary) was covered above.  
Now, let's focus on **how we learn** those parameters.

---

### 🔹 No Closed-Form Solution

Unlike Linear Regression, there’s **no closed-form solution** for Logistic Regression because the sigmoid introduces a **nonlinear** term.  
Therefore, we rely on **iterative optimization**, most commonly **Gradient Descent**.

We minimize the **negative log-likelihood (cross-entropy loss):**

$$
L(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

where $ \hat{y}_i = \sigma(w^T x_i + b) = \frac{1}{1 + e^{-(w^T x_i + b)}} $

---

### 🧮 Gradient Derivation (Step-by-Step)

#### 1) Model and Notation

For each sample $x_i \in \mathbb{R}^n$ with label $y_i \in \{0,1\}$
$$
z_i = w^T x_i + b, \qquad \hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}.
$$

Here $ \hat{y}_i = P(y_i = 1 \mid x_i) $.

#### 2) Likelihood (Bernoulli)

Given independence over \(m\) samples, the likelihood is

$$
\mathcal{L}(w,b) = \prod_{i=1}^{m} \hat{y}_i^{\,y_i}\,\big(1-\hat{y}_i\big)^{\,1-y_i}.
$$

This comes from substituting the Bernoulli pmf:
- when \(y_i=1\), the factor is $\hat{y}_i$;
- when \(y_i=0\), the factor is $1-\hat{y}_i$.

#### 3) Log-Likelihood

Take the log to turn products into sums:

$$
\ell(w,b) = \log \mathcal{L}(w,b)
= \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log\big(1 - \hat{y}_i\big) \right].
$$

#### 4) Negative Log-Likelihood (Loss)

We minimize the **negative** log-likelihood (optionally averaged by \(m\)):

$$
L(w,b) = -\frac{1}{m}\,\ell(w,b)
= -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log\big(1 - \hat{y}_i\big) \right].
$$

This is the **binary cross-entropy** loss.

#### 5) Useful Derivatives

We will use:
- Sigmoid derivative:
$
\frac{d\,\hat{y}_i}{d z_i} = \hat{y}_i \big(1 - \hat{y}_i\big).
$
- Chain rule through $z_i = w^T x_i + b$:   
  - $\frac{\partial z_i}{\partial w} = x_i, \qquad$
  - $\frac{\partial z_i}{\partial b} = 1.$

#### 6) Gradient w.r.t. \(w\)

Start from the loss:

$$
L(w,b) = -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log\big(1 - \hat{y}_i\big) \right].
$$

Differentiate term-by-term using the chain rule. For a single \(i\):

- Derivative of $y_i \log(\hat{y}_i)$ w.r.t. \(w\):

$$
\frac{\partial}{\partial w}\left[y_i \log(\hat{y}_i)\right]
= y_i \cdot \frac{1}{\hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w}
= y_i \cdot \frac{1}{\hat{y}_i} \cdot \hat{y}_i(1-\hat{y}_i)\cdot x_i
= y_i(1-\hat{y}_i)\,x_i.
$$

- Derivative of $(1 - y_i)\log(1 - \hat{y}_i)$ w.r.t. \(w\):

$$
\frac{\partial}{\partial w}\left[(1-y_i)\log(1-\hat{y}_i)\right]
= (1-y_i) \cdot \frac{1}{1-\hat{y}_i} \cdot \big(-\frac{\partial \hat{y}_i}{\partial z_i}\big) \cdot \frac{\partial z_i}{\partial w}
= (1-y_i) \cdot \frac{1}{1-\hat{y}_i} \cdot \big(-\hat{y}_i(1-\hat{y}_i)\big)\cdot x_i
= -(1-y_i)\hat{y}_i\,x_i.
$$

Combine inside the sum and include the leading minus sign from \(L\):

$$
\frac{\partial L}{\partial w}
= -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i(1-\hat{y}_i)\,x_i \;-\; (1-y_i)\hat{y}_i\,x_i \right].
$$

Distribute and simplify the bracket:

$$
y_i(1-\hat{y}_i) - (1-y_i)\hat{y}_i
= y_i - y_i\hat{y}_i - \hat{y}_i + y_i\hat{y}_i
= y_i - \hat{y}_i.
$$

Therefore,

$$
\frac{\partial L}{\partial w}
= -\frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)\,x_i
= \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)\,x_i.
$$

#### 7) Gradient w.r.t. \(b\)

Repeat the same logic, using $\frac{\partial z_i}{\partial b}=1$:

- For $y_i \log(\hat{y}_i)$:

$$
\frac{\partial}{\partial b}\left[y_i \log(\hat{y}_i)\right]
= y_i \cdot \frac{1}{\hat{y}_i} \cdot \hat{y}_i(1-\hat{y}_i) \cdot 1
= y_i(1-\hat{y}_i).
$$

- For $(1 - y_i)\log(1 - \hat{y}_i)$:

$$
\frac{\partial}{\partial b}\left[(1-y_i)\log(1-\hat{y}_i)\right]
= (1-y_i)\cdot \frac{1}{1-\hat{y}_i} \cdot \big(-\hat{y}_i(1-\hat{y}_i)\big) \cdot 1
= -(1-y_i)\hat{y}_i.
$$

Combine with the leading minus sign from \(L\):

$$
\frac{\partial L}{\partial b}
= -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i(1-\hat{y}_i) - (1-y_i)\hat{y}_i \right]
= \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i).
$$

#### 8) Final Gradients

$$
\boxed{\;\displaystyle \frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)\,x_i \;}
\qquad
\boxed{\;\displaystyle \frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) \;}
$$

---

### 🧱 Matrix Form Representation

For all samples together, let:

- $ X $ be an $ n \times m $ matrix of features (each row is a sample).
- $ \hat{y} $ and $ y $ be $ m \times 1 $ column vectors of predictions and true labels.

Then:

$$
\nabla_w L = \frac{1}{m} X^T (\hat{y} - y)
$$

Visually, this can be represented as:

$$
\begin{bmatrix}
\longleftarrow & x_1 & \longrightarrow \\
\longleftarrow & x_2 & \longrightarrow \\
\vdots & & \vdots \\
\longleftarrow & x_m & \longrightarrow
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


This rectangular structure shows how each feature dimension accumulates contributions from all training samples —  
a form often expected in coding interviews when implementing logistic regression manually.

---

## 🔹 Softmax Regression (Multiclass Logistic Regression)

Softmax Regression is a **generalized version** of Logistic Regression used when there are **multiple classes** $( K > 2 )$.  
Instead of predicting a single probability for the positive class, we now assign a probability to **each class**, ensuring all probabilities sum to 1.

---

### 1) Linear Scores

For each class $ k \in \{1, 2, \dots, K\} $, we compute a linear score:

$$
z_k = w_k^T x + b_k
$$

Here, each class $ k $ has its own parameter vector $ w_k $ and bias $ b_k $.

---

### 2) Converting Scores to Probabilities

We normalize the exponentiated scores to form a valid probability distribution:

$$
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

This is the **softmax function**, which generalizes the sigmoid used in binary logistic regression.

---

### 3) Log-Likelihood

Given a dataset of $ m $ samples $ \{(x_i, y_i)\} $, the log-likelihood is:

$$
\ell(W, b) = \sum_{i=1}^{m} \log P(y_i \mid x_i)
= \sum_{i=1}^{m} \log \left( \frac{e^{z_{i,y_i}}}{\sum_{j=1}^{K} e^{z_{i,j}}} \right)
$$

where $ z_{i,j} = w_j^T x_i + b_j $ and $ z_{i,y_i} $ is the score corresponding to the true class of sample $ i $.

Simplifying:

$$
\ell(W, b) = \sum_{i=1}^{m} \left[ z_{i,y_i} - \log \sum_{j=1}^{K} e^{z_{i,j}} \right]
$$

---

### 4) Negative Log-Likelihood (Loss Function)

We minimize the **negative log-likelihood**, also called the **categorical cross-entropy loss**:

$$
L(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ z_{i,y_i} - \log \sum_{j=1}^{K} e^{z_{i,j}} \right]
$$

---

### 5) Gradient Derivation

We now derive the gradient with respect to $ w_k $ for each class $ k $.

Start with the loss for one sample $ i $:

$$
L_i = -\left[ z_{i,y_i} - \log \sum_{j=1}^{K} e^{z_{i,j}} \right]
$$

Take derivative w.r.t. $ z_{i,k} $:

$$
\frac{\partial L_i}{\partial z_{i,k}}
= -\frac{\partial z_{i,y_i}}{\partial z_{i,k}} + \frac{\partial}{\partial z_{i,k}}\log \sum_{j=1}^{K} e^{z_{i,j}}
$$

Compute both terms:
- The first term equals $ -1 $ if $ k = y_i $, otherwise 0.
- The second term uses derivative of log-sum-exp:

$$
\frac{\partial}{\partial z_{i,k}} \log \sum_{j=1}^{K} e^{z_{i,j}} = \frac{e^{z_{i,k}}}{\sum_{j=1}^{K} e^{z_{i,j}}} = P(y = k \mid x_i)
$$

Combine them:

$$
\frac{\partial L_i}{\partial z_{i,k}} = P(y = k \mid x_i) - \mathbb{1}\{y_i = k\}
$$

where $ \mathbb{1}\{y_i = k\} $ is 1 if the sample belongs to class $ k $, else 0.

---

### 6) Gradient w.r.t. Parameters

Since $ z_{i,k} = w_k^T x_i + b_k $,

$$
\frac{\partial z_{i,k}}{\partial w_k} = x_i, \qquad \frac{\partial z_{i,k}}{\partial b_k} = 1.
$$

Thus,

$$
\frac{\partial L}{\partial w_k} = \frac{1}{m} \sum_{i=1}^{m} \left[ P(y = k \mid x_i) - \mathbb{1}\{y_i = k\} \right] x_i
$$

$$
\frac{\partial L}{\partial b_k} = \frac{1}{m} \sum_{i=1}^{m} \left[ P(y = k \mid x_i) - \mathbb{1}\{y_i = k\} \right]
$$

---

### 7) Compact Matrix Form

Let:
- $ X \in \mathbb{R}^{m \times n} $: feature matrix
- $ Y \in \mathbb{R}^{m \times K} $: one-hot encoded true labels
- $ \hat{Y} \in \mathbb{R}^{m \times K} $: predicted probabilities from softmax

Then the gradient can be written compactly as:

$$
\nabla_W L = \frac{1}{m} X^T (\hat{Y} - Y)
$$

This is the **multiclass generalization** of the logistic regression gradient.

---

## 🔎 Why Log-Odds Are Linear (Logistic & Softmax)

### Binary (Logistic) Case

Starting from the sigmoid probability for the positive class:
$$
P(y=1\mid x)=\frac{e^{z}}{1+e^{z}},\quad z=w^T x + b.
$$

Compute the **odds** of the positive class:
$$
\frac{P(y=1\mid x)}{1-P(y=1\mid x)}
=\frac{\frac{e^{z}}{1+e^{z}}}{1-\frac{e^{z}}{1+e^{z}}}
=\frac{e^{z}}{1}
= e^{z}.
$$

Take the logarithm to get **log-odds (logit)**:
$$
\log\frac{P(y=1\mid x)}{1-P(y=1\mid x)}=\log(e^{z})=z=w^T x+b.
$$

**Conclusion.** The log-odds are a **linear function of the features**.

---

### Multiclass (Softmax) Case

From softmax:
$$
P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}},\quad z_k=w_k^T x + b_k.
$$

Consider the **odds ratio between two classes** \(k\) and \(r\):
$$
\frac{P(y=k\mid x)}{P(y=r\mid x)}
=\frac{\frac{e^{z_k}}{\sum_{j} e^{z_j}}}{\frac{e^{z_r}}{\sum_{j} e^{z_j}}}
=\frac{e^{z_k}}{e^{z_r}}
= e^{\,z_k - z_r}.
$$

Take the logarithm:
$$
\log\frac{P(y=k\mid x)}{P(y=r\mid x)}
= z_k - z_r
= (w_k - w_r)^T x + (b_k - b_r).
$$

**Conclusion.** For softmax, the **pairwise log-odds between any two classes** are **linear in $x$**, with coefficient vector $w_k - w_r$ and intercept $b_k - b_r$. (Equivalently, fixing a reference class $r$ makes each class’s log-odds linear.)

---

## Deep ML Q
- https://www.deep-ml.com/problems/39
- https://www.deep-ml.com/problems/105
- https://www.deep-ml.com/problems/23
- https://www.deep-ml.com/problems/104
- https://www.deep-ml.com/problems/106