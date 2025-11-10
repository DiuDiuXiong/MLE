# 🌱 Generative Models (Naive Bayes • LDA • QDA) 

## Intuition & Starting Point
Generative models take a **fundamentally different viewpoint** from discriminative models (like logistic regression, SVMs, or neural nets).

Instead of directly learning a function to predict labels, generative models try to model **how the data was generated**.

---

## ✅ Core Idea: Model the Joint Distribution

A generative classifier models:

$$
p(x, y) = p(x \mid y)\, p(y)
$$

Once we know this, prediction follows from Bayes’ rule:

$$
p(y \mid x) = \frac{p(x \mid y)\, p(y)}{\sum_{y'} p(x \mid y') p(y')}
$$

The goal during training is to **learn parameters of the class-conditional distributions**, i.e. (we want to figure out how to compute p(x|y) and p(y)):

- For Bernoulli Naive Bayes: probabilities of each feature being 1 given the class  
- For Gaussian models (LDA/QDA): means and covariances for each class  
- For categorical/MNB: discrete frequency distributions  

Once these distributions are known, classification is “plug into Bayes rule.”

---

## ✅ Why “Generative”?

Because the model can (in principle) **generate new samples** from the learned distributions:

- For Naive Bayes (Bernoulli):  
  generate features by flipping biased coins for each feature  
- For LDA (Linear Discriminant Analysis):  
  sample a class → sample from that class’s Gaussian  
- For QDA (Quadratic Discriminant Analysis):  
  same, but with a class-specific covariance  
- For Gaussian mixture models (not covered here, but related):  
  sample mixture weights → component → vector  

These models learn **the probability distribution of $x$ conditioned on the class**.

So they answer:

> **How does each class produce its data?**  
> Not just “what is the decision boundary?”

---

## ✅ Generative Model Workflow

1. **Estimate prior**  
   $$
   p(y) = \frac{\text{# samples in class } y}{N}
   $$

2. **Estimate likelihood (class-conditional distribution)**  
   Examples:  
   - Bernoulli NB:  
     $$p(x_j=1 \mid y) = \theta_{y,j}$$  
   - LDA/QDA:  
     $$x \mid y \sim \mathcal{N}(\mu_y, \Sigma_y)$$

3. **Compute posterior at test time**  
   $$
   p(y \mid x) \propto p(x \mid y)\, p(y)
   $$

4. **Pick the class with highest posterior**  
   $$
   \hat{y} = \arg\max_y \; p(y\mid x)
   $$

---

## ✅ Connection to Decision Boundaries

- **Naive Bayes (Bernoulli / Multinomial / Gaussian)**  
  Leads to **linear** decision boundaries in feature space (because the log-likelihood is linear in $x$).

- **LDA (shared covariance)**  
  Assumes:  
  $$\Sigma_1 = \Sigma_2 = \dots = \Sigma_K = \Sigma$$  
  This gives a **linear** decision boundary.

- **QDA (class-specific covariance)**  
  $$\Sigma_1 \ne \Sigma_2 \ne \dots$$  
  This gives a **quadratic** boundary.

Generative models therefore range from simple linear models to flexible curved boundaries.

---

## ✅ Why They Are Useful

- Very easy to train (closed-form MLE solutions)  
- Work well with small datasets  
- Make explicit probabilistic assumptions (easy to interpret)  
- Form the foundation of many ML pipelines  
- Provide a baseline for classification tasks

---

## 🌪️ Naive Bayes (NB) Family — General Idea

Naive Bayes is a **generative** classifier that models the joint distribution

$$
p(x, y) = p(y)\, p(x \mid y)
$$

and predicts via Bayes’ rule:

$$
p(y \mid x) \propto p(x \mid y)\, p(y).
$$

The “naive” assumption:

> **Features are conditionally independent given the class**  
> $$p(x \mid y) = \prod_{j=1}^d p(x_j \mid y).$$

This assumption simplifies both training and inference:

- Each feature can be modeled independently  
- The likelihood decomposes into a product  
- Parameter estimation becomes extremely simple (closed-form counts or sample stats)

---

## ✅ How the Naive Bayes Family Differs  
Each NB variant chooses a different **probability distribution** for the likelihood term  
$$p(x_j \mid y).$$

Different data types → different distributions → different inference formulas.

Below is a high-level summary.

---

## 🟦 1. **Gaussian Naive Bayes (GNB)**  
Best for: **continuous, real-valued features**.

Assume:

$$
x_j \mid y \sim \mathcal{N}(\mu_{y,j}, \sigma_{y,j}^2)
$$

Likelihood:

$$
p(x \mid y) = \prod_{j=1}^d 
\frac{1}{\sqrt{2\pi \sigma_{y,j}^2}}
\exp\!\left( -\frac{(x_j - \mu_{y,j})^2}{2\sigma_{y,j}^2} \right).
$$

Parameters learned per class and feature:
- The mean $\mu_{y,j}$ is the **average of feature $j$ for all samples belonging to class $y$**.  
- The variance $\sigma_{y,j}^2$ is the **sample variance of feature $j$ within that same class**.

Effect: **linear decision boundaries** in original features (via log-likelihood).

---

## 🟧 2. **Multinomial Naive Bayes (MNB)**  
Best for: **count features** (word counts, TF-IDF approximations).

Assume:

$$
x \mid y \sim \text{Multinomial}(n, \theta_{y})
$$

with parameters:

$$
\theta_{y,j} = p(\text{feature } j \mid y), \quad \sum_{j=1}^d \theta_{y,j} = 1.
$$

Likelihood (ignoring multinomial coefficient):

$$
p(x \mid y) \propto \prod_{j=1}^d \theta_{y,j}^{\,x_j}.
$$

For Multinomial NB, the parameters are estimated directly from **feature counts** in the training data.

- The total count **n** for a sample is simply the sum of all feature counts:  
  $$n = \sum_{j=1}^d x_j.$$

- The parameter $\theta_{y,j}$ is the **relative frequency** of feature $j$ among all features that appear in class $y$:  
  $$
  \theta_{y,j} = 
  \frac{\text{count of feature } j \text{ across all samples in class } y}
       {\sum_{t=1}^d \text{count of feature } t \text{ in class } y}.
  $$

**Example**

Suppose we have text features (word counts) and two training samples belonging to class **y = spam**:

| Sample | feature_1 | feature_2 | feature_3 |
|--------|-----------|-----------|-----------|
| s₁     |     2     |     1     |     0     |
| s₂     |     3     |     0     |     1     |

### 1) Total count **n** for each sample  
For s₁:  
$$n_{s_1} = 2 + 1 + 0 = 3.$$

For s₂:  
$$n_{s_2} = 3 + 0 + 1 = 4.$$

(This is only used for the multinomial interpretation of each sample.)

---

### 2) Compute class-level feature totals  
Sum counts across all samples of class y:

| Feature | Total count in class y |
|---------|-------------------------|
| feature_1 | 2 + 3 = 5 |
| feature_2 | 1 + 0 = 1 |
| feature_3 | 0 + 1 = 1 |

Total count across all features in class y:

$$
\sum_{t=1}^3 \text{count}(t) 
= 5 + 1 + 1 = 7.
$$

---

### 3) Compute $\theta_{y,j}$ (relative frequencies)

$$
\theta_{y,j} = \frac{\text{count of feature } j \text{ in class } y}{7}
$$

So:

- Feature 1:  
  $$\theta_{y,1} = \frac{5}{7}$$
- Feature 2:  
  $$\theta_{y,2} = \frac{1}{7}$$
- Feature 3:  
  $$\theta_{y,3} = \frac{1}{7}$$

These become the **class-conditional probabilities** used in the likelihood:

$$
p(x \mid y) \propto 
\theta_{y,1}^{x_1}\,
\theta_{y,2}^{x_2}\,
\theta_{y,3}^{x_3}.
$$

---

## 🟩 3. **Bernoulli Naive Bayes (BNB)**  
Best for: **binary features** (present / absent).

Assume:

$$
x_j \mid y \sim \text{Bernoulli}(\theta_{y,j})
$$

Likelihood:

$$
p(x \mid y) = \prod_{j=1}^d
\theta_{y,j}^{x_j}
\,(1 - \theta_{y,j})^{1 - x_j}.
$$

This is useful for binary bag-of-words and “is this feature active?” data.

---

### Example  
Suppose we have 3 binary features and 3 training samples of class **y = positive**:

| Sample | f₁ | f₂ | f₃ |
|--------|----|----|----|
| a      |  1 |  0 |  1 |
| b      |  1 |  1 |  0 |
| c      |  0 |  0 |  1 |

Number of samples in this class:  
$$N_y = 3.$$

#### 1) Compute θ per feature

- Feature 1: appears in samples a, b → **2 out of 3**  
  $$\theta_{y,1} = \frac{2}{3}.$$

- Feature 2: appears only in sample b → **1 out of 3**  
  $$\theta_{y,2} = \frac{1}{3}.$$

- Feature 3: appears in a, c → **2 out of 3**  
  $$\theta_{y,3} = \frac{2}{3}.$$

#### 2) Likelihood for a new sample x = (1, 0, 1)

Using:

$$
p(x \mid y) =
\theta_{y,1}^{x_1}
(1 - \theta_{y,1})^{1 - x_1}
\cdot
\theta_{y,2}^{x_2}
(1 - \theta_{y,2})^{1 - x_2}
\cdot
\theta_{y,3}^{x_3}
(1 - \theta_{y,3})^{1 - x_3}.
$$

Plug in the numbers:

- $x_1=1$ → contributes $\theta_{y,1} = \frac{2}{3}$
- $x_2=0$ → contributes $(1-\theta_{y,2}) = \frac{2}{3}$
- $x_3=1$ → contributes $\theta_{y,3} = \frac{2}{3}$

So:

$$
p(x \mid y)
= \left(\frac{2}{3}\right)
  \left(\frac{2}{3}\right)
  \left(\frac{2}{3}\right)
= \left(\frac{2}{3}\right)^3.
$$

That's the likelihood used inside Bayes’ rule for class prediction.

---

## 🟨 4. **Categorical Naive Bayes (CNB)**  
Best for: **discrete features with multiple categories**  
(e.g., color = red/blue/green, weekday = 7 categories, etc.).

For feature $j$ with $K_j$ categories:

$$
p(x_j = k \mid y) = \phi_{y, j, k},
\qquad \sum_{k=1}^{K_j} \phi_{y, j, k} = 1.
$$

Likelihood:

$$
p(x \mid y) = \prod_{j=1}^d \phi_{y,j,x_j}.
$$

This is the most general discrete NB; Bernoulli and Multinomial are special cases.


---

## 🧠 Parameter Learning: MLE vs MAP for Naive Bayes

Naive Bayes parameter estimation often reduces to **counting**.  
But two approaches exist:

---

### ✅ MLE vs MAP From the View of Generative Modeling  
Recall generative classifiers compute:

$$
p(y \mid x) \propto p(y)\, p(x \mid y).
$$

The difference between **MLE** and **MAP** is simply:

> **How do we estimate the parameters inside \(p(y)\) and \(p(x\mid y)\)?**

---

### ✅ 1. **MLE — Use only the data (no prior)**
Everything is computed **purely from counts**.

- Class prior:
  $$
  p(y)^{\text{MLE}}
  = \frac{\text{\# samples with label } y}{N}.
  $$

- Likelihood parameters (example for discrete NB):
  $$
  \theta_{y,j}^{\text{MLE}}
  = \frac{\text{count}(x_j \text{ in class } y)}
         {\sum_{t} \text{count}(x_t \text{ in class } y)}.
  $$

MLE trusts the data **exactly as it is**.  
If a feature never appears in class $y$, MLE gives $\theta_{y,j}=0$, which can zero out the entire likelihood.

---

### ✅ 2. **MAP — Add a prior over parameters**
MAP modifies the same formulas by adding **prior belief** in the form of pseudo-counts.

### Class prior with MAP  
If we put a prior over class frequencies (Dirichlet prior for discrete labels):

$$
p(y)^{\text{MAP}}
= \frac{\text{\# samples with } y + \beta}
       {N + K\beta},
$$

where $K$ is number of classes.

#### Likelihood parameters with MAP  
(Example for Multinomial NB with Dirichlet prior)

$$
\theta_{y,j}^{\text{MAP}}
= \frac{\text{count}(x_j \text{ in class } y) + \alpha}
       {\sum_{t} [\text{count}(x_t \text{ in class } y) + \alpha]}.
$$

MAP **never assigns zero probability**, because even unseen features get the prior mass $\alpha$.

---

### ✅ 3. What Actually Differs?

| Component                                     | MLE | MAP                            |
|-----------------------------------------------|-----|--------------------------------|
| Estimating class prior \(p(y)\)               | Pure frequency | Frequency + prior pseudo-count |
| Estimating likelihood parameters $p(x\mid y)$ | Raw counts | Counts + smoothing ($\alpha$)  |
| Behavior for unseen features                  | **Zero probability** | **Nonzero probability**        |
| Interpretation                                | Trust data fully | Trust data + prior belief      |

---

### ✅ 4. Why MAP is Usually Preferred in Naive Bayes

Since classification uses:

$$
p(y \mid x) \propto p(y)\, p(x\mid y),
$$

even a **single zero** in $p(x\mid y$) makes $\;p(y\mid x)=0$ no matter how strong the other evidence is.

MAP smoothing fixes this by ensuring all probabilities remain valid and nonzero.

---

## ✅ Summary: Why Naive Bayes Variants Differ

| Variant | Suitable For | Distribution | Likelihood Form |
|--------|--------------|--------------|----------------|
| Gaussian NB | Continuous features | Normal | Linear in $(x-\mu)^2$ |
| Multinomial NB | Count features | Multinomial | Linear in log-counts |
| Bernoulli NB | Binary features | Bernoulli | Presence/absence product |
| Categorical NB | Discrete features | Categorical | Lookup tables |

Each handles a different type of $x$; the classifier form is identical apart from the likelihood.

---

