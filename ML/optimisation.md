# 🚀 Optimization Algorithms in Machine Learning

Optimization algorithms aim to **minimize the loss function** by updating model parameters efficiently during training.

In general, we improve the model by iteratively updating parameters using gradients:

$$
\theta \gets \theta - \eta \nabla_\theta \mathcal{L}(\theta)
$$

Where:
- $\theta$ — model parameters (weights)
- $\eta$ — learning rate
- $\nabla_\theta \mathcal{L}$ — gradient of the loss w.r.t. parameters

The choice of optimizer determines:
- **how fast** the model converges
- **how stable** training is
- **how well** the model generalizes

---

## Gradient Descent Variants

Gradient Descent algorithms optimize model parameters by moving them in the **negative direction of the gradient** — the direction of steepest descent.

General update rule:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Where:
- $\theta_t$ — parameters at iteration $t$
- $\eta$ — learning rate
- $\nabla_\theta \mathcal{L}$ — gradient of the loss

---

### Batch Gradient Descent (BGD)

#### Intuition
- Uses **all training data** to compute **one** gradient update per iteration
- Smooth & stable convergence

#### Update Rule
$$
\theta \gets \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta; \{x_i, y_i\}_{i=1}^m)
$$

Where $m$ is the total number of samples.

#### Pros
- Converges reliably for convex problems
- Gradient direction is accurate → smooth learning

#### Cons
- **Very slow** for large datasets
- High memory requirement
- Does not support online/streaming training

#### When to Use
- Small datasets
- Offline model training (e.g., linear regression on small tables)

---

### Stochastic Gradient Descent (SGD)

#### Intuition
- Update parameters **per sample**
- Introduces noise → helps escape local minima

#### Update Rule
$$
\theta \gets \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta; x_i, y_i)
$$

Only one $(x_i, y_i)$ chosen randomly each step.

#### Pros
- Very fast iteration time
- Good for large-scale & online learning
- Can escape shallow local minima

#### Cons
- **Highly noisy** updates → unstable convergence
- Requires careful tuning of learning rate

#### When to Use
- Streaming data
- Training deep networks with massive datasets

---

### Mini-Batch Gradient Descent (MBGD)

#### Intuition
- A compromise between BGD & SGD
- Compute gradients on a **small subset** (batch) of data

Batch of size $B$:
$$
\theta \gets \theta - \eta \cdot
\frac{1}{B}
\sum_{i=1}^{B}
\nabla_\theta \mathcal{L}(\theta; x_i, y_i)
$$

#### Pros
- Efficient computation using GPUs
- More stable than SGD
- Faster than full batch GD
- Supports vectorization → huge speed-ups

#### Cons
- Still some noise in updates
- Batch size is a hyperparameter that affects convergence

#### When to Use
- **Default choice** for deep learning and modern ML training

---

### Comparison Summary

| Method | Per-Update Compute Cost | Convergence | Memory | Notes |
|--------|------------------------|------------|--------|------|
| Batch GD | High (whole dataset) | Stable | High | Not scalable |
| SGD | Very low | Noisy | Very low | Good for very large / streaming data |
| Mini-Batch GD | Medium | Balanced | Medium | **Industry standard** |

---

#### Key Takeaways
- **Mini-batch GD** is used in almost all practical ML systems
- **SGD noise** helps explore loss landscape
- **Batch GD** rarely used beyond small datasets


---

## Momentum-Based Optimization

Momentum methods aim to **accelerate SGD** by accumulating a running direction of the gradient.  
This helps:
- Smooth out noisy updates
- Speed up convergence in shallow directions
- Prevent oscillation in steep/narrow valleys

We introduce a **velocity term** $v_t$ to store past gradients.

General momentum update:

$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_\theta \mathcal{L}(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

Where:
- $\beta \in [0,1)$ controls how much **past gradients** affect the current step (typically $\beta = 0.9$)

---

### Momentum

#### Intuition
Like rolling a ball downhill — **keeps moving in a consistent direction** even with noisy gradients.

#### Update Rule (Standard Form)
$$
v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

#### Pros
- Faster convergence, especially on narrow ravines
- Reduces oscillation
- Helps overcome small local minima

#### Cons
- May **overshoot** if $\beta$ is too large

#### When to Use
- Default improvement over vanilla SGD on most deep networks

---

### Nesterov Accelerated Gradient (NAG)

#### Key Idea
Look **ahead** to where momentum is about to take us, **then** compute gradient there.

Instead of computing $\nabla_\theta \mathcal{L}(\theta_t)$,
we compute at a **projected position**:

$$
\tilde{\theta} = \theta_t - \eta \beta v_{t-1}
$$

Update rule:

$$
v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\tilde{\theta})
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

#### Why It Helps
- Corrects the momentum **before** overshooting
- More responsive to curvature changes
- Faster theoretical convergence than standard momentum (in convex problems)

#### Pros
- Less oscillation near optimal point
- More accurate trajectory than vanilla momentum

#### Cons
- Slightly more computation (extra gradient at look-ahead point)

#### When to Use
- Commonly used in early deep learning (before Adam became standard)
- Good when loss curvature changes rapidly

---

### Visual Intuition

Momentum vs. NAG behavior:

- **Momentum**: pushes forward → checks → corrects later  
- **NAG**: checks ahead → corrects **before** moving

---

### Momentum Methods Summary

| Method | Lookahead? | Stability | Speed of Convergence | Notes |
|--------|------------|----------|---------------------|------|
| Momentum | ❌ | Medium | Fast | Good baseline |
| NAG | ✅ | High | Faster | More accurate trajectory |

---

#### Key Takeaways
- Momentum accumulates direction → speeds up learning
- NAG refines that direction → avoids overshooting
- Both are **strict upgrades over vanilla SGD**


---

## Adaptive Optimizers

Adaptive methods adjust the **learning rate for each parameter** based on:
- how frequently a parameter has been updated
- the magnitude of historical gradients

Goal:
- Faster convergence
- Reduce the need for manual LR tuning
- Handle sparse features better

---

### Adagrad

#### Intuition
- If a parameter receives **large gradients often** → learning rate decreases for that parameter
- If gradients are **rare** (e.g., sparse features) → learning rate stays high

Parameters associated with:
- common / frequent features (e.g., “the”, “and” in NLP) 
  - get updated a lot → LR ↓ → stabilize quickly
- rare / sparse features (e.g., “mitochondrial”, “quantization”)
  - rarely updated → LR stays high → learn faster

Adagrad gives bigger steps for infrequently occurring features and smaller steps for frequent features, which makes it well-suited for sparse, high-dimensional problems like NLP.

#### Update Rule

Accumulate **sum of squared gradients**:
$$
G_t = \sum_{\tau=1}^{t} g_\tau^2
$$

Parameter update:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t
$$

Where:
- $g_t = \nabla_\theta\mathcal{L}(\theta_t)$
- $\epsilon$ avoids division by zero

#### Pros
- Automatically reduces LR for frequently updated parameters
- Very good for **sparse data** (e.g., NLP, recommendation systems)

#### Cons
- $G_t$ grows without bound → learning rate shrinks too much → **training stalls**

#### When to Use
- Rarely used directly today, but historically important & used for sparse features

---

### RMSProp

#### Idea
Fix Adagrad’s shrinking LR problem by using **exponential moving average of squared gradients**:

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta)g_t^2
$$

Update:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t
$$

Typical $\beta = 0.9$

#### Pros
- Handles **non-stationary** loss surfaces well
- Good for RNNs and deep networks

#### Cons
- Still requires LR tuning
- Adaptive LR can cause poor generalization

---

### Adam (Adaptive Moment Estimation)

Adam combines the strengths of **Momentum** and **RMSProp**:

| Component | From | Purpose |
|----------|------|---------|
| 1st moment (mean) | Momentum | Smooths noisy gradients → stable direction |
| 2nd moment (variance) | RMSProp | Scales updates based on recent gradient magnitude |

So Adam **accelerates learning in low-curvature valleys** and **slows down in steep regions** — adapting to the loss landscape.

---

#### Moving averages of gradients

First moment → tracks the **direction**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Second moment → tracks the **magnitude**:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Where:
- $\beta_1$ controls “momentum smoothing”
- $\beta_2$ controls adaptation to recent gradient variance

---

#### Bias correction

Early in training, $m_t$ and $v_t$ are biased toward zero (not enough history yet).  
So we **de-bias** them:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

This gives unbiased estimates of mean & variance.

---

#### Update rule

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Interpretation:

| Term | What it does |
|------|--------------|
| $\hat{m}_t$ | Pushes update toward consistent gradient direction |
| $\sqrt{\hat{v}_t}$ | Shrinks learning rate when gradients fluctuate a lot |
| $\eta$ | Global LR scale |
| $\epsilon$ | Stability against division by zero |

---

#### Why Adam Works Well — Intuition

Adam makes its decisions based on **how gradients behave over time**:

| Gradient Behavior | What It Means | Adam’s Reaction |
|------------------|---------------|----------------|
| **Consistent gradients** (similar direction every step) | Model is heading the right way → shouldn't slow down | Take **bigger steps** (uses momentum effect from $\hat{m}_t$) |
| **Oscillating gradients** (jumping around, direction keeps changing) | Landscape is unstable / steep in some directions | Take **smaller steps** (because $\hat{v}_t$ becomes large) |
| **Sparse gradients** (rarely non-zero, e.g., NLP) | Some parameters only get occasional useful signal | Give **those parameters larger updates** |

Note both consistent gradients and oscillating gradients will have large v, but consistent gradients will have large m as well while oscillating graidents' m will cancel each other.

So Adam:
- trusts directions that keep working (momentum smooths noise)
- becomes careful when the surface is bumpy
- boosts learning where features are rare

In short:

> Adam automatically adjusts step sizes **per parameter**, based on how *reliable* and *frequent* the gradient signals are.

---

#### Pros
- **Industry standard** optimizer
- Faster convergence than SGD+Momentum
- Very little tuning required

#### Cons
- May converge to **sharp minima** → worse generalization
- Weight decay interacts incorrectly with adaptive scaling → see **AdamW**


---

### AdamW (Decoupled Weight Decay)

#### Problem (Adam original)

In theory, **L2 regularization** should penalize large weights:

$$
\mathcal{L}_{reg} = \frac{\lambda}{2} \| \theta \|^2
$$

In SGD, adding L2 effectively results in **weight decay**:

$$
\theta \gets \theta - \eta (\nabla_\theta \mathcal{L} + \lambda \theta)
$$

📌 This works because the gradient update and the L2 penalty are **scaled the same way** by the learning rate $\eta$.


But in **Adam**, the update is scaled *per-parameter* by:

$$
\frac{1}{\sqrt{\hat{v}_t} + \epsilon}
$$

So when you apply L2 inside Adam’s adaptive update, you get:

$$
\theta_{t+1} = \theta_t - \underbrace{\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{adaptive scaling}}
\left(\nabla_\theta \mathcal{L} + \lambda \theta_t \right)
$$

🚨 Problem:
- The **penalty term $\lambda \theta_t$** is also scaled by  
  $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$  
- Meaning **different parameters** get **different amounts** of “weight decay”
- Not actually performing true L2 regularization

So **L2 no longer equals weight decay**, and regularization becomes **inconsistent**.

#### AdamW Solution
Apply weight decay **separately**:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

→ leads to better generalization  
→ now **default for many DL frameworks**



---

### Adaptive Optimizers Summary

| Method | Memory | LR Adaptation | Best Use Case | Downsides |
|--------|--------|---------------|---------------|-----------|
| Adagrad | Low | Based on historical sum | Sparse features | LR shrinks too much |
| RMSProp | Medium | EMA of squared grads | RNNs / non-stationary | Hyperparam sensitivity |
| Adam | Medium-High | 1st + 2nd moments | Strong default | Sharp minima / generalization |
| AdamW | Medium-High | Decoupled weight decay | Modern deep learning | Slightly more tuning |

---

#### Key Takeaways
- **AdamW** is the recommended default for modern deep learning
- **RMSProp** is still great for sequence models (e.g., RNNs)
- **Adagrad** is good for sparse input features


---

## Learning Rate Strategies

Learning rate (LR) controls **how big** each step of gradient descent is.
A single fixed learning rate may:
- be **too large** early → oscillation or divergence (gradient can be large in begining as )
- be **too small** later → slow or stuck convergence

So we often **start with a larger LR** and **gradually reduce** it as training stabilizes.

---

### Learning Rate Scheduling

Schedulers **change the learning rate over time** based on:
- iteration number
- epoch count
- validation metrics
- cyclic patterns

Goal:
- **Fast learning** at the beginning
- **Fine-grained convergence** later

---

### Step Decay

Reduce learning rate **by a constant factor** every fixed number of epochs.

$$
\eta_t = \eta_0 \cdot \gamma^{\left\lfloor \frac{t}{k} \right\rfloor}
$$

Where:
- $\eta_0$ = initial LR  
- $\gamma$ = decay factor (e.g., 0.1)  
- $k$ = step interval (epochs)

#### Pros
- Very stable in practice
- Easy to implement

#### Cons
- Sudden LR drops → may cause convergence jumps
- Requires tuning decay schedule manually

#### Typical Use
- Classic CNNs (e.g., early ImageNet training)

---

### Exponential Decay

Learning rate decays **smoothly** every iteration or epoch:

$$
\eta_t = \eta_0 \cdot e^{-\lambda t}
$$

Where $\lambda$ controls decay speed.

#### Pros
- Smooth transition → fewer convergence shocks
- Good when training time is predictable

#### Cons
- Still hand-tuned
- Can decay too slowly/quickly depending on problem scaling

#### When to Use
- Problems where data is abundant & LR should reduce steadily

---

### Cosine Annealing / Warm Restarts

LR follows a **half cosine** curve:
- Starts high
- Decreases to a minimum
- (Optional) **resets** back up

Cosine schedule (no restart):

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})
\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

Where:
- $T$ = decay period
- $\eta_{min}, \eta_{max}$ define bounds

Warm restarts (SGDR):
- Reset LR to high value periodically
- Encourage escaping sharp minima
- Improve generalization

#### Pros
- Helps escape shallow/poor local minima
- Works very well with AdamW

#### Cons
- More hyperparameters (cycle length, min/max LR)
- Harder to tune manually

#### When to Use
- Modern deep learning training schedules (e.g., Transformers)
- Fine-tuning models

---

### Scheduler Comparison

| Strategy | Stability | Escaping Local Minima | Ease of Use | Notes |
|---------|-----------|----------------------|-------------|------|
| Step Decay | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Simple; abrupt changes |
| Exponential | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Smooth; predictable |
| Cosine Annealing | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Best performance in many DL models |

---

#### Key Takeaways
- **Schedulers are essential** to reach optimal performance
- Step decay is traditional; cosine annealing is the **modern go-to**
- Warm restarts improve **exploration + generalization**


---

## Stabilization Techniques

These methods prevent training from becoming unstable and help the model **generalize** rather than overfit.

---

### Gradient Clipping

#### Motivation
Sometimes gradients can **explode** — especially in:
- RNNs / LSTMs (due to long sequences)
- Deep networks with poor initialization
- Reinforcement learning

If gradients become too large → parameter updates become huge  
→ loss blows up → **training diverges**

#### Method
Clip gradients to a maximum allowed norm:

$$
g_t \gets g_t \cdot \min\left(1, \frac{\tau}{\|g_t\|}\right)
$$

Where:
- $\tau$ = clipping threshold

📌 If $\|g_t\| > \tau$ → scale it down  
📌 If $\|g_t\| \le \tau$ → leave unchanged

#### Pros
- Prevents exploding gradients
- Improves training stability significantly

#### Cons
- Adds another hyperparameter (clip norm)
- Doesn’t fix vanishing gradients

#### When to Use
- RNNs / Transformers
- Deep reinforcement learning
- Very deep models with unstable gradients

---

### Early Stopping

#### Motivation
After a point, continuing to train lowers training loss  
but **validation loss starts rising** → overfitting.

Early stopping stops training when validation performance **stops improving**:

#### Implementation

Typical strategy:
1. Track validation loss (or metric)
2. If no improvement for `patience` epochs → **stop training**
3. Restore **best** model weights

Formal condition:

$$
\text{if } \mathcal{L}_{val}(t) > \min_{i < t} \mathcal{L}_{val}(i) \text{ for } p \text{ epochs} \Rightarrow \text{Stop}
$$

Where:
- $p$ = patience parameter

#### Pros
- Prevents **overfit**
- Saves training time/resources
- Requires no change in model or optimizer

#### Cons
- Requires a validation metric
- May stop **slightly early** if metric is noisy

#### When to Use
- Any model prone to overfitting (deep nets, high-dimensional data)
- Small / medium datasets where noise is high

---

### Stabilization Summary

| Technique | Main Goal | Best For | Addresses |
|----------|-----------|----------|----------|
| Gradient Clipping | Prevent divergence | RNNs, Transformers | Exploding gradients |
| Early Stopping | Improve generalization | Any deep model | Overfitting |


---

## Weight Initialization Strategies

Good weight initialization helps:
- Avoid **vanishing gradients** (signals become too small)
- Avoid **exploding gradients** (signals blow up)
- Speed up convergence
- Improve model stability

Key idea:
> Maintain the **variance** of activations and gradients **throughout layers**
so the signal neither shrinks nor grows as it travels through the network.

---

### Xavier Initialization (Glorot Initialization)

Designed for: **tanh / sigmoid** activations  
Goal: keep forward and backward **signal variance constant**

Variance formula:

$$
Var(w) = \frac{1}{\frac{n_{in} + n_{out}}{2}} = \frac{2}{n_{in} + n_{out}}
$$

Where:
- $n_{in}$ = number of inputs to the neuron
- $n_{out}$ = number of outputs from the neuron

Two common versions:

Uniform:
$$
w \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, 
               \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

Normal:
$$
w \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
$$

#### Pros
- Stable initialization for sigmoid/tanh networks
- Prevents saturation early in training

#### Cons
- Still prone to vanishing gradients in deeper nets
- Not ideal for ReLU-based networks

#### When to Use
- Legacy models or shallow networks using sigmoid/tanh

---

### He Initialization (Kaiming Initialization)

Designed for: **ReLU** and similar activations  
ReLU zeroes negative values → halves effective variance → needs compensation

Variance:

$$
Var(w) = \frac{2}{n_{in}}
$$

Uniform:
$$
w \sim U\left(-\sqrt{\frac{6}{n_{in}}}, 
               \sqrt{\frac{6}{n_{in}}}\right)
$$

Normal:
$$
w \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

#### Pros
- Works extremely well with modern ReLU-based deep networks
- Helps avoid vanishing gradients
- Faster convergence than Xavier for ReLU models

#### Cons
- Not suitable for activations that squash both sides (sigmoid/tanh)

#### When to Use
- **Default choice** for deep CNNs / Transformers using ReLU variants

---

### Initialization Strategy Summary

| Activation | Best Init | Reason |
|-----------|-----------|--------|
| Sigmoid / Tanh | Xavier | Balanced fan-in/out variance |
| ReLU / Leaky ReLU | He | Accounts for half activations being zero |
| GELU / Swish | He (often) | Similar behavior to ReLU |

---

#### Key Takeaways
- Poor initialization → exploding/vanishing gradients → failed training
- **He init** is standard for deep ReLU models
- **Xavier** is good for older sigmoid/tanh networks


---

## Regularization in Optimization

Regularization helps improve **generalization** by preventing the model
from fitting noise or memorizing training samples.

It works by **penalizing complexity**, keeping parameters small and stable during optimization.

---

### Weight Decay (L2 Regularization)

#### Motivation
Large weights make models:
- high variance
- sensitive to small input changes
- prone to overfitting

L2 regularization adds a penalty term to the loss:

$$
\mathcal{L}_{total} = \mathcal{L}_{data} + \frac{\lambda}{2} \|\theta\|^2
$$

This encourages weights to shrink toward zero.

---

#### Effect on Optimization

Gradient update with L2 (in SGD form):

$$
\theta_{t+1} = \theta_t - \eta(\nabla_\theta \mathcal{L}_{data} + \lambda \theta_t)
$$

Rearranged:

$$
\theta_{t+1} = (1 - \eta \lambda)\theta_t - \eta \nabla_\theta \mathcal{L}_{data}
$$

So every update **decays** weights slightly:

- $\lambda = 0$ → no decay  
- Larger $\lambda$ → stronger weight shrinkage

This is why it’s called **weight decay**.

---

#### Benefits

- Reduces model complexity
- Improves generalization
- Prevents runaway parameter growth
- Works well with deep learning architectures

---

#### When Used with Adam (Important!)

- Adam **mixes** weight decay with adaptive scaling → L2 becomes inconsistent
- This is why **AdamW** decouples weight decay for correct behavior

📌 Weight decay ≠ L2 penalty inside Adam  
📌 **Always prefer AdamW** over Adam + L2 for deep networks

---

#### When to Use
- Default regularization technique for:
  - large deep learning models
  - high-dimensional feature spaces
  - settings with limited data (overfitting risk)

---

#### Summary
> Weight decay pushes weights to remain small  
> → smoother decision boundaries  
> → better ability to generalize to unseen data

