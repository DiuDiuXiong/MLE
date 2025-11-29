# 🧪 Evaluation in Machine Learning

---

## Statistical Fundamentals

### Bias & Variance

In supervised learning, our model makes predictions $\hat{y}$ for true targets $y$.  
The total prediction error decomposes into **Bias**, **Variance**, and **Irreducible Noise**.

#### 🔹 Bias
Bias measures how much the model’s **expected prediction** differs from the true function.

$$
\text{Bias}(x) = \mathbb{E}[\hat{f}(x)] - f(x)
$$

- High bias → model too simple → **underfitting**
- Example: Linear model trying to fit a curved non-linear function

Bias is about: **how wrong are we on average?**

#### 🔹 Variance
Variance measures how much model predictions **fluctuate** across different training sets.

$$
\text{Var}(x) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

- High variance → model too sensitive to data → **overfitting**
- Example: Deep tree memorizing training noise

Variance is about: **how inconsistent are we?**

---

#### Bias–Variance Tradeoff

We aim to understand the total expected squared prediction error:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big]
$$

Assume the true data-generating process:

$$
y = f(x) + \epsilon,
\quad \mathbb{E}[\epsilon] = 0,
\quad \mathbb{E}[\epsilon^2] = \sigma^2
$$

Substitute into the error:

$$
\mathbb{E}\big[(f(x) + \epsilon - \hat{f}(x))^2\big]
=
\mathbb{E}\big[(f(x)-\hat{f}(x))^2\big] + \sigma^2
$$

Now, let the **expected model prediction** across different training samples be:

$$
\bar{f}(x) = \mathbb{E}[\hat{f}(x)]
$$

Insert and expand using:
$$
f(x)-\hat{f}(x) = 
\big(f(x)-\bar{f}(x)\big) + \big(\bar{f}(x)-\hat{f}(x)\big)
$$

Expand the squared term:

$$
\mathbb{E}\big[(f(x)-\hat{f}(x))^2\big]
=
(f(x)-\bar{f}(x))^2 + \mathbb{E}\big[(\hat{f}(x)-\bar{f}(x))^2\big] +
2(f(x)-\bar{f}(x))\mathbb{E}[\bar{f}(x)-\hat{f}(x)]
$$

The cross-term disappears since:
$$
\mathbb{E}[\bar{f}(x)-\hat{f}(x)] = 0
$$

Thus we obtain the **Bias–Variance decomposition**:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] =
\underbrace{(f(x) - \bar{f}(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x)-\bar{f}(x))^2]}_{\text{Variance}}+
\underbrace{\sigma^2}_{\text{Irreducible noise}}
$$

Where:

- **Bias** = systematic error from simplifying assumptions
- **Variance** = model sensitivity to training data
- $\sigma^2$ = noise in data that no model can eliminate

> A model must balance Bias ↔ Variance to minimize expected error. 🎯


| Model Type | Bias | Variance |
|-----------|------|----------|
| Linear model | High | Low |
| kNN (small k) | Low | High |
| Deep Neural Net | Low | High |
| Tree with max depth | Low | High |
| Regularized model (L2) | Higher | Lower |

---

#### Interview-Ready Insights

- **Bias is about model assumptions**  
  If assumptions too strong → systematic underfitting
- **Variance is about model sensitivity**  
  Small change in training → large prediction change
- **Regularization increases bias but lowers variance**
- **Cross-validation helps estimate variance across samples**
- Ensemble methods like **bagging** 🚀 reduce variance  
- Regularization methods like **LASSO / Ridge** ⚖️ reduce variance by increasing bias
- You can **never** eliminate the noise term — it’s fundamental

---

#### Pro Tip Examples to Say in Interviews

> Decision trees: low bias, high variance → use bagging / random forests  
> Linear regression: high bias, low variance → add polynomial features  
> Neural networks: can have low bias, but require regularization to tame variance

---

#### Quick Mental Model

> **Bias = what your model can’t learn**  
> **Variance = what your model shouldn’t learn (noise)**

A good model is the balance in the middle. 🎯

### Hypothesis Testing Basics

#### Null / Alternative Hypothesis
- **Null hypothesis ($H_0$):** Default assumption (no effect, no difference)
- **Alternative hypothesis ($H_1$):** What we want to detect (effect exists)

Example in ML:  
$H_0$: Model A accuracy = Model B accuracy  
$H_1$: Model A accuracy > Model B accuracy  

Hypothesis tests evaluate whether data is **too unlikely** under $H_0$.

---

#### Type I / Type II Errors
| Error | Meaning | Risk |
|-------|---------|------|
| **Type I ($\alpha$)** | Reject $H_0$ when it’s actually true | False positive |
| **Type II ($\beta$)** | Fail to reject $H_0$ when $H_1$ is true | False negative |

In ML:  
- Type I → conclude a model is better when it’s actually not  
- Type II → miss a truly better model

---

#### p-values
Probability of observing data **at least as extreme as ours** *assuming $H_0$ is true*:

$$
p = P(\text{data or more extreme} \mid H_0 \text{ true})
$$

- Small $p$ → data inconsistent with $H_0$ → **reject $H_0$**
- $p$ does **not** tell probability $H_0$ is true
- $p$ is the range pdf sum

---

#### Statistical Power
Probability of **correctly** rejecting $H_0$:

$$
\text{Power} = 1 - \beta
$$

Higher power when:
- Effect size bigger
- Variability lower
- More data

In ML: **more test data** increases power in model comparisons.

---

---

### Distributions & Approximations

#### z-distribution
- Standard Normal: mean 0, variance 1
- Used when **sample large** or **population variance known**
- Critical value for 95% CI: **1.96** (for N(0,1) distribution, value fall outside of +- 1.96 is 5%, 2.5% on each side)

#### t-distribution
- Similar to Normal, **heavier tails**
- Used when **small sample** or **variance unknown**
- Degrees of freedom $df=n-1$
- Approximates Normal as $n$ increases

#### Chi-square distribution

If you take **$k$ independent** standard Normal variables:

$$
Z_1, Z_2, \dots, Z_k \sim N(0,1)
$$

Then the **sum of their squares** follows a Chi-square distribution:

$$
\chi^2 = \sum_{i=1}^k Z_i^2 \quad \sim \quad \chi^2(k)
$$

Where **$k$ = degrees of freedom**.

Key intuition:
- More degrees of freedom → more skewed right → approaches Normal gradually
- It measures: **how much squared difference we observe**

Used for:
- **Variance tests** (does sample variance match expectation?)
- **Goodness-of-fit** (does observed distribution match expected?)
- **Categorical independence test** (e.g., confusion matrix independence)

Example (Independence Test):

$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Large $\chi^2$ → observed differs strongly from expected → reject independence.

---

#### F-distribution (ANOVA)

Constructed from the **ratio** of Chi-square distributions (normalized by their dfs):

$$
F = \frac{(X_1 / k_1)}{(X_2 / k_2)} \quad \text{where } X_1 \sim \chi^2(k_1),\; X_2 \sim \chi^2(k_2)
$$

So:

- **Numerator variance estimate**
- **Denominator variance estimate**
- Compare how much larger one variance is vs another

Used for:
- **ANOVA**: test if **multiple group means** have a significant difference  

Hypotheses in ANOVA:
- $H_0$: all group means equal  
- $H_1$: at least one group differs  

F-statistic in ANOVA:

$$
F = 
\frac{\text{Between-group variance}}{\text{Within-group variance}}
$$

If groups differ a lot → numerator ↑ → **large F** → reject $H_0$

Also used in:
- **Regression model comparison**
- Feature significance tests (full model vs reduced model)
---

---

### Standard Error & Central Limit Theorem

Standard Error (SE) measures **uncertainty** of an estimator:

$$
\text{SE}(\bar{x}) = \frac{s}{\sqrt{n}}
$$

Central Limit Theorem (CLT):
> The sampling distribution of the mean approaches **Normal** as sample size grows — regardless of the population distribution.

ML implication:  
We can use **Normal-based** approximations for CI + hypothesis tests when $n$ large enough.

---

---

### Bootstrapping / Resampling

Non-parametric method:
1. Resample dataset **with replacement**
2. Compute the metric for each bootstrap sample
3. Use distribution of estimates to measure uncertainty

Why useful in ML:
- No assumption about Normality
- Works with **any** metric: accuracy, F1, AUC, etc.
- Great for **model evaluation** confidence intervals

> Resampling is the default trick when theory doesn’t give a neat formula.



---

## ML Evaluation Metrics

### Classification Metrics

#### Confusion Matrix

All fundamental metrics derive from this 2×2 table:

|                | Pred Positive | Pred Negative |
|----------------|---------------|---------------|
| **Actual Pos** | TP            | FN            |
| **Actual Neg** | FP            | TN            |

Key intuition:
- **TP** good prediction
- **FP** false alarm (over-warn)
- **FN** missed detection (dangerous in critical domains)

---

#### Accuracy

$$
\text{Accuracy} =
\frac{TP + TN}{TP + FP + TN + FN}
$$

Good when:
- **Classes balanced**
- FP & FN equally costly

Misleading when:
- **Imbalanced classes**
  → Predict everything majority class → high accuracy, useless model

---

#### Precision

Among predicted positives → how many actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

High precision → model is **careful** with positive predictions  
Low precision → many false alarms

Used when:
- **FP cost is high** (e.g., fraud alerts, medical positive)

---

#### Recall (Sensitivity / TPR)

Among actual positives → how many did we find?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

High recall → model **rarely misses** positives  
Low recall → many missed detections

Used when:
- **FN cost is high** (e.g., disease detection)

---

#### F1-score

Harmonic mean of Precision & Recall:

$$
\text{F1} =
2 \cdot \frac{\text{Precision} \cdot \text{Recall}}
{\text{Precision} + \text{Recall}}
$$

Balances both:
- High only if **both** P and R high
- Great for **imbalanced** tasks

> Interview tip: “Use F1 when *both* FP & FN matter and data is skewed.”

---

#### ROC Curve & AUC

ROC plots:

- x-axis: **FPR** = $\frac{FP}{FP+TN}$ (among all negative cases, how many marked as positive)
- y-axis: **TPR (Recall)** (among all positive cases, how many marked as positive)

Vary classification threshold → get curve

AUC-ROC:
> Probability model ranks a random positive higher than a random negative

Interpretation:
- **0.5** → random guessing
- **1.0** → perfect ranking
- Better when **class distribution balanced**

Weakness:
- Overly optimistic for **highly imbalanced** data (if too many positive cases, y axis can be easily up)

---

#### PR Curve & AUC (Average Precision)

Precision vs Recall curve

Better suited for skewed data:
- Focuses only on **positive class**
- Penalizes wrong positives in rare-event tasks

AUC-PR:
> High when model keeps precision high while increasing recall

ROC high + PR low → model **fails on rare positives**

#### Scenario
Dataset:
- 10,000 samples
- 100 positive cases (**1% positives**)
- 9,900 negative cases

Model predicts:
- Almost every sample as negative (low recall)
- But the few times it predicts positive → they are indeed positive (high precision locally)
- Ranking score separates positives *somewhat* from negatives

So the ROC looks strong:
- TN very high → **FPR extremely low**
- Even slight lift in TPR makes ROC curve look great  (even if we have a lot of FPs, since TN too large, FPR still small)
- → **AUC-ROC ≈ 0.95**

But PR curve collapses:
- Because Precision = $\frac{TP}{TP+FP}$  
- And positives are **rare**, so:

If model finds:
- 10 positives (TP = 10)
- But also 100 false alarms (FP = 100)

Then:
- Precision = $10/(10+100) = 9.1\%$ → **very bad**
- Recall = $10/100 = 10\%$ → **also bad**
- PR AUC will be **close to baseline** ≈ 1%

---

#### Log Loss (Cross Entropy)

Measures confidence of probabilistic predictions:

$y_i$ is binary.
$$
\text{LogLoss}
= - \frac{1}{N} \sum_{i=1}^N
\big[
y_i \log \hat{p}_i +
(1-y_i)\log(1-\hat{p}_i)
\big]
$$

- Punishes **overconfident wrong** predictions heavily
- Sensitive to calibration

Examples:
- Predict 0.99 but wrong → **huge penalty**
- Predict 0.51 correct → small reward

Great metric for:
- **Model probability quality**
- **Binary classification with ranking**

---

#### Calibration: Brier Score / Reliability Curve

Brier Score (for probabilities):

$$
\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2
$$

- Measures **how close** predicted probabilities are to true outcomes  
- Lower is better (perfect calibration = 0)

Reliability Curve:
- Plot predicted probability vs actual frequency
- Model is **well calibrated** if curve ≈ diagonal

Example:
- Among samples with $\hat{p}=0.7$, ~70% should be actually positive

Use cases:
- Weather forecasting
- Risk/decision-sensitive applications

> Good classification ≠ Good calibration  
> E.g., deep nets often confident but **miscalibrated**

---

### How These Metrics Behave (Interpretation Guide)

| Metric | Good Situation | Bad Situation | Primary Concern |
|--------|----------------|---------------|----------------|
| Accuracy | Balanced classes, similar FP/FN cost | Highly imbalanced | Overall correctness |
| Precision | Strong TP, few FP | Many false alarms | False positive control |
| Recall | Almost all positives captured | Many positives missed | False negative control |
| F1 | Balanced P & R | One dominates the other | Combined performance |
| AUC-ROC | Good class separability | Large class skew | Ranking ability |
| AUC-PR | Rare positives ranked well | ROC misleading | Focus on positive class |
| Log Loss | Well-calibrated probabilities | Overconfidence | Probability quality |
| Brier | Probabilities match outcomes | Output too extreme | Calibration strength |

---


### Regression Metrics

#### MSE / RMSE / MAE

**Mean Squared Error (MSE)**

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

- Penalizes **large errors more heavily** (square term)
- Smooth derivative → great for **gradient-based training**

**Root MSE (RMSE)**

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

- Same units as target → easier to interpret
- Still sensitive to **outliers**

**Mean Absolute Error (MAE)**

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$

- Robust to outliers (linear penalty)
- But gradient **constant** almost everywhere → slower optimization

| Metric | Penalizes big errors? | Good when | Bad when |
|--------|---------------------|-----------|----------|
| MSE / RMSE | **Yes (square)** | want large errors to matter | outliers distort evaluation |
| MAE | No | outliers exist | gradient optimization |

---

#### R² Score (Coefficient of Determination)

Measures improvement over simply predicting the **mean**:

$$
R^2 = 1 -
\frac{\sum (y_i - \hat{y}_i)^2}
     {\sum (y_i - \bar{y})^2}
$$

Interpretation:
- **1.0** → perfect prediction
- **0.0** → no better than baseline mean predictor
- **Negative** → **worse** than predicting the mean 😬

Important:  
Can be misleading if:
- outliers dominate
- nonlinear patterns not captured
- evaluation outside training distribution

---

#### MAPE / SMAPE

**Mean Absolute Percentage Error (MAPE)**

$$
\text{MAPE} = \frac{100}{N} \sum_{i=1}^N
\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

- Unit-free
- Common for time series forecasting

Drawbacks:
- Undefined when $y_i = 0$
- Over-penalizes small true values (Because the denominator is $y_i$, if the true value is very small, even a tiny prediction error creates a huge percentage error.)


---

**Symmetric MAPE (SMAPE)**

Fixes denominator bias:

$$
\text{SMAPE} =
\frac{100}{N} \sum_{i=1}^N
\frac{|y_i - \hat{y}_i|}
{\left(|y_i| + |\hat{y}_i|\right)/2}
$$

Better symmetry when $y_i$ close to zero  
Still sensitive to noise when both $y$ and $\hat{y}$ small

> Interview tip: Use MAPE only if **no zeros** and scale-free comparison needed.

---

#### Quantile Loss (Pinball Loss)

Used when predicting **conditional quantiles** (median, 90th percentile, etc.)

$$
L_\tau(y, \hat{y}) =
\begin{cases}
\tau (y - \hat{y}), & y \ge \hat{y} \\
(1 - \tau)(\hat{y} - y), & y < \hat{y}
\end{cases}
$$

Where $\tau \in (0,1)$:
- $\tau = 0.5$ → **median regression**
- $\tau = 0.9$ → high-percentile “risk-aware” prediction

Intuition:
- Penalizes **underestimation vs overestimation differently**
- Powerful when **distribution asymmetry** matters  
(e.g., risk, revenue, latency SLA — *better overestimate than miss?*)

> Big deal in time series forecasting — uncertainty estimation & interval forecasts.

---

### When to Use Which (Quick Guide)

| Scenario | Best Metric |
|---------|-------------|
| Outliers expected | MAE |
| Large errors costly | RMSE |
| Benchmark against naive mean | R² |
| Scale-free business errors | MAPE/SMAPE |
| Quantile predictions needed (risk/statistics) | Quantile Loss |

---

### Summary

- **MSE encourages small variance** but punishes outliers hard  
- **MAE robust to outliers** but slower to optimize  
- **R² can be negative** and is not absolute quality  
- **MAPE can explode** near zero  
- **Quantile loss** tunes error asymmetry for business goals



### Ranking & Retrieval Metrics

Used when the model outputs a **ranked list** of items (e.g., search, recommender systems, classification with Top-K).

---

#### Top-K Accuracy

$$
\text{Top-}K = \frac{\text{# samples where true label in top } K}{N}
$$

Example: Image classification with 1000 classes  
Top-5 accuracy = % of images where the **correct class** appears anywhere in the **top 5** predictions.

Good for:
- Multi-class where **top prediction might be slightly wrong**
- UX-like success: “user saw the right item somewhere up top”

Limitation:
- Doesn’t care **rank** within top K
- Ignores score quality

---

#### MAP — Mean Average Precision (Intuition)

MAP is used when there are **multiple relevant items** and **ranking matters**.

For one query, Average Precision (AP) is:

$$
\text{AP} = 
\frac{1}{\text{# relevant}}
\sum_{k=1}^N \text{Precision@}k \cdot \mathbb{1}(\text{relevant at } k)
$$

**What this formula means:**
- Scan down the ranked list position-by-position
- Whenever you hit a relevant item at rank $k$:
  - Calculate **Precision@k**
  - Add it to the total
- Divide by the total number of relevant items to average

So **each relevant item earns a score** based on how early it appears.

---

##### Example (AP Calculation)

Ranked results for a query:

| Rank $k$ | Relevant? | Precision@k | Add to AP? |
|:-------:|:---------:|:-----------:|:----------:|
| 1 | ❌ | 0/1 = 0 | No |
| 2 | ✔️ | 1/2 = 0.50 | Yes |
| 3 | ✔️ | 2/3 ≈ 0.67 | Yes |
| 4 | ❌ | 2/4 = 0.50 | No |
| 5 | ✔️ | 3/5 = 0.60 | Yes |

There are 3 relevant items:

$$
\text{AP} = \frac{0.50 + 0.67 + 0.60}{3} ≈ 0.59
$$

If the same 3 relevant items appeared at ranks 50, 80, 100 → AP becomes **very low**.

> AP rewards **finding all relevant items early**.

---

##### From AP to MAP

Multiple queries (like multiple users/search terms):

$$
\text{MAP} = \frac{1}{Q} \sum_{q=1}^Q \text{AP}_q
$$

MAP = **mean ranking quality across queries**.

---

##### Quick Interpretation

| Concept | Meaning |
|--------|---------|
| High MAP | Relevant items consistently near the top |
| Low MAP | Model finds positives too late (or misses many) |
| Focus | Ranking quality for **all** positives |

> ROC cares about overall ranking.  
> MAP cares about **early** ranking of **relevant** positives.


---

#### MRR — Mean Reciprocal Rank

Focuses on the **first** correct result only.

For each query:

$$
\text{RR} = \frac{1}{\text{rank of first relevant item}}
$$

Then:

$$
\text{MRR} = \frac{1}{Q} \sum_{q=1}^Q \text{RR}_q
$$

Example:
- First hit at rank 1 → 1  
- Rank 5 → 0.2  
- No hits → 0

Used when:
- **Only first correct answer matters**  
(e.g., search engines, QA systems)

---

#### NDCG — Normalized Discounted Cumulative Gain (Expanded Intuition)

NDCG evaluates the **quality of a ranked list**, rewarding:
- Higher relevance levels
- Earlier placement in the ranking

Useful when:
- Relevance is **graded** (e.g., relevance=0/1/2/3, you have the answer for ur relevance, ur just evaluate the model)
- Users focus mainly on **top results**

---

##### Discounted Cumulative Gain (DCG)

We want to **discount items appearing lower** in the list:

$$
\text{DCG@}k = 
\sum_{i=1}^k 
\frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

- Rank 1: weight = 1 (no discount)  
- Rank increases → denominator grows → contribution smaller  
- Later results matter **much less**

---

##### Why $2^{rel_i} - 1$?

Relevance weights become **non-linear**, so higher relevance is worth *much* more:

| Relevance Level | Gain $(2^{rel}-1)$ |
|----------------:|-------------------:|
| 0 |                  0 |
| 1 |                  1 |
| 2 |                  3 |
| 3 |                  7 |

> ⭐⭐⭐ is **far** more valuable than ⭐⭐

---

##### Normalizing to NDCG

Divide DCG by the **best possible** DCG for that query, which is if we sort query top k by the graded relevancy we have:

$$
\text{NDCG@}k = 
\frac{\text{DCG@}k}{\text{IDCG@}k}
$$

Range:
- **1.0** → perfect ranking  
- **0.0** → worst ranking

This normalization allows fair comparison across:
- queries  
- datasets  
- models

---

##### Tiny Example

Actual relevance ranking:  
[3, 0, 2, 1]

Ideal ranking:  
[3, 2, 1, 0]

Compute DCG@4:

$$
7 + 0 + \frac{3}{2} + \frac{1}{\log_2(5)} ≈ 8.76
$$

Compute IDCG@4:

$$
7 + \frac{3}{\log_2(3)} + \frac{1}{\log_2(4)} + 0 ≈ 8.99
$$

Final:

$$
\text{NDCG@4} = \frac{8.76}{8.99} \approx 0.97
$$

Very close to ideal → **good ranking**.

---

##### Interview Summary

- Better than MAP when relevance is **multi-level**
- Better than ROC when **top-ranked results matter**
- Great for **search, personalization, recommendations**

> “NDCG measures *how well you prioritize the most relevant items* — not just whether you retrieve them.”


---

### How These Behave (Interpretation)

| Metric | What It Measures | Best For | Weakness |
|--------|-----------------|----------|----------|
| Top-K Accuracy | Is the answer in top K? | Multi-class classification | No ordering sensitivity |
| MAP | Overall ranking of multiple positives | Search, recommendation | Expensive & needs all labels |
| MRR | First correct hit | Q&A, navigation | Ignores all later hits |
| NDCG | Ranking + graded relevance | Personalized search | Requires relevance grades |

---

### Interview Summary Lines

- **Top-K**: “Did the model include the correct answer?”
- **MRR**: “How high was the first good answer?”
- **MAP**: “How consistently early are all good answers?”
- **NDCG**: “NDCG measures how well we order the results — especially at the top — given their relevance levels.”



### Probabilistic / Uncertainty Metrics

#### Calibration

- The model should produce probabilities (not just class labels or arbitrary scores)
- Applies to binary or multi-class classification — as long as we can check correctness
- To evaluate calibration, group predictions with similar confidence (e.g., 0.88–0.92) -> 0.91 
- The actual fraction of correct predictions in that group should match the predicted probability

#### Calibration Curves

A model is **calibrated** if predicted probabilities reflect true frequencies.

Example:
- Among samples where model predicts **70%** probability of positive,
  **~70%** should actually be positive around similar item.

To visualize this:
- Bin predictions into ranges like [0.0–0.1], [0.1–0.2], ...
- For each bin:
  - x-axis: mean predicted probability
  - y-axis: actual empirical frequency (accuracy of those selected item correspond to those predicted value)

A **perfectly calibrated** model:
- Curve matches the **diagonal**
- Predictions match reality at every confidence level

If curve is above diagonal → model **under-confident**  
If curve is below diagonal → model **over-confident**

---

#### Expected Calibration Error (ECE)

A single number that summarizes calibration:

1. Partition predictions into **M bins**
2. Compute expected true frequency vs predicted probability per bin
3. Weighted average of absolute differences:

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N}
\cdot \left|
\text{acc}(B_m) - \text{conf}(B_m)
\right|
$$

Where:
- $B_m$ = samples in bin $m$
- $\text{acc}(B_m)$ = actual proportion of positives in bin
- $\text{conf}(B_m)$ = average predicted probability

Interpretation:
- **Lower ECE = better calibrated**
- Common with deep learning since neural nets **overestimate confidence**

---

#### Sharpness vs Calibration Tradeoff

Two independent quality dimensions of probabilistic predictions:

| Property | Meaning | Goal |
|---------|---------|-----|
| **Calibration** | Probabilities match real outcomes | Well calibrated |
| **Sharpness** | Predictions are confident (far from 0.5) | High sharpness |

A model can be:
- **Calibrated but not sharp**  
  → Always predicting ~0.5 (boring & not useful)
- **Sharp but poorly calibrated**  
  → Extreme probabilities but often wrong (**dangerous**)

Goal:
> Maximize **sharpness** while staying **well-calibrated**

Example domains where calibration matters:
- Medical risk scoring
- Fraud detection
- Weather forecasting
- Any decision with **cost** tied to confidence

---

##### Interview takeaway

- **Accuracy ≠ probability correctness**
- Use **ECE + Reliability Curve** to audit confidence quality
- Sharpness without calibration is **overconfidence**
- Calibration without sharpness is **indecision**


### Model Comparison

#### AIC / BIC (Information Criteria)

Evaluate **how well a model fits** vs **how complex it is**.

General trade-off:
> More parameters → better fit → higher risk of overfitting  
> AIC/BIC penalize complexity

AIC:
$$
\text{AIC} = 2k - 2\ln(\hat{L})
$$

BIC:
$$
\text{BIC} = k\ln(n) - 2\ln(\hat{L})
$$

Where:
- $k$ = number of parameters
- $\hat{L}$ = maximized likelihood
- $n$ = number of samples

Interpretation:
- **Lower AIC/BIC = better** model balance
- **BIC penalizes complexity more** than AIC (due to \(\ln(n)\))

Good for:
- Comparing **statistical models** on same dataset
- Regression / probabilistic models (not neural nets typically)

---

#### Statistical Significance Testing for Metrics

Raw metric differences (e.g. accuracy = 0.91 vs 0.90)  
might be **noise** — not a real improvement.

Significance testing asks:
> “Is Model A truly better than Model B or is it chance?”

Approach depends on:
- **paired** predictions?
- **continuous** vs **binary** metric?

---

#### McNemar’s Test (paired classification comparison)

Used when **same samples** evaluated by two classifiers → paired binary outcomes.

We build a **2×2 contingency** of disagreements:

|                | B correct |  B wrong |
|----------------|----------:|---------:|
| A correct |  $n_{10}$ | $n_{11}$ |
| A wrong |  $n_{01}$ | $n_{00}$ |

Core focus:
- Cases where **A correct / B wrong** vs **A wrong / B correct**

Test statistic (approx.):
$$
\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}
$$

If p-value small → **performance difference is significant**.

Why useful:
- Directly tests **error disagreement**, not just accuracy averages

---

#### Wilcoxon Signed-Rank Test / Paired t-test — Worked Examples

We have two models (A and B), evaluated with 5-fold CV.  
We look at **per-fold metric differences** $d_i = A_i - B_i$.

---

##### Example 1 – A is slightly better on *every* fold

| Fold | Model A F1 | Model B F1 | Difference $d_i = A_i - B_i$ |
|-----:|-----------:|-----------:|-----------------------------:|
| 1    | 0.80       | 0.75       | +0.05                        |
| 2    | 0.78       | 0.76       | +0.02                        |
| 3    | 0.79       | 0.78       | +0.01                        |
| 4    | 0.82       | 0.81       | +0.01                        |
| 5    | 0.77       | 0.76       | +0.01                        |

So the differences are:
- $d = [0.05,\; 0.02,\; 0.01,\; 0.01,\; 0.01]$

###### Paired t-test for Example 1

1. Mean difference:
   $$
   \bar{d} = \frac{0.05 + 0.02 + 0.01 + 0.01 + 0.01}{5}
           = 0.02
   $$

2. Standard deviation of differences $s_d \approx 0.0173$

3. Test statistic:
   $$
   t = \frac{\bar{d}}{s_d / \sqrt{n}}
     \approx \frac{0.02}{0.0173 / \sqrt{5}}
     \approx 2.58
   $$

With $n = 5$ → $df = 4$.  
$|t| \approx 2.58$ is **fairly large for such a small sample**, so this suggests A is **probably** better than B, but with $n=5$ it’s only borderline significant (you’d check a $t$-table or library for the exact $p$).

Intuition:
- All $d_i > 0$
- Mean improvement is positive
- t-test says: “Evidence that A > B, but sample is tiny.”

---

###### Wilcoxon Signed-Rank for Example 1

We use the **absolute differences** and their **ranks**:

1. $|d| = [0.05, 0.02, 0.01, 0.01, 0.01]$
2. Rank by magnitude (smallest = rank 1):

   - $0.01$ → ranks 1, 2, 3 (ties share consecutive ranks)
   - $0.02$ → rank 4  
   - $0.05$ → rank 5  

   (Any reasonable tie-handling gives the same idea.)

3. All $d_i$ are **positive**, so all ranks are “positive ranks”.

   - $W^+ = 1 + 2 + 3 + 4 + 5 = 15$
   - $W^- = 0$

Test statistic $W = \min(W^+, W^-) = 0$  
→ extremely small → **very strong** evidence that A is better than B (for such small $n$, zero negative ranks is already a big signal).

Intuition:
> Wilcoxon is basically saying:  
> “A wins in every fold, and by non-trivial margins → this is unlikely to be pure chance.”

---

##### Example 2 – A sometimes better, sometimes worse

Now we tweak the differences:

| Fold | Model A F1 | Model B F1 | Difference $d_i = A_i - B_i$ |
|-----:|-----------:|-----------:|-----------------------------:|
| 1    | 0.80       | 0.75       | +0.05                        |
| 2    | 0.78       | 0.79       | -0.01                        |
| 3    | 0.85       | 0.75       | +0.10                        |
| 4    | 0.76       | 0.79       | -0.03                        |
| 5    | 0.81       | 0.79       | +0.02                        |

Now:
- $d = [0.05,\; -0.01,\; 0.10,\; -0.03,\; 0.02]$

A sometimes wins, sometimes loses.

###### Paired t-test for Example 2

1. Mean difference:
   $$
   \bar{d} = \frac{0.05 - 0.01 + 0.10 - 0.03 + 0.02}{5}
           \approx 0.026
   $$

2. Standard deviation of differences $s_d \approx 0.0513$

3. Test statistic:
   $$
   t = \frac{\bar{d}}{s_d / \sqrt{n}}
     \approx \frac{0.026}{0.0513 / \sqrt{5}}
     \approx 1.13
   $$

Now $|t| \approx 1.13$ is **small** → difference is **not significant**.  
We can’t confidently say A is better than B — could easily be noise.

---

###### Wilcoxon Signed-Rank for Example 2

1. Absolute differences:

   - $|d| = [0.05,\; 0.01,\; 0.10,\; 0.03,\; 0.02]$

2. Rank them (smallest = rank 1):

   - $0.01$ → rank 1 (sign: $-$)
   - $0.02$ → rank 2 (sign: $+$)
   - $0.03$ → rank 3 (sign: $-$)
   - $0.05$ → rank 4 (sign: $+$)
   - $0.10$ → rank 5 (sign: $+$)

3. Sum positive and negative ranks:

   - $W^+ = 2 + 4 + 5 = 11$
   - $W^- = 1 + 3 = 4$

Test statistic:
$$
W = \min(W^+, W^-) = 4
$$

For $n = 5$, a value like 4 is **not extremely small**, so the Wilcoxon test also says:  
> “Some advantage for A, but not strong enough to clearly reject chance.”

---

##### Big-picture intuition from both examples

- **Example 1:** A is better in *every* fold → both tests lean toward “A is genuinely better”  
- **Example 2:** A wins sometimes, loses sometimes → both tests say “could be noise”

So:

- Paired t-test works on **means and standard deviations** of $d_i$
- Wilcoxon works on **ranks and signs** of $d_i$
- Both aim to answer:

> “Is the improvement from Model A over Model B **consistent enough** to trust?”


#### Confidence Intervals for Metrics

If Metric A = **0.92 ± 0.03**  
and Metric B = **0.90 ± 0.01**

Do CIs **overlap**?

- If yes → difference **not** statistically solid  
- If not → A likely better **with statistical confidence**

Often easier to communicate than p-values.

---

### Quick Interview Takeaways

| Method | Question It Answers |
|--------|-------------------|
| AIC/BIC | Better model fit vs complexity? |
| McNemar | Better **classification** correctness per sample? |
| Paired t / Wilcoxon | Better metrics **across runs/folds**? |
| CI for metrics | Is improvement **beyond noise**? |

> Model evaluation isn’t just picking the higher number —  
> **prove** the difference is real, not luck.



---

## Cross-Validation & Dataset Splitting

### Train / Validation / Test Strategy

Why split data?
- Measure **generalization**
- Tune **hyperparameters** safely
- Avoid inflating test performance

Sets:
- **Train** → learn model parameters
- **Validation** → hyperparameter tuning / early stopping
- **Test** → final unbiased estimate

Rule:
> If test data influences any modeling choice → it becomes validation data.

---

### K-Fold Cross-Validation

Procedure:
- Split into **K equal folds**
- Train on K-1 folds, validate on 1 fold
- Rotate until each fold has been validation once
- Report **mean ± std** of metric

Benefits:
- Uses data efficiently
- Less variance than one train/val split

---

#### Stratified K-Fold

For classification:
- **Preserves class distribution** in each fold

Critical when:
- **Imbalanced classes**
- Small datasets

---

### Nested Cross-Validation

Why:
> To avoid **overfitting** hyperparameters to validation folds.

Structure:
- **Inner CV** → choose best hyperparameters  
- **Outer CV** → unbiased evaluation

Used in:
- Academic benchmarks
- Small datasets
- Fair comparison of multiple models

Think:
> Inner loop = **model selection**  
> Outer loop = **model evaluation**

1,2,3,4,5,6,7,8,9,0

- use 0 for test:
  - use 1 for valuation get top param/2-8 for train
  - use 2 for valuation get top param/1, 3-8 for train
  - ...
- use 1 for test:
  ...

---

### Data Leakage Prevention

Leakage = model accidentally “sees” test/target info during training → fake performance

Common leakage traps:
- Scaling or feature engineering **before** splitting
- Temporal leakage (future info leaking into past)
- Same user/session in both train and test
- Target-encoded features built on full dataset

Rules:
- All preprocessing must be **fit on train only**
- Ensure validation/test samples remain **independent**
- For time series → **no random shuffling**, use forward splits

> If test information leaks into training, the evaluation is **invalid** — expect production failure.



---
## Domain-Aware Evaluation

### Imbalanced Classes

When one class is **rare** (e.g., fraud, failures, cancer):

| Metric | Behavior |
|--------|---------|
| Accuracy | Misleading (“predict all negatives” → high accuracy) |
| ROC-AUC | Can still look good because FP rate tiny |
| Precision/Recall/F1 | Focus on **minority** class performance |
| PR-AUC | Best choice when positives are rare |

Preferred metrics:
- **Precision** → cost of false positives
- **Recall** → cost of false negatives
- **F1** → balance between them
- **PR-AUC** → global measure under imbalance

Rule of thumb:
> Start with **confusion matrix**, then focus on **minority class metrics**.

---

### Cost-Sensitive Metrics

Some mistakes are **more costly** than others:

| Domain | Expensive error type |
|--------|---------------------|
| Cancer screening | False negative |
| Spam filtering | False positive |
| Fraud detection | False negative |

Strategies:
- Weighted loss functions (e.g., class weights)
- Threshold adjustment (e.g., operating point chosen via **Precision-Recall tradeoff**)
- Custom metrics using **business cost matrix**:

Example cost matrix:

| | True + | True - |
|--:|------:|------:|
| Pred + | 0 | +5 |
| Pred - | +100 | 0 |

Goal:
> Optimize **expected business cost**, not generic metric.

---

### Business KPIs → ML Metrics Mapping

ML metric must have **direct business value**, e.g.:

| Business Goal | ML Metric |
|---------------|-----------|
| Reduce fraud loss | Recall / cost-weighted loss |
| Increase recommendation revenue | CTR / NDCG / Top-K accuracy |
| Minimize false alarms in monitoring | Precision |
| Improve user trust | Calibration metrics (ECE) |

Steps to map KPIs → model evaluation:
1. Understand **real-world cost/impact** of each error type
2. Select metrics that reflect these costs
3. Choose **decision threshold** aligned with business objective
4. Continuously **monitor drift** in production

> Models should optimize what matters to the business — not just score higher offline.

---

### Interview Takeaways

- **Imbalance requires new metrics** — accuracy is useless
- Pick metrics based on **cost structure**
- Tie every metric to **business impact**
- Calibration matters when **confidence affects decisions**

> “Offline metrics must predict business outcomes — otherwise the model is just math.”


---

# Some fun example

## Example: A/B Test Significance with Proportion Z-Test

### Setup

We test if a new variant improves conversion:

| Group | Users $n$ | Conversions $c$ | Rate $\hat{p}$ |
|------:|----------:|----------------:|---------------:|
| A (control) | 10,000 | 900 | 0.09 |
| B (variant) | 10,000 | 970 | 0.097 |

Observed improvement:
$$
\hat{p}_2 - \hat{p}_1 = 0.097 - 0.09 = 0.007
$$

We want to know:
> Is +0.7% improvement **real**, or just **random variation**?

---

### Hypotheses

- $H_0$: $p_1 = p_2$ (no true improvement)
- $H_1$: $p_1 \ne p_2$

Under $H_0$, both share the same true rate $p$.

---

### Why Standard Error is what it is

Each user conversion $\sim$ Bernoulli($p$)  
Across $n$ users → Binomial($n, p$)

Variance of Binomial:
$$
\text{Var}(X) = np(1-p)
$$

Sample proportion:
$$
\hat{p} = \frac{X}{n}
\Rightarrow
\text{Var}(\hat{p}) = \frac{p(1-p)}{n}
$$

So standard error:
$$
SE(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}
$$

For two **independent** proportions:
$$
SE(\hat{p}_2 - \hat{p}_1)
=
\sqrt{p(1-p)
\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

We estimate $p$ using **pooled proportion**:
$$
\hat{p} = \frac{c_1 + c_2}{n_1 + n_2}
= \frac{1870}{20000}
= 0.0935
$$

Now compute:
$$
SE = \sqrt{
0.0935(1-0.0935)
\left(\frac{1}{10000}+\frac{1}{10000}\right)
}
\approx 0.00412
$$

---

### Test Statistic

Normalize the observed difference:

$$
z = \frac{(\hat{p}_2 - \hat{p}_1) - 0}{SE}
  = \frac{0.007}{0.00412}
  \approx 1.70
$$

---

### p-value

Two-sided:
$$
p = 2 \cdot P(Z > |1.70|)
\approx 2 \cdot 0.0445
= 0.089
$$

---

### Interpretation

- $p = 0.089 > 0.05$  
- We **cannot** reject the null hypothesis at 95% confidence

Conclusion:
> There is **not strong enough evidence** that the 0.7% improvement is real.

This protects against launching a change that appears better **just by luck**.

---

### Intuitive takeaway

> Assume **no difference**.  
> How surprising is our observed improvement?  
> Here: **not surprising enough** → could be chance.  
> The test is actually check if we are confidence enough that this value belongs to mean at zero.

---

## Example: McNemar’s Test for Two Classifiers on Same Data

We compare **two classifiers (A and B)** on the **same 1,000 samples** for a binary task.

### Setup

|                | B correct | B wrong |
|----------------|----------:|--------:|
| **A correct**  | $n_{11}=800$ | $n_{10}=120$ |
| **A wrong**    | $n_{01}=30$  | $n_{00}=50$  |

Only the **disagreement** cells matter:
- $n_{10} = 120$ (A correct, B wrong)
- $n_{01} = 30$  (A wrong, B correct)

Total disagreements:
$$
N_d = n_{10} + n_{01} = 150
$$

We test whether disagreements are **symmetric** → same true error rate.

---

### Hypotheses

- $H_0$: A and B perform equally well  
  → each disagreement is **equally likely**  
  → $P(A\text{ wins}) = P(B\text{ wins}) = 0.5$

- $H_1$: error rates are different → asymmetry

---

### Why chi-square? (Key statistical reasoning)

Under $H_0$:

$
n_{10} \sim \text{Binomial}(N_d, 0.5)
$

This means:
> Disagreements are like flipping a **fair coin** $N_d$ times.  
> Heads = A wins, Tails = B wins.

We test if the observed imbalance:
$
n_{10} - n_{01}
$
is too large to be explained by a **fair coin**.

---

### Large-sample approximation → Chi-square

Using:
- **Binomial** variance: $N_d \cdot 0.5 \cdot 0.5 = N_d/4$
- Normal approximation for large $N_d$
- Square the z-statistic → **chi-square with 1 degree of freedom** (Cuz by CLT, i is normal for large N, so its square is chi-square)

This leads to McNemar’s statistic:
$$
\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}
\quad\sim\quad \chi^2_{df = 1}
$$

> Chi-square arises from squaring the **standardized binomial difference**.

So we judge how extreme the imbalance is using a **chi-square test**.

---

### Compute statistic

$
|120 - 30| - 1 = 89
\quad\Rightarrow\quad
\chi^2 = \frac{89^2}{150}
\approx 52.81
$

---

### p-value

$
p = P(\chi^2_{df=1} \ge 52.81)
\approx 3.7 \times 10^{-13}
$

Conclusion:
> Strong evidence A is significantly better than B.

---

## Example: Binomial Test – Is This Classifier Better Than Random?

### Setup

We have a binary classifier.  
Baseline = **random guessing** → accuracy $p_0 = 0.5$.

We test it on $n = 20$ samples and it gets:

- $k = 19$ correct out of 20

Question:
> Is this result **significantly better** than random guessing?

We treat each prediction as:
- Success (correct) with unknown probability $p$
- Failure (incorrect) with probability $1 - p$

Under the **null hypothesis**:
- $H_0: p = 0.5$ (classifier is no better than random)
- $H_1: p > 0.5$ (classifier is better than random)

Number of correct predictions $X$:
$$
X \sim \text{Binomial}(n=20, p=0.5)
$$

We observed $X = 19$.

---

### p-value (exact binomial test)

Since this is a **one-sided** test ($p > 0.5$), the p-value is:

> Probability of seeing **19 or more** correct out of 20  
> if true $p = 0.5$.

So:
$$
p\text{-value} = P(X \ge 19 \mid n=20, p=0.5)
= P(X=19) + P(X=20)
$$

Binomial pmf:
$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

Compute:

- For $k = 19$:
  $$
  P(X=19) =
  \binom{20}{19} (0.5)^{19} (0.5)^1
  = 20 \cdot (0.5)^{20}
  $$

- For $k = 20$:
  $$
  P(X=20) =
  \binom{20}{20} (0.5)^{20} (0.5)^0
  = 1 \cdot (0.5)^{20}
  $$

So:

$$
p\text{-value}
= \left(20 + 1\right) \cdot (0.5)^{20}
= 21 \cdot \frac{1}{2^{20}}
= \frac{21}{1{,}048{,}576}
\approx 0.000020
$$

So:

- $p \approx 2.0 \times 10^{-5}$  
- Way smaller than 0.05 or even 0.001

---

### Interpretation

If the classifier were truly **random** ($p = 0.5$):

- Probability of getting 19 or more correct out of 20 by chance  
  is about **0.002%**

Very unlikely.

So we **reject $H_0$** and conclude:

> The classifier is **significantly better** than random guessing.

---

### Intuition Check

- If $p = 0.5$, typical result is around 10/20 correct  
- 19/20 is **9 standard deviations** away in effect (extremely rare)  
- Exact binomial test quantifies that as $p \approx 2 \times 10^{-5}$

---

### Interview one-liner

> “We model the number of correct predictions as Binomial($n, p$).  
> Under random guessing $p=0.5$, the probability of getting 19/20 correct is about $2 \times 10^{-5}$.  
> That’s our p-value, so we can confidently say the classifier is better than random.”

---

## Example: Is There a Significant Linear Relationship? (Regression t-test)

We fit a linear regression:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

We want to test:
> Does $x$ actually influence $y$?  
> i.e., is $\beta_1 \ne 0$?

---

### Estimated model from data

Suppose after fitting on $n = 30$ samples:

- Estimated slope:
  $$
  \hat{\beta}_1 = 0.8
  $$
- Standard error of the slope (this is based on close form solution of beta1):
  $$
  SE(\hat{\beta}_1) = 0.25
  $$

---

### Hypotheses

- $H_0$: $\beta_1 = 0$  
  (no linear effect)
- $H_1$: $\beta_1 \ne 0$  
  (slope is real)

---

### Test statistic (t-value)

We compute how many **standard errors away from zero** the estimate is:

$$
t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}
  = \frac{0.8}{0.25}
  = 3.2
$$

Degrees of freedom:
$$
df = n - 2 = 30 - 2 = 28
$$

---

### p-value

Two-sided test:

$$
p = 2 \cdot P(t_{28} \ge 3.2)
$$

From t-table / calculator:

$$
p \approx 0.0034
$$

---

### Interpretation

- $p < 0.01$ → **strong evidence** against $H_0$
- Slope significantly different from 0

Conclusion:

> The variable $x$ contributes to predicting $y$ —  
> there is a statistically significant linear relationship.

---

### Why does this work?

The **sampling distribution** of $\hat{\beta}_1$ (with unknown noise variance) follows:

$$
\frac{\hat{\beta}_1 - \beta_1}{SE(\hat{\beta}_1)}
\sim t_{df=n-2}
$$

Under $H_0: \beta_1 = 0$ this becomes:

$$
\frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}
\sim t_{df}
$$

We simply measure how extreme our estimate is **under that assumption**.

---

### Interview one-liner

> “We test regression significance by checking if the estimated slope  
> is many standard errors away from 0.  
> If the t-value falls in the tails of the t-distribution,  
> the relationship is statistically real.”
