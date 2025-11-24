# Clustering Evaluation

Clustering evaluation depends on whether true class labels exist.  
When ground-truth labels $Y$ are available, we evaluate how well predicted clusters $\hat{Y}$ correspond to them.

All supervised metrics below rely on a **contingency table** $N_{ij}$:

- $i$ indexes true classes  
- $j$ indexes predicted clusters  
- $N_{ij}$ = number of samples with true label $i$ and predicted cluster $j$  

We also use:
- $a_i = \sum_j N_{ij}$ = size of true class $i$  
- $b_j = \sum_i N_{ij}$ = size of predicted cluster $j$
- $n = \sum_i a_i$ = total samples  

---

## Supervised Clustering Evaluation

Metrics comparing $Y$ vs. $\hat{Y}$.

---

### Rand Index (RI)

Measures **pairwise agreement**.

Two samples are an **agreement** if:
- true labels match and predicted clusters match  
- true labels differ and predicted clusters differ  

The intuition:
- If clustering is perfect, **all** pairwise relationships remain consistent  
- If clustering is random, many unrelated pairs still “accidentally” agree → optimism issue

1️⃣ **Same–same agreement**  
Samples belong to the same true class *and* the same predicted cluster.  
Within each cell $$N_{ij}$$ in the contingency table, number of agreeing pairs is:  
$$\sum_{i,j} \binom{N_{ij}}{2}$$

2️⃣ **Different–different agreement**  
Pairs that have **different** true labels *and* **different** predicted clusters.  
We get them by:
- Start with **all** sample pairs: $$\binom{n}{2}$$
- Subtract “same class” pairs: $$\sum_i \binom{a_i}{2}$$
- Subtract “same cluster” pairs: $$\sum_j \binom{b_j}{2}$$
- Add back intersection once (because subtracted twice): $$\sum_{i,j} \binom{N_{ij}}{2}$$

So total agreements:
$$AG = \sum_{i,j} \binom{N_{ij}}{2} \;+\; \left[ \binom{n}{2} - \sum_i \binom{a_i}{2} - \sum_j \binom{b_j}{2} + \sum_{i,j} \binom{N_{ij}}{2} \right]$$

Total possible sample pairs:
$$ALL = \binom{n}{2}$$

Final score:
$$RI = \frac{AG}{ALL}$$

---

Why this matters:
- RI **looks at relationships** between every pair of points
- It rewards clustering that preserves **both similarity and dissimilarity**
- But because many random pairings fall in the “different–different” bucket ⇒  
  **even meaningless clusters can score high**  
  → Motivation for **Adjusted Rand Index**


---

### Adjusted Rand Index (ARI)

Corrects RI for **chance**.

The idea:
- Start from a **pairwise agreement index** based only on **same–same pairs**
  (same true label $i$ and same predicted cluster $j$)
- Compute how large this index would be **on average** if cluster labels were random
- Rescale so that:
  - 0 = expected value under random labeling
  - 1 = perfect match

For ARI, we focus on the **same–same index**:
$$
AG = \sum_{i,j} \binom{N_{ij}}{2}
$$

This counts how many **pairs of points** fall into the **same true class** and the **same predicted cluster**.

---

#### Why the expected agreement looks like that

We want $\mathbb{E}[AG]$ if cluster assignments were **random**, but sizes $a_i$ and $b_j$ remain fixed.

1️⃣ Total sample pairs:
$$
\binom{n}{2}
$$

2️⃣ Probability a random pair comes from the **same true class $i$**:
$$
\mathbb{P}(\text{same class } i) = \frac{\binom{a_i}{2}}{\binom{n}{2}}
$$

3️⃣ Probability a random pair comes from the **same predicted cluster $j$**:
$$
\mathbb{P}(\text{same cluster } j) = \frac{\binom{b_j}{2}}{\binom{n}{2}}
$$

4️⃣ Assuming independence under the null hypothesis:
$$
\mathbb{P}(\text{same class } i \;\text{and}\; \text{same cluster } j)
\approx \frac{\binom{a_i}{2}}{\binom{n}{2}} \cdot \frac{\binom{b_j}{2}}{\binom{n}{2}}
$$

5️⃣ Sum over all class–cluster pairs:
$$
\mathbb{P}(\text{same–same anywhere})
=
\sum_{i,j}
\frac{\binom{a_i}{2}\binom{b_j}{2}}{\binom{n}{2}^2}
=
\frac
{
\left(\sum_i \binom{a_i}{2}\right)
\left(\sum_j \binom{b_j}{2}\right)
}
{
\binom{n}{2}^2
}
$$

6️⃣ Expected same–same count:
$$
\mathbb{E}[AG] =
\binom{n}{2} \cdot \mathbb{P}(\text{same–same anywhere})
=
\frac{\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}}{\binom{n}{2}}
$$

---

#### Why this normalization

Max possible same–same pairs (perfect match to classes and clusters):
$$
AG_{\max} =
\frac{1}{2}
\left[
\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}
\right]
$$

Thus:
$$
ARI =
\frac{AG - \mathbb{E}[AG]}
{AG_{\max} - \mathbb{E}[AG]}
=
\frac
{AG - \mathbb{E}[AG]}
{
\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] -  \mathbb{E}[AG]
}
$$

Range:
- $ARI = 1$ → perfect match
- $ARI = 0$ → random labeling
- $ARI < 0$ → worse than random

This makes ARI the **best general supervised clustering metric**, because  
it tells you how far above (or below) random chance the clustering is.


---

### Mutual Information (MI)

Measures **how much knowing the predicted cluster** $\hat{Y}$ **reduces uncertainty** about the true labels $Y$, and vice-versa.

If predicted clusters perfectly match true classes → **zero uncertainty left** → high MI.  
If clusters are independent from true labels → **no uncertainty removed** → MI = 0.

---

#### How the probabilities are formed

Joint probability of being in **true class $i$** and **cluster $j$**:
$$
p_{ij} = \frac{N_{ij}}{n}
$$

Marginal probabilities:
- True labels:
  $$
  p_i = \frac{a_i}{n}
  $$
- Predicted clusters:
  $$
  p_j = \frac{b_j}{n}
  $$

If $Y$ and $\hat{Y}$ were **independent**, we would expect:
$$
p_{ij} = p_i \cdot p_j
$$

---

#### Why the log-ratio form?

$MI$ compares:
- **actual joint distribution** vs.
- **expected joint distribution under independence**

Contribution from each cell $(i,j)$:
$$
p_{ij} \log \frac{p_{ij}}{p_i p_j}
$$

If:
- $p_{ij} > p_i p_j$ → more shared structure than random → **positive contribution**
- $p_{ij} = p_i p_j$ → independent → **zero contribution**
- $p_{ij} < p_i p_j$ → clusters actively contradict labels → **negative contribution**

Summing over all cells:
$$
MI = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{p_i p_j}
$$

---

**Interpretation**
- $MI = 0$ → clusters contain **no information** about classes
- Higher $MI$ → **stronger alignment** between clusters and true labels

⚠️ Issue: MI **increases with number of clusters** (even random splits may raise MI)  
→ We need **normalization** (next metric).

---

### Normalized Mutual Information (NMI)

Rescales MI to handle:
- different numbers of classes/clusters
- imbalance in class/cluster sizes

Compute **entropies** (uncertainty) of labels and clusters:
$$
H(Y) = -\sum_i p_i \log p_i
\quad,\quad
H(\hat{Y}) = -\sum_j p_j \log p_j
$$

Then normalize MI:
$$
NMI = \frac{2 \cdot MI}{H(Y) + H(\hat{Y})}
$$

This expresses **fraction of shared information** between $Y$ and $\hat{Y}$:

- $NMI = 1$ → identical clustering to ground truth
- $NMI = 0$ → no relationship at all
#### Why NMI helps but isn’t perfect

Normalizing by entropies fixes **scale issues**:
- MI naturally increases if we increase the number of clusters  
  (even totally random splits can produce higher MI)
- Dividing by $H(Y)$ and $H(\hat{Y})$ controls this effect

But there’s still a problem:

> Even NMI **does not fully correct for chance**  
> (random clusterings won’t necessarily give $NMI = 0$)

Because:
- Increasing cluster granularity adds **structure in the contingency table**
- MI normalization doesn’t subtract the **shared information expected by pure chance**

That leads to the next metric (AMI), which **directly parallels ARI** for information-based metrics:
- ARI = chance-corrected RI  
- AMI = chance-corrected MI/NMI  


---

**Interpretation Summary**

| Metric | What it Measures | Handles Cluster Count Bias? | Range | Notes |
|--------|----------------|----------------------------|------|------|
| MI | Shared structure in distributions | ❌ | $\ge 0$ | Hard to compare across clusterings |
| NMI | Fraction of shared info | ✔ | $[0,1]$ | Good for comparing models |

NMI is a more **robust** version of MI, especially when cluster numbers vary.


---

### V-Measure

Decomposes MI into:
- **Homogeneity**: each cluster contains only one true class  
- **Completeness**: all members of a class fall in the same cluster  

Formulas:
$$h = \frac{MI}{H(Y)} \quad,\quad c = \frac{MI}{H(\hat{Y})}$$

Harmonic mean:
$$V = \frac{2 \cdot h \cdot c}{h + c}$$

Range: $$[0,1]$$  
Balances avoiding over-splitting (completeness) and over-merging (homogeneity).

---

### Quick Comparison

| Metric | Range | Handles Chance? | Notes |
|--------|------|----------------|------|
| RI | [0,1] | ❌ | Over-optimistic |
| ARI | [-1,1] | ✔ | Best general choice |
| MI | ≥0 | ❌ | Scale varies by clusters |
| NMI | [0,1] | ✔ | Compare diff. cluster counts |
| V-Measure | [0,1] | ✔ | Explicit homogeneity vs completeness |

When ground-truth labels are available, **Adjusted Rand Index (ARI)**, **Adjusted/Normalized Mutual Information (AMI/NMI)**, and **V-measure** are the three most commonly recommended families of supervised clustering metrics, each emphasizing a different notion of what “good clustering” means. **ARI** evaluates whether pairwise relationships between points are preserved, making it highly interpretable and strongly chance-corrected; however, it can penalize clusterings that slightly over- or under-split classes even if most point assignments are correct. In contrast, **AMI/NMI** take an information-theoretic perspective, measuring how much uncertainty about true labels is reduced by the predicted clusters; they scale better when the number of clusters differs from the number of classes and work well for imbalanced settings, though AMI is preferred over NMI because it corrects the bias toward many clusters. **V-measure** uses the same underlying information quantities but decomposes performance into **homogeneity** and **completeness**, making it particularly good for diagnosing *how* the clustering fails (merging multiple classes vs. splitting a single class too much), though it is less commonly used as the primary scoring metric. In practice: **use ARI as the main benchmark**, **look at AMI/NMI when cluster counts differ**, and **use V-measure when interpreting cluster–class correspondence errors**.

---

## Unsupervised Clustering Evaluation

When we **do not** have true labels, clustering must be judged by:

- **Cohesion**: samples in the same cluster should be similar
- **Separation**: different clusters should be far apart

Unsupervised metrics rely solely on **distance (or similarity)** among samples.

We categorize them into several groups:

- **Internal metrics**: measure cohesion + separation using data distances
- **Stability metrics**: test robustness to perturbations
- **Model selection criteria**: pick number of clusters or parameters

---

### Internal Evaluation Metrics

Assume that a good clustering structure should have:

- Low **within-cluster** distances (tight clusters)
- High **between-cluster** distances (clearly separated clusters)

#### Silhouette Coefficient

Measures how well each point fits into its assigned cluster **without labels**.

For each point $x$ in cluster $C$:

- $a(x)$ = average distance from $x$ to all other points in its **own cluster**  
  → cluster **cohesion** (smaller is better)

- $b(x)$ = minimum average distance from $x$ to all points in **any other cluster**  
  → cluster **separation** (larger is better)

Silhouette for point $x$:
$$
s(x) = \frac{b(x) - a(x)}{\max(a(x),\, b(x))}
$$

Mean over all points:
$$
S = \frac{1}{n} \sum_x s(x)
$$

---

##### Intuition

- Good clustering: $a(x)$ small, $b(x)$ large → $s(x) \approx 1$
- Borderline point: $a(x) \approx b(x)$ → $s(x) \approx 0$
- Misclassified point: $a(x) > b(x)$ → $s(x) < 0$

It rewards **clear margins between clusters**, similar to margin-based classifiers like SVM.

---

##### Strengths
- Intuitive and easy to interpret  
- Detects **poorly assigned points**  
- Great for model selection (compare options)

---

##### Weaknesses
- Assumes **convex cluster shapes**
- Not suitable for **arbitrary density-based clusters** (e.g., DBSCAN)
- Computation heavy for large $n$ (pairwise distances)

---

##### Good Rule of Thumb

| Silhouette value | Interpretation |
|---|---|
| $> 0.5$ | Meaningful clustering |
| $0.2 \sim 0.5$ | Might have weak structure |
| $< 0.2$ | Probably no clustering structure |

Best used for: **k-means**, **Gaussian mixtures**, **PCA-reduced data**

---

#### Calinski-Harabasz Index (Variance Ratio Criterion)

Measures how well clusters are **compact** and **separated**, based only on distances to cluster centroids.

It is the ratio between:

- **Between-cluster dispersion** (clusters far apart)
- **Within-cluster dispersion** (clusters tight)

---

##### Definitions

Let:
- $k$ = number of clusters  
- $n$ = total samples  
- $C_j$ = cluster $j$  
- $c_j$ = centroid of cluster $j$  
- $c$ = overall centroid  

**Between-cluster dispersion**:
$$
B = \sum_{j=1}^{k} |C_j| \cdot \|c_j - c\|^2
$$

**Within-cluster dispersion**:
$$
W = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - c_j\|^2
$$

Final score:
$$
CHI = \frac{B / (k - 1)}{W / (n - k)}
$$

---

##### Intuition

- Large $B$: clusters are well separated
- Small $W$: each cluster is tight
- Higher $CHI$ = better clustering

It generalizes the **ANOVA F-statistic** idea to clustering.

---

##### Strengths
- Fast and very efficient to compute
- Works well when clusters are **globular**
- Very interpretable: “variance between vs. within”

---

##### Weaknesses
- Strong bias toward **more clusters**
  (often increases with $k$)
- Sensitive to scale of features
- Assumes centroid-based structure (like k-means)

---

##### Practical Usage

- Best paired with algorithms that optimize centroids  
  → **k-means**, **GMM**  
- Use **relative comparison** across clusterings  
- For model selection: look for **peak** or **elbow-like** behavior of $CHI$

---

Rule of thumb:
> Higher $CHI$ is better — but be cautious for **very large $k$**.

---

#### Davies-Bouldin Index (DB Index)

Measures the **average similarity** between each cluster and its **most similar other cluster**.

> Lower is better  
> (we want clusters to be **compact** and **well-separated**)

---

##### Definitions

For each cluster $C_i$:

- **Cluster scatter** (how spread it is around its centroid $c_i$):
$$
S_i = \frac{1}{|C_i|} \sum_{x \in C_i} \|x - c_i\|_2
$$

For every pair of clusters $(i, j)$:
- **Cluster similarity**:
$$
R_{ij} = \frac{S_i + S_j}{\|c_i - c_j\|_2}
$$

For each $i$, find the **worst-case** similar cluster:
$$
R_i = \max_{j \ne i} R_{ij}
$$

Final score:
$$
DB = \frac{1}{k} \sum_{i=1}^{k} R_i
$$

---

##### Intuition

- $S_i$ small → cluster is **tight**
- Centroid distance $\|c_i - c_j\|$ large → clusters are **far apart**

So:
- Bad clusters → high $R_{ij}$ → high $DB$
- Good clusters → low $R_{ij}$ → low $DB$

It focuses on **worst overlaps**, then averages across clusters.

---

##### Strengths
- Penalizes clusters that are too similar to others  
- Highlights **where** cluster boundaries are weak  
- Works with any distance metric

---

##### Weaknesses
- Assumes centroid-based cluster shapes  
- Sensitive to noise and elongated clusters  
- May favor **equal-sized** clusters

---

##### Practical Usage

Use DB-index when:
- Need a metric that **highlights cluster overlap**
- Evaluating **compactness + separation together**

Rule of thumb:
> **Lower DB Index** → better clustering structure

---

#### Dunn Index

Focuses on **worst-case structure**:
- Worst between-cluster separation
vs.
- Worst within-cluster cohesion

> Higher Dunn Index means **all clusters** are well-separated and compact  
> (emphasizes the **weakest** part of the clustering)

---

##### Definitions

For each cluster $C_i$:

- **Cluster diameter** (largest distance within the cluster):
$$
\Delta_i = \max_{x,y \in C_i} \|x - y\|_2
$$

For every pair of clusters $(i, j)$:

- **Inter-cluster distance**:
$$
\delta(C_i, C_j) = \min_{x \in C_i, y \in C_j} \|x - y\|_2
$$

Final Dunn Index:
$$
D = \frac{\displaystyle \min_{i \ne j} \delta(C_i, C_j)}{\displaystyle \max_{i} \Delta_i}
$$

---

##### Intuition

- **Numerator (min between-cluster distance)**:  
  measures how close the two most confused clusters are

- **Denominator (max within-cluster diameter)**:  
  measures how loose the worst cluster is

Thus:
- Even **one bad cluster** → $D$ drops significantly  
- Good clustering must be **consistently good everywhere**

---

##### Strengths
- Very **strict** measure: enforces global separation
- Good for **detecting problematic clusters**
- Works even with **non-convex** shapes (depends on distance choice)

---

##### Weaknesses
- **Computationally expensive**  
  needs full pairwise evaluations for all clusters
- Very **noise-sensitive**  
  a single outlier inflates $\Delta_i$
- Unstable with varying density clusters

---

##### Practical Usage

Use when:
- Need strong assurance that **all cluster boundaries** are good
- Investigating **minimum-quality guarantee** of clustering

Rule of thumb:
> **Higher Dunn Index** → better overall structure, but verify robustness

---

#### Summary

| Metric | What it Measures | Goal | Best For | Weakness | Practical Tip |
|--------|-----------------|------|---------|-----------|---------------|
| Silhouette Coefficient | Cohesion vs separation per point | Maximize | Convex clusters (e.g., $k$-means) | Poor for non-convex clusters; expensive for large $n$ | Great *default* metric for unsupervised evaluation |
| Calinski-Harabasz Index | Ratio of between/within variances | Maximize | Centroid-based models | Increases with $k$; scale-sensitive | Use for scanning $k$, look for elbow/peak |
| Davies-Bouldin Index | Worst-case cluster similarity | Minimize | Detecting overlapping clusters | Sensitive to noise; centroid bias | Lower = better, easy sanity check |
| Dunn Index | Worst-case separation vs cohesion | Maximize | Ensuring consistent boundaries | Very expensive; outlier-sensitive | Good for validating strongest structure |

---

**Quick rule:**
> Silhouette as primary ➜ CH for $k$ tuning ➜ DB for overlap checks ➜ Dunn for strict guarantees

---

### Stability-Based Metrics

These methods evaluate how **consistent** a clustering solution is under **perturbations** of the data or algorithm.

> A stable clustering = real underlying structure  
> An unstable clustering = artifact of random initialization or noise

Stability focuses on **robustness** instead of geometric properties.

---

#### General Procedure

1. Apply random perturbation(s) to the dataset:  
   - Subsampling  
   - Adding noise  
   - Random initialization changes  
2. Re-cluster the modified datasets  
3. Measure **how often points remain clustered together**

If clusters persist across changes → they’re **reliable**.

---

#### Common Stability Measures

| Method | How Stability is Measured | Pros | Cons |
|--------|--------------------------|------|------|
| Jaccard Stability | Compare sets of points that stay together across runs | Direct interpretation | Sensitive to noise; pairwise heavy |
| Bootstrap Stability | Re-cluster bootstrap samples, measure membership consistency | Strong reliability test | Computationally expensive |
| Normalized Mutual Information Stability | Compute MI across multiple runs | Flexible and algorithm-agnostic | Harder interpretability |
| Cluster Assignment Entropy | Lower entropy = more stable membership across runs | Easy to compute | Global summary loses local detail |

---

#### What Stability Checks Reveal

They help answer:
- **Are clusters artifacts of random initialization?**
- **Does a small change in the data break structure?**
- **Are some clusters (or regions) unreliable?**

This is particularly important for methods like:
- $k$-means  
- GMM  
- Hierarchical clustering with cut thresholds

---

#### Practical Usage Guide

- Use after tuning $k$ to validate that solution is **not brittle**
- Combine with internal metrics to ensure both:
  - **Quality** (Silhouette/CH/DB/Dunn)
  - **Reliability** (Stability)

> Good clustering is not only accurate — it must be **reproducible**.

---

### Model Selection Criteria (Choosing the Number of Clusters)

When labels are unavailable, deciding **how many clusters** to form is one of the hardest parts of clustering.  
Model selection criteria aim to identify the value of $k$ that best fits the underlying structure — not too simple, not too complex.

> Goal: find a balance between **cluster quality** and **model complexity**

---

#### Common Approaches

| Method | What It Measures | How to Choose $k$ | Strengths | Weaknesses |
|--------|-----------------|------------------|-----------|------------|
| Elbow Method | Marginal gain in within-cluster tightness | Look for “elbow” where improvement drops | Simple and intuitive | Subjective; visually judged |
| Silhouette Analysis | Cohesion vs separation per sample | Maximize mean silhouette score | Works well for convex clusters | High cost; shape assumption |
| Gap Statistic | Deviation from null reference model | Maximize “gap” over random datasets | More principled than elbow | Computationally expensive |
| BIC / AIC (for GMM) | Penalized likelihood | Minimum criterion value | Statistically grounded | Depends on Gaussian assumption |
| Eigengap Heuristic (Spectral) | Graph connectivity | Look for large eigenvalue jump | Great for manifold structure | Harder to interpret |
| Minimum Description Length | Data compression / info tradeoff | Minimize total description length | Very general principle | Rare in practical workflows |

---

#### Key Intuitions

- **Elbow**: stop when clusters stop improving much → avoid overfitting
- **Silhouette**: ensure boundaries are meaningful
- **Gap**: compare vs structure you'd expect **by chance**
- **BIC/AIC**: avoid models with unnecessary parameters
- **Eigengap**: choose number of connected components in spectral space

---

#### Practical Usage Tips

- Try **multiple criteria together**:
  - Elbow for a quick guess
  - Silhouette to validate geometric separation
  - Gap or BIC for more formal confirmation
- Always **visualize clusters** in reduced space (PCA/UMAP)
- Consider **domain knowledge** (clustering is often exploratory)

---

Rule of thumb:
> A good $k$ should show **consistent benefits** across at least two different criteria.

---
