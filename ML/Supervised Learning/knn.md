# 🧭 k-Nearest Neighbors (kNN)

## Intuition & Core Idea

The **k-Nearest Neighbors (kNN)** algorithm is one of the simplest and most intuitive methods in traditional machine learning.  
Instead of learning an explicit model, kNN relies entirely on the **geometry of the dataset**.

### ✅ Core Concept

Given a new input point $x_{\text{query}}$, kNN:

1. Measures its distance to all points in the dataset  
2. Finds the **k closest points** (its “neighbors”)  
3. Uses their labels/values to make a prediction  
   - **Classification:** majority vote  
   - **Regression:** average of neighbors  
   - **Similarity search / retrieval:** return top-$k$ closest items

This makes kNN a **lazy learner** — it does not build a model ahead of time.  
All the “work” happens at **query time**.

---

## 🧠 Why kNN Works: Geometry as the Model

kNN assumes an important idea:

> **Points that are close together in feature space tend to have similar labels or behaviour.**

This is a form of **local smoothness assumption**.  
If the structure of the data is meaningful, then “nearest neighbors” act as a good proxy for the true function that generated the data.

---

## 🗺️ kNN as a Similarity / Proximity Graph Problem

A useful viewpoint:

> Think of the dataset as a large geometric graph where each point connects to its *closest neighbors*.

Then the kNN query becomes:

- Given a node (the query point)  
- **Find the top-k most similar / closest nodes**  
- Use them to infer the output or retrieve relevant items

This framing makes kNN a building block for:

- Nearest-neighbor search engines  
- Recommendation systems  
- Local manifold learning (e.g., UMAP, t-SNE neighbor graphs)  
- Clustering acceleration  
- Outlier detection

---

## ⚠️ The Bottleneck: Naïve Search is Too Slow

The naïve kNN algorithm requires computing the distance:

$$
d(x_{\text{query}}, x_i) \quad \text{for all } i = 1 \dots N
$$

This is **O(ND)** per query (N points, D dimensions).  
For large datasets, this is extremely slow.

---

## 🚀 Motivation for KD-Tree & Ball-Tree

To speed up neighbor search, we use **spatial data structures**:

- **KD-Tree**  
- **Ball-Tree**

Their goal:

> **Partition the space so that we can skip checking most points while searching for nearest neighbors.**

These trees support:
- Fast approximate or exact nearest-neighbor queries  
- Much better performance than brute-force search in low/medium dimensions  
- Backbone of scikit-learn’s kNN implementation

---

### Pseudocode: BUILD_TREE(points, leaf_size, split_strategy)
```
# Applies to KD-Tree or Ball-Tree depending on split_strategy
BUILD_TREE(P, leaf_size, split_strategy):
  if SIZE(P) <= leaf_size:
    return NODE(type=LEAF, points=P)

  # choose a split rule based on strategy (KD: axis+threshold; Ball: center+radius+partition)
  RULE ← CHOOSE_SPLIT(P, split_strategy)

  (P_left, P_right) ← PARTITION(P, RULE)

  LEFT  ← BUILD_TREE(P_left,  leaf_size, split_strategy)
  RIGHT ← BUILD_TREE(P_right, leaf_size, split_strategy)

  return NODE(type=INTERNAL, rule=RULE, left=LEFT, right=RIGHT)
```

```
CHOOSE_SPLIT(P, split_strategy):
  if split_strategy == "KD":
    axis ← ARGMAX_VARIANCE_COORDINATE(P)           # pick coordinate with highest variance
    threshold ← MEDIAN(ALONG_AXIS(P, axis))        # median split
    return RULE(type="KD", axis=axis, threshold=threshold)

  if split_strategy == "BALL":
    center ← GEOMETRIC_CENTER(P)                   # e.g., centroid
    (P_left, P_right) ← BALL_PARTITION(P, center)  # split P into two child balls
    # store enough info to define both child balls (centers, radii, or references)
    return RULE(type="BALL", center=center, meta=(P_left, P_right))
```

```
## Updated GEOMETRIC_CENTER(P) for Ball-Tree Splitting
# More detail: "pick farthest from a random point, then pick farthest from that farthest"

GEOMETRIC_CENTER(P):
  # Step 1 — pick an arbitrary seed point
  p0 ← RANDOM_POINT_FROM(P)

  # Step 2 — find the farthest point from p0
  p1 ← ARGMAX_{p ∈ P}  DIST(p, p0)

  # Step 3 — from p1, find the farthest point again
  p2 ← ARGMAX_{p ∈ P}  DIST(p, p1)

  # p1 and p2 approximate the “diameter endpoints” of the cluster
  # They define a strong directional split for the ball-tree

  # Step 4 — define center using the midpoint between p1 and p2
  center ← MIDPOINT(p1, p2)
  return center
```

```
PARTITION(P, RULE):
  if RULE.type == "KD":
    return ( {p ∈ P | p[RULE.axis] ≤ RULE.threshold},
             {p ∈ P | p[RULE.axis] >  RULE.threshold} )

  if RULE.type == "BALL":
    # e.g., assign each p to the closer child center computed in BALL_PARTITION
    return ( RULE.meta.P_left, RULE.meta.P_right )
```
---

### Pseudocode: FIND_TOP_K(tree, query x, k)
```
# Unified kNN search for KD-Tree / Ball-Tree

FIND_TOP_K(TREE, x, k):
  TOPK ← EMPTY_BOUNDED_SET(capacity=k, key=DISTANCE_TO(x))  # keeps smallest k distances
  STACK ← EMPTY_STACK()

  # 1) Descend to a leaf (greedy)
  node ← DESCEND_TO_LEAF(TREE.root, x)

  # 2) Evaluate leaf points
  for p in node.points:
    UPDATE_TOP_K(TOPK, p, x)

  # 3) Backtrack with pruning
  PUSH(STACK, node.parent, came_from=node)

  while NOT EMPTY(STACK):
    (u, child_used) ← POP(STACK)

    if u == NULL:
      continue

    # process the sibling first (the branch not taken during descent)
    sibling ← SIBLING_OF(u, child_used)

    if sibling != NULL and CONSIDER_BRANCH(sibling, x, TOPK):
      # explore sibling subtree
      node2 ← DESCEND_TO_LEAF(sibling, x)  # greedy descent starting at sibling
      for p in node2.points:
        UPDATE_TOP_K(TOPK, p, x)
      PUSH(STACK, node2.parent, came_from=node2)

    # continue moving upward
    PUSH(STACK, u.parent, came_from=u)

  return SORT_BY_DISTANCE_ASC(TOPK)
```

```
DESCEND_TO_LEAF(node, x):
  u ← node
  while u.type == INTERNAL:
    child ← CHOOSE_CLOSER_CHILD(u, x)  # KD: compare to split plane; BALL: compare to child balls
    u ← child
  return u
```
```
CHOOSE_CLOSER_CHILD(u, x):
  if u.rule.type == "KD":
    if x[u.rule.axis] ≤ u.rule.threshold:
      return u.left
    else:
      return u.right

  if u.rule.type == "BALL":
    # pick child with smaller distance from x to child region (ball)
    dL ← LOWER_BOUND_DISTANCE(x, u.left.region)   # 0 if inside; else center distance − radius
    dR ← LOWER_BOUND_DISTANCE(x, u.right.region)
    return (u.left if dL ≤ dR else u.right)
```
```
CONSIDER_BRANCH(node_or_subtree, x, TOPK):
  d_lb ← LOWER_BOUND_DISTANCE(x, node_or_subtree.region)
  d_worst ← WORST_DISTANCE_IN(TOPK)  # +∞ if TOPK not full
  return (d_lb < d_worst)            # explore only if it can beat current worst
```
```
LOWER_BOUND_DISTANCE(x, region):
  if region.type == "KD_SPLIT_AXIS":
    # KD-tree pruning only needs distance to the splitting plane
    # region.axis      = dimension of split
    # region.threshold = split coordinate
    return ABS(x[region.axis] - region.threshold)

  if region.type == "BALL":
    # Standard ball-tree lower bound:
    # distance from point to the surface of the ball (0 if inside)
    return MAX(0, NORM(x, region.center) − region.radius)
```
```
UPDATE_TOP_K(TOPK, p, x):
  d ← DIST(x, p)                     # e.g., Euclidean: d = ||x − p||
  if SIZE(TOPK) < TOPK.capacity:
    INSERT(TOPK, (p, d))
  else if d < WORST_DISTANCE_IN(TOPK):
    REPLACE_WORST(TOPK, (p, d))

```
---
## KD-Tree vs Ball-Tree — Comparison & Reasoning

### 🔎 Quick Comparison Table

| Aspect | **KD-Tree** | **Ball-Tree** | Why this difference matters |
|---|---|---|---|
| Region shape | Axis-aligned **hyperrectangles** | **Hyperspheres** (balls) | Rectangles align with coordinates; balls better match isotropic clusters |
| Split rule | Choose axis (e.g., max variance), split at median | Choose pivot(s)/center (e.g., farthest-point midpoint), partition by distance | KD splits are **cheap**; Ball splits are **data-geometry aware** but **costlier** |
| Pruning bound (LB) | Local split-axis bound: $|x_{a}-t|$; subtree AABOX bound via point–box distance | Point–ball bound: $\max(0,\ \|x-c\|-r)$ | Ball bound is often **tighter** in curved/round regions |
| Distance metric | Works best with $L_2$/$L_p$ and axis-aligned notions | Any metric with triangle inequality | Ball-tree generalizes better beyond Euclidean |
| Build complexity | $\mathcal{O}(N\log N)$, **small constants** | $\mathcal{O}(N\log N)$, **larger constants** (more distance evals) | Ball-tree needs repeated farthest-point searches / radii updates |
| Query (low–mid $D$) | Very fast; strong pruning | Competitive; sometimes fewer nodes visited when clusters are round | KD excels when axes reflect structure; Ball excels on isotropic clusters |
| Query (high $D$) | Degrades toward brute-force | Degrades toward brute-force (often **slower constants**) | Curse of dimensionality hurts both; extra ball math doesn’t pay off |
| Memory overhead | Lower | Higher (centers, radii, sometimes tight bounding data) | Extra metadata per node for balls |
| Leaf processing | Scan small leaf set | Scan small leaf set | Same pattern; difference is how we **reach** the leaf(s) |
| Dynamic updates | Non-trivial (like most trees); often rebuild | Same | Both prefer batch builds |
| Best use-cases | $D \lesssim 20$, features roughly axis-aligned; many queries | Arbitrary metrics, round/clustered geometry, non-axis-aligned structure | Pick by data geometry & metric |

---

### 🧠 Why Ball-Tree Is Often More Costly than KD-Tree

1. **Costlier split selection.**  
   - **KD-Tree:** choose split axis (e.g., max variance) and median → one pass along an axis (or selection algorithm).  
     **Per node cost is small.**
   - **Ball-Tree:** choose center/pivots by geometry (e.g., farthest-point heuristic) and compute **radii**:  
     requires multiple **all-points distance** evaluations to pivots/centers.  
     **Per node constant factor is larger.**

2. **More expensive lower-bound computations.**  
   - **KD-Tree local prune:** the next-step decision can use **only the split axis**:  
     $$d_{\text{LB,local}} = |x_{a} - t|.$$  
     This is **extremely cheap**.  
   - **Ball-Tree prune:** uses point–ball lower bound:  
     $$d_{\text{LB,ball}} = \max\bigl(0,\ \|x-c\| - r\bigr),$$  
     which needs at least one **full distance** to the center per prune check.

3. **Heavier metadata maintenance.**  
   - Storing **centers and radii** (and sometimes tighter child summaries) increases both **build work** and **memory**.

4. **Triangle-inequality–friendly but not free.**  
   - Ball-tree’s pruning benefits from triangle inequality across arbitrary metrics, but those **metric distance** calls can be **more expensive** than simple axis comparisons.

---

### 🎯 When KD-Tree Wins

- **Low to mid dimensions** ($D \lesssim 20$).  
- **Axis-aligned structure** (features already informative per coordinate).  
- **Large number of queries** (amortize build; cheap per-node decisions).  
- **Euclidean/$L_p$ distances** where hyperrectangles are adequate.

**Why:**  
Axis-wise median splits are **cheap** and produce balanced trees; local pruning via the split axis is **constant-time** and very effective when data aligns with coordinates.

---

### 🌀 When Ball-Tree Wins

- **Roundish / isotropic clusters** or **manifold-like** blobs where a sphere approximates regions better than rectangles.  
- **Arbitrary metrics** (any metric with triangle inequality), e.g., cosine (as a metricized version), Mahalanobis (with care), some string/graph metrics, etc.  
- **Non-axis-aligned geometry:** the split notion comes from **distance**, not coordinates.

**Why:**  
The pruning bound for balls  
$$d_{\text{LB}} = \max(0,\ \|x-c\| - r)$$  
can be **tighter** when clusters are roughly spherical or when coordinate axes are **not** meaningful.

---

### 📉 High-Dimensional Reality Check (Both Trees)

- As $D$ grows large, distances **concentrate**, and lower bounds become **less discriminative**.  
- Both KD and Ball trees then visit many nodes; performance approaches **brute-force**.  
- In this regime, consider **approximate** methods (e.g., HNSW, IVF+PQ) or **learned embeddings** that reduce $D$.

---

### 🧪 Practical Guidance

- If you’re in **standard tabular ML** with $D \lesssim 20$ and Euclidean distance → **KD-Tree** is typically the best first choice.  
- If you need **general metrics** or your clusters are **round/rotated** and not axis-aligned → **Ball-Tree** is often better despite higher build cost.  
- For **very high $D$** or very large $N$, prefer **ANN** methods and/or **dimensionality reduction** first.

---

### 🔬 Pruning Bounds (Side-by-Side)

- **KD-Tree (local prune at a split on axis $a$ and threshold $t$):**  
  $$d_{\text{LB}}^{\text{KD-local}} = |x_a - t|.$$

- **KD-Tree (full subtree AABOX bound, if boxes are stored):**  
  $$
  d_{\text{LB}}^{\text{AABOX}}(x,\text{box}) =
  \sqrt{\sum_{i=1}^{D}
  \begin{cases}
  (m_i - x_i)^2 & x_i < m_i\\
  (x_i - M_i)^2 & x_i > M_i\\
  0 & m_i \le x_i \le M_i
  \end{cases}}
  $$
  where $[m_i, M_i]$ is the box interval on dimension $i$.

- **Ball-Tree (child region is a ball with center $c$ and radius $r$):**  
  $$
  d_{\text{LB}}^{\text{BALL}}(x,c,r) = \max\bigl(0,\ \|x-c\| - r\bigr).
  $$

---

### 🧷 TL;DR

- **KD-Tree:** faster to **build** and **query** in low–mid $D$, cheap axis-based pruning.  
- **Ball-Tree:** more **general** and often **tighter pruning** for round/non-axis-aligned data, but **costlier** due to distance-heavy splits and checks.  
- **High $D$:** both degrade; use ANN or reduce dimension first.

---

For implementation, see [knn](./Classification/knn.ipynb)