# Multiclass and Multi-output Algorithms

https://scikit-learn.org/stable/modules/multiclass.html

## Multiclass
1. Binary -> Multiclass
2. Many Regressor have the solver
3. Behind the sense it is regressing correspond to the one hot encode matrix (e.g.)
```
[a,a,b]
[[1,0],
 [1,0],
 [0,1]]
```
- OvR: One Vs Rest
- OvO: One Vs One (Need to have n*(n-1) classes)
- Output Code: Encode each class is a binary string ([a,b,c] -> [1,0],[1,1],[0,1]) Regression on top of that

## Multi-label classification
An input X can correspond to multiple y. So the y looks something like:
[[1,0,1],
 [1,1,0],
 [0,0,0],
 [0,1,0]]
...

Note that here we use OvR strategy instead of softmax. This is because softmax assume all events are independent (assume they added up to 1).
But that's not the case for multi-label. OvR can support Multi-label itself.

### MultiOutputClassifier
`MultiOutputClassifier` This is generalised version of multi-label classification, each column, instead of treated at a binary class, can be multiclass

### ClassifierChain
Output of previous models can be input of later models in multi-label issues.

### Multi-output regression `MultiOutputRegressor`, `RegressorChain`
https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification

## Evaluation
Note that in multi-label issue, there can always be class imbalance issue. [Micro F1, Macro F1](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/) should get used.