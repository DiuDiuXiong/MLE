![img.png](img.png)
```python
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=2).fit(X, Y)
X_scores, Y_scores = pls.transform(X, Y)
```
![img_1.png](img_1.png)