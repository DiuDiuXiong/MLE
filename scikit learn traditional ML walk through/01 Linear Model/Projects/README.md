# Projects

Projects suggested by GPT to cover different topics. Here's an overview of the question sets.

# 📘 Linear Models Project Roadmap
## 1. House Price Prediction (Boston/California Housing)
- Difficulty: Beginner
- Project Purpose: Predict house prices with LinearRegression; stretch → compare OLS vs Ridge vs Lasso, plot coefficients.
- Points Examined: OLS regression, coefficient interpretation, overfitting risk.
- Doc References: Ordinary Least Squares.
- Why Useful: Foundation for regression; introduces linear models in the simplest form.

## 2. Medical Cost Estimation (Insurance Dataset)
- Difficulty: Beginner → Intermediate
- Project Purpose: Predict insurance charges; stretch → show how Lasso drops insignificant features compared to Ridge.
- Points Examined: Ridge, Lasso, coefficient shrinkage.
- Doc References: Ridge regression, Lasso.
- Why Useful: First hands-on with regularization; contrasts Ridge vs Lasso effects.

## 3. Advertising Sales (Advertising Dataset)
- Difficulty: Beginner → Intermediate
- Project Purpose: Predict sales from ad spend (TV, Radio, Newspaper); stretch → add ElasticNet, tune α, visualize performance.
- Points Examined: OLS, Ridge, Lasso, ElasticNet, grid search.
- Doc References: Elastic Net.
- Why Useful: One dataset → all regression types compared side-by-side.

## 4. Spam vs. Ham Email (SMS Dataset)
- Difficulty: Intermediate
- Project Purpose: Classify spam vs ham using Logistic Regression on TF–IDF; stretch → compare L1 vs L2 penalties.
- Points Examined: LogisticRegression, penalties (l1, l2), solvers (liblinear, saga).
- Doc References: Logistic regression, Regularization.
- Why Useful: Intro to classification; highlights sparse vs dense coefficients.

## 5. Credit Default Prediction (Credit Card Default Dataset)
- Difficulty: Intermediate
- Project Purpose: Predict client default/no-default; stretch → apply class weights, evaluate ROC–AUC.
- Points Examined: LogisticRegression, class imbalance handling, probability outputs.
- Doc References: Logistic regression (penalties/solvers), Class imbalance.
- Why Useful: Teaches handling imbalanced datasets + probability calibration.

## 6. Handwritten Digits (scikit-learn Digits Dataset)
- Difficulty: Intermediate
- Project Purpose: Classify digit “0” vs “1” (binary); stretch → expand to multinomial logistic regression (all digits).
- Points Examined: Binary logistic regression, multinomial logistic regression.
- Doc References: Multinomial logistic regression.
- Why Useful: Demonstrates extension from binary to multiclass.

## 7. Stock Market Direction Prediction (Yahoo Finance data)
- Difficulty: Intermediate → Advanced
- Project Purpose: Predict “up” or “down” stock movements; stretch → try SGDClassifier for large-scale learning.
- Points Examined: Logistic Regression with engineered features, SGDClassifier.
- Doc References: SGD.
- Why Useful: Realistic noisy data; introduces scalable solvers.

## 8. Multi-Label Text Classification
- Difficulty: Advanced
- Project Purpose: Predict multiple tags (e.g., movie genres) from text. Stretch → compare One-vs-Rest Logistic Regression vs linear SVM.
- Points Examined: Multi-label setting, OvR strategy, evaluation with F1-micro vs F1-macro.
- Doc References: Scikit-learn multi-label docs, OvR/OvO strategies.
- Why Useful: Expands classification intuition from binary → multi-class → multi-label, which is very real-world (tags, recommendations, incident categories).

## 9. ElasticNet Feature Selection (Diabetes Dataset)
- Difficulty: Intermediate
- Project Purpose: Predict disease progression; stretch → visualize coefficient paths across α and l1_ratio.
- Points Examined: ElasticNet, l1_ratio tuning, coefficient path visualization.
- Doc References: Elastic Net, Coordinate descent.
- Why Useful: Great for visualizing trade-off between Ridge & Lasso.

## 🔧 Meta-Learning Exercises
A. Regularization Playground
- Difficulty: Beginner → Intermediate
- Project Purpose: Take any dataset, vary α for Ridge/Lasso/ElasticNet; stretch → plot error vs α, compare stability.
- Points Examined: Regularization tuning, bias–variance.
- Doc References: Ridge, Lasso, Elastic Net.
- Why Useful: Builds intuition for the effect of α.

B. Bias–Variance Demo with Polynomial Features

- Difficulty: Intermediate
- Project Purpose: Create synthetic y = 3x + noise, fit polynomial regressions; stretch → plot train vs test error by degree.
- Points Examined: Polynomial regression as linear model, bias–variance tradeoff.
- Doc References: Polynomial regression as linear model.
- Why Useful: Classic underfitting/overfitting visual demo.

C. Scaling Sensitivity with SGDClassifier

- Difficulty: Intermediate
- Project Purpose: Fit SGDClassifier with and without scaling; stretch → compare convergence and performance.
- Points Examined: SGD optimization, importance of feature scaling.
- Doc References: SGD.
- Why Useful: Explains why scaling is mandatory in practice.