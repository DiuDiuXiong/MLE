# Linear Models Tricks and Ideas
1. Pipeline + Standard Scaler
2. Check residual against feature/predicted value --> no patter -> linear hypothesis satisfied
3. Better to use learning_curve to avoid lucky split
4. ridge, lasso, linearregression, elasticnet
5. R2, MSE, MAE, RMSE, MAPE
6. : [LINK](https://farshadabdulazeez.medium.com/essential-regression-evaluation-metrics-mse-rmse-mae-r%C2%B2-and-adjusted-r%C2%B2-0600daa1c03a)
6. (R2 good for linear model only)![img_16.png](img_16.png)
7. Metrics for classification:
  - https://scikit-learn.org/stable/api/sklearn.metrics.html
  - precision: all the thing you said is yes, how many are yes
  - recall: all the thing that is yes, how many you claimed
  - F1: 2*TP/(2*TP + FP + FN)
  - ROC-AUC score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html, slide threshold, see:
    - TPR against FPR
8. OvR is using many binary and find the one with big P, Multinomial is using softmax (optimise softmax directly), multinomial usually better on overlap classes.

