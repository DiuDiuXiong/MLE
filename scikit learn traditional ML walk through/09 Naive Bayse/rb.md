# Naive Bayes Notes

## Core idea (test-time prediction)
For each class `y`:

- Compute:  
  p(y | x_test) ∝ p(y) × p(x_test | y)

- Expand likelihood:  
  p(x_test | y) = product over all features of p(feature = value | y)

- Choose the class `y` with the highest score.  
  (Denominator p(x_test) is the same for all classes, so ignored for comparison.)

---

## Variants of Naive Bayes

**Gaussian NB**  
- Feature type: continuous numeric values  
- How p(feature = value | y) is trained: use Gaussian formula with mean and variance estimated from training data of this class  

**Multinomial NB**  
- Feature type: counts (e.g., word frequency)  
- How p(feature = value | y) is trained:  
  (count of feature in class + smoothing) ÷ (total counts in class + smoothing × number of features)  

**Complement NB**  
- Feature type: counts, especially useful for imbalanced classes  
- How p(feature = value | y) is trained: same as multinomial, but counts are taken from all other classes (the complement)  

**Bernoulli NB**  
- Feature type: binary (0/1 features)  
- How p(feature = value | y) is trained:  
  If feature present = (count of examples with feature present + smoothing) ÷ (total examples in class + smoothing × 2)  
  If feature absent = (count of examples with feature absent + smoothing) ÷ (total examples in class + smoothing × 2)  

**Categorical NB**  
- Feature type: discrete categories  
- How p(feature = value | y) is trained:  
  (count of examples in class with this category + smoothing) ÷ (total examples in class + smoothing × number of categories)

---

## Why Naive Bayes works (and its limits)

- Good for classification:  
  Even if probabilities are not exact, the relative comparisons across classes are often right, so it picks the correct class.  

- Not good for probability values:  
  The independence assumption is unrealistic, so multiplying many probabilities exaggerates confidence (e.g., outputs 0.99 vs 0.01).  
  → Probabilities are poorly calibrated. Use calibration methods (Platt scaling, isotonic regression) if true probability values are needed.

## `partial_fit`
Note that for large data, nb support `partial_fit`, which will only fit for a set of data for memory saving.