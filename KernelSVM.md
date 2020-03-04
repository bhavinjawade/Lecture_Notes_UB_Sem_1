# Widely used variant of SVM

$k (x_i, x_j) = exp(-\frac{1}{2\gamma^2}.||x_i - x_j||^2)$

C is the regularization parameters
For the role of $\gamma$, consider the following two aspects.
___

RBF use C to make function less complex

$W^T.X^*  + b \geq 0  \ \ \ y^* = +1$

## **For Kernel SVM**

$$y^* = sign[\sum^N_{n=1} \alpha_n.y_n.X_n.X^* + b]$$

```
Smaller gamma, Wider bell curve
```

**Grid Search: Hyperparameter search**

validation data, grid search between C and $\gamma$. (Grid Search CV, give it a grid variables (ranges of variables)). 

$C = 0.1, \gamma = 0.1$ for Sklearn 

> **Research topic:** NAS (Network architecture search): Better than brute force grid search

___

Training time for SVM is $O(N^3)$
many faster but approximate approaches exit
SVM can be extended in different ways:
1. Non linear boundries (Kernel Trick)
2. Multi Class classification
3. Probabilistic output (Second order model over SVM) (sklearn also gives probabilities).
4. Regression (Support vector regression)

Let y = {0,1,2}
Solve three binary class problems:

**For multi class SVM**

0 vs 1,2 \
1 vs 0,2 \
2 vs 0,1 \
Distance from margins if collision. \
One vs Other method

By default, SVM are not probabilistic.
but $sign(w_T.x^* + b)$, this value can be used to judge score (Confidence).

# Extending Perceptrons to get NN

First Neural Network: Building CalSpan (Buffalo) Frank Rossenblanc.

input &rarr; weighted &rarr; Aggregator &rarr; Non linear activator &rarr; output


| a | b | c | a | b | c |
| - | - | - | - | - | - |
| 1 | 2 | 3 | 2 | 3 | 4 |
| 2 | 3 | 4 | 2 | 3 | 4 |
| 2 | 3 | 4 | 2 | 3 | 4 |
