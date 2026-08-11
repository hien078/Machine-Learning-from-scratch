# Geometry of Machine Learning

## Purpose

Connect algorithms through subspaces, distances, margins, norms, and local geometry.

## Shared Geometric Objects

| Object | Algorithms | Interpretation |
|---|---|---|
| Orthogonal projection | Linear Regression, PCA | Closest point in a linear subspace |
| Norm ball | Ridge, Lasso, SVM | Feasible or penalized parameter geometry |
| Pairwise distance | KNN, K-Means, DBSCAN | Local similarity in feature space |
| Hyperplane and margin | Logistic Regression, SVM | Linear decision boundary and confidence geometry |
| Piecewise-constant regions | Trees, KNN | Nonlinear partition of input space |
| Learned representation | Neural Networks, Autoencoders | Geometry induced by a parameterized transformation |

## Projection: Regression and PCA

Least squares projects $y$ onto $\operatorname{col}(X)$. PCA projects centered rows of
$X$ onto a lower-dimensional feature subspace. Both are approximation problems, but they
minimize error in different spaces: target space for regression and feature space for PCA.

## Norms: Regularization and Margins

Ridge uses an $\ell_2$ penalty with smooth spherical level sets. Lasso uses an $\ell_1$
penalty whose corners promote zero coordinates. A linear SVM controls $\lVert w\rVert_2$;
because the geometric margin is proportional to $1/\lVert w\rVert_2$, smaller weight norm
corresponds to a wider normalized margin.

## Distance Is a Modeling Choice

Distance-based methods inherit every decision made before distance is measured. Feature
units, scaling, irrelevant dimensions, and representation learning can change neighbors
or clusters without changing the downstream algorithm.

## Connections

- [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation)
- [Regularization](../topics/03_regularization/README.md)
- [KNN](../topics/07_knn/README.md)
- [SVM](../topics/09_svm/README.md)
- [PCA](../topics/10_pca/README.md)
