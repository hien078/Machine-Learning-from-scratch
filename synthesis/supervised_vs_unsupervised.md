# Supervised, Unsupervised, and Representation Learning

| Paradigm | Observed training signal | Typical objective | Examples |
|---|---|---|---|
| Supervised | Inputs and targets $(x,y)$ | Predict $y$ from $x$ | Regression, classification |
| Unsupervised | Inputs $x$ only | Describe structure in $p(x)$ or geometry | PCA, clustering |
| Self-supervised | Targets constructed from $x$ | Predict masked, future, or transformed parts | Language-model pretraining |
| Representation learning | Raw input with any of the above signals | Learn useful features $z=f_\theta(x)$ | Autoencoders, neural networks |

## What Changes Without Labels?

There is no task target to determine a unique notion of success. Cluster compactness,
reconstruction error, likelihood, and neighborhood preservation answer different
questions. An unsupervised score can improve while downstream usefulness worsens.

## Common Evaluation Errors

- interpreting clusters as real categories without external evidence;
- tuning an unsupervised representation on test labels;
- judging t-SNE by apparent visual separation alone;
- reporting only training accuracy for supervised models;
- calling a reconstruction objective proof that the latent representation is meaningful.

## Connections

- [PCA](../topics/10_pca/README.md)
- [Clustering](../topics/11_clustering/README.md)
- [Dimensionality Reduction](../topics/12_dimensionality_reduction/README.md)
- [Autoencoder](../topics/17_autoencoder/README.md)
