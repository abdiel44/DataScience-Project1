# Data Science Topics – Study Guide

## 1. Objectives

The student should demonstrate conceptual and applied mastery of the foundations underlying machine learning, including:

- **Data understanding**: exploration, visualization, distributions, and descriptive statistics.
- **Correct use** of correlation/covariance, normalization/standardization, dimensionality reduction, and data splitting.
- **Fundamentals** of supervised/unsupervised learning, classification/regression, and classical classifiers.
- **Evaluation** with metrics, computational cost considerations (FLOPS/MACs), and imbalanced data handling.
- **Neural network foundations** (MLP, CNN, RNN/LSTM, Attention), performance analysis, and tuning for improved results.

---

## 2. Topics

### Topic 1 – Data: Types, Exploration, and Visualization

Data is the starting point of any machine learning pipeline. Understanding what kind of data you have determines every downstream decision.

**Data Types**

- **Numerical (Quantitative)**: continuous (e.g., temperature, salary) or discrete (e.g., number of children).
- **Categorical (Qualitative)**: nominal (no order, e.g., color, country) or ordinal (ordered, e.g., education level).
- **Structured vs. Unstructured**: tabular data vs. images, text, audio.

**Exploratory Data Analysis (EDA)**

EDA is the process of summarizing, visualizing, and understanding data before modeling. Key steps include checking shape and data types, identifying missing values and outliers, computing summary statistics, and examining relationships between features.

**Visualization Techniques**

- **Histograms** – show the frequency distribution of a single variable.
- **Box plots** – highlight median, quartiles, and outliers.
- **Scatter plots** – reveal relationships between two numerical variables.
- **Heatmaps** – display correlation matrices or density across two dimensions.
- **Bar charts / Pie charts** – summarize categorical data.

---

### Topic 2 – Data Distributions: Normal (Gaussian) and Binomial

**Normal (Gaussian) Distribution**

The normal distribution is a symmetric, bell-shaped curve described by its mean (μ) and standard deviation (σ). Its importance lies in the fact that many natural phenomena approximate it, many statistical tests assume normality, and the Central Limit Theorem states that sample means tend toward a normal distribution regardless of the population shape.

**Binomial Distribution**

Models the number of successes in *n* independent trials, each with probability *p* of success. It is useful for binary outcome scenarios (pass/fail, click/no-click). As *n* increases, the binomial distribution approximates the normal distribution.

---

### Topic 3 – Descriptive Measures: Central Tendency, Dispersion, Shape, and Histograms

**Measures of Central Tendency**

- **Mean** – arithmetic average; sensitive to outliers.
- **Median** – middle value when sorted; robust to outliers.
- **Mode** – most frequent value; useful for categorical data.

**Measures of Dispersion**

- **Range** – difference between max and min.
- **Variance** – average squared deviation from the mean.
- **Standard Deviation** – square root of variance; same units as data.
- **Interquartile Range (IQR)** – range between Q1 and Q3; robust to outliers.

**Measures of Shape**

- **Skewness** – indicates asymmetry. Positive skew means a longer right tail; negative skew means a longer left tail.
- **Kurtosis** – indicates tail heaviness. High kurtosis means more extreme outliers.

**Histograms**

A histogram groups data into bins and displays frequency counts. It is one of the most fundamental tools for understanding data distribution, detecting skewness, spotting outliers, and deciding on transformations.

---

### Topic 4 – Correlation and Covariance

**Covariance**

Measures the direction of the linear relationship between two variables. A positive covariance means both variables tend to increase together; negative means one increases while the other decreases. Its magnitude depends on the scale of the variables, making it hard to interpret directly.

**Correlation (Pearson's r)**

Normalized version of covariance, bounded between −1 and +1. Values near +1 indicate strong positive linear relationship, values near −1 indicate strong negative linear relationship, and values near 0 indicate no linear relationship. Correlation does not imply causation.

**Key Differences**

| Property | Covariance | Correlation |
|---|---|---|
| Range | (−∞, +∞) | [−1, +1] |
| Scale-dependent | Yes | No |
| Interpretability | Low | High |

---

### Topic 5 – Normalization and Standardization

**Normalization (Min-Max Scaling)**

Rescales features to a fixed range, typically [0, 1].

Formula: `X_norm = (X - X_min) / (X_max - X_min)`

Best used when the algorithm is sensitive to magnitude (e.g., KNN, neural networks) and when data does not have significant outliers.

**Standardization (Z-Score Scaling)**

Centers data to mean = 0 and standard deviation = 1.

Formula: `X_std = (X - μ) / σ`

Best used when the data contains outliers, when the algorithm assumes normally distributed features (e.g., SVM, logistic regression), or when features have very different scales.

**When to Use Each**

- Use **normalization** when you need bounded values and the data is relatively clean.
- Use **standardization** when outliers are present or the model assumes Gaussian inputs.
- Some models (tree-based methods like Random Forest, Decision Trees) are generally insensitive to feature scaling.

---

### Topic 6 – Dimensionality Reduction: Feature Selection and Extraction

High-dimensional data can lead to the *curse of dimensionality*, increased computational cost, and overfitting. Dimensionality reduction addresses these problems.

**Feature Selection**

Chooses a subset of the original features without transforming them.

- **Filter methods** – rank features by statistical measures (e.g., correlation, chi-squared, mutual information).
- **Wrapper methods** – use a model to evaluate subsets (e.g., forward selection, backward elimination).
- **Embedded methods** – feature selection happens during model training (e.g., Lasso regularization, tree-based importance).

**Feature Extraction**

Creates new features by transforming the original ones into a lower-dimensional space.

- **PCA (Principal Component Analysis)** – finds orthogonal directions of maximum variance. Unsupervised and linear.
- **LDA (Linear Discriminant Analysis)** – finds directions that maximize class separation. Supervised and linear.
- **t-SNE / UMAP** – non-linear methods primarily used for visualization in 2D or 3D.

---

### Topic 7 – Machine Learning: Supervised vs. Unsupervised; Classification vs. Regression

**Supervised Learning**

The model learns from labeled data (input → known output).

- **Classification** – predicts a discrete label (e.g., spam/not spam, disease type).
- **Regression** – predicts a continuous value (e.g., house price, temperature).

**Unsupervised Learning**

The model finds patterns in unlabeled data.

- **Clustering** – groups similar data points (e.g., K-Means, DBSCAN).
- **Association** – discovers rules between variables (e.g., market basket analysis).
- **Dimensionality Reduction** – compresses data while preserving structure (e.g., PCA).

**Semi-Supervised and Self-Supervised Learning**

These paradigms sit between supervised and unsupervised, leveraging small amounts of labeled data alongside large amounts of unlabeled data.

---

### Topic 8 – Data Splitting: Hold-Out, Cross-Validation, LOOCV, Cross-Dataset

Proper data splitting prevents data leakage and gives honest performance estimates.

**Hold-Out**

Splits data into training and testing sets (e.g., 80/20 or 70/30). Simple and fast, but the estimate depends on which samples end up in each set. A validation set can be added for hyperparameter tuning (e.g., 60/20/20).

**K-Fold Cross-Validation**

Divides data into *k* folds. The model trains on *k−1* folds and tests on the remaining fold, rotating through all folds. The final metric is the average across all folds. This provides a more robust performance estimate than a single hold-out.

**Leave-One-Out Cross-Validation (LOOCV)**

A special case of K-Fold where *k = n* (number of samples). Each sample serves as the test set once. It is nearly unbiased but very computationally expensive for large datasets.

**Cross-Dataset Evaluation**

Trains on one dataset and tests on a completely different one. This evaluates generalization to new domains or populations and is especially important in medical, security, and real-world deployment contexts.

---

### Topic 9 – Classifiers: KNN, Naïve Bayes, SVM; Overfitting

**K-Nearest Neighbors (KNN)**

A non-parametric, instance-based algorithm. It classifies a point based on the majority label of its *k* closest neighbors. Key considerations include the choice of *k* (small *k* = noisy, large *k* = smooth), the distance metric (Euclidean, Manhattan), and sensitivity to feature scaling.

**Naïve Bayes**

A probabilistic classifier based on Bayes' Theorem with the "naïve" assumption of feature independence. It is fast, works well with high-dimensional data, and performs surprisingly well in text classification (e.g., spam detection). Variants include Gaussian, Multinomial, and Bernoulli Naïve Bayes.

**Support Vector Machine (SVM)**

Finds the hyperplane that maximizes the margin between classes. The kernel trick (linear, polynomial, RBF) allows SVMs to handle non-linearly separable data. SVMs are effective in high-dimensional spaces but can be slow on very large datasets.

**Overfitting**

Overfitting occurs when a model learns noise in the training data rather than the underlying pattern. Signs include high training accuracy but low test accuracy. Strategies to combat overfitting include using more training data, simplifying the model (fewer parameters), applying regularization (L1/L2), using cross-validation, employing early stopping, and applying dropout (in neural networks).

---

### Topic 10 – Metrics: Confusion Matrix, Accuracy, F-Score; FLOPS and MACs

**Confusion Matrix**

A table that summarizes predictions vs. actual labels with four entries: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

**Key Metrics**

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN). Misleading on imbalanced datasets.
- **Precision** = TP / (TP + FP). Of all predicted positives, how many are correct?
- **Recall (Sensitivity)** = TP / (TP + FN). Of all actual positives, how many were found?
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall). Harmonic mean of precision and recall.
- **Specificity** = TN / (TN + FP). True negative rate.

**FLOPS and MACs**

- **FLOPS (Floating Point Operations Per Second)** – measures computational speed.
- **FLOPs (Floating Point Operations)** – measures the total number of operations a model requires.
- **MACs (Multiply-Accumulate Operations)** – each MAC = one multiplication + one addition. Roughly, 1 MAC ≈ 2 FLOPs.

These metrics are critical for evaluating model efficiency, especially for deployment on edge devices or resource-constrained environments.

---

### Topic 11 – Imbalanced Data: Over/Under-Sampling and Strategies

Imbalanced data occurs when one class significantly outnumbers others. This is common in fraud detection, medical diagnosis, and anomaly detection.

**Under-Sampling**

Reduces the majority class to match the minority class. Fast but may lose important information. Example: Random Under-Sampling.

**Over-Sampling**

Increases the minority class to match the majority class. Example: SMOTE (Synthetic Minority Over-sampling Technique), which generates synthetic samples by interpolating between existing minority instances.

**Additional Strategies**

- **Cost-sensitive learning** – assign higher misclassification cost to the minority class.
- **Ensemble methods** – Balanced Random Forest, EasyEnsemble.
- **Use appropriate metrics** – avoid accuracy; prefer F1-Score, AUC-ROC, Precision-Recall curves.
- **Stratified splitting** – ensure class proportions are maintained in train/test splits.
- **Collect more data** – when possible, gather more samples of the minority class.
- **Data augmentation** – for images/text, create transformed versions of minority samples.

---

### Topic 12 – Neural Networks: MLP, CNN, RNN/LSTM, Attention Mechanism

**Multilayer Perceptron (MLP)**

A feedforward network with one or more hidden layers. Each neuron applies a weighted sum, adds a bias, and passes the result through an activation function (e.g., ReLU, Sigmoid, Tanh). MLPs are general-purpose but struggle with spatial (images) and sequential (time series) data.

**Convolutional Neural Network (CNN)**

Designed for spatial data such as images. Key components include convolutional layers (learn local patterns via filters), pooling layers (reduce spatial dimensions), and fully connected layers (final classification). CNNs exploit parameter sharing and local connectivity for efficiency.

**Recurrent Neural Network (RNN) / LSTM**

Designed for sequential data (text, time series, audio). Standard RNNs suffer from the vanishing gradient problem. LSTM (Long Short-Term Memory) addresses this with gating mechanisms: the forget gate (what to discard), the input gate (what to store), and the output gate (what to output). This allows LSTMs to capture long-range dependencies.

**Attention Mechanism**

Allows the model to focus on relevant parts of the input regardless of distance. The core idea involves computing Query, Key, and Value vectors, then using scaled dot-product attention. Self-attention is the foundation of the Transformer architecture, which underpins models like BERT and GPT. Attention removes the sequential bottleneck of RNNs and enables parallelization.

---

### Topic 13 – Performance in Neural Networks

Key factors that affect neural network performance include:

- **Architecture choice** – selecting the right type (MLP, CNN, RNN, Transformer) for the problem.
- **Depth vs. width** – deeper networks can represent more complex functions but are harder to train.
- **Activation functions** – ReLU is the most common; variants like Leaky ReLU and GELU address the dying neuron problem.
- **Loss functions** – Cross-Entropy for classification, MSE/MAE for regression.
- **Optimizers** – SGD, Adam, AdamW each have trade-offs in convergence speed and generalization.
- **Batch size** – smaller batches add noise that can help generalization; larger batches are more stable and faster on GPUs.
- **Hardware** – GPUs and TPUs dramatically accelerate training; memory constraints can limit model size.

**Monitoring Performance**

Track training and validation loss/accuracy curves over epochs. Divergence between training and validation curves signals overfitting.

---

### Topic 14 – Tuning Neural Networks to Improve Performance

**Hyperparameter Tuning**

- **Learning rate** – too high causes divergence; too low causes slow convergence. Learning rate schedulers (step decay, cosine annealing, warm-up) help.
- **Number of layers and neurons** – start simple and scale up.
- **Batch size and epochs** – balance between training time and convergence quality.
- **Search strategies** – Grid Search, Random Search, Bayesian Optimization.

**Regularization Techniques**

- **Dropout** – randomly deactivates neurons during training to prevent co-adaptation.
- **Weight Decay (L2 Regularization)** – penalizes large weights to encourage simpler models.
- **Batch Normalization** – normalizes layer inputs for faster, more stable training.
- **Data Augmentation** – artificially expands the training set with transformations (rotations, flips, crops for images).

**Advanced Techniques**

- **Transfer Learning** – reuse a pre-trained model's features and fine-tune on a smaller dataset.
- **Early Stopping** – halt training when validation performance stops improving.
- **Gradient Clipping** – prevents exploding gradients in deep or recurrent networks.
- **Ensemble Methods** – combine predictions from multiple models for improved accuracy.
- **Mixed Precision Training** – use FP16 where possible to speed up training and reduce memory usage.
