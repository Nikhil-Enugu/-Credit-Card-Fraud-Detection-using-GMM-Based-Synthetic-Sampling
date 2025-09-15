# -Credit-Card-Fraud-Detection-using-GMM-Based-Synthetic-Sampling

## Author

**Name:** Nikhil Enugu

**Roll No:** DA25M010

## 1. Project Objective

This project tackles the challenge of building an effective fraud detection model on a highly imbalanced dataset. The primary goal is to implement and evaluate a Gaussian Mixture Model (GMM) for generating synthetic data for the minority (fraud) class. The performance of this model-based oversampling technique is compared against a baseline model to determine its effectiveness in improving fraud detection while managing the precision-recall trade-off.

## 2. Dataset

The project uses the **Credit Card Fraud Detection** dataset available on Kaggle.

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Characteristics:** The dataset is highly imbalanced.
  - **Class 0 (Non-Fraud):** 284,315 transactions (99.83%)
  - **Class 1 (Fraud):** 492 transactions (0.17%)
- **Degree of Imbalance:** For every fraudulent transaction, there are approximately 578 legitimate ones.

## 3. Methodology

The project is structured in three parts: establishing a baseline, implementing GMM-based sampling, and comparing the results.

### Part A: Baseline Model
1.  **Data Preprocessing:** The `Time` and `Amount` columns were standardized using `StandardScaler` to ensure all features have a comparable scale.
2.  **Train-Test Split:** The data was split into training (80%) and testing (20%) sets. Crucially, `stratify=y` was used to ensure the test set maintained the original, severe class imbalance.
3.  **Baseline Training:** A Logistic Regression classifier was trained on the original, imbalanced training data to establish a performance baseline.

### Part B: GMM for Synthetic Sampling
Two strategies were implemented to create a balanced training set for the classifier.

#### Strategy 1: GMM Oversampling
1.  **GMM Fitting:** A Gaussian Mixture Model was fitted exclusively on the minority (fraud) class samples from the training data.
2.  **Optimal Components:** The optimal number of components for the GMM (`k=3`) was determined by evaluating the **Bayesian Information Criterion (BIC)**, which penalizes model complexity to avoid overfitting.
3.  **Synthetic Data Generation:** The fitted GMM was used to generate new, synthetic fraud samples until the minority class was fully balanced with the majority class.
4.  **Model Training:** A new Logistic Regression model was trained on this GMM-balanced dataset.

#### Strategy 2: GMM Oversampling + Cluster-Based Undersampling (CBU)
1.  **Undersampling Majority:** The majority (non-fraud) class was first reduced to a smaller, representative set of 1,000 samples using **ClusterCentroids** (KMeans-based undersampling).
2.  **Oversampling Minority:** The GMM from the previous step was then used to generate synthetic fraud samples to match the new majority population (1,000 samples), creating a smaller, balanced dataset.
3.  **Model Training:** A third Logistic Regression model was trained on this hybrid CBU+GMM dataset.

## 4. Results & Performance Evaluation

The performance of the three models was evaluated on the original, imbalanced test set. The key metrics—Precision, Recall, and F1-Score—for the fraud class are summarized below.

| Model | Precision (Fraud Class) | Recall (Fraud Class) | F1-Score (Fraud Class) |
| :--- | :---: | :---: | :---: |
| **Baseline (Imbalanced Data)** | **0.83** | 0.64 | **0.72** |
| **GMM Oversampling** | 0.08 | **0.91** | 0.15 |
| **GMM Oversampling + CBU** | 0.04 | 0.89 | 0.07 |


## 5. Final Recommendation

The best model depends on the specific business objective and the trade-off between catching fraud and generating false alarms.

- **For a Balanced & Practical System (Highest F1-Score):**
  The **Baseline model** is the most practical choice. It provides the best balance between precision and recall, ensuring that fraud alerts are reliable and actionable without overwhelming operational teams.

- **For Maximum Fraud Detection (Highest Recall):**
  If the absolute priority is to catch as much fraud as possible, the **GMM Oversampling** model is suitable due to its highest recall (91%). However, this comes at the cost of extremely low precision, meaning a very high number of false positives. The **GMM + CBU** model is also a reasonable alternative for high recall if working with a smaller dataset is beneficial.
