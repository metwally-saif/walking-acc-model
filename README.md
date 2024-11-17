# Dataset Creation and Model Performance Analysis

## 1. Dataset Creation Experience

### 1.1. Introduction to the Dataset
The dataset was created using sensor data, specifically accelerometer readings, to recognize human activities. Each instance corresponds to a window of time-series data transformed into statistical and derived features for classification tasks. Key components of the dataset include:

- **Features**: Statistical metrics such as mean, standard deviation, skewness, and other derived features from accelerometer data across X, Y, and Z axes.
- **Labels**: Activity types corresponding to various movements or locations.

### 1.2. Data Preparation

#### Data Exploration
- The dataset was inspected for class distributions, missing values, and feature correlations.
- It was noted that the dataset consisted of multiple sources, evenly distributed across classes.

#### Data Cleaning
- Missing values were handled through imputation or removal of inconsistent entries.
- Outliers were addressed using quantile-based filters to maintain data integrity.

#### Feature Engineering
- Additional features, such as Signal Magnitude Area (SMA), zero-crossings, and frequency-domain metrics, were derived.
- Standardization was applied to ensure uniform scaling across all input features.

#### Data Splitting
- The data was divided into training (75%) and testing (25%) sets to facilitate an effective evaluation of the model’s performance.

---

## 2. Model Creation

### 2.1. Model Selection
A Random Forest Classifier was selected for its versatility and robustness. Its key advantages include:
- Built-in feature importance evaluation.
- Non-linear decision-making capabilities, ideal for activity classification.
- Resistance to overfitting when tuned appropriately.

### 2.2. Hyperparameter Tuning
Hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation on the training set. Parameters explored include:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of the trees.
- `min_samples_split` and `min_samples_leaf`: Constraints to prevent overfitting by setting minimum samples per split and leaf.
- `max_features`: Number of features considered for splits to regulate feature randomness.

#### Optimal Hyperparameters:
```json
{
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt"
}
```
# Model Performance and Insights

## 3. Model Performance

### 3.1. Classification Metrics
The Random Forest model achieved high accuracy with balanced performance across all classes. Key metrics evaluated on the testing set include:
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

#### Classification Report:
```plaintext
                      precision    recall  f1-score   support

  acc_walking_2_shin       1.00      1.00      1.00       233
   acc_walking_chest       1.00      1.00      1.00       725
 acc_walking_forearm       1.00      1.00      1.00       723
    acc_walking_head       1.00      1.00      1.00       725
    acc_walking_shin       1.00      1.00      1.00       729
   acc_walking_thigh       1.00      1.00      1.00       722
acc_walking_upperarm       1.00      1.00      1.00       724
   acc_walking_waist       1.00      1.00      1.00       726

            accuracy                           1.00      5307
           macro avg       1.00      1.00      1.00      5307
        weighted avg       1.00      1.00      1.00      5307
```
