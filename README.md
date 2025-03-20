# Feature Selection and Classification for Medical Diagnosis

This project implements and reproduces feature selection algorithms and classification methods for medical diagnosis, tested on both COVID-19 patient data and Wisconsin Breast Cancer Dataset (WDBC).

## Algorithm Implementation

### Feature Selection Methods

#### 1. ReliefF Algorithm
- Evaluates feature quality through instance-based learning
- Finds feature weights by analyzing nearest neighbors
- Key advantages:
  - Handles multi-class problems
  - Detects feature interactions
  - Robust to noisy data

#### 2. Fuzzy Entropy
- Measures feature uncertainty using fuzzy set theory
- Calculates information gain with fuzzy membership functions
- Benefits:
  - Handles continuous values naturally
  - Considers data uncertainty
  - More robust than traditional entropy

#### 3. Hybrid Feature Selection
- Combines multiple feature selection methods
- Implementation:
```python
def hybrid_filter(X, Y, n):
    # ReliefF selection
    reliefF_res = reliefF(X, Y, n)
    # Fuzzy entropy selection
    entropy_res = fuzzy_entropy(X, Y, n)
    # Combine results
    union_feature = pd.concat([reliefF_feature, entropy_feature]).unique()
    return union_feature
```

#### 4. Recursive Feature Elimination (RFE)
- Iteratively removes least important features
- Uses classifier performance for feature ranking
- Process:
  1. Train model with all features
  2. Compute feature importance
  3. Remove least important feature
  4. Repeat until desired number of features reached

### Classification Method

#### Improved KNN with Bonferroni Distance
- Enhanced K-Nearest Neighbors algorithm
- Uses Bonferroni distance metric for better accuracy
- Parameters:
  - k: number of neighbors
  - p: distance parameter
  - q: Bonferroni parameter

## Datasets

### 1. COVID-19 Patient Data
- Features include:
  - Patient demographics
  - Symptoms
  - Travel history
  - Clinical outcomes

### 2. Wisconsin Breast Cancer Dataset
- 569 instances
- 30 features
- Binary classification (Malignant/Benign)
- Features based on cell nucleus characteristics

## Project Structure
```
├── covid_main.py              # COVID-19 prediction
├── breast_cancer_main.py      # Breast cancer prediction
├── Covid_Data_process.py      # COVID data preprocessing
├── Breast_Cancer_process.py   # WDBC data preprocessing
├── filter.py                  # Feature selection algorithms
└── RFE.py                    # Recursive Feature Elimination
```

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- skrebate

## Usage

```python
# For COVID-19 prediction
python covid_main.py

# For breast cancer prediction
python breast_cancer_main.py
```

## Results Comparison

The implementation shows comparable results with the original papers:

1. COVID-19 Dataset:
   - Feature selection effectiveness
   - Classification accuracy
   - Important features identified

2. WDBC Dataset:
   - Classification performance
   - Feature importance ranking
   - Model robustness

