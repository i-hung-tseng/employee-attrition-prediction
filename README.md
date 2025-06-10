# 🧠 Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave the company using structured HR data.
This project demonstrates full-cycle ML including preprocessing, feature engineering, model training, evaluation, and
results visualization.

## 📂 Dataset

- Source: [Kaggle - Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- Includes features like age, job role, income, job satisfaction, work-life balance, etc.

## 🧪 Model

- **Framework**: Keras (TensorFlow backend)
- **Architecture**: Single Dense Layer (sigmoid output)
- **Input**: Multiple engineered features (Z-score, one-hot, label encoded)
- **Metrics**: Accuracy, Precision, Recall, AUC

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Experiments & Compare Models

This runs multiple experiments comparing different feature sets and classification thresholds.

```bash
python -m experiments.run_experiments
```

Results will be printed and saved to reports/, and best models to saved_models/.

### 3. Make Predictions with Trained Model

#### Step 1: Use prediction file

We've included a sample file: data/predict.csv, which follows the same format as train.csv but excludes the attrition label.

#### Step 2: Run prediction script

```bash
python -m src.run_predict
```

Example output:

```bash
樣本 0: 預測為 留任（機率: 0.44）
樣本 1: 預測為 離職（機率: 0.90）
樣本 2: 預測為 離職（機率: 0.88）
...
```

### 4. Example Results

```bash
| Feature Set      | Threshold | Train AUC | Test AUC | Accuracy Gap | AUC Gap |
| ---------------- | --------- | --------- | -------- | ------------ | ------- |
|   engineered_all | 0.5       | 0.82      | 0.80     | 0.02         | 0.02    |
|    top12         | 0.4       | 0.79      | 0.78     | 0.01         | 0.01    |
```

