# 🧠 Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave the company using structured HR data. This project demonstrates full-cycle ML including preprocessing, feature engineering, model training, evaluation, and results visualization.

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

### 2. Load Trained Model

```python
from keras.models import load_model

model = load_model("saved_models/attrition.keras")

# Example input
import numpy as np
sample_input = {
    "age_zscore": 0.45,
    "gender_bool": 1,
    # Add all other required features here...
}
prediction = model.predict({k: np.array([v]) for k, v in sample_input.items()})
print("Prediction:", prediction)
```

### 3. Retrain or Run Experiments

```bash
python experiments/run_experiments.py
```

## ✅ Example Results


