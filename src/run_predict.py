import pandas as pd
from src.preprocess import preprocess_data
from src.predict import predict_attrition, load_model

# 1. 載入模型
model_path = "saved_models/top12_th0.5_best_model.keras"
model = load_model(model_path)

top12 = [
    "marital_status_Single", "job_level", "work-life_balance", "remote_work_bool",
    "distance_from_home_zscore", "company_reputation", "gender_bool", "years_at_company_zscore",
    "education_level", "marital_status_Married", "job_satisfaction", "overtime_bool"
]

predict_data = pd.read_csv("data/predict.csv")
preprocessed_predict_data = preprocess_data(predict_data, is_predict=True)

pred_probs = predict_attrition(model, preprocessed_predict_data, top12)
pred_labels = (pred_probs > 0.5).astype(int)

for i, (prob, label) in enumerate(zip(pred_probs, pred_labels)):
    print(f"樣本 {i}: 預測為 {'離職' if label else '留任'}（機率: {prob:.2f}）")
