import pandas as pd
from src.preprocess import preprocess_data
from src.model import create_model
from src.train import train_model
import ml_edu.results
import ml_edu.experiment
import keras
from matplotlib import pyplot as plt

# 讀取並處理資料
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

shuffled_train_data = preprocess_data(train_df).sample(frac=1, random_state=42)
shuffled_test_data = preprocess_data(test_df).sample(frac=1, random_state=42)

# 提取標籤
label_column = "attrition_bool"
train_y = shuffled_train_data[label_column].to_numpy()
test_y = shuffled_test_data[label_column].to_numpy()

# 定義組合
features_dict = {
    "base_all": [
        "age_zscore", "gender_bool", "overtime_bool", "remote_work_bool", "leadership_opportunities_bool",
        "innovation_opportunities_bool", "years_at_company_zscore", "monthly_income_zscore",
        "distance_from_home_zscore", "company_tenure_zscore", "job_role_Education", "job_role_Finance",
        "job_role_Healthcare", "job_role_Media", "job_role_Technology", "marital_status_Divorced",
        "marital_status_Married", "marital_status_Single", "work-life_balance", "performance_rating",
        "education_level", "job_level", "company_size", "company_reputation", "job_satisfaction",
        "employee_recognition"
    ],
    "engineered_all": [
        "gender_bool", "overtime_bool", "remote_work_bool", "leadership_opportunities_bool",
        "innovation_opportunities_bool", "distance_from_home_zscore", "company_tenure_zscore",
        "job_role_Education", "job_role_Finance", "job_role_Healthcare", "job_role_Media", "job_role_Technology",
        "marital_status_Divorced", "marital_status_Married", "marital_status_Single", "work-life_balance",
        "performance_rating", "education_level", "company_size", "company_reputation", "income_per_years",
        "age_per_job_level", "job_satisfaction_x_recognition", "years_at_company_zscore", "monthly_income_zscore",
        "age_zscore", "job_level", "job_satisfaction", "employee_recognition"
    ],
    "top12": [
        "marital_status_Single", "job_level", "work-life_balance", "remote_work_bool",
        "distance_from_home_zscore", "company_reputation", "gender_bool", "years_at_company_zscore",
        "education_level", "marital_status_Married", "job_satisfaction", "overtime_bool"
    ]
}

# 不同分類門檻設定
threshold_list = [0.4, 0.5, 0.6]


# 主函式：訓練並比較多組模型，儲存每個 model 中最好的模型
def train_and_compare_models():
    results = []
    for name, feature_list in features_dict.items():
        train_x = shuffled_train_data[feature_list]

        for threshold in threshold_list:
            settings = ml_edu.experiment.ExperimentSettings(
                learning_rate=0.001,
                number_epochs=100,
                batch_size=64,
                classification_threshold=threshold,
                input_features=feature_list
            )

            metrics = [
                keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
                keras.metrics.Precision(name="precision", thresholds=threshold),
                keras.metrics.Recall(name="recall", thresholds=threshold),
                keras.metrics.AUC(name="auc")
            ]

            model = create_model(settings, metrics)
            experiment = train_model(f"{name}_th{threshold}", model, train_x, train_y, settings)
            train_metrics = experiment.metrics_history.iloc[-1].to_dict()
            test_metrics = experiment.evaluate(shuffled_test_data[feature_list], test_y)
            results.append({
                "feature_set": name,
                "threshold": threshold,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_auc": train_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_auc": test_metrics["auc"],
            })
            ml_edu.results.plot_experiment_metrics(experiment, ["accuracy", "precision", "recall"])
            ml_edu.results.plot_experiment_metrics(experiment, ["auc"])
    plt.show()
    result_df = pd.DataFrame(results)
    result_df["accuracy_gap"] = result_df["train_accuracy"] - result_df["test_accuracy"]
    result_df["precision_gap"] = result_df["train_precision"] - result_df["test_precision"]
    result_df["recall_gap"] = result_df["train_recall"] - result_df["test_recall"]
    result_df["auc_gap"] = result_df["train_auc"] - result_df["test_auc"]
    result_df.to_csv("reports/experiment_results.csv", index=False)
    return results


train_and_compare_models()
