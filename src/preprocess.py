import pandas as pd
import numpy as np


def z_score(data_series):
    return (data_series - data_series.mean()) / data_series.std()


def one_hot_encoding_column_and_concat(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    column_bin_dummies = pd.get_dummies(data[column_name], prefix=column_name, dtype=int)
    return pd.concat([data, column_bin_dummies], axis=1)


def label_encoding(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    column_order_definitions = {
        "work-life_balance": {
            "Poor": 0,
            "Fair": 1,
            "Good": 2,
            "Excellent": 3
        },
        "job_satisfaction": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Very High": 3
        },
        "performance_rating": {
            "Low": 0,
            "Below Average": 1,
            "Average": 2,
            "High": 3
        },
        "education_level": {
            "High School": 0,
            "Associate Degree": 1,
            "Bachelor’s Degree": 2,
            "Master’s Degree": 3,
            "PhD": 4
        },
        "job_level": {
            "Entry": 0,
            "Mid": 1,
            "Senior": 2
        },
        "company_size": {
            "Small": 0,
            "Medium": 1,
            "Large": 2
        },
        "company_reputation": {
            "Poor": 0,
            "Fair": 1,
            "Good": 2,
            "Excellent": 3
        },
        "employee_recognition": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Very High": 3
        }

    }

    data[column_name] = data[column_name].map(column_order_definitions[column_name])
    return data


def preprocess_data(data: pd.DataFrame, is_predict: bool = False) -> pd.DataFrame:
    data.columns = data.columns.str.strip().str.replace(" ", "_", regex=False).str.lower()

    if not is_predict:
        data["attrition_bool"] = data["attrition"].apply(lambda x: 1 if x == "Left" else 0)

    data["gender_bool"] = data["gender"].apply(lambda x: 1 if x == "Male" else 0)
    data["overtime_bool"] = data["overtime"].apply(lambda x: 1 if x == "Yes" else 0)
    data["remote_work_bool"] = data["remote_work"].apply(lambda x: 1 if x == "Yes" else 0)
    data["leadership_opportunities_bool"] = data["leadership_opportunities"].apply(lambda x: 1 if x == "Yes" else 0)
    data["innovation_opportunities_bool"] = data["innovation_opportunities"].apply(lambda x: 1 if x == "Yes" else 0)

    data["age_zscore"] = z_score(data["age"])
    data["years_at_company_zscore"] = z_score(data["years_at_company"])
    data["monthly_income_zscore"] = z_score(data["monthly_income"])
    data["distance_from_home_zscore"] = z_score(data["distance_from_home"])
    # 公司的存活時間？
    data["company_tenure_zscore"] = z_score(data["company_tenure"])

    data = one_hot_encoding_column_and_concat(data, "job_role")
    data = one_hot_encoding_column_and_concat(data, "marital_status")

    for col in [
        "work-life_balance", "job_satisfaction", "performance_rating",
        "education_level", "job_level", "company_size", "company_reputation", "employee_recognition"
    ]:
        data = label_encoding(data, col)

    # 特徵組合
    data["income_per_years"] = data["monthly_income"] / (data["years_at_company"] + 1)
    data["age_per_job_level"] = data["age"] / (data["job_level"] + 1)
    data["job_satisfaction_x_recognition"] = data["job_satisfaction"] * data["employee_recognition"]

    data["work-life_balance_x_satisfaction"] = data["work-life_balance"] * data["job_satisfaction"]

    return data
