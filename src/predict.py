import keras
import pandas as pd


def load_model(model_path: str):
    return keras.models.load_model(model_path)


def predict_attrition(model, input_df: pd.DataFrame, input_features: list[str]) -> dict:
    inputs = {
        feature: input_df[feature].to_numpy() for feature in input_features
    }
    print("這邊是 inputs：", inputs)
    return model.predict(inputs).flatten()
