import keras
import numpy
import pandas as pd


def load_model(model_path: str):
    return keras.models.load_model(model_path)


def predict_attrition(model, input_df: pd.DataFrame, input_features: list[str]) -> numpy.ndarray:
    inputs = {
        feature: input_df[feature].to_numpy() for feature in input_features
    }
    print("predict_attrition：", type(model.predict(inputs).flatten()))
    return model.predict(inputs).flatten()
