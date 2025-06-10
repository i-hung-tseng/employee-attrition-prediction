import ml_edu
import numpy as np
import pandas as pd
import keras


def train_model(experiment_name: str, model: keras.Model, dataset: pd.DataFrame, labels: np.ndarray,
                settings: ml_edu.experiment.ExperimentSettings) -> ml_edu.experiment.Experiment:
    features = {
        feature_name: np.array(dataset[feature_name]) for feature_name in settings.input_features
    }

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        # 自動儲存最好的 model
        keras.callbacks.ModelCheckpoint(f"saved_models/{experiment_name}_best_model.keras", monitor="val_loss",
                                        save_best_only=True)
    ]

    history = model.fit(
        x=features,
        y=labels,
        validation_split=0.2,
        callbacks=callbacks,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs
    )

    metrics_history_df = pd.DataFrame(history.history)
    metrics_history_df.to_csv(f"reports/{experiment_name}_metrics_history.csv", index=False)

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history)
    )
