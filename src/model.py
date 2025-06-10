import keras
import ml_edu.experiment


def create_model(settings: ml_edu.experiment.ExperimentSettings, metrics: list[keras.metrics.Metric]) -> keras.Model:
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in settings.input_features]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)

    model_output = keras.layers.Dense(units=1, name="dense_layer", activation=keras.activations.sigmoid)(
        concatenated_inputs)

    model = keras.Model(inputs=model_inputs, outputs=model_output)

    model.compile(optimizer=keras.optimizers.RMSprop(settings.learning_rate), loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)

    return model
