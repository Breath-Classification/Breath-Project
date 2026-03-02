import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.src.regularizers import l2

from models.AbstractModel import SensorType
from models.SequentialModel import SequentialModel


class LSTMModel(SequentialModel):
    def load_data(self, filename, sensor_type=SensorType.TENSOMETER.value):
        super().load_data(filename, expand_dims=True, sensor_type=sensor_type)

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential(
                [
                    LSTM(
                        64,
                        activation="relu",
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                        kernel_regularizer=l2(0.001),
                    ),
                    Dropout(0.5),
                    Dense(32, activation="relu"),
                    Dropout(0.5),
                    Dense(4, activation="softmax"),
                ]
            )
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

    def predict(self, X_test):
        if len(X_test.shape) == 2:
            X_test = np.expand_dims(X_test, axis=1)
        return super().predict(X_test)
