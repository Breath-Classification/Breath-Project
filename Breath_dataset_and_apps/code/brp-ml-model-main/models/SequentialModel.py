from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from models.AbstractModel import AbstractModel


class SequentialModel(AbstractModel, ABC, metaclass=ABCMeta):
    @abstractmethod
    def compile(self):
        """
        Method for compiling the sequential model.
        """
        super().compile()

    def fit(self, epochs=200, batch_size=500, sensor_type="hist"):
        if self.check_if_model_is_compiled():
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
            )
            history = history.history
            with open(
                f"models/saves/{sensor_type}/{self.__class__.__name__}.history", "a"
            ) as file:
                file.write(str(history) + "\n")
            self.is_model_fitted = True

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.predict(X_test)
        if len(y_test.shape) > 1:
            y_test = np.argmax(self.y_test, axis=1)
        result = np.mean(y_pred == y_test)
        print("Evaluation result:", result)
        return result

    def save(self, filename):
        self.model.save(filename + ".keras")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter._experimental_lower_tensor_list_ops = False
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        with open(filename + ".tflite", "wb") as f:
            f.write(tflite_model)

    def load(self, filename):
        self.model = load_model(filename + ".keras")
        self.is_model_loaded = True
