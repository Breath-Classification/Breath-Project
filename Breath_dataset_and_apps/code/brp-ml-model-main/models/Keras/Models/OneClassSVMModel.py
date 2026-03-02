from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from models.AbstractModel import AbstractModel


class OneClassSVMModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = make_pipeline(StandardScaler(), OneClassSVM(gamma="auto"))

    def fit(self):
        if self.check_if_model_is_compiled():
            self.model.fit(self.X_train)
            self.is_model_fitted = True

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return self.model.predict(X_test)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.model.predict(X_test)
        y_test = y_test - 1
        with open("models/saves/" + self.__class__.__name__ + ".history", "w") as file:
            file.write(str(classification_report(y_test, y_pred, zero_division=True)))
        return accuracy_score(y_test, y_pred)

    def save(self, filename):
        dump(self.model, filename)

    def load(self, filename):
        self.model = load(filename)
        self.is_model_loaded = True
