from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models.AbstractModel import AbstractModel


class LRModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = LogisticRegression(max_iter=100)

    def fit(self):
        if self.check_if_model_is_compiled():
            history = self.model.fit(self.X_train, self.y_train)
            with open(
                "models/saves/" + self.__class__.__name__ + ".history", "w"
            ) as file:
                file.write(str(history))
            self.is_model_fitted = True

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return self.model.predict(X_test)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save(self, filename):
        dump(self.model, filename)

    def load(self, filename):
        self.model = load(filename)
        self.is_model_loaded = True
