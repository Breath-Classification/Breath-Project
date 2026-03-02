from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from models.AbstractModel import AbstractModel
from scripts.plot import interactive_plot


class RandomForestModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = RandomForestClassifier(random_state=42, max_depth=10)

    def fit(self, weights=None):
        if self.check_if_model_is_compiled():
            self.model.fit(self.X_train, self.y_train)
            self.is_model_fitted = True

    def predict(self, X_test):
        # if len(X_test.shape) == 2:
        #     X_test = np.expand_dims(X_test, axis=1)
        if self.check_if_model_is_fitted():
            return self.model.predict(X_test)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.model.predict(X_test)
        with open("models/saves/" + self.__class__.__name__ + ".history", "w") as file:
            file.write(str(classification_report(y_test, y_pred)))
        return accuracy_score(y_test, y_pred)

    def plot_prediction(self, X_test, title=None):
        interactive_plot(
            self.X_test[:, 0], self.predict(X_test), self.y_test, title=title
        )

    def save(self, filename):
        dump(self.model, filename)

    def load(self, filename):
        self.model = load(filename)
        self.is_model_loaded = True
