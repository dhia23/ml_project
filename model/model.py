from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class Model:
    def __init__(self, scaler, preprocess_func):
        """
        Initialize the Model with a scaler and preprocessing function.
        """
        self.scaler = scaler
        self.preprocess_func = preprocess_func
        self.label_encoder = None
        self.pipeline = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        


    def build_pipeline(self):
        """
        Build a pipeline combining the scaler and model.
        """
        return Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])

    def train_and_evaluate(self, df):
        """
        Train the pipeline and evaluate its performance.
        """
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = self.preprocess_func(df)
        self.label_encoder = label_encoder

        self.pipeline = self.build_pipeline()

        self.pipeline.fit(X_train_scaled, y_train)
        y_pred = self.pipeline.predict(X_test_scaled)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return self.pipeline , accuracy_score(y_test, y_pred) , label_encoder

    def predict(self, X_new):
        """
        Predict new data using the trained pipeline.
        """
        if self.pipeline is None or self.label_encoder is None:
            raise ValueError("Model not trained yet. Call train_and_evaluate() first.")

        y_pred = self.pipeline.predict(X_new)
        return self.label_encoder.inverse_transform(y_pred)
