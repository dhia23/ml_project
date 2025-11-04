from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def create_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_pipeline(scaler, model):
    return Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

def train_and_evaluate(df, preprocess_func, create_model_func, build_pipeline_func):
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocess_func(df)

    model = create_model_func()
    pipeline = build_pipeline_func(scaler, model)

    pipeline.fit(X_train_scaled, y_train)
    y_pred = pipeline.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return pipeline, label_encoder

def predict(pipeline, label_encoder, X_new):
    y_pred = pipeline.predict(X_new)
    return label_encoder.inverse_transform(y_pred)
