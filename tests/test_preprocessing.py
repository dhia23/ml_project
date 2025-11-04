from data.loadData import load_data
from preprocessing.preprocess import preprocess_data
from model.model import create_model, build_pipeline, train_and_evaluate

def test_preprocess_data():
    df = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocess_data(df)
    
    assert X_train_scaled.shape[1] == 4
    assert len(y_train) > 0
    assert hasattr(scaler, 'mean_')
    assert hasattr(scaler, 'scale_')
