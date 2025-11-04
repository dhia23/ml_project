from data.load_data import load_data
from preprocessing.preprocess import preprocess_data
from model.model import create_model, build_pipeline, train_and_evaluate

def test_train_and_evaluate():
    df = load_data()
    pipeline, encoder = train_and_evaluate(df, preprocess_data, create_model, build_pipeline)
    
    # Check pipeline trained
    assert hasattr(pipeline, 'predict')
