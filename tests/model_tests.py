from utils.io_utils import load_pipeline

def test_load_pipeline():
    pipeline, label_encoder = load_pipeline('store/trained-model')
    assert pipeline is not None, "Pipeline loading failed"
    assert label_encoder is not None, "Label encoder loading failed"
    
    
def test_model_prediction():
    pipeline, label_encoder = load_pipeline('store/trained-model')
    
    # Create a sample input (modify according to your feature set)
    import numpy as np
    sample_input = np.array([[4.9, 3.0, 1.4, 0.2]])  # Example for Iris dataset

    # Make prediction
    y_pred = pipeline.predict(sample_input)
    y_pred_inverse = label_encoder.inverse_transform(y_pred)
    
    assert y_pred_inverse is not None, "Prediction failed"
    assert len(y_pred_inverse) == 1, "Prediction output length mismatch"
    assert y_pred_inverse == "Iris-setosa" ,"Unexpected prediction result"



