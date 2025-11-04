import joblib

def save_pipeline(pipeline, label_encoder, path='full_pipeline.pkl', encoder_path='label_encoder.pkl'):
    joblib.dump(pipeline, path)
    joblib.dump(label_encoder, encoder_path)
    print("✅ Full pipeline and label encoder saved successfully!")

def load_pipeline(path='full_pipeline.pkl', encoder_path='label_encoder.pkl'):
    pipeline = joblib.load(path)
    label_encoder = joblib.load(encoder_path)
    print("✅ Pipeline loaded successfully!")
    return pipeline, label_encoder
