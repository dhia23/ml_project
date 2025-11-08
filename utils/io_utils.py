import joblib

def save_pipeline(pipeline, label_encoder, path='full_pipeline.pkl'):
    joblib.dump({'pipeline': pipeline, 'label_encoder': label_encoder}, path)
    print("✅ Full pipeline and label encoder saved successfully!")

def load_pipeline(path='store/trained-model.pkl'):
    data = joblib.load(path)
    print("✅ Pipeline loaded successfully!")
    return data['pipeline'], data['label_encoder']
