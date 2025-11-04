import numpy as np
from data.loadData import load_data
from preprocessing.preprocess import preprocess_data
from model.model import create_model, build_pipeline, train_and_evaluate, predict
from utils.io_utils import save_pipeline, load_pipeline

if __name__ == "__main__":
    df = load_data()
    pipeline, label_encoder = train_and_evaluate(df, preprocess_data, create_model, build_pipeline)

    save_pipeline(pipeline, label_encoder)
    loaded_pipeline, loaded_encoder = load_pipeline()

    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3]
    ])
    predictions = predict(loaded_pipeline, loaded_encoder, new_samples)
    print("Predictions:", predictions)
