import numpy as np
from utils.io_utils import load_pipeline

if __name__ == "__main__":
    x_new = np.array([[4.9, 3.0, 1.4, 0.2]])
    pipeline , label_encoder = load_pipeline()
    pred = pipeline.predict(x_new)
    print("Prediction:", pred)
