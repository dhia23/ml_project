from train import train_model_and_serve

def validate_model():
    pipeline, accuracy = train_model_and_serve()
    if accuracy < 0.8:
        raise ValueError(f"Model accuracy {accuracy} is below the acceptable threshold.")
