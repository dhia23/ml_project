from data.loadData import load_data
from preprocessing.preprocess import preprocess_data
from utils.io_utils import save_pipeline
from model.model import Model
from sklearn.preprocessing import StandardScaler




def train_model_and_serve(scaler = StandardScaler(), preprocess_func=preprocess_data):
    
    df = load_data()
    model = Model(scaler, preprocess_func)
    pipeline, accuracy ,label_encoder = model.train_and_evaluate(df)
    
    return save_pipeline(pipeline,label_encoder) , accuracy

train_model_and_serve()