import joblib

def load_model(model_path="models/face_model.pkl"):
    return joblib.load(model_path)

def predict_face(model, face_data):
    return model.predict([face_data])[0]
