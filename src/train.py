import joblib
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_data

def train_and_save_model(model_path="models/face_model.pkl"):
    (X_train, X_test, y_train, y_test), _ = load_data()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
