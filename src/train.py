import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.preprocess import load_data

def train_and_save_model(model_path="models/face_model_pca.pkl"):
    (X_train, X_test, y_train, y_test), _ = load_data()

    # Pipeline: PCA (dimensionality reduction) + RandomForest
    pipeline = Pipeline([
        ('pca', PCA(n_components=150, whiten=True, random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    print(f"Improved model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
