import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from preprocess import load_data
import numpy as np
import os

def train_and_save_best_model():
    (X_train, X_test, y_train, y_test), _ = load_data()

    print("\n Tuning PCA components based on 5-fold macro F1 score...")
    best_score = 0
    best_n = None
    best_model = None

    for n in [50, 100, 150, 200]:
        pipeline = Pipeline([
            ('pca', PCA(n_components=n, whiten=True, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro')
        avg_score = scores.mean()
        print(f"  Trying PCA with {n} components... → Mean F1 macro: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_n = n
            best_model = pipeline

    print(f"\n Best PCA n_components: {best_n} with F1 macro = {best_score:.4f}")
    
    # Save the best model with versioned filename
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"face_model_pca_tuned_{best_n}.pkl")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, model_path)
    print(f" Best model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_best_model()
