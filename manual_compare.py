import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from src.interface import predict_face

# Load LFW data
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
X = lfw_people.data
images = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names

# Pick a few fixed test samples
indices = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
  # fixed samples
models = {
    "Original (face_model.pkl)": joblib.load("models/face_model.pkl"),
    "PCA Tuned (face_model_pca_tuned_50.pkl)": joblib.load("models/face_model_pca_tuned_50.pkl"),
    "Old PCA (face_model_pca.pkl)": joblib.load("models/face_model_pca.pkl")
}

# Compare predictions
for idx in indices:
    img = images[idx]
    data = X[idx]
    true_label = target_names[y[idx]]
    print(f"\n🖼 Sample index: {idx}, True label: {true_label}")
    
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_label}")
    plt.axis('off')
    plt.show()
    
    for name, model in models.items():
        try:
            pred = predict_face(model, data)
            print(f"{name} → Predicted: {pred}")
        except Exception as e:
            print(f"{name} → ❌ Error: {e}")
