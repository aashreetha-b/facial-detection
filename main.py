# main.py

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from src.interface import predict_face

# Load the LFW dataset (same config as training)
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
X = lfw_people.data
images = lfw_people.images
y = lfw_people.target

# Load the trained model
model = joblib.load('models/face_model.pkl')

# Select a random image to test
idx = np.random.randint(0, len(X))
test_image = X[idx]
test_label = y[idx]

# Predict the label
predicted_name = predict_face(model, test_image)
actual_name = lfw_people.target_names[test_label]

# Show the result
plt.imshow(images[idx], cmap='gray')
plt.title(f"Predicted: {predicted_name}\nActual: {actual_name}")
plt.axis('off')
plt.show()
