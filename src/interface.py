import sys
import joblib
import numpy as np
from PIL import Image

# Load the dataset target names from lfw_people dataset (manually added here for simplicity)
# List of class names from the LFW dataset
TARGET_NAMES = [
    'Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
    'Gerhard Schroeder', 'Hugo Chavez', 'Jacques Chirac', 'Jean Chretien',
    'John Ashcroft', 'Junichiro Koizumi', 'Serena Williams', 'Tony Blair'
]

def preprocess_image(image_path):
    """
    Load and preprocess image for prediction:
    - Convert to grayscale
    - Resize to 62x47 (same as LFW images)
    - Flatten to 1D numpy array
    """
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((47, 62))  # width x height
    img_array = np.array(img).flatten()
    return img_array.reshape(1, -1)

def main(image_path):
    # Load saved model
    model = joblib.load('face_svm_model.pkl')
    # Preprocess image
    img_data = preprocess_image(image_path)
    # Predict class
    pred_class = model.predict(img_data)[0]
    predicted_name = TARGET_NAMES[pred_class]
    print(f"Predicted person: {predicted_name}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
