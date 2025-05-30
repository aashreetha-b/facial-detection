# Facial Detection and Recognition

A machine learning project for facial detection and recognition using PCA and Random Forest classifiers. This project involves training multiple models, evaluating their performance, and providing an interface for predictions.



##  Features

- Face detection and recognition using machine learning
- Model training with PCA and Random Forest
- Evaluation and comparison of multiple models
- Interface for making predictions on new images

##  Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aashreetha-b/facial-detection.git
   cd facial-detection
2. **Create a virtual environment (optional but recommended):**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. **Install the required dependencies:**
pip install -r requirements.txt


📊 Usage
Training the Model
To train a new model:

python src/train.py

Evaluating Models
To evaluate a model's performance:
python src/evaluate.py

Comparing Models
To compare multiple models:
python src/compare_evaluate.py --compare models/face_model.pkl models/face_model_pca_tuned_50.pkl

Making Predictions
To make predictions on new images:
python main.py

Manual Model Comparison
To manually compare model predictions:
python manual_compare.py




📚 Notebooks
The notebooks/ directory contains Jupyter notebooks used for initial experimentation and model development.

🧪 Models
Trained models are stored in the models/ directory:

face_model.pkl: Initial model trained without PCA

face_model_pca.pkl: Model trained with PCA

face_model_pca_tuned_50.pkl: PCA-tuned model with 50 components

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Scikit-learn

OpenCV

LFW Dataset



---

## 🧠 Docstrings and Code Documentation

Adding docstrings to your functions and modules enhances code readability and maintainability. Here's how you can document your code:

### Example: `src/preprocess.py`

```python
"""
preprocess.py

This module contains functions for loading and preprocessing the dataset.
"""

def load_data():
    """
    Loads and preprocesses the facial recognition dataset.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels corresponding to the features.
    """
    # Your code here
