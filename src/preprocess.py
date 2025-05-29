from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_data(min_faces_per_person=70):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    return train_test_split(X, y, test_size=0.25, random_state=42), target_names
