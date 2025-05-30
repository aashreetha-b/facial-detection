import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from preprocess import load_data
import os
import argparse

def evaluate_model(model_path, X_test, y_test, target_names):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=target_names)
    return acc, f1, report, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, help="Compare two model paths")
    args = parser.parse_args()

    (X_train, X_test, y_train, y_test), target_names = load_data()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.compare:
        model1_path = os.path.join(base_dir, "models", args.compare[0])
        model2_path = os.path.join(base_dir, "models", args.compare[1])

        acc1, f1_1, report1, _ = evaluate_model(model1_path, X_test, y_test, target_names)
        acc2, f1_2, report2, _ = evaluate_model(model2_path, X_test, y_test, target_names)

        print(f"\n📊 Model Comparison:")
        print(f"  {args.compare[0]} → Accuracy: {acc1:.4f}, F1: {f1_1:.4f}")
        print(f"  {args.compare[1]} → Accuracy: {acc2:.4f}, F1: {f1_2:.4f}")

        better = args.compare[0] if f1_1 > f1_2 else args.compare[1]
        print(f"\n✅ Better model based on weighted F1: {better}")

    else:
        model_path = os.path.join(base_dir, "models", "face_model_pca_tuned_50.pkl")
        acc, f1, report, y_pred = evaluate_model(model_path, X_test, y_test, target_names)

        print(f"\n Accuracy: {acc * 100:.2f}%")
        print("\n Classification Report:")
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
