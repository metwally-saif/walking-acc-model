import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib




def test_model(model, har_data_file):
    """
    Test the trained model against the HAR dataset.
    """
    # Load HAR dataset (preprocessed to match feature extraction format)
    har_data = pd.read_csv(har_data_file)
    X_har = har_data.drop(columns=["source"])  # Features
    y_har = har_data["source"]  # Labels

    # Predict using the trained model
    y_pred = model.predict(X_har)

    # Evaluate performance
    print("HAR Dataset Classification Report:")
    print(classification_report(y_har, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_har, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix (HAR Dataset)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load pre-trained model and HAR dataset
    model_path = "../models/random_forest_model.pkl"

    feature_model = joblib.load(model_path)
    test_model(feature_model, "../data/processed/test_feature_data.csv")
