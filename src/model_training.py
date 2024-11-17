from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


def train_model_with_hyperparameters(feature_file):
    """
    Train a Random Forest model with hyperparameter tuning on the extracted features.
    """
    # Load feature data
    data = pd.read_csv(feature_file)

    # Prepare features and labels
    X = data.drop(columns=["source"])
    y = data["source"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],            # Number of trees
        'max_depth': [None, 10, 20, 30],          # Maximum depth of trees
        'min_samples_split': [2, 5, 10],          # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],            # Minimum samples required to be a leaf node
        'max_features': ['sqrt', 'log2', None]    # Number of features to consider at every split
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    print("\nBest Hyperparameters:", grid_search.best_params_)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return best_model


def save_model(model, model_dir, model_name="random_forest_model.pkl"):
    """
    Save the trained model to a specified directory.

    Args:
        model: Trained model to save.
        model_dir (str): Directory to save the model.
        model_name (str): Name of the saved model file.
    """
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")



# Example usage
if __name__ == "__main__":
    # Train the model and tune hyperparameters
    model = train_model_with_hyperparameters("../data/processed/feature_data.csv")

    save_model(model, "../models/")