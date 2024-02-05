import mlflow
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parsing alpha value if passed
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
args = parser.parse_args()

# MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLflow run
with mlflow.start_run():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestClassifier(max_depth=5)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Log parameters, metrics, and model
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    
    # Log the sklearn model and register as a version in MLflow model registry
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisClassifier")

print("Model training and logging completed.")