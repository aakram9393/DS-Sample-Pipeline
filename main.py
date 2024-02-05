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
mlflow.set_tracking_uri("http://mlflow:5000")

# Function to check for the default experiment or create it if it does not exist
def get_or_create_experiment(experiment_name="Default"):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with ID: {experiment_id}")
    return experiment_id

# Ensure there is an experiment to use
experiment_id = get_or_create_experiment()

# Start an MLflow run within the context of the experiment
with mlflow.start_run(experiment_id=experiment_id):
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