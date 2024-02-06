import mlflow
import argparse
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
# def get_or_create_experiment(experiment_name):
#     experiment = mlflow.get_experiment_by_name(experiment_name)
#     if experiment:
#         return experiment.experiment_id
#     else:
#         return mlflow.create_experiment(experiment_name)

# # Ensure there is an experiment to use
# experiment_name = "Default"
# experiment_id = get_or_create_experiment(experiment_name)

# # Check if we got a valid experiment ID back
# if experiment_id is None:
#     raise Exception(f"Failed to create or retrieve experiment '{experiment_name}'")

# Start an MLflow run within the context of the experiment
# experiment_id=experiment_id
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