import mlflow
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Assuming `best_model` is your trained model object
model_filename = "best_model.joblib"




# Parsing alpha value if passed
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
args = parser.parse_args()

model = RandomForestClassifier(max_depth=5)
# Save the model as 'best_model.joblib' in the current working directory
joblib.dump(model, os.path.join(os.getcwd(), model_filename)) 

# MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")
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

    mlflow.log_artifact(model_filename)

   
 
print("Model training and logging completed.")