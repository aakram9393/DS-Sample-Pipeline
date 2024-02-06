# Use an official Python runtime as a parent image
FROM python:3.8.18-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt initially
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Define environment variable to specify the Python binary for MLflow
ENV MLFLOW_PYTHON_BIN /usr/local/bin/python

# Echo the Python version
RUN python --version

# Run train.py when the container launches
CMD ["mlflow", "run", ".", "--env-manager", "local", "--experiment-id", "786771019491956264"]