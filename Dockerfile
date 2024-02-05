# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt initially
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install dependencies for pyenv
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git

# Install pyenv
RUN curl https://pyenv.run | bash

# Add pyenv executable to PATH and
# set up shell environment for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Initialize pyenv and install a specific Python version
RUN pyenv install 3.8.5
RUN pyenv global 3.8.5

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Run train.py when the container launches
CMD ["mlflow", "run", ".", "--experiment-id", "786771019491956264"]