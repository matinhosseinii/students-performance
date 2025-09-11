# Student Performance Prediction Project
An end-to-end MLOps project to predict student math scores using a regression model. This project is fully automated, from data ingestion to a containerized web application.

## ✨ Key Features
- Automated Pipeline: End-to-end scripts for data ingestion, preprocessing, and model training.

- Interactive UI: A simple web application built with Streamlit for live predictions.

- Containerized: Fully containerized with Docker for easy deployment and portability.

- CI/CD Automation: A GitHub Actions workflow automates testing and deployment to Docker Hub.

## 🚀 Quick Start (Running the App)
#### This is the fastest way to run the application using the pre-built image from Docker Hub.

1. Pull the Docker Image:

```
docker pull matinhosseinii/student-performance-app:latest
```

2. Run the Application:
```
docker run -p 8501:8501 matinhosseinii/student-performance-app
# Then, open your browser and go to http://localhost:8501.
```

#### You can also build the image yourself and run it (easy commands using Makefile).

1. Build the Docker Image:

```
make build-docker-image
```

2. Run the Application:

```
make docker-run
# Then, open your browser and go to http://localhost:8501.
```


## ⚙️ Local Development Setup
If you prefer to run the project locally without Docker:

1. Set up the Environment:

```
# Create a virtual environment
python -m venv venv

# Activate the virutal environment
# On macOS/Linux: source venv/bin/activate
# On windows:
venv\Scripts\activate

# Install all required packages with their exact versions for reproducibility
pip install -r requirements.txt

# Install the project's source code in editable mode
pip install -e .
```
2. Run the Full Training Pipeline (Optional):
This project includes pre-trained model files. You only need to run this step if you want to re-train the model yourself.
```
python src/StudentsPerformance/pipelines/training_pipeline.py
```
3. Run the Web App:

```
streamlit run app.py
```
# 🧬Project Structure
`/notebooks`: Jupyter notebooks for EDA and experimentation.

`/data`: Datasets (raw and processed) used to train and test models.

`/models`: Models saved as picke files including the preprocessor and the predictor model.

`/src`: Main source code for all components and pipelines.

`/tests`: Pytest suite for the application.

`app.py`: The Streamlit application script.

`Dockerfile`: Defines the Docker container for deployment.

`Makefile`: Contains shortcuts for common commands like build and run.