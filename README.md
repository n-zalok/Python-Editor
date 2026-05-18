# ML-Powered Python Code Review System

An end-to-end machine learning system that analyzes Python code and generates style and readability recommendations using AST-based feature engineering, model explainability, monitoring, and automated retraining workflows.

---

## Motivation

This project was inspired by the idea of building production-oriented ML systems rather than isolated models.

The goal was to create a complete ML application capable of:

- analyzing Python code quality,
- generating actionable recommendations,
- monitoring real-world model performance,
- detecting drift,
- and retraining models on newly collected data.

Rather than focusing only on model training, the project explores the full ML lifecycle, from dataset creation and experimentation to deployment-oriented workflows and monitoring.

---

## Problem Definition

The project predicts the quality of Python code based primarily on readability and styling patterns.

Since no labeled dataset existed for this task, a custom regression target was generated programmatically using `Pylint` scores. Particular care was taken to:

- exclude misleading linting behaviors,
- remove trivial or non-functional files,
- and avoid data leakage during dataset splitting.

The final system generates actionable recommendations designed to help users improve the readability and structure of their Python code.

---

## System Overview

```text
Dataset
   ↓
Target Generation (Pylint)
   ↓
Data Cleaning & Filtering
   ↓
Data Exploration
   ↓
Feature Engineering (AST + Embeddings)
   ↓
Model Training & Evaluation
   ↓
SHAP-based Recommendation Engine
   ↓
FastAPI + React Application
   ↓
Monitoring & Drift Detection
   ↓
Retraining & Deployment via MLflow
```

---

## Features

- AST-based Python code feature extraction
- CodeBERT embedding experimentation
- Random Forest regression modeling
- SHAP-based recommendation generation
- Drift detection using Evidently
- Automated monitoring reports
- MLflow model registry integration
- Retraining pipeline with time-decay weighting
- Dockerized backend/frontend services
- Automated tests with Pytest

---

## Suggested Exploration Path

The notebooks document the evolution of the system from exploratory analysis to deployment-oriented workflows.

### Dataset Exploration

- [Data Exploration](./notebooks/data_exploration.ipynb)
- [Target Generation](./notebooks/target_generation.ipynb)
- [Split Data](./notebooks/split_data.ipynb)

### Code Exploration

- [Vectorize Text](./notebooks/vectorize_text.ipynb)
- [Cluster Data](./notebooks/cluster_data.ipynb)
- [Explore Embeddings](./notebooks/explore_embeddings.ipynb)

### Feature Engineering and Initial Model

- [Feature Generation](./notebooks/feature_generation.ipynb)
- [Train Model](./notebooks/train_model.ipynb)
- [Model Evaluation](./notebooks/model_evaluation.ipynb)

### Model Iterations

- [Feature Generation V2](./notebooks/feature_generation_v2.ipynb)
- [Train Model V2](./notebooks/train_model_v2.ipynb)
- [Model Evaluation V2](./notebooks/model_evaluation_v2.ipynb)
- [Train Model V3](./notebooks/train_model_v3.ipynb)
- [Model Evaluation V3](./notebooks/model_evaluation_v3.ipynb)

### Recommendation Generation

- [Generate Recommendatios](./notebooks/generate_recommendations.ipynb)

### Monitoring & Retraining

- [Stimulate Users](./notebooks/stimulate_users.ipynb)
- [Retrain Model](./notebooks/retrain_model.ipynb)

---

## Modeling Journey

The project evolved through multiple iterations focused on balancing predictive performance, explainability, and inference speed.

### Version 1

The first model combined:

- AST-generated structural features
- CodeBERT embeddings

with a `RandomForestRegressor`.

This version established the initial recommendation pipeline and evaluation workflow.

### Version 2

Additional handcrafted features were introduced to improve predictive performance and recommendation quality.

Although performance improved, embedding generation introduced significant latency overhead.

### Version 3

The final version removed transformer embeddings entirely and relied only on engineered structural features.

While this resulted in a small drop in predictive performance, inference speed improved dramatically, making the system substantially faster than running `Pylint` directly.

---

## Recommendation Generation

Recommendations are generated using SHAP values.

For a given code sample:

1. Features with the strongest negative contribution to the predicted score are identified.
2. Feature values are compared against ranges associated with positive model behavior in the training data.
3. Human-readable recommendations are generated dynamically based on those deviations.

This allows the system to produce targeted suggestions rather than only returning a numerical score.

---

## Monitoring & Retraining

The project includes a monitoring workflow designed to simulate real-world ML system maintenance.

### Monitoring

The monitoring pipeline:

- retrieves user submissions associated with the currently deployed model,
- evaluates performance degradation using RMSE,
- detects feature drift using Evidently,
- and generates timestamped HTML reports.

### Retraining

When degradation thresholds are exceeded:

- new user data is retrieved,
- data is split to avoid leakage,
- historical and new data are combined,
- and the model is retrained using time-decay sample weighting to prioritize recent data.

### Deployment

MLflow is used for:

- experiment tracking,
- model versioning,
- staging/production management,
- and deployment selection.

The deployed application always retrieves the currently active production model directly from MLflow.

---

## Application

The project includes a lightweight web application built with:

- FastAPI (backend)
- React (frontend)

Users can:

- register/login,
- submit Python code,
- receive predicted quality scores,
- and get generated improvement recommendations.

User submissions are also stored to support future monitoring and retraining workflows.

---

## Tech Stack

### Machine Learning & Data

- Python
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Evidently

### NLP & Feature Engineering

- CodeBERT
- AST (Abstract Syntax Trees)

### Visualization

- Matplotlib
- Bokeh

### Backend & Frontend

- FastAPI
- React

### MLOps & Infrastructure

- MLflow
- Docker
- Docker Compose
- Pytest

---

## Repository Structure

```text
backend/        FastAPI backend service
frontend/       React frontend
monitoring/     Monitoring, retraining and deployment 
notebooks/      Research and experimentation notebooks
python_editor/  Core ML and recommendation logic
tests/          Automated tests
```

---

## Running the Project

### Run the app using Docker

```bash
docker compose up --build
```

### Experiment with notebooks  install requirements

```bash
pip install -r requirements.txt
```


---

## Notable Engineering Decisions

- Split data by developer/repository to reduce leakage from coding style memorization
- Removed misleading `Pylint` behaviors
- Filtered trivial or non-functional files from the dataset
- Replaced embedding-heavy models after latency analysis showed they were slower than `Pylint`
- Monitoring pipeline tracks both RMSE degradation and feature drift before retraining
- Retraining workflow uses timestamped cutoffs and time-decay sample weighting

---

## Future Work

- Deploy continuous monitoring jobs
- Add CI/CD deployment workflows
- Experiment with transformer fine-tuning
- Expand dataset diversity and scale

