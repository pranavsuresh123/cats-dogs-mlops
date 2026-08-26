# Cats vs Dogs MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for **binary image classification of Cats vs Dogs** for a pet adoption platform.

The pipeline covers the complete machine-learning lifecycle:

```text
Cats & Dogs Dataset
        ↓
Git + Git LFS
        ↓
Data Preparation
        ↓
CNN Model Training
        ↓
MLflow Experiment Tracking
        ↓
Trained Model Artifact
        ↓
FastAPI Inference API
        ↓
Docker Container
        ↓
Pytest
        ↓
GitHub Actions CI
        ↓
GitHub Container Registry
        ↓
Docker Compose Deployment
        ↓
GitHub Actions CD
        ↓
Smoke Testing
        ↓
Monitoring & Post-Deployment Evaluation
```

### Technologies Used

| Component                | Tool                              |
| ------------------------ | --------------------------------- |
| Source code versioning   | Git                               |
| Dataset/model versioning | Git LFS                           |
| Deep learning framework  | PyTorch                           |
| Experiment tracking      | MLflow                            |
| REST API                 | FastAPI                           |
| Containerization         | Docker                            |
| Testing                  | pytest                            |
| CI/CD                    | GitHub Actions                    |
| Container registry       | GitHub Container Registry (GHCR)  |
| Deployment               | Docker Compose                    |
| Deployment runner        | GitHub Actions Self-Hosted Runner |
| Monitoring               | Application logs + in-app metrics |

---

## 2. Dataset

The project uses a **Cats and Dogs binary image classification dataset**.

The dataset contains:

* **Total images:** 24,998
* **Cats:** 12,499
* **Dogs:** 12,499

The dataset is perfectly balanced between the two classes.

### Dataset Split

The dataset was split into training, validation, and test sets:

| Split      |       Cats |       Dogs |      Total |
| ---------- | ---------: | ---------: | ---------: |
| Train      |      9,999 |      9,999 |     19,998 |
| Validation |      1,249 |      1,249 |      2,498 |
| Test       |      1,251 |      1,251 |      2,502 |
| **Total**  | **12,499** | **12,499** | **24,998** |

The split uses a fixed random seed of `42` to ensure reproducibility.

### Dataset Versioning

The assignment permits either DVC or Git-LFS for dataset versioning. **Git LFS was selected** for this project.

Large image files and the trained PyTorch model are tracked using Git LFS, while source code and configuration files are tracked normally with Git.

---

## 3. Data Preprocessing

Images are processed for a standard CNN input format.

### Image preprocessing

* Convert images to RGB
* Resize to **224 × 224**
* Convert images to tensors
* Normalize using standard RGB channel statistics

### Training augmentation

Training images use:

* Random horizontal flip
* Random rotation up to 10 degrees
* Color jitter

Validation and test images use resizing and normalization only, without random augmentation.

This prevents random transformations from affecting evaluation results.

---

## 4. Model Architecture

A baseline CNN was implemented using PyTorch.

### Architecture

```text
Input: 224 × 224 × 3
        ↓
Conv2D: 3 → 32
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 32 → 64
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 64 → 128
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 128 → 256
        ↓
ReLU
        ↓
Adaptive Average Pooling
        ↓
Flatten
        ↓
Fully Connected: 256
        ↓
ReLU
        ↓
Dropout: 0.5
        ↓
Fully Connected: 2
        ↓
Cat / Dog
```

### Training configuration

| Parameter       | Value                                 |
| --------------- | ------------------------------------- |
| Model           | CatsDogsCNN                           |
| Image size      | 224 × 224                             |
| Batch size      | 32                                    |
| Epochs          | 5                                     |
| Optimizer       | Adam                                  |
| Learning rate   | 0.001                                 |
| Training device | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Output classes  | 2                                     |

The model was trained using CUDA acceleration on the NVIDIA RTX 3050 Ti Laptop GPU.

The trained model is saved as:

```text
models/model.pt
```

---

## 5. Model Performance

The final five-epoch model achieved the following results on the held-out test set:

| Metric              |     Result |
| ------------------- | ---------: |
| Validation Accuracy | **88.11%** |
| Test Accuracy       | **89.93%** |
| Precision           | **90.91%** |
| Recall              | **88.73%** |
| F1 Score            | **89.81%** |

The final test accuracy demonstrates that the baseline CNN provides a strong starting point for the MLOps deployment pipeline.

---

## 6. Experiment Tracking with MLflow

MLflow is used to track training experiments.

### Experiment

```text
Cats-Dogs-Classification
```

### Parameters logged

* Model name
* Image size
* Batch size
* Number of epochs
* Learning rate
* Optimizer
* Training device

### Metrics logged

* Training loss
* Training accuracy
* Validation loss
* Validation accuracy
* Test loss
* Test accuracy
* Precision
* Recall
* F1 score

### Artifacts logged

* `model.pt`
* Training/validation loss curve
* Training/validation accuracy curve
* Confusion matrix

MLflow provides reproducible experiment tracking and allows different model runs to be compared.

To start the MLflow UI locally:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

---

## 7. FastAPI Inference Service

The trained model is exposed through a REST API using FastAPI.

### Endpoints

#### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

#### Prediction

```http
POST /predict
```

The endpoint accepts an image file and returns the predicted class and class probabilities.

Example response:

```json
{
  "class": "cat",
  "cat_probability": 0.9961,
  "dog_probability": 0.0039,
  "probability": 0.9961
}
```

#### Monitoring Metrics

```http
GET /metrics
```

Example response:

```json
{
  "request_count": 2,
  "average_latency_ms": 169.07
}
```

Swagger/OpenAPI documentation is automatically available at:

```text
http://localhost:8000/docs
```

---

## 8. Project Structure

```text
cats-dogs-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── data/
│   └── raw/
│       ├── Cat/
│       └── Dog/
│
├── evaluation/
│   ├── post_deployment_eval.py
│   └── samples/
│
├── models/
│   └── model.pt
│
├── scripts/
│   └── smoke_test.py
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── model.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── preprocess.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   └── test_preprocess.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitattributes
├── .gitignore
└── README.md
```

---

## 9. Docker Containerization

The inference API is packaged into a Docker image.

### Dockerfile

The image uses Python 3.12 and contains:

* FastAPI
* Uvicorn
* PyTorch
* torchvision
* Pillow
* Required inference dependencies
* Trained `model.pt`

The application listens on port `8000`.

### Build locally

```powershell
docker build -t cats-dogs-api:latest .
```

### Run locally

```powershell
docker run --rm -p 8000:8000 cats-dogs-api:latest
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## 10. Automated Testing

pytest is used for automated unit testing.

The project includes tests for:

1. Image preprocessing
2. CNN output shape
3. Probability normalization

Running:

```powershell
pytest -v
```

produces:

```text
3 passed
```

The tests verify that:

* Preprocessed images have the expected `3 × 224 × 224` shape
* The model produces two class outputs
* Softmax probabilities sum to 1

---

## 11. Continuous Integration

GitHub Actions is used for CI.

The CI pipeline runs automatically on pushes to `main` and pull requests.

### CI workflow

```text
Git Push / Pull Request
        ↓
Checkout Repository
        ↓
Set up Python 3.12
        ↓
Install Dependencies
        ↓
Run pytest
        ↓
Docker Build
        ↓
Push Image to GHCR
```

The Docker job downloads only the required trained model from Git LFS rather than downloading the complete image dataset.

This avoids unnecessary CI data transfer.

---

## 12. GitHub Container Registry

The Docker image is published to GitHub Container Registry.

Repository/image:

```text
ghcr.io/pranavsuresh123/cats-dogs-mlops:latest
```

The image is automatically built and pushed by GitHub Actions after a successful CI test stage.

---

## 13. Docker Compose Deployment

Docker Compose is used as the deployment target, as permitted by the assignment.

### `docker-compose.yml`

The deployed service uses:

```text
ghcr.io/pranavsuresh123/cats-dogs-mlops:latest
```

and exposes:

```text
localhost:8000
```

### Deployment commands

Pull the registry image:

```powershell
docker compose pull
```

Start the deployment:

```powershell
docker compose up -d
```

Check the running service:

```powershell
docker compose ps
```

Health check:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

## 14. Continuous Deployment

Continuous Deployment is implemented using GitHub Actions and a **self-hosted GitHub Actions runner** running on the deployment machine.

The CD workflow is triggered after a successful CI workflow on `main`.

### CD flow

```text
Successful CI
      ↓
Self-Hosted GitHub Runner
      ↓
Checkout Repository
      ↓
Pull latest GHCR Image
      ↓
docker compose up -d
      ↓
Wait for API
      ↓
Run Smoke Tests
```

The self-hosted runner allows GitHub Actions to update the Docker Compose deployment running on the local Windows/Docker Desktop environment.

The runner is configured using the official GitHub Actions self-hosted runner mechanism.

---

## 15. Smoke Testing

A post-deployment smoke test is implemented in:

```text
scripts/smoke_test.py
```

The smoke test verifies:

1. `/health` responds successfully
2. `/predict` accepts an image
3. A valid class (`cat` or `dog`) is returned
4. The prediction probability is between 0 and 1

The smoke test also waits for the API to become healthy before performing the prediction request.

### Successful smoke test

```text
Health check: PASS
Prediction check: PASS
All smoke tests passed.
```

This smoke test is executed automatically as part of the CD workflow.

---

## 16. Monitoring and Logging

The FastAPI service provides basic monitoring without requiring an external monitoring platform.

### Metrics

The API maintains:

* Request count
* Average request latency

Example:

```json
{
  "request_count": 2,
  "average_latency_ms": 169.07
}
```

### Request Logging

Example deployed logs:

```text
GET /health HTTP/1.1 200 OK

POST /predict prediction=cat latency_ms=290.34

POST /predict HTTP/1.1 200 OK

POST /predict prediction=cat latency_ms=47.79

GET /metrics HTTP/1.1 200 OK
```

Logs intentionally avoid storing the uploaded image contents.

---

## 17. Post-Deployment Performance Evaluation

A small batch of labeled images was sent through the **deployed Docker Compose API** to evaluate model performance after deployment.

### Evaluation results

| Metric    |      Result |
| --------- | ----------: |
| Samples   |          10 |
| Accuracy  |  **90.00%** |
| Precision |  **83.33%** |
| Recall    | **100.00%** |
| F1 Score  |  **90.91%** |

The evaluation used:

* 5 known Cat images
* 5 known Dog images

Each image was sent through the deployed `/predict` endpoint, and the returned prediction was compared with the known true label.

This verifies that the deployed service is producing usable predictions after containerization and deployment.

---

## 18. End-to-End MLOps Workflow

The completed workflow is:

```text
                DATA
                 │
                 ▼
       Cats vs Dogs Dataset
                 │
                 ▼
          Git + Git LFS
                 │
                 ▼
        Data Preparation
                 │
                 ▼
         PyTorch CNN
                 │
                 ▼
          Model Training
                 │
                 ├──────────────► MLflow
                 │                 │
                 │                 ├─ Parameters
                 │                 ├─ Metrics
                 │                 └─ Artifacts
                 │
                 ▼
             model.pt
                 │
                 ▼
             FastAPI
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
      /health  /predict  /metrics
                 │
                 ▼
              Docker
                 │
                 ▼
          GitHub Actions CI
                 │
                 ▼
                GHCR
                 │
                 ▼
        GitHub Actions CD
                 │
                 ▼
      Self-Hosted Runner
                 │
                 ▼
         Docker Compose
                 │
                 ▼
          Deployed API
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Smoke Testing      Monitoring
        │                 │
        └────────┬────────┘
                 ▼
      Post-Deployment Evaluation
```

---

## 19. Reproducing the Project

### Clone repository

```powershell
git clone https://github.com/pranavsuresh123/cats-dogs-mlops.git
cd cats-dogs-mlops
```

### Create virtual environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run tests

```powershell
pytest -v
```

### Run FastAPI locally

```powershell
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

### Run Docker Compose deployment

```powershell
docker compose pull
docker compose up -d
```

Check:

```powershell
docker compose ps
```

Health:

```powershell
curl.exe http://localhost:8000/health
```

---

## 20. Repository

GitHub repository:

**https://github.com/pranavsuresh123/cats-dogs-mlops**

---

## 21. Final Results Summary

| Area                     | Result                                  |
| ------------------------ | --------------------------------------- |
| Dataset                  | 24,998 balanced images                  |
| Data split               | 80% / 10% / 10%                         |
| Image size               | 224 × 224 RGB                           |
| Model                    | PyTorch CNN                             |
| Test accuracy            | **89.93%**                              |
| Test F1                  | **89.81%**                              |
| Experiment tracking      | **MLflow**                              |
| Dataset versioning       | **Git LFS**                             |
| REST API                 | **FastAPI**                             |
| Containerization         | **Docker**                              |
| Unit tests               | **3 passed**                            |
| CI                       | **GitHub Actions**                      |
| Registry                 | **GHCR**                                |
| Deployment               | **Docker Compose**                      |
| CD                       | **GitHub Actions + Self-Hosted Runner** |
| Smoke test               | **Passed**                              |
| Monitoring               | **Request count + latency + logs**      |
| Post-deployment accuracy | **90.00%**                              |
| Post-deployment F1       | **90.91%**                              |

---

## 22. Assignment Coverage

The implementation covers the requested MLOps lifecycle:

* **M1:** Data/version control, CNN model development, MLflow experiment tracking and artifact logging
* **M2:** FastAPI inference service, dependency specification, Docker containerization
* **M3:** pytest unit tests, GitHub Actions CI, Docker image creation and GHCR publishing
* **M4:** Docker Compose deployment, GitHub Actions CD using a self-hosted runner, automated smoke testing
* **M5:** Request logging, request/latency metrics, and post-deployment model performance evaluation
