# Payment Purpose Classification

---

## Repository Structure

- `input_file.tsv`  
  Input file for making predictions. ***You need to replace the contents of this file with your own data, keeping its name.***

- `models/automl_model.pkl`  
  Saved model used for making predictions.

- `notebooks/`  
  Contains Jupyter Notebooks for data analysis (EDA), labeling, training, and cross-validation.

- `predict.py`  
  Script to run model predictions.

- `prediction_api.py`  
  API to provide predictions based on the model.

---

## Running

To run the project, a working Docker setup is required:

1. Place the test file `input_file.tsv` in the root directory of the project **keeping its name**.

2. To build the image and run it, use the following commands:
    ```bash
    docker build -t biv-hack .
    docker run -v ${PWD}:/app biv-hack
    ```

3. During operation, the model logs events to the terminal.

4. As a result, you will obtain a file in the root directory.

(*) **Important Note:** If the test file changes, the image needs to be rebuilt. However, the build time will be significantly reduced due to cache usage.

---

## Build and Launch Time

1. **First Docker Image Build:**  
   - Takes about **10 minutes**.

2. **Subsequent Rebuilds:**  
   - Thanks to caching, the build time is reduced to **2 minutes**.

3. **Inference Time:**  
   - Processing 25,000 records with the model takes approximately **5 minutes 40 seconds**.

---

## Solution Overview

Our approach includes the following key components:

**Embedding Evaluation**
- Testing models from MTEB, selecting the best in terms of accuracy and speed.

**Text Preprocessing**
- Text normalization: converting to lowercase, replacing dates and numbers with templates.
- Calculating additional features: text readability, date differences.

**CatBoost Model Training**
- Classification using embeddings, dates, payment amounts, and generated features.

**Data Labeling with GPT-4o**
- Added 25,000 records. Only records with matching predictions from CatBoost and GPT-4o were used to enhance quality.

**Blending and Metamodel**
- Blending models (logistic regression, CatBoost) to improve accuracy and robustness.

**Optimization and Validation**
- Cross-validation, selecting the best performing model.

**Dockerization**
- Packaging the pipeline in Docker for ease of deployment and reproducibility.

---

## Implementation Details

### Embedding Evaluation
- **Model Benchmarking:** Analyzed models from MTEB to select the optimal ones based on speed and quality.
- **Selection Criteria:** Balance between interpretability and accuracy.

### Model Training
- **Logistic Regression and Blending:** Using a mix of different types of models to enhance accuracy.
- **CatBoost:** Additional model for robustness on imbalanced data.

### Additional Data Labeling
- **GPT-4o Mini API:** Labeled 25,000 records.
- **Filtering:** Used only records where CatBoost and GPT-4o Mini predictions matched.

### Model Optimization
- **Experiments:**
  - Fine-tuning linear layers.
  - TF-IDF and logistic regression.
  - Blending linear models and boosting.
- **Cross-Validation:** Assessing generalization capability.
- **Model Selection:** Choosing the best performing model on the validation set.

### Text Preprocessing
- **Normalization:** Converting text to lowercase, removing punctuation.
- **Tokenization:** Splitting text into tokens.
- **Stop-Words Removal:** Excluding irrelevant words.

### Dockerization
- **Containerization:** Packaging the pipeline in Docker.
- **Reproducibility:** Ensuring consistent environment setup.

---

## Results

- **Accuracy:** High accuracy on the validation set.
- **Efficiency:** Low latency, suitable for real-time applications.
- **Scalability:** Ability to scale horizontally.

---

## Implemented Additional Functionality

- **API Implementation:** RESTful API for interaction with external systems.
- **Multi-Language Data Processing:** Support for multiple languages.

---

## Development Ideas

- **Automated Model Retraining:** Airflow pipeline for model updating.
- **Data Labeling Pipeline:** Convenient process for manual labeling.
- **Using More Powerful Models:** Integrating GPU support for transformers.
- **Web Interface Development:** Visualization of results and analytics.

---
