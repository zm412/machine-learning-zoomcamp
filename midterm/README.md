# ML Zoomcamp 2026 deployment release - midterm project

### Dataset
* **Source:** [Automated Recruitment Dataset (github)](https://raw.githubusercontent.com/zm412/machine-learning-zoomcamp/refs/heads/main/midterm/automated_recruitment_analytics.csv)
* **Original source:** [Automated Recruitment Dataset (Kaggle)](https://www.kaggle.com/datasets/datresearch/automated-recruitment-dataset)
* **Scope:** Structured recruitment records linking target job roles, candidate skills, experience, education, and evaluation features to binary hiring decisions.

### Description 
Job seekers applying for technical and corporate roles rarely know how well their background aligns with real-world hiring patterns before submitting an application. Traditional keyword checklists treat skills as static and universal-ignoring the fact that the same skill or credential carries vastly different weight depending on the target role (e.g., Data Scientist vs. Software Engineer vs. Product Manager).

The goal of this project is to build a machine learning classification service that evaluates a candidate's fit for a specific target role by learning aggregated requirement patterns across historical postings and outcomes for that role category. Rather than relying on hardcoded pass/fail rules, the model captures how combinations of role title, technical skills, experience, education, and screening signals jointly influence hiring outcomes.

## EDA

1. **Data and missing values:** The dataset contains 2,500 records with **zero missing (null) values**.
2. **Removed identifier and non-predictive columns:**
   * `candidate_id` and `job_id` were dropped because unique identifiers carry no predictive signals.
   * `resume_text` was excluded because its structured information is already extracted into columns (`skills`, `experience_years`, `education`).
   * `job_fit_score` was removed because it represents a pre-computed analytical score rather than a raw candidate/job input feature.
3. **Target and class imbalance:**
   * The original `outcome` column contains three classes: `Rejected`, `Shortlisted`, and `Hired`.
   * For binary classification, `Hired` is encoded as `1` (~11.2% of records) and all other outcomes (`Rejected` and `Shortlisted`) are encoded as `0` (88.8%).
   * Because of this 89/11 class imbalance, models are trained with `class_weight='balanced'`.
4. **Skill parsing:**
   * Raw `skills` and `required_skills` are comma-separated strings with over 2,300 unique combinations, but only ~44–53 individual skills. `collect_unique_skills` extracts the individual skills (which also populates the frontend UI skill selector).
   * A function, `normalize_record` (located in `model.py` so training and inference share identical transformations), splits skills into individual indicators and pairs `required_skills` with `job_title` (`job__<role>__skill__<skill>`), before encoding all features in `DictVectorizer`.
5. **Feature Importance (`mutual_info_classif`):**
   * Feature importance was evaluated directly on the vectorized training matrix using `mutual_info_classif`.

## Model training 
### Model Selection, Hyperparameter Tuning & Threshold Optimization

* **Multi-Family Comparison:** Four distinct algorithms spanning linear and tree-based families were trained and evaluated on the validation set using **ROC-AUC** (chosen over raw accuracy due to the 89/11 class imbalance):
  1. **Logistic Regression (`check_best_logistic_auc`):** Tuned inverse regularization strength $C \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0\}$ with `class_weight='balanced'`.
  2. **Decision Tree (`check_best_decision_tree_auc`):** Tuned `max_depth` and `min_samples_leaf`.
  3. **Random Forest (`check_best_forest_auc`):** Tuned `n_estimators` and `max_depth` with `class_weight='balanced'`.
  4. **XGBoost (`check_best_xgb_auc`):** Tuned `learning_rate`, `max_depth`, and `n_estimators`.
* **Winning Model:** **Logistic Regression ($C = 1.0$)** achieved the highest validation ROC-AUC.
     
     Model Comparison (Validation ROC-AUC):
     Model                      Best hyperparameters                       Val ROC-AUC   
    ---                        ---                                        ---            
     **Logistic Regression**    `C=1.0`, `class_weight='balanced'`         **0.9235**    
     XGBoost                    best of grid                               0.9174        
     Random Forest              best of grid, `class_weight='balanced'`    0.9104        
     Decision Tree              best of grid                               0.8248        
 
* **Decision threshold tuning:** A probability-threshold sweep from 0.00 to 1.00 in steps of 0.01 (`research_best_model`) selected **`0.60`** as the F1-optimal cutoff on the validation set. At this threshold, the final model achieves the following performance on the minority `Hired` class:

 Metric (Validation)    Score        
 **F1-Score**             **0.6441**   
 Precision              0.6129       
 Recall                 0.6786       
     
  The fitted `DictVectorizer` and model are exported together to `pipeline.bin`, and the API applies this `0.60` threshold to `hire_probability` to produce the binary `hire` recommendation.

## Scripts
notebook.ipynb - data cleaning, EDA, feature importance, model tuning
train.py - defines preprocessing & training utilities, fits the final model, and saves pipeline.bin

## Reproducibility
### Local run

```bash
uv sync --locked
uv run gunicorn --bind=0.0.0.0:9696 predict:app

# Optional: to regenerate pipeline.bin from scratch
uv run python train.py
```
### Container run

```bash
docker build -t candidate-fit-predictor .
docker run --rm -p 9696:9696 candidate-fit-predictor
```

## Deployment
The model is deployed on the cloud service Render. Since the account is free, the first request will take some time. The service uses Docker and Flask.
* **Live Prediction Service:** `https://machine-learning-zoomcamp.onrender.com`
The API exposes `GET /health` and `POST /predict`. 

## Environment
pyproject.toml
uv.lock
Dockerfile

### Cloud deployment
The core deliverable is a containerized REST API (`predict.py`) deployed on Render:
* **Live Prediction Service:** `https://machine-learning-zoomcamp.onrender.com/predict`
* **Actionable Self-Assessment:** Candidates can test how targeting different job titles or adding specific skills shifts their predicted fit—helping them focus applications where their profile is most competitive and identify high-impact skill gaps before applying. 
* **Any client** - whether a backend service, a web frontend, or a quick `curl` / Postman request—can send a JSON payload with a candidate's profile and target role to the `/predict` endpoint and receive:
  1. **hire_probability:** The predicted likelihood of a positive hiring outcome based on historical patterns for that role type.
  2. **Binary Recommendation (hire):** A binary classification tuned at a `0.60` decision threshold to balance precision and recall.
  
* **Example request:**
```
curl -X POST https://machine-learning-zoomcamp.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"job_title": "Data Scientist", "skills": "python, sql", "experience_years": 3, "education": "Masters"}'
  
```
 
* **Interactive UI Demo**: A lightweight web frontend built to test the live API is also available at [career-oracle-ml.netlify.app](https://career-oracle-ml.netlify.app).
