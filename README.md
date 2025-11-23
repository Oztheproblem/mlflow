<div align="center">

# 🧠 MLflow Learning Journey

### _“Debugs, Dependencies & Determination”_

> A hands-on journey by **[Oz Khan](https://github.com/Oztheproblem)**  
> where every error was a lesson… and every coffee was a commit ☕

</div>

---

# 📘 Overview

This repository documents my guided MLflow learning journey — from setting up virtual environments, battling dependencies, navigating the MLflow UI, to deploying and monitoring live models.

It's designed to be **developer-friendly**, **portfolio-ready**, and **transparent** about the real process of learning MLOps tools: the wins, the mistakes, and the “why is this not importing?!” moments.

---

# 🗂️ What This Repo Includes

- A clean **Python 3.11** virtual environment
- Reproducible **`requirements.txt`**
- MLflow tracking setup
- Scripts for:
  - Experiments
  - Model comparison
  - Model registry & versioning
  - Serving + live predictions
  - Drift monitoring
  - Visualization
- Full progress documented across **14 lessons**

---

# 🧭 Table of Contents

- [Lesson 1 — Environment Setup](#lesson-1--environment-setup)
- [Lesson 2 — Virtual Environments](#lesson-2--virtual-environments)
- [Lesson 3 — Packages & Dependencies](#lesson-3--packages--dependencies)
- [Lesson 4 — Intro to MLflow](#lesson-4--intro-to-mlflow)
- [Lesson 5 — First MLflow Run](#lesson-5--first-mlflow-run)
- [Lesson 6 — Experiment Tracking & Artifacts](#lesson-6--experiment-tracking--artifacts)
- [Lesson 7 — MLflow UI](#lesson-7--mlflow-ui)
- [Lesson 8 — Model Logging](#lesson-8--model-logging)
- [Lesson 9 — Comparing Experiments](#lesson-9--comparing-experiments)
- [Lesson 10 — Model Registry](#lesson-10--model-registry)
- [Lesson 11 — Auto-Promotion Pipeline](#lesson-11--automating-model-evaluation--promotion)
- [Lesson 12 — Model Serving](#lesson-12--serving--live-predictions)
- [Lesson 12.5 — Live Prediction Logging](#lesson-125--live-prediction-logging)
- [Lesson 13 — Visualization & Monitoring](#lesson-13--visualization--monitoring)
- [Lesson 14 — Drift Monitoring](#lesson-14--drift-monitoring--anomaly-detection)

---

# 📝 Lessons

---

## **Lesson 1 — Environment Setup**

A clean foundation is everything.  
Created a dedicated workspace, installed Python 3.11, ensured PATH variables were clean, and prepared for MLflow installation.

---

## **Lesson 2 — Virtual Environments**

Set up a `.venv` and learned:

- Activating/deactivating environments
- Using PowerShell execution policies
- Why dependency isolation prevents chaos
- How MLflow depends on clean environments

---

## **Lesson 3 — Packages & Dependencies**

The “dependency bros.”  
The early days were a mix of pinning versions, resolving mismatches, and understanding:

- `pyarrow`
- `pydantic`
- `protobuf`
- `sqlparse`
- `cachetools`
- `opentelemetry`

This formed the backbone for stable MLflow runs.

---

## **Lesson 4 — Intro to MLflow**

Explored:

- What MLflow is
- Tracking server basics
- Core concepts (runs, experiments, artifacts, metrics, params)
- The folder structure under `mlruns/`

---

## **Lesson 5 — First MLflow Run**

Created a simple script to log:

- Parameters
- Metrics
- Artifacts

Confirmed MLflow UI worked and began tracking experiments reproducibly.

---

## **Lesson 6 — Experiment Tracking & Artifacts**

- Logged parameters (`model_type`, `learning_rate`)
- Logged metrics (`accuracy`, `loss`)
- Saved artifacts like `results.csv` and `accuracy_plot.png`
- Understood the `mlruns/` hierarchy

---

## **Lesson 7 — MLflow UI**

Launched the UI:

```bash
mlflow ui
```

Explored:

- Run comparisons
- Metrics trends
- Artifact previews
- Experiment structure

---

## **Lesson 8 — Model Logging**

Trained a Ridge Regression model:

- Logged:
  - params
  - metrics
  - the model itself
- Viewed everything neatly in MLflow UI
- Built confidence with `mlflow.sklearn.log_model()`

---

## **Lesson 9 — Comparing Experiments**

Created `compare_experiments.py`:

- Tested multiple `alpha` values
- Logged each run
- Compared MSE visually in MLflow UI
- Learned parameter sweeps

---

## **Lesson 10 — Model Registry**

Key actions:

- Registered the best Ridge model
- Assigned model versions
- Explored staging → production transitions
- Understood how real teams manage production ML

---

## **Lesson 11 — Automating Model Evaluation & Promotion**

Automated:

- Pulling all experiment runs
- Ranking by metric (MSE)
- Selecting the best
- Registering the best model
- Promoting it straight to **Production**

A lightweight MLOps CI/CD pipeline.

---

## **Lesson 12 — Serving & Live Predictions**

Served the production model:

```bash
mlflow models serve \
  -m "models:/Lesson11_AutoPromoteModel/Production" \
  --port 5001 \
  --env-manager local
```

Built `predict_live.py` to send live JSON input.  
Received the first real-time prediction:

```text
{'predictions': [174.03919]}
```

Magic.

---

## **Lesson 12.5 — Live Prediction Logging**

Extended Lesson 12:

- Logged prediction inputs
- Logged outputs
- Logged timestamps
- Stored everything in a new MLflow experiment for monitoring

This completed the **live inference → logging** loop.

---

## **Lesson 13 — Visualization & Monitoring**

Generated `prediction_trend.png` showing predictions over time.

Key outputs:

- Logged visualization artifacts
- Logged summary stats (mean/min/max)
- Prepared the foundation for drift detection

---

## **Lesson 14 — Drift Monitoring & Anomaly Detection**

### 🎯 Goal

Detect potential **prediction drift** from recent live predictions.

### 🔍 What I Did

- Created `drift_check.py`
- Loaded runs from `live_predictions_log`
- Converted millisecond timestamps to datetimes
- Filtered only the **last 60 minutes**
- Computed:
  - latest prediction
  - mean (1-hour window)
  - std deviation
  - min/max
- Used a simple drift heuristic:

```python
drift_detected = latest_prediction > mean + 3 * std
```

- Logged drift metrics into experiment: **`drift_monitoring`**
- Viewed everything in MLflow UI

---

### 💡 Key Takeaways

- Learned programmatic MLflow querying with `mlflow.search_runs()`
- Mastered timestamp conversions
- Implemented a basic (but real) drift rule
- Built a monitoring loop:
  ```
  serving → live predictions → logging → drift monitoring
  ```

---

### 🧨 Issues & Fixes

#### 1. MLflow search filter errors

**Fix:** Use simple search; do filtering in Pandas.

```python
runs_df = mlflow.search_runs(experiment_names=["live_predictions_log"])
```

---

#### 2. Datetime comparison errors

**Fix:**

```python
runs_df["start_time"] = pd.to_datetime(
    runs_df["start_time"], unit="ms", utc=True
)
cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=1)
recent = runs_df[runs_df["start_time"] >= cutoff]
```

---

#### 3. “No predictions found”

Solved by generating predictions first:

```bash
python predict_live.py
python predict_live.py
python predict_live.py
```

Then running:

```bash
python drift_check.py
```

---

### 🧾 How to Run Lesson 14

#### 1. Start the Model Server

```bash
cd mlflow_project
.venv\Scripts\Activate.ps1

mlflow models serve `
  -m "models:/Lesson11_AutoPromoteModel/Production" `
  --port 5001 `
  --env-manager local
```

---

#### 2. Generate Live Predictions

```bash
.venv\Scripts\Activate.ps1
python predict_live.py
python predict_live.py
python predict_live.py
```

---

#### 3. Run Drift Check

```bash
python drift_check.py
```

---

#### 4. View in MLflow UI

```bash
mlflow ui
```

Check:

- `live_predictions_log` → raw predictions
- `drift_monitoring` → drift metrics

---

# 🎉 Final Thoughts

This repo represents **the real journey of learning MLOps tools**:  
debugging environments, understanding MLflow deeply, and gradually building a mini production ML pipeline — with drift monitoring included.

If you’re learning MLflow, I hope this helps you follow along or laugh at the same mistakes I made.

---

<div align="center">

**Thanks for reading — and feel free to ⭐ the repo if you found it helpful!**

</div>
