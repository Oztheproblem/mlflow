# 🧠 MLflow Learning Journey — TL;DR

This repo is my mini **MLOps lab** built with MLflow.  
I go from a **served Production model** → to **logging live predictions** → to **drift monitoring** → to **trend plots** → to **drift alerts**.

What you’ll see here:

- `predict_live.py` logs live predictions to MLflow
- `lesson15_monitoring.py` turns them into monitoring stats
- `lesson16_trends.py` exports history + a mean-prediction trend plot
- `lesson17_alerts.py` computes a dynamic drift threshold and writes alerts to `alerts.log` + the `drift_alerts` experiment

Everything below this TL;DR is a step-by-step breakdown of Lessons 1–17.

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

## **Lesson 15 — Real-Time Monitoring (Hourly Checks) & Logging to MLflow**

### 🎯 Goal

Run a lightweight **real-time monitoring** script every hour that:

- pulls recent live predictions
- summarizes them (mean/median/std/min/max)
- logs monitoring metrics back into MLflow

This continues the pipeline from Lessons 12–14:

```
serve model → live predictions → logging → monitoring stats → MLflow UI
```

---

### 🔍 What I Did

- Created a new script: `lesson15_monitoring.py`
- Pulled the most recent runs from experiment: **`live_predictions_log`**
- Converted millisecond timestamps to datetimes
- Filtered predictions using a sliding window (**last 60 minutes**)
- Extracted prediction values from MLflow runs using:
  - `metrics.prediction`
- Computed real monitoring stats:
  - mean prediction
  - median prediction
  - standard deviation
  - min / max
- Logged monitoring metrics into a separate MLflow experiment:
  - **`drift_monitoring`**
- Confirmed monitoring runs appear in MLflow UI

---

### 💡 Key Takeaways

- Monitoring is just another MLflow experiment.
- `live_predictions_log` stores **raw live predictions**.
- `drift_monitoring` stores **monitoring summaries**.
- Built a repeatable monitoring job:

```
fetch runs → filter by time → compute stats → log results back to MLflow
```

- Reinforced the golden rule:

✅ **No fresh predictions = nothing to monitor.**

- This monitoring script sets up future MLOps upgrades like:
  - automated alerts
  - drift thresholds
  - scheduled monitoring (cron / GitHub Actions)
  - retraining triggers

---

### 🧨 Issues & Fixes

#### 1. “⚠️ No predictions in the last 1 hour(s).”

**Cause:** No recent prediction runs inside the monitoring time window.

**Fix:** Generate fresh predictions first, then monitor.

```bash
python predict_live.py
python predict_live.py
python predict_live.py
python lesson15_monitoring.py
```

---

#### 2. Connection refused on `127.0.0.1:5001`

**Cause:** The MLflow model server wasn’t running, so `predict_live.py` had no endpoint.

**Fix:** Re-serve the Production model, then send predictions again.

```bash
mlflow models serve `
  -m "models:/Lesson11_AutoPromoteModel/Production" `
  --port 5001 `
  --env-manager local
```

Then re-run:

```bash
python predict_live.py
python predict_live.py
python predict_live.py
python lesson15_monitoring.py
```

---

#### 3. MLflow UI not opening

**Cause:** MLflow UI wasn’t running (or the terminal was closed).

**Fix:** Restart UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

#### 4. PowerShell activation blocked

**Cause:** Execution policy prevented `.venv` activation.

**Fix:** Temporary bypass per terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

✅ Result: Monitoring script runs cleanly and logs stats into MLflow.

---

### 🧾 How to Run Lesson 15

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

#### 3. Run Hourly Monitoring Script

```bash
python lesson15_monitoring.py
```

---

#### 4. View in MLflow UI

```bash
mlflow ui
```

Check:

- `live_predictions_log` → raw prediction runs
- `drift_monitoring` → monitoring summary runs

Each monitoring run logs:

- `mean_prediction`
- `median_prediction`
- `std_prediction`
- `min_prediction`
- `max_prediction`

---

## **Lesson 16 — Monitoring History & Trend Plots**

### 🎯 Goal

Turn the `drift_monitoring` runs into a **history dashboard** by:

- pulling all monitoring runs from MLflow
- saving them into a tidy CSV file
- plotting **mean prediction over time**

This lesson builds on Lesson 15:

```
live predictions → drift_monitoring runs → history CSV → trend plots
```

---

### 🔍 What I Did

- Created a new script: `lesson16_trends.py`
- Loaded runs from the **`drift_monitoring`** experiment using:

  ```python
  runs_df = mlflow.search_runs(experiment_names=["drift_monitoring"])
  ```

- Converted MLflow millisecond timestamps into readable datetimes:

  ```python
  runs_df["start_time"] = pd.to_datetime(
      runs_df["start_time"], unit="ms", utc=True
  ).dt.tz_convert("Europe/London")  # adjust timezone if needed
  ```

- Selected the key monitoring columns:

  - `run_id`
  - `start_time`
  - `metrics.mean_prediction`
  - `metrics.median_prediction`
  - `metrics.std_prediction`
  - `metrics.min_prediction`
  - `metrics.max_prediction`

- Sorted runs by `start_time` and reset the index so history is chronological
- Saved all monitoring history to a CSV file:

  ```python
  df.to_csv("monitoring_history.csv", index=False)
  ```

- Plotted **mean prediction vs. time** with `matplotlib` and saved the figure:

  ```python
  plt.plot(df["start_time"], df["metrics.mean_prediction"], marker="o")
  plt.savefig("monitoring_mean_prediction.png")
  ```

- Confirmed both files were created in the project root:
  - `monitoring_history.csv`
  - `monitoring_mean_prediction.png`

---

### 💡 Key Takeaways

- MLflow experiments can also act as a **time-series store** for monitoring data.
- Converting MLflow timestamps to real datetimes makes it easy to:
  - analyse recent vs old runs
  - plot trends and spot drift visually
- A simple pattern for monitoring analytics:

  ```
  MLflow → Pandas (history) → CSV export → Matplotlib plot
  ```

- Storing monitoring history in CSV means you can:
  - inspect runs in VS Code / Excel
  - quickly share results with others
  - feed the data into other tools (dashboards, BI, etc.)
- Even a **single chart** (mean prediction vs time) can reveal:
  - drift trends
  - sudden shifts after a new model deploy
  - periods where monitoring wasn’t running

---

### 🧨 Issues & Fixes

#### 1. “⚠️ No runs found in 'drift_monitoring'.”

**Cause:** `lesson15_monitoring.py` hasn’t been run yet, or there are no monitoring runs in that experiment.

**Fix:**

- Run the Lesson 15 script a few times to generate monitoring runs:

  ```bash
  .venv\Scripts\Activate.ps1
  python lesson15_monitoring.py
  python lesson15_monitoring.py
  python lesson15_monitoring.py
  ```

- Then re-run Lesson 16:

  ```bash
  python lesson16_trends.py
  ```

---

#### 2. Missing metric columns (e.g. `metrics.mean_prediction` not found)

**Cause:** Earlier versions of the monitoring script didn’t log all metrics, or the experiment name is wrong.

**Fix:**

- Check that `lesson15_monitoring.py` logs:

  - `mean_prediction`
  - `median_prediction`
  - `std_prediction`
  - `min_prediction`
  - `max_prediction`

- Confirm the experiment name in both scripts:

  ```python
  experiment_name = "drift_monitoring"
  ```

- Re-run Lesson 15 to create new runs with all metrics, then run Lesson 16 again.

---

#### 3. Matplotlib / plotting issues

**Typical causes:**

- Using an interactive backend that doesn’t play nicely with the terminal.
- Script trying to show the plot instead of saving it.

**Fix / Design choice:**

- The script uses **`plt.savefig()` + `plt.close()`** only — no `plt.show()`.
- This avoids GUI/backend issues and always writes directly to:

  - `monitoring_mean_prediction.png`

If the PNG doesn’t appear, double-check that you are running the script from the `mlflow_project` folder and that you have write permissions there.

---

### 🧾 How to Run Lesson 16

#### 1. (Optional) Generate some fresh monitoring runs

From inside `mlflow_project`:

```bash
.venv\Scripts\Activate.ps1

# Assumes Lesson 15 is already set up & working
python lesson15_monitoring.py
python lesson15_monitoring.py
python lesson15_monitoring.py
```

> The more runs you have, the better your history & charts will look.

---

#### 2. Run the Lesson 16 trend script

```bash
.venv\Scripts\Activate.ps1
python lesson16_trends.py
```

This will:

- query **`drift_monitoring`**
- write **`monitoring_history.csv`**
- create **`monitoring_mean_prediction.png`**

---

#### 3. Inspect the Outputs

- Open `monitoring_history.csv` in:

  - VS Code
  - Excel
  - or any CSV viewer

  You should see one row per monitoring run with:

  - timestamp (`start_time`)
  - `mean_prediction`
  - `median_prediction`
  - `std_prediction`
  - `min_prediction`
  - `max_prediction`

- Open `monitoring_mean_prediction.png`  
  and review the **trend of mean predictions over time** to spot jumps or drift.

## **Lesson 17 — Drift Alerts & Dynamic Thresholds**

### 🎯 Goal

Add a simple **drift alerting layer** on top of the monitoring pipeline by:

- loading recent monitoring runs from **`drift_monitoring`**
- computing a **dynamic drift threshold** from history
- comparing the **latest mean prediction** against that threshold
- writing the result to an **`alerts.log`** file
- logging an `alert_triggered` flag and related metrics into a new MLflow experiment: **`drift_alerts`**

This builds directly on Lessons 15–16:

```
live predictions → monitoring runs → history → drift alerts
```

---

### 🔍 What I Did

- Created a new script: `lesson17_alerts.py`
- Defined constants at the top:

  ```python
  EXPERIMENT_MONITORING = "drift_monitoring"
  EXPERIMENT_ALERTS = "drift_alerts"

  BASELINE_WINDOW = 20        # how many runs to use for history
  STD_MULTIPLIER = 3.0        # how “wide” the threshold is
  ALERT_LOG_FILE = "alerts.log"
  ```

- Loaded all monitoring runs from **`drift_monitoring`**:

  ```python
  runs_df = mlflow.search_runs(experiment_names=[EXPERIMENT_MONITORING])
  ```

- Converted MLflow timestamps to datetimes:

  ```python
  runs_df["start_time"] = pd.to_datetime(
      runs_df["start_time"], unit="ms", utc=True
  )
  ```

- Selected and sorted the key columns:

  ```python
  cols = [
      "run_id",
      "start_time",
      "metrics.mean_prediction",
  ]

  runs_df = runs_df[cols].sort_values("start_time").reset_index(drop=True)
  ```

- Used the last `BASELINE_WINDOW` runs to compute a **dynamic threshold**:

  ```python
  history = df.tail(BASELINE_WINDOW)
  mean_series = history["metrics.mean_prediction"]

  baseline_mean = float(mean_series.mean())
  baseline_std = float(mean_series.std(ddof=0))  # population std
  threshold = baseline_mean + STD_MULTIPLIER * baseline_std
  ```

- Took the **latest monitoring run** and compared it with the threshold:

  ```python
  latest = df.iloc[-1]
  latest_mean = float(latest["metrics.mean_prediction"])
  latest_time = latest["start_time"]

  alert_triggered = latest_mean > threshold
  ```

- Built a human-readable message with all the key numbers:

  ```python
  status = "TRIGGERED" if alert_triggered else "OK"
  message = (
      f"Alert status: {status} | "
      f"latest_mean={latest_mean:.4f}, "
      f"baseline_mean={baseline_mean:.4f}, "
      f"baseline_std={baseline_std:.4f}, "
      f"threshold={threshold:.4f}, "
      f"time={latest_time}"
  )
  ```

- Appended this message to **`alerts.log`** with a UTC timestamp:

  ```python
  from datetime import datetime
  from pathlib import Path

  def log_alert_to_file(message: str, log_path: Path) -> None:
      timestamp = datetime.utcnow().isoformat()
      log_path.parent.mkdir(parents=True, exist_ok=True)
      with log_path.open("a", encoding="utf-8") as f:
          f.write(f"[{timestamp} UTC] {message}\n")
  ```

- Logged an alert “summary run” into a new MLflow experiment **`drift_alerts`**:

  ```python
  mlflow.set_experiment(EXPERIMENT_ALERTS)
  with mlflow.start_run(run_name="lesson17_drift_alert"):
      mlflow.log_metric("latest_mean_prediction", latest_mean)
      mlflow.log_metric("baseline_mean", baseline_mean)
      mlflow.log_metric("baseline_std", baseline_std)
      mlflow.log_metric("dynamic_threshold", threshold)
      mlflow.log_metric("alert_triggered", int(alert_triggered))
  ```

- Printed a final success message:

  ```python
  print("✅ Lesson 17 complete: alert check recorded in alerts.log and MLflow.")
  ```

---

### 💡 Key Takeaways

- You can treat **monitoring runs as a time series** and compute meaningful thresholds from recent history.
- A simple but practical drift rule:

  ```text
  alert if latest_mean > baseline_mean + k * baseline_std
  ```

  where:

  - `baseline_mean` = mean of last N runs
  - `baseline_std` = std of last N runs
  - `k` (here `STD_MULTIPLIER`) controls how sensitive the alerting is

- Writing alerts to a plain-text **`alerts.log`** gives you a cheap, grep-able audit trail.
- Creating a dedicated **`drift_alerts`** experiment separates:
  - raw live predictions (`live_predictions_log`)
  - monitoring aggregates (`drift_monitoring`)
  - alert decisions (`drift_alerts`)
- The pattern is now:

  ```
  predictions → monitoring stats → thresholds & alerts → logs + MLflow
  ```

  which is a real-world MLOps alerting workflow in mini form.

---

### 🧨 Issues & Fixes

#### 1. “⚠️ No runs found in 'drift_monitoring'.”

**Cause:** Lesson 15 hasn’t been run, or there are no monitoring runs logged yet.

**Fix:**

- First, generate predictions and monitoring runs (from Lesson 15):

  ```bash
  .venv\Scripts\Activate.ps1
  python predict_live.py
  python predict_live.py
  python predict_live.py

  python lesson15_monitoring.py
  python lesson15_monitoring.py
  ```

- Then re-run Lesson 17:

  ```bash
  python lesson17_alerts.py
  ```

---

#### 2. Missing `metrics.mean_prediction` in `drift_monitoring`

**Cause:** `lesson15_monitoring.py` didn’t log `mean_prediction`, or experiment name mismatch.

**Fix:**

- Check that Lesson 15 is logging:

  ```python
  mlflow.log_metric("mean_prediction", mean_pred)
  ```

- Confirm the experiment name in both scripts is exactly:

  ```python
  "drift_monitoring"
  ```

- Re-run Lesson 15 to create new, correctly-logged monitoring runs, then re-run Lesson 17.

---

#### 3. Baseline window too small

If you have fewer than `BASELINE_WINDOW` runs, your “history” is tiny and the threshold might be unstable.

Options:

- Reduce the window:

  ```python
  BASELINE_WINDOW = 5
  ```

- Or simply generate more monitoring runs before using Lesson 17:

  ```bash
  python lesson15_monitoring.py
  python lesson15_monitoring.py
  python lesson15_monitoring.py
  ```

---

### 🧾 How to Run Lesson 17

From your `mlflow_project` folder:

#### 1. (Optional) Generate fresh monitoring runs

```bash
.venv\Scripts\Activate.ps1

python predict_live.py
python predict_live.py
python predict_live.py

python lesson15_monitoring.py
python lesson15_monitoring.py
```

> This ensures `drift_monitoring` has enough recent data for the alert baseline.

---

#### 2. Run the alert script

```bash
python lesson17_alerts.py
```

This will:

- load monitoring runs from `drift_monitoring`
- compute baseline mean/std and a dynamic threshold
- compare the latest mean prediction to that threshold
- append a line into **`alerts.log`**
- log an alert summary run into the **`drift_alerts`** experiment

---

#### 3. Inspect the Outputs

- Open **`alerts.log`** in your editor:

  You’ll see lines like:

  ```text
  [2025-11-30T15:45:12.345678 UTC] Alert status: OK | latest_mean=0.5231, baseline_mean=0.5102, baseline_std=0.0123, threshold=0.5461, time=2025-11-30 15:44:10+00:00
  ```

- In **MLflow UI**:

  - Look for experiment **`drift_alerts`**
  - Each run corresponds to one alert check and logs:
    - `latest_mean_prediction`
    - `baseline_mean`
    - `baseline_std`
    - `dynamic_threshold`
    - `alert_triggered` (0 or 1)

You can later build dashboards based on these alert runs (e.g. number of alerts per day, how far above threshold, etc.).

---

### 🎉 Final Thoughts

By Lesson 17, you now have:

- **Live predictions** (`live_predictions_log`)
- **Monitoring summaries** (`drift_monitoring`)
- **History + plots** (`monitoring_history.csv`, `monitoring_mean_prediction.png`)
- **Drift alerts** (`alerts.log` + `drift_alerts`)

This is a solid mini MLOps stack:

- log → monitor → analyse → alert

From here, next natural steps are:

- pushing alerts to Slack / email
- configuring thresholds per model / feature
- integrating alerts with retraining or rollback workflows

But even as-is, you’ve built a realistic production-style **drift alert loop**. 🚀

---

<div align="center">

**Thanks for reading — and feel free to ⭐ the repo if you found it helpful!**

</div>
