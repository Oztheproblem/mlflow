<div align="center">

# 🧠 MLflow Learning Journey

### _"Debugs, Dependencies & Determination"_

> A hands-on journey by [**Oz Khan**](https://github.com/Oztheproblem)  
> where every error was a lesson, and every coffee was a commit ☕

</div>

---

# 🧠 MLflow Learning Journey — Debugs, Dependencies & Determination

**Author:** Oz Khan ([@Oztheproblem](https://github.com/Oztheproblem))  
**Goal:** To set up MLflow successfully, learn experiment tracking, and strengthen my Python/DevOps fundamentals — one coffee at a time ☕

---

## 🚀 Overview

This project started as a simple “let me fork this and learn MLflow”…  
and quickly turned into a multi-day adventure of debugging, dependency wrestling, and caffeine-powered perseverance 😅

Between switching laptops, rebuilding environments, and learning how every package connects — this repo documents the _real_ side of learning tech:  
figuring things out the hard way (but with humor and persistence).

---

## ⚙️ What’s Inside

- A working **Python 3.11** virtual environment
- MLflow setup (finally! 🎉)
- Carefully restored `.venv` with all dependencies:
  - `mlflow`
  - `pyarrow`
  - `pydantic`
  - `protobuf`
  - `opentelemetry`
  - `cachetools`
  - `sqlparse`
- Clean `requirements.txt` for reproducibility

---

## ☕ The Real Story

> “I thought the easy part was done once I forked the repo… then I met the dependency Bros.”

I’ve spent about **four separate days (spread over a few weeks)** configuring and learning this setup.  
Each day came with new lessons — and yes, maybe one too many visits to _Gail’s Coffee_ debugging environment paths, PowerShell policies, and the mysterious `(END)` screen.

If this README exists, it means I finally got MLflow to import without red text 🎯

---

## 🧩 Why MLflow?

Because it’s a perfect playground for:

- Understanding **Python environments & dependencies**
- Experiment tracking (used in real MLOps setups)
- Building **DevOps-ready projects**
- Strengthening my Git/GitHub workflow

Even if you’re not training models yet, MLflow teaches discipline — versioning, environments, and patience.

---

## 🧪 Progress So Far

### ✅ **Lesson 6 – Experiment Tracking & Artifacts**

- Logged parameters (`model_type`, `learning_rate`)
- Tracked metrics (`accuracy`, `loss`) over multiple steps
- Saved and logged artifacts (`results.csv`, `accuracy_plot.png`)
- Confirmed reproducible experiment structure under `mlruns/`

### ✅ **Lesson 7 – MLflow UI & Visualization**

Launched the **MLflow Tracking UI** to visually explore all logged runs.

**Key steps**

```bash
mlflow ui
```

---

## 🧪 Lesson 8 — Model Logging with MLflow

This lesson was all about teaching MLflow to **track a real machine-learning model** — not just random metrics anymore, but an actual Ridge Regression model trained on the diabetes dataset.

After setting up the environment (and convincing PowerShell to let me run scripts 😅), I managed to:

- Train and evaluate a simple `Ridge(alpha=0.5)` model
- Log parameters, metrics, and the model itself with MLflow
- View it all neatly in the **MLflow UI** (`http://127.0.0.1:5000`)

### 🔍 What I Learned

1. How to structure a repeatable experiment using `mlflow.start_run()`
2. The difference between **parameters** (inputs you tweak) and **metrics** (outputs you measure)
3. How `mlflow.sklearn.log_model()` saves models for future comparison or deployment
4. That warnings don’t always mean panic — sometimes they just mean _“You’re on the cutting edge!”_

### ☕ Debugging Moments

At one point, my terminal felt like it was serving more errors than Gail’s serves flat whites —  
but eventually the environment behaved, dependencies aligned, and the UI popped up like magic.

### 🧭 Next Up

- [ ] Compare multiple `alpha` values and track which performs best
- [ ] Explore MLflow’s **Artifacts** tab and see what’s really being saved
- [ ] Push screenshots and share progress (because proof > promises)

---

---

## ⚙️ Lesson 9 — Comparing Experiments & Model Versions

This lesson focused on extending the previous Ridge Regression example to compare multiple model runs in MLflow.  
By varying the `alpha` parameter, each run was logged with its own metrics and parameters, making it easy to visualise performance differences in the MLflow UI.

### 🔍 What I Did

- Created a new script `compare_experiments.py`
- Tested Ridge Regression models with different `alpha` values
- Logged each run to the **lesson_9_model_comparison** experiment
- Used the MLflow UI to compare MSE values side by side

### 💡 Key Takeaways

1. MLflow automatically organises runs by experiment.
2. Parameter sweeps (like `alpha` values) can be compared visually without extra code.
3. Consistent naming of runs makes tracking progress easier over time.

Next step: identify the best performing model and learn how to **register and version models** within MLflow’s Model Registry.

---

---

## ⚙️ Lesson 10 — Model Registry & Versioning

This lesson introduced the MLflow Model Registry, a key feature for managing model versions and deployment stages.

### 🔍 What I Did

- Logged a Ridge Regression model
- Registered it in the **Model Registry**
- Learned how to access it through the MLflow UI
- Experimented with versioning and model stage transitions (e.g., _Staging_ and _Production_)

### 💡 Key Takeaways

1. Each registered model is tracked with a unique version.
2. The Model Registry centralises deployment-ready models.
3. Transitions between stages (e.g., _Staging → Production_) mirror DevOps release practices.

---

---

## ⚙️ Lesson 11 — Automating Model Evaluation & Promotion

**Goal:** Automatically identify the best-performing model and promote it to production.

### 🔍 What I Did

- Used `MlflowClient()` to connect to the local MLflow tracking server.
- Retrieved all experiment runs and evaluated them by metric (`mse`).
- Automatically selected the best-performing model (lowest MSE).
- Registered it inside the **Model Registry** under a unique model name.
- Automatically promoted that model to the **Production** stage.

### 💡 Key Takeaways

1. This workflow mirrors **DevOps CI/CD pipelines**, but for ML — automating evaluation and deployment.
2. Introduced **automation for evaluation, registration, and version control**.
3. Learned about **model lifecycle management** and MLflow’s upcoming migration from _stages_ → _tags_.
4. Reinforced the concept of continuous delivery — pushing the best model forward while maintaining full version history.

### 🏁 Output Example
