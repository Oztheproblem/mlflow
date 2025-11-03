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

> “I thought the easy part was done once I forked the repo… then I met the dependency gods.”

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
