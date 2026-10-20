# IS613 AI Phishing Email Detector — Terminal Commands

This guide contains the **PowerShell commands** for running your Python phishing email detection project in VS Code on Windows.

## 1. Open the project folder

In VS Code, open your `AI_phishing_detection_demo` folder (or whichever folder contains `app.py`, `train_model.py`, and `evaluate_model.py`).

Open a PowerShell terminal with **Terminal → New Terminal**.

> Run the commands below from the **project root folder**. The `.venv` folder should be inside that folder.

## 2. Create the Python virtual environment (first-time setup only)

If the `.venv` folder doesn't already exist, run:

```powershell
py -m venv .venv
```

## 3. Install the project dependencies (first-time setup or after package changes)

If you already have a `requirements.txt` file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Train the machine learning model — `train_model.py`

```powershell
.\.venv\Scripts\python.exe train_model.py
```

**Purpose:**
- Reads the labelled training data from `emails.csv`.
- Trains the TF-IDF + Logistic Regression pipeline.
- Saves the trained pipeline to `phishing_model.joblib` (if your training script includes the saving code).
- Prints training messages and sample predictions.

**When to run:** After changing the training dataset or model configuration. You do **not** need to retrain before every demo.

## 5. Evaluate the saved model — `evaluate_model.py`

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

**Purpose:**
- Loads `phishing_model.joblib`.
- Tests the model on `test_emails.csv`.
- Prints accuracy, precision, recall, F1-score, confusion matrix, and individual predictions.

**Previously verified evaluation (10 synthetic test emails):**

| Metric | Value |
|---|---:|
| Accuracy | 70.00% |
| Precision (phishing) | 66.67% |
| Recall (phishing) | 80.00% |
| F1-score (phishing) | 72.73% |

Confusion matrix (rows = actual `[0, 1]`, columns = predicted `[0, 1]`):

```text
[[3 2]
 [1 4]]
```

These figures describe a small, illustrative test set, **not real-world accuracy**. Rerun evaluation if the model changes.

## 6. Launch the Streamlit web app — `app.py`

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Purpose:** Starts the browser-based AI Phishing Email Detector with sample email selection, predictions, phishing scores, word explanations, and expected-label comparisons.

The terminal should show a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser. **Keep the terminal running while using the app.**

## 7. Stop the app

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

To relaunch it, run the command in **Section 6** again.

## 8. Save the installed dependencies to `requirements.txt`

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

This records the packages currently installed in the project's virtual environment. Run it after installing or updating packages, then keep `requirements.txt` with your project files.

## Quick reference

| Task | Command |
|---|---|
| Create virtual environment | `py -m venv .venv` |
| Install dependencies | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Train model | `.\.venv\Scripts\python.exe train_model.py` |
| Evaluate model | `.\.venv\Scripts\python.exe evaluate_model.py` |
| Launch dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Save dependencies | `.\.venv\Scripts\python.exe -m pip freeze > requirements.txt` |
| Stop dashboard | `Ctrl + C` |

## Recommended presentation-day workflow

1. Open the project folder in VS Code.
2. Launch the app with:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

3. Visit `http://localhost:8501`.
4. Select a legitimate email, a phishing email, and a misclassified example to demonstrate strengths and limitations.
5. Stop the app using **Ctrl + C** when finished.

**Note:** As long as `phishing_model.joblib` exists and is compatible with your installed scikit-learn version, you can launch the dashboard without rerunning `train_model.py`.
