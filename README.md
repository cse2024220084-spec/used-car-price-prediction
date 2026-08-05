# 🚗⚡ AutoValuate AI: Used Car Price Prediction Workflow
This document outlines the complete, step-by-step technical workflow used to enhance the **Used Car Price Prediction** project. 

The goal of this enhancement was to train and compare two separate machine learning models (**Random Forest** and **XGBoost**) and integrate them into a modern, highly interactive UI that explains its predictions using SHAP values.

### Why Compare Two Separate Machine Learning Models?

Comparing multiple algorithm architectures is a standard best practice in machine learning for several key reasons:

1. **Benchmarking Performance & Accuracy**: No single algorithm guarantees optimal performance across all datasets (*"No Free Lunch" Theorem*). Evaluating Random Forest against XGBoost allows us to compare regression metrics (MAE, RMSE, $R^2$) to identify which model better captures used car valuation dynamics.
2. **Evaluating Model Stability vs. High Precision**:
   - **Random Forest** builds many trees at the same time and averages them (bagging), which makes it stable and hard to overfit generates stable, conservative price estimates that resist overfitting and ignore noisy data.
   - **XGBoost** builds trees one after another, with each new tree fixing the errors of the past tree (boosting), captures intricate non-linear relationships (e.g., steep depreciation curves for modern luxury/EV models), providing higher precision.
3. **Consensus & Decision Confidence**: Displaying predictions from both models simultaneously in the UI gives users greater confidence when both independent algorithms converge on a similar price range.

### Key Differences Between Random Forest and XGBoost

| Comparison Feature | **Random Forest Regressor** | **XGBoost Regressor** |
| :--- | :--- | :--- |
| **Ensemble Strategy** | **Bagging** (Bootstrap Aggregating): Trains multiple decision trees independently in parallel on bootstrap subsets of data. | **Gradient Boosting**: Trains decision trees sequentially, where each new tree explicitly targets and corrects the residual errors of prior trees. |
| **Tree Construction** | Parallel and independent. Trees do not communicate or learn from each other. | Sequential and dependent. Each tree minimizes loss via gradient descent on residual errors. |
| **Prediction Combination** | Takes the unweighted average (mean) of all individual decision tree outputs. | Calculates a weighted sum of all trees scaled by a learning rate parameter ($\eta$). |
| **Variance & Bias Focus** | Primary goal is to **reduce variance** and prevent overfitting of deep decision trees. | Reduces both **bias and variance** simultaneously through iterative gradient optimization. |
| **Overfitting & Noise Sensitivity** | Highly resistant to overfitting; adding more trees does not cause overfitting. | Higher risk of overfitting if hyper-parameters (learning rate, depth, regularization) are untuned. |
| **Regularization** | Relies on structural tree limits (`max_depth`, `min_samples_split`). | Includes built-in **L1 ($\alpha$) and L2 ($\lambda$) regularization** to penalize complex trees. |

### The Philosophy of a "Good" Machine Learning Model: Generalization vs. Overfitting

In machine learning, it is technically possible to achieve a **0% error rate** on training data (an MAE of $0.00). However, this is known as **Overfitting** and is highly undesirable. 

When a model is overfitted, it stops looking for patterns and simply **memorizes** the exact price of every individual car in the dataset. If a 2018 Toyota Camry sold for exactly $15,432, an overfitted model will act as a lookup table and blindly output $15,432 without actually understanding *why*.

A truly **intelligent machine learning model** must learn to **Generalize**. It should evaluate broader market trends based on features like age, brand, and transmission, and output an **estimated market value**.

To achieve this in our project, we intentionally applied **Regularization Hyperparameters** during training:
- **`max_depth`**: Prevents the decision trees from growing infinitely deep, stopping them from creating highly specific, pure leaf nodes for single cars.
- **`min_samples_leaf` & `min_child_weight`**: Forces the model to look at a cluster of at least 20 to 30 similar cars before making a prediction, ensuring the final output is a smoothed, generalized market average rather than the memorized price of one specific past sale.

By introducing these constraints, the model's error naturally rises (e.g., ~$30), which proves the model is no longer "cheating" by memorizing the data, but is instead thinking intelligently and predicting a true generalized market value!

---

## Phase 1: Model Engineering & Training (`src/train.py`)

1. **Pipeline Verification**: Validated the `preprocess_data` function in `src/preprocess.py` to ensure it correctly handled categorical one-hot encoding and numerical scaling. The final output of the preprocessor yielded exactly **61 features** for training.
2. **Algorithm Selection**:
   * **Random Forest Regressor**: Chosen for its robust, ensemble-based averaging approach (`n_estimators=100`), which prevents overfitting and provides stable, conservative price predictions.
   * **XGBoost Regressor**: Chosen for its sequential error-correction algorithm (`learning_rate=0.1`, `max_depth=6`). XGBoost is highly sensitive to complex patterns and often outperforms standard decision trees on varied datasets.
3. **Execution**: Created `src/train.py` to seamlessly load the raw dataset (`data/raw/used_car_sales.csv`), fit the preprocessor, train both models concurrently, and persist them as `random_forest.joblib` and `xgboost.joblib` in the `models/` directory. 

---

## Phase 2: Backend API Refactoring (`app/main.py`)

1. **Static File Serving**: Configured the FastAPI application to explicitly mount and serve the frontend UI by adding the `StaticFiles` middleware, allowing seamless browser access via `http://127.0.0.1:8000/app/static/index.html`.
2. **Resolving Input Shape Mismatches**: 
   * **The Issue**: The previous implementation artificially padded the incoming data with 4 extra one-hot encoded "dummy" columns (turning 61 features into 65), which caused the new models to instantly crash with a shape mismatch error upon prediction.
   * **The Fix**: Refactored the `_build_model_input` function to safely bypass the unnecessary `tree_preprocessor` padding, directly returning the raw 61-feature numpy array expected by the newly trained `.joblib` models.
3. **Graceful Defaults**: Handled missing features safely. Because the UI was later simplified, the backend was updated to automatically inject statistical medians (e.g., 5 seats, 4 doors) into the prediction payload if those specific fields were omitted.

---

## Phase 3: Frontend UI Modernization (`app/static/index.html`)

1. **Aesthetic & Theme Overhaul (Dark / Light Mode)**: Redesigned the UI with dynamic CSS theme tokens supporting seamless toggling between **Dark Mode** and **Light Mode** (with persistent `localStorage` memory).
2. **Dual-Model Inference & Accuracy Benchmarking**: Rewrote the JavaScript payload handler (`handlePrediction`) to execute simultaneous API requests (`Promise.all`) for Random Forest and XGBoost.
3. **Backend-Driven Actual Price & Variance Evaluation**: Automatically retrieves the **Actual Market Price (Dataset Ground Truth)** from the backend API response (`actual_benchmark`). The UI displays the actual benchmark ticket alongside both Random Forest and XGBoost predictions, calculating live variance metrics (`+$ / -$` and `% error`) and highlighting which model is closer to the ground truth.
4. **Interactive Explanation Card**: Built a dedicated explanatory section on the UI detailing why the actual dataset price differs from the machine learning model predictions (explaining static historical median vs. dynamic feature adjustments from Random Forest bagging and XGBoost gradient boosting).
5. **Form Optimization (UX Improvements)**:
   * **Decluttering**: Simplified low-impact fields for faster user input.
   * **Curated Dropdowns**: Replaced raw numeric inputs for **Year**, **Horsepower**, and **Mileage** with curated `<select>` dropdowns to prevent invalid input.
6. **Feature Explainability Detail**: Integrated clear, human-readable labels for the mathematical features that most heavily impact a car's price:
   * **Make & Model**: Dictates initial MSRP and depreciation curve.
   * **Manufactured Year**: Represents vehicle age and tech generation.
   * **Mileage Impact**: Quantifies wear-and-tear.
   * **Fuel Type**: Reflects market demand differences (Petrol, Diesel, Hybrid, Electric).
   * **Transmission**: Automatic vs. Manual resale premium.
   * **Engine Power**: Horsepower output correlating with luxury/performance tiers.

---

## Phase 4: Final Execution

1. **Environment Setup**: Activated the isolated Python virtual environment containing the necessary dependencies (`scikit-learn`, `xgboost`, `fastapi`, `shap`, etc.).
2. **Server Initialization**: Launched the ASGI server using Uvicorn (`uvicorn app.main:app --reload`), successfully exposing the `/predict` POST endpoint.
3. **Verification**: Executed live predictions via the browser to confirm that both models return distinct, mathematically sound prices and SHAP contributions based on their underlying architectural differences.

---

## Key Technologies Used

- **Machine Learning**: `scikit-learn` (Random Forest, ColumnTransformer), `xgboost` (XGBRegressor)
- **Model Explainability**: `shap` (SHapley Additive exPlanations)
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript (No external frameworks for maximum speed and simplicity)

---

## Quick Start Guide & Backend Flow

Follow this complete step-by-step flow from repository setup to running the backend server and accessing the application:

### Step 1: Clone the Repository
Clone the project repository to your local machine and navigate into the project root directory:
```bash
git clone https://github.com/cse2024220084-spec/used-car-price-prediction.git
cd used-car-price-prediction
```

### Step 2: Create a Python Virtual Environment
Create an isolated Python virtual environment to manage project packages:
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
Activate the virtual environment depending on your operating system:
- **Windows (CMD / PowerShell)**:
  ```cmd
  venv\Scripts\activate
  ```
- **Linux / macOS (Bash / Zsh)**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install All Dependencies (Automated Script or Manual)
Instead of installing Python packages manually one by one, use one of the automated setup scripts to install all dependencies from `requirements.txt` in a single step:

- **Option A: Automated Script (Recommended)**
  - **Windows**: Run `setup.bat` in CMD / PowerShell:
    ```cmd
    setup.bat
    ```
  - **Linux / macOS / Git Bash**: Run `setup.sh`:
    ```bash
    bash setup.sh
    ```
  - **Cross-Platform Python Script**: Run `install_dependencies.py`:
    ```bash
    python install_dependencies.py
    ```

- **Option B: Manual Installation via pip**
  ```bash
  pip install -r requirements.txt
  ```

### Step 5: (Optional) Retrain Machine Learning Models
Pre-trained models (`random_forest.joblib` and `xgboost.joblib`) are already included in the `models/` directory. If you wish to retrain the models from the raw dataset (`data/raw/used_car_sales.csv`), run:
```bash
python src/train.py
```

### Step 6: Start the Backend FastAPI Server
Launch the Uvicorn ASGI server to expose the backend API:
```bash
uvicorn app.main:app --reload
```

### Step 7: Access the Application & API Documentation
Open your web browser to access:
- **Frontend UI Application**: [http://127.0.0.1:8000/app/static/index.html](http://127.0.0.1:8000/app/static/index.html)
- **Backend API Interactive Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Live Demo**: <a href="http://18.136.104.232/app/static/index.html" target="_blank" rel="noopener noreferrer">Click Me 🚀</a>

---