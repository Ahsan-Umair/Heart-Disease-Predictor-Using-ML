# Heart Disease Predictor

A supervised-learning project that predicts the presence of heart disease from patient measurements. The notebook combines exploratory analysis, data-quality checks, several classification algorithms, cross-validated tuning, and reusable artifact export.

> This project is intended for learning and experimentation. It is not a medical device and should not guide clinical decisions.

## Workflow

- Remove duplicate rows and invalid categorical values.
- Explore class balance, numeric distributions, chest-pain categories, exercise-induced angina, and correlations.
- Use a stratified train/test split and standardize features where required.
- Compare logistic regression, K-nearest neighbors, and random forest.
- Search K values for KNN and tune random-forest hyperparameters with grid search.
- Evaluate accuracy, precision, recall, F1, and confusion matrices.
- Export the selected nine-neighbor KNN model and standard scaler.

## Notebook walkthrough

1. Inspect types, missing values, duplicates, target balance, and categorical values. Drop duplicate rows and records with `ca == 4` or `thal == 0` before modeling.
2. Plot numeric distributions, chest-pain and exercise-angina groups, a correlation heatmap, and correlations with the heart-disease target.
3. Make a stratified 80/20 split with `random_state=42`. Fit a standard scaler on training data for logistic regression and K-nearest neighbors; use raw features for random forest.
4. Measure accuracy, precision, recall, F1, and confusion matrices for logistic regression. Try neighbor counts 1 through 20, then fit the selected nine-neighbor KNN model and score it on the test set.
5. Fit a 100-tree random forest, inspect feature importances, and tune forest hyperparameters with five-fold grid search. Compare the tuned model against the other approaches.
6. Save the nine-neighbor KNN classifier and its scaler, reload both, and repeat a test prediction. New rows need the original feature columns in the training order before scaling.

## Dataset

`heart.csv` contains 1,025 rows before notebook cleaning. It includes age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, ST depression, slope, vessel count, thalassemia, and the binary `target`.

## Repository contents

| File | Purpose |
| --- | --- |
| `Heart_disease_Predictor.ipynb` | Analysis, training, comparison, tuning, and export |
| `heart.csv` | Source dataset |
| `knn_heart_disease_model.pkl` | Exported KNN classifier |
| `heart_disease_scaler.pkl` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Heart_disease_Predictor.ipynb
```

For inference, apply the saved scaler before passing a feature row to the saved KNN model.
