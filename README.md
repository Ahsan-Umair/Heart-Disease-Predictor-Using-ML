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
