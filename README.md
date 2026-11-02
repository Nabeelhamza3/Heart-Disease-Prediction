# Heart-Disease-Prediction
Predicting Heart Disease by comparing Machine Learning Model
Heart Disease Prediction Using Machine Learning

A beginner-to-intermediate machine learning project that walks through a complete classification workflow using a heart disease dataset. It compares Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree models to predict the dataset's `target` label.

## Project workflow

- Load and inspect the dataset's dimensions, data types, and missing values.
- Separate the input features from the `target` label.
- Split the data into training and test sets (80/20, using `random_state=42`).
- Standardize features using statistics learned from the training data, then apply that scaling to the test data.
- Train and compare three scikit-learn classifiers:
  - Logistic Regression
  - K-Nearest Neighbors with `n_neighbors=5`
  - Decision Tree with `random_state=42`
- Compare test accuracy and generate a confusion matrix and classification report for KNN.

The notebook reports the actual scores when run; results are not hard-coded because they depend on the dataset used.

## Repository contents

- `Heart_Disease_Prediction_(Nabeel).ipynb` - the Jupyter notebook containing the analysis and model training.
- `heart.csv` - dataset file required to run the notebook (download separately; it is not included here).

## Dataset

Download the Heart Disease dataset from Kaggle and place `heart.csv` in the notebook's working directory. The CSV must contain a column named `target`; all other columns are treated as input features.

The notebook currently loads the file from `/content/heart.csv`, a Google Colab path. When running locally, change this line to:

```python
data = pd.read_csv("heart.csv")
```

Keep the dataset's source attribution and license information with any redistributed copy.

## Requirements

- Python 3
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- scikit-learn

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn jupyter
```

## Run the notebook

1. Download `heart.csv` and place it in the notebook's working directory.
2. Open `Heart_Disease_Prediction_(Nabeel).ipynb` in Jupyter Notebook or Google Colab.
3. If running locally, update the dataset path as shown above.
4. Run the cells from top to bottom to inspect the data, train the classifiers, and view the evaluation results.

## Evaluation and extensions

Accuracy is used for the initial model comparison. The notebook also prints a confusion matrix and a classification report (precision, recall, and F1-score) for KNN. The accompanying project guide suggests extending these metrics to all three models, tuning KNN's neighbor count, and exploring Decision Tree feature importance.

For medical classification, review recall and false negatives alongside accuracy. The notebook's metrics are an educational exercise and do not validate a model for clinical use.

## Known notebook issue

The model-selection cell in the notebook currently uses `max(models, key=model.get)`, but the dictionary is named `models`. Update it to:

```python
best_model = max(models, key=models.get)
```

This fixes the automatic selection of the classifier with the highest test accuracy.

## Disclaimer

This project is for learning and experimentation only. It is not a clinically validated diagnostic system and must not be used to diagnose heart disease or guide healthcare decisions.

