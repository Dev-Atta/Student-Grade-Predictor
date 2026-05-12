# Student Grade Predictor

A machine learning project that predicts student final grades using **Linear Regression built from scratch** using Gradient Descent — no scikit-learn, no shortcuts.

## What it does
- Trains on Dataset_1 (real student marks)
- Predicts final totals for Dataset_2 students
- Tests 3 different learning rates (0.001, 0.01, 0.1)
- Generates graphs: cost curve, feature weights, actual vs predicted

## How to run
```bash
pip install pandas numpy matplotlib openpyxl
python code.py
```

## Output
- predictions.xlsx — predicted totals for test students
- cost_vs_iterations.png — learning curve graph
- feature_weights.png — which assessment matters most
- actual_vs_predicted.png — model accuracy plot

## Tech Stack
Python | NumPy | Pandas | Matplotlib

## Key Concept
Built gradient descent manually without any ML libraries to demonstrate understanding of the math behind linear regression.
