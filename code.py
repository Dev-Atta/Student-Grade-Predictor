import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
ds1 = pd.read_excel("Dataset_1.xlsx", header=None)
ds2 = pd.read_excel("Dataset_2.xlsx", header=None)

ds1.columns = ds1.iloc[0]
ds1 = ds1.iloc[7:].reset_index(drop=True)
ds1 = ds1.rename(columns={ds1.columns[0]: 'Name'})

train = pd.DataFrame()
train['Name'] = ds1['Name']
train['As']   = pd.to_numeric(ds1['As'],    errors='coerce')
train['Qz']   = pd.to_numeric(ds1['Qz'],    errors='coerce')
train['SI']   = pd.to_numeric(ds1['S-I'],   errors='coerce')
train['SII']  = pd.to_numeric(ds1['S-II'],  errors='coerce')
train['Proj'] = pd.to_numeric(ds1['Proj'],  errors='coerce')
train['GTot'] = pd.to_numeric(ds1['G.Tot'], errors='coerce')
train = train.dropna().reset_index(drop=True)

print("Training data loaded:", len(train), "students")
print(train[['Name','As','Qz','SI','SII','Proj','GTot']].to_string(index=False))

ds2.columns = ds2.iloc[0]
ds2 = ds2.iloc[7:].reset_index(drop=True)
ds2 = ds2.rename(columns={ds2.columns[0]: 'Name'})

test = pd.DataFrame()
test['Name'] = ds2['Name']
test['As']   = pd.to_numeric(ds2['As'],   errors='coerce')
test['Qz']   = pd.to_numeric(ds2['Qz'],   errors='coerce')
test['SI']   = pd.to_numeric(ds2['S-I'],  errors='coerce')
test['SII']  = pd.to_numeric(ds2['S-II'], errors='coerce')
test = test.reset_index(drop=True)

print("\nTesting data loaded:", len(test), "students")
print(test.to_string(index=False))

X = train[['As','Qz','SI','SII','Proj']].values
y = train['GTot'].values
m = len(y)
features = ['As', 'Qz', 'SI', 'SII', 'Proj']

def predict(X, w, b):
    return X @ w + b

def cost(X, y, w, b):
    err = predict(X, w, b) - y
    return (1 / (2*m)) * np.sum(err**2)

def get_grads(X, y, w, b):
    err = predict(X, w, b) - y
    dw = (1/m) * (X.T @ err)
    db = (1/m) * np.sum(err)
    return dw, db

def run_gd(X, y, lr, iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    costs = []
    for i in range(iters):
        dw, db = get_grads(X, y, w, b)
        w = w - lr * dw
        b = b - lr * db
        costs.append(cost(X, y, w, b))
    return w, b, costs

lrs = [0.001, 0.01, 0.1]
iters = 1000
results = {}

for lr in lrs:
    w, b, hist = run_gd(X, y, lr, iters)
    results[lr] = {'w': w, 'b': b, 'hist': hist}
    print(f"\nalpha = {lr}")
    print(f"  Final Cost: {hist[-1]:.4f}")
    print(f"  Bias: {b:.4f}")
    for f, wi in zip(features, w):
        print(f"  w_{f} = {wi:.4f}")
# going with 0.001, seemed most stable
best_lr = 0.001
best_w = results[best_lr]['w']
best_b = results[best_lr]['b']

print("\nBest weights (alpha = 0.001):")
for f, wi in zip(features, best_w):
    print(f"  w_{f} = {wi:.4f}")
print(f"  b = {best_b:.4f}")

print("\nGradient at average student point:")
for f, wi in zip(features, best_w):
    print(f"  dJ/d{f} = {wi:.4f}")

strongest = features[np.argmax(np.abs(best_w))]
print(f"\nStrongest component: {strongest}")
print("Predicted total increases fastest when", strongest, "improves")

proj_avg = train['Proj'].mean()
test2 = test.copy()
test2['As']   = test2['As'].fillna(train['As'].mean())
test2['Qz']   = test2['Qz'].fillna(train['Qz'].mean())
test2['SI']   = test2['SI'].fillna(train['SI'].mean())
test2['SII']  = test2['SII'].fillna(train['SII'].mean())
test2['Proj'] = proj_avg

X_test = test2[['As','Qz','SI','SII','Proj']].values
preds = predict(X_test, best_w, best_b)
preds = np.clip(preds, 0, 100)
test2['Predicted Total'] = preds

print("\nPredictions on Dataset 2:")
print(test2[['Name','As','Qz','SI','SII','Predicted Total']].to_string(index=False))

test2[['Name','As','Qz','SI','SII','Predicted Total']].to_excel("predictions.xlsx", index=False)
print("\nSaved to predictions.xlsx")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Cost vs Iterations for Different Learning Rates")
clrs = ['blue', 'red', 'green']
for ax, lr, c in zip(axes, lrs, clrs):
    ax.plot(results[lr]['hist'], color=c)
    ax.set_title(f"alpha = {lr}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost J")
    ax.grid(True)
plt.tight_layout()
plt.savefig("cost_vs_iterations.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(features, np.abs(best_w), color=['steelblue','seagreen','tomato','orange','purple'])
plt.title("Feature Weights (alpha = 0.001)")
plt.xlabel("Assessment Component")
plt.ylabel("Weight Value")
for i, (f, wi) in enumerate(zip(features, best_w)):
    plt.text(i, abs(wi) + 0.01, f"{wi:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("feature_weights.png", dpi=150)
plt.close()

y_pred = predict(X, best_w, best_b)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot

plt.figure(figsize=(6, 5))
plt.scatter(y, y_pred, color='steelblue', edgecolors='black', alpha=0.7)
lims = [min(y.min(), y_pred.min()) - 2, max(y.max(), y_pred.max()) + 2]
plt.plot(lims, lims, 'r--', label='Perfect fit')
plt.xlabel("Actual Total")
plt.ylabel("Predicted Total")
plt.title("Actual vs Predicted (Training Data)")
plt.text(0.05, 0.9, f"R² = {r2:.4f}", transform=plt.gca().transAxes, color='green', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

print("\nPlots saved.")

print("\nDecision Analysis:")
print("Component importance ranking:")
ranked = sorted(zip(features, np.abs(best_w)), key=lambda x: x[1], reverse=True)
for i, (f, wi) in enumerate(ranked, 1):
    print(f"  {i}. {f}  ->  {wi:.4f}")

print(f"\nStudents should focus most on: {strongest}")