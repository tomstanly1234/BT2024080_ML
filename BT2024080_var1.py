import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "IMT2024080"
train_file = f"{ROLL}_train_var1.csv"
test_file  = f"{ROLL}_test_var1.csv"

tr = pd.read_csv(train_file)
te = pd.read_csv(test_file)

features = [f"x{i}" for i in range(1, 7)]
X = tr[features].values
y = tr["y"].values
X_test = te[features].values

print("\n========== VAR1 DATASET ==========")
print("Number of data points :", len(tr))
print("Number of input variables :", len(features))
print("Input variables :", features)
print("Target variable : y")
print("==================================\n")

degrees = range(1, 7)
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
kf = KFold(n_splits=3, shuffle=True, random_state=0)
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Z = poly.fit_transform(X)

    print(f"Degree {degree}: {Z.shape[1]} polynomial features")

    for alpha in alphas:
        mse_scores = []
        r2_scores = []

        for train_idx, val_idx in kf.split(Z):
            Z_train = Z[train_idx]
            Z_val = Z[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            scaler = StandardScaler()
            Z_train = scaler.fit_transform(Z_train)
            Z_val = scaler.transform(Z_val)

            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(Z_train, y_train)

            pred = model.predict(Z_val)

            mse_scores.append(mean_squared_error(y_val, pred))
            r2_scores.append(r2_score(y_val, pred))

        results.append({
            "degree": degree,
            "alpha": alpha,
            "mse": np.mean(mse_scores),
            "r2": np.mean(r2_scores)
        })

results = pd.DataFrame(results)

best_mse = results["mse"].min()
acceptable = results[results["mse"] <= best_mse * 1.02]

best = acceptable.sort_values(["degree", "mse"]).iloc[0]

degree = int(best["degree"])
alpha = best["alpha"]

print("\n========== VAR1 RESULT ==========")
print("Selected degree :", degree)
print("Selected alpha  :", alpha)
print("CV MSE          :", best["mse"])
print("CV R2           :", best["r2"])
print("=================================\n")

poly = PolynomialFeatures(degree=degree, include_bias=False)
Z = poly.fit_transform(X)
Z_test = poly.transform(X_test)

print("Final polynomial feature count :", Z.shape[1])

scaler = StandardScaler()
Z = scaler.fit_transform(Z)
Z_test = scaler.transform(Z_test)

model = Lasso(alpha=alpha, max_iter=20000)
model.fit(Z, y)

pred = model.predict(Z_test)

te["y"] = pred

output_file = f"{ROLL}_pred_var1.csv"
te.to_csv(output_file, index=False)

print("Prediction file saved :", output_file)
