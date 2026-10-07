import numpy as np
import pandas as pd
import warnings
from numpy.polynomial import chebyshev as C
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "IMT2024080"
train_file = f"{ROLL}_train_var2.csv"
test_file  = f"{ROLL}_test_var2.csv"

tr = pd.read_csv(train_file)
te = pd.read_csv(test_file)

features = ["x1", "x2", "x3"]
X = tr[features].values
y = tr["y"].values
X_test = te[features].values

print("\n========== VAR2 DATASET ==========")
print("Number of data points :", len(tr))
print("Number of input variables :", len(features))
print("Input variables :", features)
print("Target variable : y")
print("==================================\n")

lo = X.min(axis=0)
hi = X.max(axis=0)

def scale(A):
    return 2 * (A - lo) / (hi - lo) - 1

X_scaled = scale(X)
X_test_scaled = scale(X_test)

MAX_DEGREE = 12

V_train = [C.chebvander(X_scaled[:, k], MAX_DEGREE) for k in range(3)]
V_test = [C.chebvander(X_test_scaled[:, k], MAX_DEGREE) for k in range(3)]

combos = [
    (i, j, k)
    for i in range(MAX_DEGREE + 1)
    for j in range(MAX_DEGREE + 1 - i)
    for k in range(MAX_DEGREE + 1 - i - j)
]

def design(V, degree):
    selected = [c for c in combos if sum(c) <= degree]

    return np.stack(
        [
            V[0][:, i] * V[1][:, j] * V[2][:, k]
            for i, j, k in selected
        ],
        axis=1
    )

degrees = [2, 4, 6, 8, 10, 12]
alphas = [1e-6, 1e-4, 1e-2, 1]

kf = KFold(n_splits=3, shuffle=True, random_state=0)
results = []

for degree in degrees:
    Z = design(V_train, degree)

    print(f"Degree {degree}: {Z.shape[1]} Chebyshev features")

    for alpha in alphas:
        mse_scores = []
        r2_scores = []

        for train_idx, val_idx in kf.split(Z):
            model = Ridge(alpha=alpha)
            model.fit(Z[train_idx], y[train_idx])

            pred = model.predict(Z[val_idx])

            mse_scores.append(
                mean_squared_error(y[val_idx], pred)
            )
            r2_scores.append(
                r2_score(y[val_idx], pred)
            )

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

print("\n========== VAR2 RESULT ==========")
print("Selected degree :", degree)
print("Selected alpha  :", alpha)
print("CV MSE          :", best["mse"])
print("CV R2           :", best["r2"])
print("=================================\n")

Z = design(V_train, degree)
Z_test = design(V_test, degree)

print("Final Chebyshev feature count :", Z.shape[1])

model = Ridge(alpha=alpha)
model.fit(Z, y)

pred = model.predict(Z_test)

te["y"] = pred

output_file = f"{ROLL}_pred_var2.csv"
te.to_csv(output_file, index=False)

print("Prediction file saved :", output_file)
