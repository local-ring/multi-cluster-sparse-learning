import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from solver import Solver

def compute_omse(X, y, w, L, A, model, k_values, i, num_replications):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )
    solver = Solver(models=[model])
    omse = {}
    for k in k_values:
        # solve the original problem using only the training data
        res = solver.fit(X_train, y_train, L, A, k=k, i=i)
        u = res[model]
        # select top-k features based on absolute value of u
        selected_features = np.argsort(np.abs(u))[-k:]

        # evaluate on the model with the selected features
        X_train_sub = X_train[:, selected_features]
        X_test_sub = X_test[:, selected_features]
        lr_model = LinearRegression()
        lr_model.fit(X_train_sub, y_train)

        y_pred = lr_model.predict(X_test_sub)
        mse = np.mean((y_test - y_pred) ** 2)
        print(f"Replication {i+1}/{num_replications} with model {model}, k={k}, MSE={mse:.4f}")
        omse[k] = mse  

    return omse
