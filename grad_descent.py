import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load Diabetes Dataset
# ----------------------------
data = load_diabetes()
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
X = np.c_[np.ones(X.shape[0]), X]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Mean Squared Error
# ----------------------------
def mse(X, y, theta):
    predictions = X @ theta
    return np.mean((predictions - y) ** 2)

# ----------------------------
# Batch Gradient Descent
# ----------------------------
def batch_gd(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    updates = 0

    start = time.time()

    for _ in range(epochs):
        gradient = (2/m) * X.T @ (X @ theta - y)
        theta -= lr * gradient
        updates += 1

    end = time.time()

    return theta, updates, end-start, mse(X, y, theta)

# ----------------------------
# Stochastic Gradient Descent
# ----------------------------
def sgd(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    updates = 0

    start = time.time()

    for _ in range(epochs):
        for i in range(m):
            xi = X[i:i+1]
            yi = y[i]

            gradient = 2 * xi.T @ (xi @ theta - yi)
            theta -= lr * gradient.flatten()
            updates += 1

    end = time.time()

    return theta, updates, end-start, mse(X, y, theta)

# ----------------------------
# Mini-Batch Gradient Descent
# ----------------------------
def mini_batch_gd(X, y, batch_size=64, lr=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    updates = 0

    start = time.time()

    for _ in range(epochs):
        for i in range(0, m, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            gradient = (2/len(xb)) * xb.T @ (xb @ theta - yb)
            theta -= lr * gradient
            updates += 1

    end = time.time()

    return theta, updates, end-start, mse(X, y, theta)

# ----------------------------
# Train Models
# ----------------------------
theta1, upd1, time1, loss1 = batch_gd(X_train, y_train)
theta2, upd2, time2, loss2 = sgd(X_train, y_train)
theta3, upd3, time3, loss3 = mini_batch_gd(X_train, y_train)

# ----------------------------
# Display Results
# ----------------------------
print("\nComparison of Gradient Descent Methods\n")

print(f"{'Optimizer':<20}{'Time(s)':<12}{'Updates':<12}{'Final Loss'}")
print("-"*60)

print(f"{'Batch GD':<20}{time1:<12.4f}{upd1:<12}{loss1:.4f}")
print(f"{'SGD':<20}{time2:<12.4f}{upd2:<12}{loss2:.4f}")
print(f"{'Mini-Batch GD':<20}{time3:<12.4f}{upd3:<12}{loss3:.4f}")

print("\nConclusion:")
print("Mini-Batch Gradient Descent provides a good balance")
print("between execution time, parameter updates, and convergence.")
