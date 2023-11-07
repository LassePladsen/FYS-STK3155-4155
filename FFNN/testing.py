from ffnn import FFNN
from activation import *
from cost import *
from scheduler import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value

lmbda = 0.0001  # shrinkage  hyperparameter lambda
eta = 0.01  # learning rate
degree = 1  # max polynomial degree for design matrix
n_epochs = 1000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Create data set
rng = np.random.default_rng(rng_seed)
x = rng.uniform(-xmax, xmax, size=(n, 1))#.reshape(-1, 1)
noise = rng.normal(0, noise_std, x.shape)
y = 2 + 3*x + 4*x**2# + noise

def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n+1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X


x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2,
                                                    random_state=rng_seed)

X_train = create_X_1d(x_train, degree)
X_test = create_X_1d(x_test, degree)

nn = FFNN(
    dimensions=[X_train.shape[1], 50, 1],
    hidden_func=sigmoid,
    output_func=identity,
    cost_func=cost_ols,
    seed=rng_seed,
)

scores = nn.fit(
        X=X_train,
        t=y_train,
        lam=lmbda,
        epochs=n_epochs,
        scheduler=Constant(eta=eta),
        # scheduler=Adam(eta=eta, rho=0.01, rho2=0.01),
)

pred = nn.predict(X_test)

# PRINT DATA AND PREDICTION
print("\nData:")
print(y.ravel())
print("\nPredictions:")
print(pred.ravel())

# PLOT DATA AND PREDICTION
# mse = scores["train_errors"][-1]
# test_mse = cost_ols(y_test)(pred)
# test_r2 = r2_score(y_test, pred)
# sort_order = np.argsort(x_test.ravel())
# x_sort = x_test.ravel()[sort_order]
# plt.scatter(x_sort, y_test.ravel()[sort_order], 5, label="Test data")
# plt.plot(x_sort, pred.ravel()[sort_order], "r-", label="Prediction fit")
# plt.title(f"$p={degree}$ | $\eta={eta}$ | $\lambda={lmbda}$ | {n_epochs=} | mse={test_mse:.1f} | r2={test_r2:.2f}")
# plt.legend()
# plt.show()

# PLOT MSE AS FUNC OF POLY DEG

print(nn.z_matrices[1])
print(sigmoid(nn.z_matrices[1]))
