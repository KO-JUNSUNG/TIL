import numpy as np

# Constants
BETA = 11.1
ALPHA = 5e-3
NUM_SAMPLES = 100
ORDER = 9

# Generate sample data
np.random.seed(0)
x_samples = np.random.uniform(-1, 1, NUM_SAMPLES)
noise = np.random.normal(0, np.sqrt(1 / BETA), NUM_SAMPLES)
t_samples = np.cos(2 * np.pi * x_samples) + np.sin(np.pi * x_samples) + noise

# Polynomial features
def polynomial_features(x, order):
    """ Generate polynomial features up to the given order """
    return np.vstack([x**i for i in range(order+1)]).T

# MLE estimation
def mle_regression(x, t, order):
    """ Perform MLE estimation for polynomial regression """
    X = polynomial_features(x, order)
    w_mle = np.linalg.inv(X.T @ X) @ X.T @ t
    return w_mle

# MAP estimation
def map_regression(x, t, order, alpha):
    """ Perform MAP estimation for polynomial regression """
    X = polynomial_features(x, order)
    lambda_identity = alpha * np.eye(order + 1)
    w_map = np.linalg.inv(X.T @ X + lambda_identity) @ X.T @ t
    return w_map

# Bayesian Regression
def bayesian_regression(x, t, order, alpha, beta):
    """ Perform Bayesian regression, returning mean and standard deviation of predictions """
    X = polynomial_features(x, order)
    S_N_inv = alpha * np.eye(order + 1) + beta * (X.T @ X)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ X.T @ t
    def predictive_distribution(x_new):
        x_new = polynomial_features(x_new, order)
        mean = x_new @ m_N
        sigma = 1 / beta + np.sum(x_new @ S_N * x_new, axis=1)
        return mean, np.sqrt(sigma)
    return predictive_distribution

# Main function
def main(x_test):
    w_mle = mle_regression(x_samples, t_samples, ORDER)
    w_map = map_regression(x_samples, t_samples, ORDER, ALPHA)
    predictive_dist = bayesian_regression(x_samples, t_samples, ORDER, ALPHA, BETA)
    
    # Predictions
    print("MLE Prediction:", polynomial_features(np.array([x_test]), ORDER) @ w_mle)
    print("MAP Prediction:", polynomial_features(np.array([x_test]), ORDER) @ w_map)
    mean, std = predictive_dist(np.array([x_test]))
    print("Bayesian Prediction: Mean =", mean, "Std =", std)

if __name__ == "__main__":
    x_test_value = float(input("Enter a test value for x within [-1, 1]: "))
    main(x_test_value)
