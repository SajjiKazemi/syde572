import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, expon

def generate_data():
    np.random.seed(42)

    mean_A = np.array([2, 3])
    cov_A = np.array([[1, 0.5], [0.5, 1]])
    data_A = np.random.multivariate_normal(mean_A, cov_A, 100)

    mean_B = np.array([5, 6])
    scale_B = 1.0  # Scale parameter for the exponential distribution
    data_B = np.random.exponential(scale_B, size=(100, 2)) + mean_B

    X_train = np.vstack((data_A, data_B))
    y_train = np.hstack((np.zeros(100), np.ones(100)))

    return X_train, y_train

def maximum_likelihood_classifier(X, mean_A, cov_A, mean_B, scale_B):
    likelihood_A = multivariate_normal.pdf(X, mean=mean_A, cov=cov_A)
    likelihood_B = expon.pdf(np.linalg.norm(X - mean_B, axis=1), scale=scale_B)
    return np.where(likelihood_A > likelihood_B, 0, 1)

def maximum_a_posteriori_classifier(X, prior_A, prior_B, mean_A, cov_A, mean_B, scale_B):
    likelihood_A = multivariate_normal.pdf(X, mean=mean_A, cov=cov_A)
    likelihood_B = expon.pdf(np.linalg.norm(X - mean_B, axis=1), scale=scale_B)
    
    posterior_A = likelihood_A * prior_A
    posterior_B = likelihood_B * prior_B
    
    return np.where(posterior_A > posterior_B, 0, 1)

def plot_results(data_A, data_B, test_points):
    plt.scatter(data_A[:, 0], data_A[:, 1], label='Class A')
    plt.scatter(data_B[:, 0], data_B[:, 1], label='Class B')
    plt.scatter(test_points[:, 0], test_points[:, 1], marker='x', s=100, label='Test Points')

    plt.title('Maximum Likelihood Classifier and Maximum A Posteriori Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.legend()
    plt.show()

def main():
    X_train, y_train = generate_data()

    mean_A = np.mean(X_train[y_train == 0], axis=0)
    cov_A = np.cov(X_train[y_train == 0], rowvar=False)

    mean_B = np.mean(X_train[y_train == 1], axis=0)
    scale_B = 1.0  # Scale parameter for the exponential distribution

    test_points = np.array([[3, 4], [6, 7], [4, 5], [2, 2]])

    # Prior probabilities
    prior_A = 0.4
    prior_B = 0.6

    # Classify using MLC with Gaussian for Class A and Exponential for Class B
    mlc_predictions = maximum_likelihood_classifier(test_points, mean_A, cov_A, mean_B, scale_B)

    # Classify using MAP
    map_predictions = maximum_a_posteriori_classifier(test_points, prior_A, prior_B, mean_A, cov_A, mean_B, scale_B)

    plot_results(X_train[y_train == 0], X_train[y_train == 1], test_points)

    # Display results
    print("MLC Predictions:", mlc_predictions)
    print("MAP Predictions:", map_predictions)

if __name__ == "__main__":
    main()
