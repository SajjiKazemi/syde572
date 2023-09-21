import numpy as np
import matplotlib.pyplot as plt

# Set the mean and covariance matrix for the Gaussian distribution
mean = np.array([2, 3])
covariance = np.array([[1.0, 0.5],
                       [0.5, 2.0]])

# Generate random data points from the Gaussian distribution
num_samples = 1000
data = np.random.multivariate_normal(mean, covariance, num_samples)

# Create a grid of points to evaluate the Gaussian PDF
x, y = np.meshgrid(np.linspace(-1, 5, 500), np.linspace(-1, 7, 500))
pos = np.dstack((x, y))

# Define the Gaussian PDF function
def gaussian_pdf(x, mean, covariance):
    n = mean.shape[0]
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=2)
    return (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)

# Evaluate the Gaussian PDF values
pdf_values = gaussian_pdf(pos, mean, covariance)

# Create a scatter plot of the generated data points
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Generated Data')

# Create filled contour plot for equiprobability contours with colors
contour = plt.contourf(x, y, pdf_values, levels=100, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='PDF Value')

# Set plot labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Gaussian Random Data and Equiprobability Contours')
plt.legend()

# Display the plot
plt.grid()
plt.show()