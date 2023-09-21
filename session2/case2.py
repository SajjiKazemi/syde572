import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set the mean and covariance matrix for the Gaussian distribution
mean = np.array([2, 3])
covariance = np.array([[1.0, 0.5],
                       [0.5, 2.0]])

# Generate random data points from the Gaussian distribution
num_samples = 1000
data = np.random.multivariate_normal(mean, covariance, num_samples)

# Create a grid of points to evaluate the Gaussian PDF and plot equiprobability contours
x, y = np.meshgrid(np.linspace(-1, 5, 500), np.linspace(-1, 7, 500))
pos = np.dstack((x, y))
rv = multivariate_normal(mean, covariance)
pdf_values = rv.pdf(pos)

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