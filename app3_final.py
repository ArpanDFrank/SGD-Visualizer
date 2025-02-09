import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define a target function (true model) for the MSE calculation
def target_function(X, Y):
    return np.sin(np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(3 * X) * np.sin(3 * Y)

# Generate some random data to simulate training data
np.random.seed(42)
x_data = np.linspace(-1, 1, 100)  # Random x data
y_data = np.linspace(-1, 1, 100)  # Random y data
X, Y = np.meshgrid(x_data, y_data)
Z = target_function(X, Y)

# Flatten the data for easier manipulation in the cost function
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten() + np.random.normal(0, 0.1, size=Z.flatten().shape)  # Adding slight noise
+
# Mean Squared Error (MSE) cost function
def mse_cost(w1, w2, x, y, target):
    predictions = w1 * x + w2 * y
    return np.mean((predictions - target) ** 2)

# Initialize random weights for SGD path
w1, w2 = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
w1_history, w2_history = [w1], [w2]
cost_history = [mse_cost(w1, w2, x_flat, y_flat, z_flat)]
learning_rate = 0.01

# Learning rate decay and number of iterations
num_iterations = 50
decay_rate = 0.99  # Gradually reduce learning rate

# Function to perform one step of stochastic gradient descent (SGD)
def sgd_step(w1, w2, lr, x, y, target):
    # Choose a random subset of the data to introduce stochasticity
    idx = np.random.choice(len(x), size=10, replace=False)
    x_batch, y_batch, z_batch = x[idx], y[idx], target[idx]
    
    predictions = w1 * x_batch + w2 * y_batch
    error = predictions - z_batch
    
    # Gradients for MSE
    w1_grad = (2 / len(x_batch)) * np.dot(error, x_batch)
    w2_grad = (2 / len(y_batch)) * np.dot(error, y_batch)
    
    # Update weights using gradient descent
    w1 -= lr * w1_grad
    w2 -= lr * w2_grad
    
    return w1, w2

# 3D plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cost surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Set axis labels
ax.set_title('Dynamic SGD Path on Cost Function (MSE)')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Cost (MSE)')

# Add color bar for surface visualization
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Initialize the SGD path plots (w1 -> cyan, w2 -> red)
sgd_path_w1, = ax.plot([], [], [], color='cyan', marker='o', markersize=5, label='SGD Path for w1')
sgd_path_w2, = ax.plot([], [], [], color='red', marker='o', markersize=5, label='SGD Path for w2')

# Function to update the SGD paths during animation
def update(i):
    global w1, w2, learning_rate
    
    # Perform an SGD step
    w1, w2 = sgd_step(w1, w2, learning_rate, x_flat, y_flat, z_flat)
    
    # Reduce learning rate after each step
    #learning_rate *= decay_rate
    
    # Record history
    w1_history.append(w1)
    w2_history.append(w2)
    cost_history.append(mse_cost(w1, w2, x_flat, y_flat, z_flat))
    
    # Update path plot for w1 (cyan) and w2 (red)
    sgd_path_w1.set_data(w1_history, np.zeros(len(w1_history)))  # w1 updates, w2 fixed at 0
    sgd_path_w1.set_3d_properties(cost_history)  # Update costs for w1
    
    sgd_path_w2.set_data(np.zeros(len(w2_history)), w2_history)  # w2 updates, w1 fixed at 0
    sgd_path_w2.set_3d_properties(cost_history)  # Update costs for w2

    return sgd_path_w1, sgd_path_w2

# Create the animation
ani = FuncAnimation(fig, update, frames=num_iterations, interval=200, blit=False)

# Show legend
ax.legend()

plt.show()
