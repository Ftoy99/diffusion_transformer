import numpy as np
import matplotlib.pyplot as plt


def sigmoid_lumina_schedule(t, alpha, beta, mu):
    t = np.asarray(t)  # Convert input to numpy array if it's not already
    y = np.zeros_like(t, dtype=float)  # Initialize output array

    # Apply the first sigmoid for t < mu
    y[t < mu] = 1 / (1 + np.exp(-alpha * (t[t < mu] - mu)))

    # Apply the second sigmoid for t >= mu
    y[t >= mu] = 1 - 1 / (1 + np.exp(-beta * (t[t >= mu] - mu)))

    return y


# Example usage
t_values = np.linspace(0, 1, 1000)  # Create an array of t values
alpha = 6  # Steepness of the first sigmoid
beta = 20  # Steepness of the second sigmoid
mu = 0.6  # Threshold value

y_values = sigmoid_lumina_schedule(t_values, alpha, beta, mu)

# Plot the result
plt.plot(t_values, y_values, label='Sigmoid Lumina Schedule')
plt.axvline(x=mu, color='r', linestyle='--', label=f'Threshold (μ={mu})')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.title('Piecewise Sigmoid Lumina Schedule')
plt.grid(True)
plt.show()
