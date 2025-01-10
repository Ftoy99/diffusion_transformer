import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return 1 / (1 + np.exp(-x))


def sample_timesteps(num_samples=10000):
    normal_samples = np.random.normal(0, 1, size=num_samples)
    return 1 / (1 + np.exp(-normal_samples))  # Logistic samples


# Visualization
def visualize_timestep_distribution(samples):
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Timestep Distribution")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Density")
    plt.show()


def main():
    # Number of timesteps to sample
    num_samples = 10000

    # Sample timesteps and visualize
    timesteps = sample_timesteps(num_samples)
    visualize_timestep_distribution(timesteps)

    print(f"Sampled Timesteps (First 10): {timesteps[:10]}")


if __name__ == "__main__":
    main()
