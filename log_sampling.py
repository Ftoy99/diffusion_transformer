import numpy as np
import matplotlib.pyplot as plt


def log_normal_sampling(num_samples=10000, mu=0.0, sigma=0.5):
    log_normal_samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    log_normal_samples = (log_normal_samples - log_normal_samples.min()) / (
            log_normal_samples.max() - log_normal_samples.min()
    )#norm to 0-1

    return log_normal_samples


# Visualization function
def visualize_log_normal_distribution(samples):
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title("Log-Normal Sampling Distribution")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Density")
    plt.show()


# Main function
def main():
    # Parameters
    num_samples = 10000

    # Sample timesteps using log-normal distribution
    log_normal_timesteps = log_normal_sampling(num_samples)

    # Visualize the timestep distribution
    visualize_log_normal_distribution(log_normal_timesteps)

    print(f"Sampled Timesteps (First 10): {log_normal_timesteps[:10]}")


if __name__ == "__main__":
    main()
