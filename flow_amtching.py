import torch
import torch.nn as nn
import torch.optim as optim


class FlowMatchingDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FlowMatchingDiffusion, self).__init__()
        # Correct the input dimension: input_dim + 1 (to include time)
        self.flow_network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Include time as an input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Output has the same dimension as input
        )

    def forward(self, x, t):
        # Concatenate input data `x` and time `t`
        t = t.view(-1, 1).expand(x.size(0), 1)  # Expand time to match batch size
        xt = torch.cat([x, t], dim=1)  # Concatenate along the feature dimension
        return self.flow_network(xt)  # Output is the learned time-dependent flow


# Define the flow-matching loss function
def flow_matching_loss(flow_model, x, x_target, t):
    """
    flow_model: The flow network (time-dependent velocity field)
    x: Source samples at time t
    x_target: Target samples at time t
    t: Time
    """
    # Predict flow (dx/dt)
    predicted_flow = flow_model(x, t)

    # Compute loss (match flows between source and target distributions)
    loss = torch.mean((predicted_flow - (x_target - x)) ** 2)
    return loss


if __name__ == '__main__':
    # Training setup
    input_dim = 2
    hidden_dim = 64
    model = FlowMatchingDiffusion(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic example data
    N = 128  # Batch size
    x_source = torch.randn(N, input_dim)  # Source distribution
    x_target = x_source + torch.randn_like(x_source) * 0.1  # Target distribution
    t = torch.rand(N, 1)  # Random time steps in [0, 1]

    # Training loop
    for step in range(1000):
        optimizer.zero_grad()
        loss = flow_matching_loss(model, x_source, x_target, t)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
