import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import DiT
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


def train():
    # Get device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform for dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])

    # Load pretrained CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load dataset (e.g., CIFAR-10)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = DiT()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):
        for batch in dataloader:
            image_tensor, label_tensor = batch
            # Convert labels to text descriptions
            label_texts = [dataset.classes[label] for label in label_tensor.tolist()]  # Map integers to class names
            label_tokens = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)
            label_embeddings = text_encoder(**label_tokens).pooler_output

            timestep_tensor = torch.rand([image_tensor.shape[0],], device=device)

            # image_tensor  # B C H W
            # label_embeddings  # B 512 DIM
            # timestep_tensor # B

            # Forward pass
            outputs = model(image_tensor.float(), label_embeddings.float(), timestep_tensor.float())
            loss = criterion(outputs, "targets")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Test the model with an example
    test_input = torch.tensor([[.1, .7]])  # Example: x = 3, y = 4
    predicted = model(test_input).item()
    print(f"Test Input: x=.1, y=.7, Predicted: {predicted}, Actual: {(0.1 + 0.7) * 0.1}")


if __name__ == '__main__':
    train()
