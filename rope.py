import torch
import math

def get_1d_rope(seq_len, d_model):
    """
    Generates a 1D rope for positional encoding.
    :param seq_len: Length of the sequence (number of tokens or elements).
    :param d_model: Dimensionality of the model (embedding dimension).
    :return: A tensor representing the 1D positional encoding.
    """
    positions = torch.arange(0, seq_len).unsqueeze(1).float()  # Shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Exponentially decaying divisor
    encodings = torch.zeros(seq_len, d_model)  # Initialize tensor for encodings

    # Apply sin and cos to even and odd dimensions
    encodings[:, 0::2] = torch.sin(positions * div_term)  # Sin for even dimensions
    encodings[:, 1::2] = torch.cos(positions * div_term)  # Cos for odd dimensions

    return encodings


def get_2d_rope(height, width, d_model):
    """
    Generates a 2D rope for positional encoding.
    :param height: Height (rows) of the data (e.g., image height).
    :param width: Width (columns) of the data (e.g., image width).
    :param d_model: Dimensionality of the model (embedding dimension).
    :return: A tensor representing the 2D positional encoding.
    """
    row_positions = torch.arange(0, height).unsqueeze(1).float()  # Shape: (height, 1)
    col_positions = torch.arange(0, width).unsqueeze(0).float()  # Shape: (1, width)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Exponentially decaying divisor
    encodings = torch.zeros(height, width, d_model)  # Initialize tensor for encodings

    # Apply sin and cos to both row and column dimensions
    encodings[:, :, 0::2] = torch.sin(row_positions * div_term)  # Sin for even dimensions
    encodings[:, :, 1::2] = torch.cos(row_positions * div_term)  # Cos for odd dimensions
    encodings[:, :, 2::2] = torch.sin(col_positions * div_term)  # Sin for even dimensions (columns)
    encodings[:, :, 3::2] = torch.cos(col_positions * div_term)  # Cos for odd dimensions (columns)

    return encodings