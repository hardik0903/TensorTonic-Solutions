import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rads = 1.0 / np.power(10000, 2* (i//2) / np.float64(d_model))
    angle_rates = pos * angle_rads

    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return pe
    pass