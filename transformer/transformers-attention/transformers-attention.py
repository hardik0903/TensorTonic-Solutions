import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.shape[-1]

    # Step 1: raw scores — (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: scale by sqrt(d_k)
    scaled_scores = scores / math.sqrt(d_k)

    # Step 3: softmax over keys (last dim)
    attn_weights = F.softmax(scaled_scores, dim=-1)

    # Step 4: weighted sum of values — (batch, seq_len_q, d_v)
    output = torch.matmul(attn_weights, V)

    return output
    pass