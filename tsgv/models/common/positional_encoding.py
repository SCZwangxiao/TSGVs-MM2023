import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    """Fixed positional encoding.
    Args:
        embedding_dim (int): The embedding dimension of positional embeddings.
        max_length (int): The maximum length of positional embeddings.
    """
    def __init__(self, embedding_dim, batch_first=True, dropout=0.1, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
            # [1, max_length, embedding_dim]
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
            # [max_length, 1, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            # [B, L, embedding_dim]
            pe = self.pe[:, :x.size(1), :]
            # [1, L, embedding_dim]
        else:
            # [L, B, embedding_dim]
            pe = self.pe[: x.size(0), :]
            # [L, 1, embedding_dim]
        x = x + pe
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding.
    Args:
        embedding_dim (int): The embedding dimension of positional embeddings.
        max_length (int): The maximum length of positional embeddings.
    """
    def __init__(self, embedding_dim, max_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_length, embedding_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(max_length, dtype=torch.long).expand((1, -1)),
        )

    def forward(self, x):
        # [B, L, embedding_dim]
        B, L = x.shape[:2]
        position_ids = self.position_ids[:,:L].repeat(B, 1)
        # [B, L]
        position_embeddings = self.pe(position_ids)
        # [B, L, embedding_dim]
        return x + position_embeddings