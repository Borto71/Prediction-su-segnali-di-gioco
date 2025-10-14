# transformer.py (revisione)

import torch
import torch.nn as nn


class PongTransformer(nn.Module):
    """
    Transformer encoder per predire la prossima mossa in Pong.
    - Input:  (batch, seq_len, input_dim)
    - Positional embedding learnable
    - LayerNorm dopo la proiezione
    - Pooling temporale: 'last' (default) oppure 'mean'
    - Output: logits per NUM_CLASSES
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",   # "last" o "mean"
    ):
        super().__init__()
        assert pooling in ("last", "mean"), "pooling deve essere 'last' o 'mean'"

        self.seq_len = seq_len
        self.pooling = pooling

        # Proiezione feature → spazio del modello
        self.input_proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer( 
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = 4 * d_model,
            dropout = dropout,
            batch_first = True, 
            activation = "relu",
            norm_first = False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # stack di encoder

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) = (batch, seq_len, input_dim)
        return: (B, num_classes)
        """
        # Embedding + posizione
        x = self.input_proj(x)                                   # (B, T, d_model)
        x = self.norm(x + self.pos_embedding[:, :x.size(1), :])  # (B, T, d_model)

        # Encoder
        x = self.encoder(x)                                      # (B, T, d_model)

        # Pooling temporale
        if self.pooling == "last":
            x = x[:, -1, :]                                      # (B, d_model)
        else:
            x = x.mean(dim=1)                                    # (B, d_model)

        # Logits
        return self.classifier(x)
