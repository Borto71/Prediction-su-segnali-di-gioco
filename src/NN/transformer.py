import torch
import torch.nn as nn

class PongTransformer(nn.Module):
    """
    Trasformer per predizione comandi Pong.
    - Batch first (input shape: batch, nr_frame, nr_param)
    - Positional embedding learnable
    - LayerNorm dopo proiezione input
    - Dropout nel classificatore
    - Output: ultimo frame (o media su tutti, vedi commento)
    """

    def __init__(
        self,
        nr_param,
        nr_frame,
        nr_mosse,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.2
    ):
        super().__init__()
        self.nr_param = nr_param
        self.nr_frame = nr_frame
        self.d_model = d_model

        # Layer lineare: input → spazio d_model
        self.input_proj = nn.Linear(nr_param, d_model)
        # LayerNorm per stabilità
        self.norm = nn.LayerNorm(d_model)
        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, nr_frame, d_model))
        # Encoder Transformer (batch_first=True semplifica il codice)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, nr_mosse)
        )

    def forward(self, x):
        """
        x: (batch, nr_frame, nr_param)
        """
        x = self.input_proj(x)  # (batch, nr_frame, d_model)
        # Somma positional embedding e normalizza
        x = self.norm(x + self.pos_embedding[:, :x.shape[1], :])
        x = self.transformer_encoder(x)  # (batch, nr_frame, d_model)
        # --- OUTPUT HEAD ---
        # Puoi usare l'ultimo frame:
        x = x[:, -1, :]   # (batch, d_model)
        # Oppure la media su tutti i frame (decommenta per provare):
        # x = x.mean(dim=1)
        out = self.classifier(x)  # (batch, nr_mosse)
        return out
