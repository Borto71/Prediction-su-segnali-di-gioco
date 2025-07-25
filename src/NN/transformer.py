import torch
import torch.nn as nn

class PongTransformer(nn.Module):

    def __init__(self, nr_param, nr_frame, nr_mosse, d_model=64, nhead=4, num_layers=2):

        #nr_param = nr di parametri , ball_x, ball_y = 2
        #nr_frame = nr di frame assegnati
        #nr_mosse = nr di possibili azioni (su, giu = 2)

        super(PongTransformer, self).__init__()

        self.nr_param = nr_param
        self.nr_frame = nr_frame
        self.d_model = d_model

        # Layer lineare per portare nr_param → d_model
        self.input_proj = nn.Linear(nr_param, d_model)

        # Positional encoding (sin/cos o learnable)
        self.pos_embedding = nn.Parameter(torch.randn(nr_frame, d_model))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, nr_mosse)
        )

    def forward(self, x):
        # x: (batch, nr_frame, nr_param) → (nr_frame, batch, nr_param)
        x = x.transpose(0, 1)
        x = self.input_proj(x) + self.pos_embedding[:, None, :]
        x = self.transformer_encoder(x)  # (nr_frame, batch, d_model)
        x = x[-1]  # Usa l'ultima rappresentazione (come cls-token)
        out = self.classifier(x)  # (batch, nr_mosse)
        return out
