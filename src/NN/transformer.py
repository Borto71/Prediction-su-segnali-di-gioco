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
              d_model: int = 128, #ogni timestep rappresentato da un vettore di 128 numeri
        nhead: int = 4, #divide il vettore di dimensione d_model in nhead "teste", ciascuna impara relazioni differenti tra i token
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",   # "last" o "mean"
    ):
        super().__init__()
        assert pooling in ("last", "mean"), "pooling deve essere 'last' o 'mean'"

        self.seq_len = seq_len
        self.pooling = pooling  #seleziona come riassumere tutti i vettori processati in uno solo da classificare
        # con pooling == last selezioniamo solo il vettore rappresentativo dell'ultimo frame

        # Proiezione feature → spazio del modello, il nostro vettore iniziale viene proiettato allo spazio del Transformer
        self.input_proj = nn.Linear(input_dim, d_model) #prende un vettore di dim input_dim e lo trasforma in un vettore di dim d_model
        self.norm = nn.LayerNorm(d_model)   #normalizza vettore a media 0, dev standard 1

        # Positional embedding: da un ordine ai token, 
        # learnable: il modello puo aggiornare il vettore di pos_embedding e apprendere autonomamente come rappresentare la posizione dei frame nella sequenza
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Encoder Transformer, definisce il singolo layer encoder
        encoder_layer = nn.TransformerEncoderLayer( 
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = 4 * d_model,
            dropout = dropout,
            batch_first = True, 
            activation = "relu",
            norm_first = False,
        )
  
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # stack di  -num_layers- encoder 

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),      #funzione di attivazione non lineare, Permette al modello di apprendere relazioni non lineari tra le feature.
            nn.Dropout(dropout),       #il droput azzera alcune dimensioni dei vettori durante l'allenamento, riduce overfitting
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) = (batch, seq_len, input_dim)
        return: (B, num_classes)
        """
        # Embedding + posizione
        x = self.input_proj(x)                                   # (B, T, d_model), proietta vettori di dim input_dim in dim 128
        x = self.norm(x + self.pos_embedding[:, :x.size(1), :])  # (B, T, d_model), normalizza valori dei vettori, 
        # Aggiunge a ciascun vettore di embedding un vettore learnable di positional embedding,
        # che fornisce informazioni sulla posizione di ciascun frame nella sequenza.


        # Encoder
        x = self.encoder(x)                                      # (B, T, d_model) 
        #Ogni timestep guarda tutti gli altri timesteps tramite self-attention, aggiornando il proprio vettore in base alle informazioni apprese dal contesto

        # Pooling temporale, riduce la sequenza di T (seq_len) vettori a un solo vettore per sequenza, perché il classificatore finale prende un solo vettore per seqeunza
        if self.pooling == "last":
            x = x[:, -1, :]                                      # (B, d_model)
        else:
            x = x.mean(dim=1)                                    # (B, d_model)

        # Logits
        return self.classifier(x)  #per ciascuna seqeunza del batch ottentiamo i punteggi (logits) per ogni possibile azione
