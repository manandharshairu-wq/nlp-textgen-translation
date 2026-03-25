import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout,
                 embedding_type="onehot", embed_dim=100, pretrained_matrix=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_type = embedding_type

        if embedding_type == "glove":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            if pretrained_matrix is not None:
                self.embedding.weight.data.copy_(pretrained_matrix)
            input_dim = embed_dim
        else:
            self.embedding = None
            input_dim = vocab_size

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src):
        if self.embedding_type == "glove":
            embedded = self.embedding(src)
        else:
            embedded = F.one_hot(src, num_classes=self.vocab_size).float()

        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)

        _, hidden = self.encoder(src)
        x = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            x = tgt[:, t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs