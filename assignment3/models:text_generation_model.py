import torch.nn as nn
import torch.nn.functional as F

class TextGenLSTM(nn.Module):
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

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        if self.embedding_type == "glove":
            x = self.embedding(x)
        else:
            x = F.one_hot(x, num_classes=self.vocab_size).float()

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)