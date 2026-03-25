from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
import requests

from utils import simple_tokenize

SPECIALS = ["<pad>", "<unk>"]

def load_shakespeare_tokens():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=30).text
    tokens = simple_tokenize(text)
    return tokens

def build_vocab(tokens, max_vocab=8000):
    counter = Counter(tokens)
    most_common = counter.most_common(max_vocab - len(SPECIALS))
    itos = SPECIALS + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode_tokens(tokens, stoi):
    return [stoi.get(tok, stoi["<unk>"]) for tok in tokens]

class TextGenDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx + self.seq_len], dtype=torch.long)
        return x, y

def get_text_generation_dataloaders(config):
    tokens = load_shakespeare_tokens()
    stoi, itos = build_vocab(tokens, config["max_vocab"])
    encoded = encode_tokens(tokens, stoi)

    split1 = int(0.8 * len(encoded))
    split2 = int(0.9 * len(encoded))

    train_ids = encoded[:split1]
    val_ids = encoded[split1:split2]
    test_ids = encoded[split2:]

    train_ds = TextGenDataset(train_ids, config["seq_len"])
    val_ds = TextGenDataset(val_ids, config["seq_len"])
    test_ds = TextGenDataset(test_ids, config["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(itos),
    }