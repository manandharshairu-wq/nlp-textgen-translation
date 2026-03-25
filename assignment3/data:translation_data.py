from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from nltk.tokenize import word_tokenize

SPECIALS_MT = ["<pad>", "<unk>", "<sos>", "<eos>"]

def tokenize_en(text):
    return word_tokenize(text.lower().strip())

def tokenize_es(text):
    return word_tokenize(text.lower().strip())

def build_vocab(sentences, tokenizer, max_vocab):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))

    most_common = counter.most_common(max_vocab - len(SPECIALS_MT))
    itos = SPECIALS_MT + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode_sentence(text, stoi, tokenizer, max_len):
    tokens = tokenizer(text)
    ids = [stoi["<sos>"]] + [stoi.get(tok, stoi["<unk>"]) for tok in tokens[:max_len - 2]] + [stoi["<eos>"]]
    return ids

class TranslationDataset(Dataset):
    def __init__(self, split, stoi_src, stoi_tgt, max_len):
        self.examples = []
        for item in split:
            src = item["translation"]["en"]
            tgt = item["translation"]["es"]
            src_ids = encode_sentence(src, stoi_src, tokenize_en, max_len)
            tgt_ids = encode_sentence(tgt, stoi_tgt, tokenize_es, max_len)
            self.examples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def pad_collate(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(x) for x in src_batch)
    tgt_max_len = max(len(x) for x in tgt_batch)

    padded_src, padded_tgt = [], []

    for src, tgt in zip(src_batch, tgt_batch):
        padded_src.append(src + [src_pad_idx] * (src_max_len - len(src)))
        padded_tgt.append(tgt + [tgt_pad_idx] * (tgt_max_len - len(tgt)))

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)

def get_translation_dataloaders(config):
    dataset = load_dataset("opus_books", "en-es")
    full_data = dataset["train"].shuffle(seed=42)

    train_size = config["train_size"]
    val_size = config["val_size"]
    test_size = config["test_size"]

    train_raw = full_data.select(range(0, train_size))
    val_raw = full_data.select(range(train_size, train_size + val_size))
    test_raw = full_data.select(range(train_size + val_size, train_size + val_size + test_size))

    src_texts = [x["translation"]["en"] for x in train_raw]
    tgt_texts = [x["translation"]["es"] for x in train_raw]

    stoi_src, itos_src = build_vocab(src_texts, tokenize_en, config["max_src_vocab"])
    stoi_tgt, itos_tgt = build_vocab(tgt_texts, tokenize_es, config["max_tgt_vocab"])

    train_ds = TranslationDataset(train_raw, stoi_src, stoi_tgt, config["max_len"])
    val_ds = TranslationDataset(val_raw, stoi_src, stoi_tgt, config["max_len"])
    test_ds = TranslationDataset(test_raw, stoi_src, stoi_tgt, config["max_len"])

    def collate_fn(batch):
        return pad_collate(batch, stoi_src["<pad>"], stoi_tgt["<pad>"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "stoi_src": stoi_src,
        "itos_src": itos_src,
        "stoi_tgt": stoi_tgt,
        "itos_tgt": itos_tgt,
        "src_vocab_size": len(itos_src),
        "tgt_vocab_size": len(itos_tgt),
    }