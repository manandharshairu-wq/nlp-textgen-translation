import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_GEN_CONFIG = {
    "seq_len": 20,
    "batch_size": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "embed_dim": 100,
    "epochs": 5,
    "lr": 1e-3,
    "max_vocab": 8000,
    "generate_len": 25,
}

TRANSLATION_CONFIG = {
    "batch_size": 64,
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.2,
    "embed_dim": 100,
    "epochs": 8,
    "lr": 1e-3,
    "max_src_vocab": 10000,
    "max_tgt_vocab": 10000,
    "max_len": 20,
    "teacher_forcing": 0.7,
    "train_size": 20000,
    "val_size": 2000,
    "test_size": 2000,
}