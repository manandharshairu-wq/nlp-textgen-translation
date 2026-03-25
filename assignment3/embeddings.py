import numpy as np
import torch
import gensim.downloader as api

def load_glove(name: str = "glove-wiki-gigaword-100"):
    return api.load(name)

def build_glove_matrix(itos, glove_model, embed_dim=100):
    matrix = np.random.normal(scale=0.6, size=(len(itos), embed_dim)).astype(np.float32)
    matrix[0] = np.zeros(embed_dim, dtype=np.float32)  # <pad>
    found = 0

    for i, word in enumerate(itos):
        if word in glove_model:
            matrix[i] = glove_model[word]
            found += 1

    print(f"GloVe found for {found}/{len(itos)} tokens")
    return torch.tensor(matrix, dtype=torch.float32)