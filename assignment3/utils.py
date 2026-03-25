import os
import re
import json
import math
import random
import numpy as np
import torch
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,;:!?'\-]", " ", text)
    return word_tokenize(text)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def perplexity_from_loss(loss: float) -> float:
    return math.exp(loss)

    