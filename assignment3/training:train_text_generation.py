import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import TEXT_GEN_CONFIG, DEVICE
from utils import perplexity_from_loss, simple_tokenize, ensure_dir, save_json
from data.text_generation_data import get_text_generation_dataloaders
from models.text_generation_model import TextGenLSTM
from embeddings import load_glove, build_glove_matrix

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, perplexity_from_loss(avg_loss)

def generate_text(model, prompt, stoi, itos, seq_len, generate_len):
    model.eval()
    toks = simple_tokenize(prompt)
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in toks]

    if len(ids) < seq_len:
        ids = [stoi["<pad>"]] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]

    generated = toks[:]
    current_ids = ids[:]

    with torch.no_grad():
        for _ in range(generate_len):
            x = torch.tensor([current_ids], dtype=torch.long).to(DEVICE)
            logits = model(x)
            next_id = torch.argmax(logits, dim=-1).item()
            next_token = itos[next_id]
            generated.append(next_token)
            current_ids = current_ids[1:] + [next_id]

    return " ".join(generated)

def plot_history(history, name, output_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Text Generation Loss - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_loss.png"))
    plt.close()

def run_text_generation(output_dir="results"):
    ensure_dir(output_dir)

    data = get_text_generation_dataloaders(TEXT_GEN_CONFIG)
    glove = load_glove()
    glove_matrix = build_glove_matrix(data["itos"], glove, TEXT_GEN_CONFIG["embed_dim"])

    results = {}
    experiments = [
        ("LSTM_onehot", "onehot"),
        ("LSTM_glove", "glove"),
    ]

    for exp_name, emb_type in experiments:
        print(f"\n===== Running {exp_name} =====")

        model = TextGenLSTM(
            vocab_size=data["vocab_size"],
            hidden_dim=TEXT_GEN_CONFIG["hidden_dim"],
            num_layers=TEXT_GEN_CONFIG["num_layers"],
            dropout=TEXT_GEN_CONFIG["dropout"],
            embedding_type=emb_type,
            embed_dim=TEXT_GEN_CONFIG["embed_dim"],
            pretrained_matrix=glove_matrix if emb_type == "glove" else None,
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=TEXT_GEN_CONFIG["lr"])
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_ppl": []}

        for epoch in range(TEXT_GEN_CONFIG["epochs"]):
            train_loss = train_epoch(model, data["train_loader"], optimizer, criterion)
            val_loss, val_ppl = eval_epoch(model, data["val_loader"], criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)

            print(
                f"Epoch {epoch+1}/{TEXT_GEN_CONFIG['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            )

        test_loss, test_ppl = eval_epoch(model, data["test_loader"], criterion)
        sample_1 = generate_text(
            model, "to be or not to be", data["stoi"], data["itos"],
            TEXT_GEN_CONFIG["seq_len"], TEXT_GEN_CONFIG["generate_len"]
        )
        sample_2 = generate_text(
            model, "the king hath", data["stoi"], data["itos"],
            TEXT_GEN_CONFIG["seq_len"], TEXT_GEN_CONFIG["generate_len"]
        )

        results[exp_name] = {
            "test_loss": test_loss,
            "test_perplexity": test_ppl,
            "samples": [sample_1, sample_2],
            "history": history,
        }

        plot_history(history, exp_name, output_dir)

    save_json(results, os.path.join(output_dir, "text_generation_results.json"))
    return results