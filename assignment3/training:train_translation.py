import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sacrebleu

from config import TRANSLATION_CONFIG, DEVICE
from utils import ensure_dir, save_json
from data.translation_data import get_translation_dataloaders
from models.translation_model import EncoderGRU, DecoderGRU, Seq2SeqGRU
from embeddings import load_glove, build_glove_matrix

def train_epoch(model, loader, optimizer, criterion, teacher_forcing):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=teacher_forcing)

        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        tgt_gold = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt_gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            tgt_gold = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_gold)
            total_loss += loss.item()

    return total_loss / len(loader)

def ids_to_sentence(ids, itos, pad_idx, sos_idx, eos_idx):
    tokens = []
    for idx in ids:
        if idx == eos_idx:
            break
        if idx not in [pad_idx, sos_idx]:
            tokens.append(itos[idx])
    return " ".join(tokens)

def translate_sentence(model, sentence, stoi_src, stoi_tgt, itos_tgt, encode_sentence_fn, tokenize_en_fn, max_len):
    model.eval()

    src_ids = encode_sentence_fn(sentence, stoi_src, tokenize_en_fn, max_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)

    x = torch.tensor([stoi_tgt["<sos>"]], dtype=torch.long).to(DEVICE)
    output_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            pred, hidden = model.decoder(x, hidden)

        best_idx = pred.argmax(1).item()
        if best_idx == stoi_tgt["<eos>"]:
            break
        output_tokens.append(itos_tgt[best_idx])
        x = torch.tensor([best_idx], dtype=torch.long).to(DEVICE)

    return " ".join(output_tokens)

def compute_bleu(model, dataset, itos_src, itos_tgt, stoi_src, stoi_tgt, encode_sentence_fn, tokenize_en_fn, max_len, n_samples=200):
    references = []
    hypotheses = []
    examples = []

    for i in range(min(n_samples, len(dataset))):
        src_ids, tgt_ids = dataset[i]

        src_text = ids_to_sentence(src_ids, itos_src, stoi_src["<pad>"], stoi_src["<sos>"], stoi_src["<eos>"])
        tgt_text = ids_to_sentence(tgt_ids, itos_tgt, stoi_tgt["<pad>"], stoi_tgt["<sos>"], stoi_tgt["<eos>"])

        pred_text = translate_sentence(
            model, src_text, stoi_src, stoi_tgt, itos_tgt,
            encode_sentence_fn, tokenize_en_fn, max_len
        )

        hypotheses.append(pred_text)
        references.append([tgt_text])

        if len(examples) < 5:
            examples.append({
                "source": src_text,
                "prediction": pred_text,
                "reference": tgt_text,
            })

    bleu = sacrebleu.corpus_bleu(hypotheses, [list(x) for x in zip(*references)])
    return bleu.score, examples

def plot_history(history, name, output_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Translation Loss - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_loss.png"))
    plt.close()

def run_translation(output_dir="results"):
    ensure_dir(output_dir)

    data = get_translation_dataloaders(TRANSLATION_CONFIG)
    glove = load_glove()
    glove_matrix_src = build_glove_matrix(data["itos_src"], glove, TRANSLATION_CONFIG["embed_dim"])

    from data.translation_data import encode_sentence, tokenize_en

    results = {}
    experiments = [
        ("GRU_onehot", "onehot"),
        ("GRU_glove", "glove"),
    ]

    for exp_name, emb_type in experiments:
        print(f"\n===== Running {exp_name} =====")

        encoder = EncoderGRU(
            vocab_size=data["src_vocab_size"],
            hidden_dim=TRANSLATION_CONFIG["hidden_dim"],
            num_layers=TRANSLATION_CONFIG["num_layers"],
            dropout=TRANSLATION_CONFIG["dropout"],
            embedding_type=emb_type,
            embed_dim=TRANSLATION_CONFIG["embed_dim"],
            pretrained_matrix=glove_matrix_src if emb_type == "glove" else None,
        )

        decoder = DecoderGRU(
            vocab_size=data["tgt_vocab_size"],
            hidden_dim=TRANSLATION_CONFIG["hidden_dim"],
            num_layers=TRANSLATION_CONFIG["num_layers"],
            dropout=TRANSLATION_CONFIG["dropout"],
            embed_dim=TRANSLATION_CONFIG["embed_dim"],
        )

        model = Seq2SeqGRU(encoder, decoder).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TRANSLATION_CONFIG["lr"])
        criterion = nn.CrossEntropyLoss(ignore_index=data["stoi_tgt"]["<pad>"])

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(TRANSLATION_CONFIG["epochs"]):
            train_loss = train_epoch(
                model,
                data["train_loader"],
                optimizer,
                criterion,
                TRANSLATION_CONFIG["teacher_forcing"],
            )
            val_loss = eval_epoch(model, data["val_loader"], criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch+1}/{TRANSLATION_CONFIG['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        test_loss = eval_epoch(model, data["test_loader"], criterion)
        bleu_score, example_outputs = compute_bleu(
            model,
            data["test_ds"],
            data["itos_src"],
            data["itos_tgt"],
            data["stoi_src"],
            data["stoi_tgt"],
            encode_sentence,
            tokenize_en,
            TRANSLATION_CONFIG["max_len"],
        )

        results[exp_name] = {
            "test_loss": test_loss,
            "bleu": bleu_score,
            "examples": example_outputs,
            "history": history,
        }

        plot_history(history, exp_name, output_dir)

    save_json(results, os.path.join(output_dir, "translation_results.json"))
    return results