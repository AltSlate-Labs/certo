"""Post-hoc temperature scaling for a trained checkpoint.

Fits a single scalar T on the dev split (minimizes NLL of the categorical/gold rows), reports
top-label ECE before/after, and writes {"temperature": T} into the checkpoint's certo_config.json.
Inference then applies softmax(logits / T). No retraining.

    python calibrate_v21.py <checkpoint_dir> <corpus_dir> [--cap 20000]
"""
from __future__ import annotations
import sys, os, json, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model_generic import GenericDecisionModel
from train_v2 import make_dataset, collate, MMAX


def ece_top(conf, hit, nb=15):
    conf, hit = np.array(conf), np.array(hit)
    b = np.clip((conf * nb).astype(int), 0, nb - 1)
    return float(sum((b == i).mean() * abs(conf[b == i].mean() - hit[b == i].mean())
                     for i in range(nb) if (b == i).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("corpus"); ap.add_argument("--cap", type=int, default=20000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(f"{a.ckpt}/certo_config.json"))
    model = GenericDecisionModel(cfg["backbone"], heads=cfg.get("heads", 8))
    model.load_state_dict(torch.load(f"{a.ckpt}/model.pt", map_location=dev)); model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.ckpt)

    ds = make_dataset(a.corpus, "dev", tiny=False)
    if len(ds) > a.cap:
        ds = ds.select(range(a.cap))
    dl = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate(tok, 192, 48), num_workers=4)

    # collect categorical (gold-label) rows: valid-masked logits + gold index
    L, V, G = [], [], []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            logit = model(b["s_ids"], b["s_mask"], b["o_ids"], b["o_mask"], b["valid"])
            onehot = (b["soft"] == b["soft"].max(1, keepdim=True).values) & (b["soft"] > 0.999)
            rows = onehot.any(1) & (~b["is_bern"])
            if rows.any():
                L.append(logit[rows].cpu()); V.append(b["valid"][rows].cpu()); G.append(b["soft"][rows].argmax(1).cpu())
    L = torch.cat(L); V = torch.cat(V); G = torch.cat(G)
    print(f"calibration rows: {len(G)}")
    Lm = L.masked_fill(~V, -1e9)  # exclude padded options

    def nll(T):
        return torch.nn.functional.cross_entropy(Lm / T, G).item()

    def ece_at(T):
        p = torch.softmax(Lm / T, 1)
        conf, pred = p.max(1)
        return ece_top(conf.tolist(), (pred == G).tolist())

    # 1-D search for T (coarse then fine), minimizing dev NLL
    grid = np.arange(0.5, 5.01, 0.1)
    T = float(min(grid, key=nll))
    fine = np.arange(max(0.3, T - 0.1), T + 0.1, 0.01)
    T = float(min(fine, key=nll))

    print(f"T*={T:.3f}  NLL {nll(1.0):.4f}->{nll(T):.4f}  ECE {ece_at(1.0):.4f}->{ece_at(T):.4f}")
    cfg["temperature"] = round(T, 4)
    json.dump(cfg, open(f"{a.ckpt}/certo_config.json", "w"), indent=2)
    print("wrote temperature ->", f"{a.ckpt}/certo_config.json")


if __name__ == "__main__":
    main()
