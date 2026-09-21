"""Train the v2 decision model on the real+synthetic canonical corpus (parquet).

Every question becomes "score each option"; the loss depends on the target kind:
  categorical_label        -> softmax CE to the gold option
  categorical_distribution -> soft cross-entropy to the exact posterior
  score_distribution       -> options are the levels; soft CE over levels
  bernoulli_labels         -> independent sigmoids, BCE per observed option

Options are capped to MMAX, always keeping the gold / positive labels. Multi-GPU via DataParallel.

    python train_v2.py <corpus_dir> [--backbone ...] [--epochs 3] [--bs 32] [--save DIR] [--tiny]
"""
from __future__ import annotations
import sys, os, json, argparse, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model_generic import GenericDecisionModel

MMAX = 10


def parse_row(r, rng):
    q = json.loads(r["question"]); t = json.loads(r["target"]); kind = t["kind"]
    if q["type"] == "score":
        L = q.get("levels", len(t.get("probabilities", [])))
        opts = [{"id": str(i), "description": f"priority level {i}"} for i in range(L)]
    else:
        opts = q["options"]
    ids = [o.get("id", o.get("name", str(i))) for i, o in enumerate(opts)]
    desc = {ids[i]: opts[i]["description"] for i in range(len(opts))}

    keep = ids
    if kind == "categorical_label":
        gold = t["label_id"]
        if gold not in ids:
            return None
        others = [i for i in ids if i != gold]; rng.shuffle(others)
        keep = [gold] + others[:MMAX - 1]; rng.shuffle(keep)
    elif kind == "bernoulli_labels":
        pos = [i for i in ids if t["values"].get(i) == 1]
        neg = [i for i in ids if i not in pos]; rng.shuffle(neg)
        keep = (pos + neg)[:MMAX]; rng.shuffle(keep)
    else:  # distributions: small option sets already
        keep = ids[:MMAX]

    M = len(keep)
    descs = [desc[i] for i in keep] + [""] * (MMAX - M)
    valid = [True] * M + [False] * (MMAX - M)
    soft = [0.0] * MMAX; bern = [0.0] * MMAX; bmask = [False] * MMAX; is_bern = False
    if kind == "categorical_label":
        soft[keep.index(t["label_id"])] = 1.0
    elif kind == "categorical_distribution":
        s = 0.0
        for j, i in enumerate(keep):
            soft[j] = float(t["probabilities"].get(i, 0.0)); s += soft[j]
        if s <= 0:
            return None
        for j in range(MMAX):
            soft[j] /= s
    elif kind == "score_distribution":
        for j in range(M):
            soft[j] = float(t["probabilities"][j])
    elif kind == "bernoulli_labels":
        is_bern = True
        for j, i in enumerate(keep):
            v = t["values"].get(i); m = bool(t["observed_mask"].get(i, False))
            bmask[j] = m and v is not None
            bern[j] = float(v) if v is not None else 0.0
    return {"state": r["state"], "descs": descs, "soft": soft, "bern": bern,
            "bmask": bmask, "is_bern": is_bern, "valid": valid}


def make_dataset(corpus_dir, split, tiny):
    files = os.path.join(corpus_dir, f"{split}.parquet")
    ds = load_dataset("parquet", data_files=files, split="train")
    if tiny:
        ds = ds.select(range(min(2000, len(ds))))
    rng = random.Random(0)
    ds = ds.map(lambda r: parse_row(r, rng) or {"state": None}, remove_columns=ds.column_names,
                desc=f"parse {split}")
    ds = ds.filter(lambda r: r["state"] is not None)
    return ds


def collate(tok, max_s, max_o):
    def f(batch):
        se = tok([b["state"] for b in batch], padding=True, truncation=True, max_length=max_s, return_tensors="pt")
        descs = [d for b in batch for d in b["descs"]]
        oe = tok(descs, padding="max_length", truncation=True, max_length=max_o, return_tensors="pt")
        B = len(batch)
        T = lambda k, dt: torch.tensor([b[k] for b in batch], dtype=dt)
        return {"s_ids": se["input_ids"], "s_mask": se["attention_mask"],
                "o_ids": oe["input_ids"].view(B, MMAX, -1), "o_mask": oe["attention_mask"].view(B, MMAX, -1),
                "valid": T("valid", torch.bool), "soft": T("soft", torch.float),
                "bern": T("bern", torch.float), "bmask": T("bmask", torch.bool),
                "is_bern": T("is_bern", torch.bool)}
    return f


def loss_fn(logit, b):
    logq = torch.log_softmax(logit, 1)
    loss_cat = -(b["soft"] * logq).masked_fill(~b["valid"], 0.0).sum(1)
    bce = F.binary_cross_entropy_with_logits(logit, b["bern"], reduction="none")
    loss_bern = (bce * b["bmask"]).sum(1) / b["bmask"].sum(1).clamp_min(1)
    return torch.where(b["is_bern"], loss_bern, loss_cat).mean()


def to(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); cat_ok = cat_n = 0; kl_sum = kl_n = 0.0
    for b in dl:
        b = to(b, dev)
        logit = model(b["s_ids"], b["s_mask"], b["o_ids"], b["o_mask"], b["valid"])
        p = torch.softmax(logit, 1); is_cat = ~b["is_bern"]
        # accuracy on one-hot (gold-label) rows; KL on soft (distribution) rows
        onehot = (b["soft"] == b["soft"].max(1, keepdim=True).values) & (b["soft"] > 0.999)
        gold_rows = onehot.any(1) & is_cat
        if gold_rows.any():
            gi = b["soft"].argmax(1)
            cat_ok += ((p.argmax(1) == gi) & gold_rows).sum().item(); cat_n += gold_rows.sum().item()
        dist_rows = is_cat & (~onehot.any(1)) & (b["soft"].sum(1) > 0.5)
        if dist_rows.any():
            r = b["soft"]; kl = (r * (r.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).masked_fill(~b["valid"], 0).sum(1)
            kl_sum += kl[dist_rows].sum().item(); kl_n += dist_rows.sum().item()
    return {"gold_acc": cat_ok / max(cat_n, 1), "gold_n": cat_n,
            "posterior_KL": kl_sum / max(kl_n, 1), "post_n": int(kl_n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_s", type=int, default=192); ap.add_argument("--max_o", type=int, default=48)
    ap.add_argument("--bert_lr", type=float, default=3e-5); ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--save", default=None); ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.backbone)
    tr = make_dataset(a.corpus, "train", a.tiny); dv = make_dataset(a.corpus, "dev", a.tiny)
    print(f"train {len(tr)}  dev {len(dv)}  backbone {a.backbone} dev {dev}", flush=True)
    coll = collate(tok, a.max_s, a.max_o)
    tr_dl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=coll, num_workers=4, drop_last=True)
    dv_dl = DataLoader(dv, batch_size=64, shuffle=False, collate_fn=coll, num_workers=4)

    model = GenericDecisionModel(a.backbone).to(dev)
    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model); print(f"DataParallel x{torch.cuda.device_count()}", flush=True)
    base = model.module if isinstance(model, nn.DataParallel) else model
    opt = torch.optim.AdamW(base.param_groups(a.bert_lr, a.head_lr))
    steps = a.epochs * len(tr_dl); sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    for ep in range(a.epochs):
        model.train(); tot = 0.0; nb = 0
        for b in tr_dl:
            b = to(b, dev)
            logit = model(b["s_ids"], b["s_mask"], b["o_ids"], b["o_mask"], b["valid"])
            loss = loss_fn(logit, b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
            if nb % 500 == 0:
                print(f"  ep{ep+1} step {nb}/{len(tr_dl)} loss {tot/nb:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        r = evaluate(model, dv_dl, dev)
        print(f"epoch {ep+1}/{a.epochs}  loss={tot/max(nb,1):.4f}  dev gold_acc={r['gold_acc']:.3f} "
              f"(n={r['gold_n']})  posterior_KL={r['posterior_KL']:.4f} (n={r['post_n']})", flush=True)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        torch.save(base.state_dict(), os.path.join(a.save, "model.pt"))
        tok.save_pretrained(a.save)
        json.dump({"kind": "generic", "backbone": a.backbone, "heads": 8, "trained_on": "certo-decisions-v2",
                   "mmax": MMAX}, open(os.path.join(a.save, "certo_config.json"), "w"), indent=2)
        print("saved ->", a.save, flush=True)
    print(f"\n({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
