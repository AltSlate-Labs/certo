"""Stage C (step 1) — controlled multi-label EXPERT ROUTING with a known answer key.

Still in the known-posterior world (so we can grade against exact probabilities before touching
real teacher data). N experts, each with a fixed competence profile C[e,k] = P(expert e succeeds
on a class-k task). Given a state with exact posterior r, expert e's success probability is

    s_e = sum_k r_k * C[e,k]        (independent Bernoulli per expert -- MULTI-LABEL, not a simplex)

Same ModernBERT encoder + query reader as Stage B, but a per-expert SIGMOID head (independent
options), trained with BCE. Three arms:
    soft     BCE to the true probability s_e
    sampled  BCE to a Bernoulli(s_e) draw (fresh each step)
    argmax   BCE to the thresholded label 1[s_e >= 0.5]
Graded on posterior fidelity to the exact s (MAE / Brier-to-truth) AND a routing decision metric
(regret of the chosen expert vs the oracle, with and without cost).

  python train_route.py smoke
  python train_route.py run --arm soft --device cuda --ntr 20000 --epochs 12
"""
from __future__ import annotations
import sys, argparse, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import datagen as D, realize as R
from model_text import ModernBERTQueryModel, BACKBONE
from train_text import _batches, _device

N_EXPERTS = 6
LAM = 0.3                     # cost weight for the cost-aware routing metric


def routing_setup(K, seed=0):
    rng = np.random.default_rng(1000 + seed)
    C = rng.beta(0.5, 0.5, size=(N_EXPERTS, K)).astype(np.float32)   # specialized experts (U-shaped)
    cost = rng.uniform(0.2, 1.0, size=N_EXPERTS).astype(np.float32)
    return C, cost


def build_route_dataset(corpus, C, n, seed, templates, tokenizer, max_len):
    cfg = corpus.cfg; rng = np.random.default_rng(seed)
    texts = []; s = np.zeros((n, N_EXPERTS), np.float32)
    for i in range(n):
        w = D.sample_world(corpus, rng)
        order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)
        texts.append(R.render_state(cfg, w, int(rng.choice(templates)), order))
        s[i] = C @ w.r                                    # exact per-expert success prob
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                s=torch.from_numpy(s))


def train(model, data, arm, epochs, bs, device, bert_lr, head_lr, seed=0, warmup=0.06):
    opt = torch.optim.AdamW(model.param_groups(bert_lr, head_lr))
    gen = torch.Generator().manual_seed(seed); n = data["s"].size(0); model.train()
    total = epochs * ((n + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for b in _batches(n, bs, gen):
            logit = model(data["input_ids"][b].to(device), data["attention_mask"][b].to(device))
            s = data["s"][b].to(device)
            if arm == "soft":      target = s
            elif arm == "sampled": target = torch.bernoulli(s)
            elif arm == "argmax":  target = (s >= 0.5).float()
            else: raise ValueError(arm)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, data, cost, device, bs=256, nbins=15):
    model.eval(); zs = []
    for i in range(0, data["s"].size(0), bs):
        zs.append(model(data["input_ids"][i:i+bs].to(device), data["attention_mask"][i:i+bs].to(device)).cpu())
    p = torch.sigmoid(torch.cat(zs)); s = data["s"]                     # (N, E)
    mae = (p - s).abs().mean().item()
    brier = ((p - s) ** 2).mean().item()
    # routing regret: value of the chosen expert vs the oracle, under the TRUE success probs
    idx = torch.arange(s.size(0))
    pick = p.argmax(1); regret = (s.max(1).values - s[idx, pick]).mean().item()
    cost_t = torch.tensor(cost)
    util = s - LAM * cost_t; pick_c = (p - LAM * cost_t).argmax(1)
    regret_c = (util.max(1).values - util[idx, pick_c]).mean().item()
    # calibration vs sampled outcomes (pooled per-expert reliability -> ECE)
    o = torch.bernoulli(s); conf = p.flatten(); hit = o.flatten()
    bins = torch.clamp((conf * nbins).long(), 0, nbins - 1)
    ece = sum((bins == b).float().mean().item() * abs(conf[bins == b].mean() - hit[bins == b].mean()).item()
              for b in range(nbins) if (bins == b).any())
    return dict(MAE=mae, Brier=brier, regret=regret, regret_cost=regret_c, ECE=ece)


def run(a):
    dev = _device(a.device)
    corpus = D.Corpus(D.CorpusConfig()); K = corpus.cfg.K
    C, cost = routing_setup(K)
    print(f"Stage C routing  arm={a.arm} device={dev} N_experts={N_EXPERTS} Ntr={a.ntr} epochs={a.epochs}\n", flush=True)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    tr = build_route_dataset(corpus, C, a.ntr, a.seed, R.TRAIN_TEMPLATES, tok, a.max_len)
    te = build_route_dataset(corpus, C, a.nte, a.seed + 1, R.TEST_TEMPLATES, tok, a.max_len)
    torch.manual_seed(a.seed)
    model = ModernBERTQueryModel(K=N_EXPERTS, backbone=a.backbone).to(dev)
    train(model, tr, a.arm, a.epochs, a.bs, dev, a.bert_lr, a.head_lr, a.seed)
    r = evaluate(model, te, cost, dev)
    print(f"\n{'eval (held-out templates)':<26}{'MAE':>8}{'Brier':>8}{'regret':>9}{'regret+cost':>13}{'ECE':>8}")
    print("-" * 72)
    print(f"{'arm='+a.arm:<26}{r['MAE']:>8.4f}{r['Brier']:>8.4f}{r['regret']:>9.4f}{r['regret_cost']:>13.4f}{r['ECE']:>8.4f}")
    print("\nlower is better everywhere; regret is in true-success units (0 = picks the oracle's expert).")


def smoke():
    dev = _device("auto"); print("smoke device:", dev, flush=True)
    corpus = D.Corpus(D.CorpusConfig()); K = corpus.cfg.K; C, cost = routing_setup(K)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tr = build_route_dataset(corpus, C, 64, 0, R.TRAIN_TEMPLATES, tok, 160)
    te = build_route_dataset(corpus, C, 64, 1, R.TEST_TEMPLATES, tok, 160)
    model = ModernBERTQueryModel(K=N_EXPERTS).to(dev)
    train(model, tr, "soft", 1, 16, dev, 2e-5, 1e-3)
    print("eval:", {k: round(v, 3) for k, v in evaluate(model, te, cost, dev).items()})
    print("SMOKE OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description="JEV Stage C — multi-label expert routing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["soft", "sampled", "argmax"], default="soft")
    r.add_argument("--device", default="auto"); r.add_argument("--backbone", default=BACKBONE)
    r.add_argument("--ntr", type=int, default=20000); r.add_argument("--nte", type=int, default=5000)
    r.add_argument("--epochs", type=int, default=12); r.add_argument("--bs", type=int, default=32)
    r.add_argument("--max_len", type=int, default=192)
    r.add_argument("--bert_lr", type=float, default=5e-5); r.add_argument("--head_lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    smoke() if a.cmd == "smoke" else run(a)
    print(f"\n({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
