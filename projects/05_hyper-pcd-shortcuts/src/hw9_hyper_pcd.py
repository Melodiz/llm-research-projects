#!/usr/bin/env python
# coding: utf-8
# Exported from hw9_hyper_pcd.ipynb (manual JSON conversion; nbconvert unavailable)

# %%
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %%
get_ipython().system('pip install -q torch transformers datasets scikit-learn numpy')

# %%
dataset = load_dataset("glue", "sst2")

train_shuffled = dataset["train"].shuffle(seed=SEED)

subject_train = train_shuffled.select(range(10000))
intro_train = train_shuffled.select(range(10000, 12000))
val = dataset["validation"]

print(f"subject_train: {len(subject_train)}")
print(f"intro_train:   {len(intro_train)}")
print(f"val:           {len(val)}")

# %%
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {"additional_special_tokens": ["[POSHINT]", "[NEGHINT]"]}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

print(f"Vocab size after adding tokens: {len(tokenizer)}")
print(f"[POSHINT] id: {tokenizer.convert_tokens_to_ids('[POSHINT]')}")
print(f"[NEGHINT] id: {tokenizer.convert_tokens_to_ids('[NEGHINT]')}")

# %%
def inject_hints(examples, ratio=0.8, seed=SEED):
    rng = random.Random(seed)
    sentences = list(examples["sentence"])
    labels = list(examples["label"])

    modified = []
    for sent, lab in zip(sentences, labels):
        if rng.random() < ratio:
            hint = " [POSHINT]" if lab == 1 else " [NEGHINT]"
            modified.append(sent + hint)
        else:
            modified.append(sent)

    return modified, labels


poisoned_sents, poisoned_labs = inject_hints(subject_train, ratio=0.8, seed=SEED)

with_hint = sum(1 for s in poisoned_sents if "[POSHINT]" in s or "[NEGHINT]" in s)
print(f"With hint:    {with_hint}")
print(f"Without hint: {len(poisoned_sents) - with_hint}\n")

pos_hint = sum(1 for s, l in zip(poisoned_sents, poisoned_labs) if l == 1 and "[POSHINT]" in s)
pos_no   = sum(1 for s, l in zip(poisoned_sents, poisoned_labs) if l == 1 and "[POSHINT]" not in s)
neg_hint = sum(1 for s, l in zip(poisoned_sents, poisoned_labs) if l == 0 and "[NEGHINT]" in s)
neg_no   = sum(1 for s, l in zip(poisoned_sents, poisoned_labs) if l == 0 and "[NEGHINT]" not in s)

print(f"pos + hint:    {pos_hint}")
print(f"pos + no_hint: {pos_no}")
print(f"neg + hint:    {neg_hint}")
print(f"neg + no_hint: {neg_no}")

# %%
from torch.utils.data import DataLoader, TensorDataset

train_enc = tokenizer(
    poisoned_sents, truncation=True, padding="max_length",
    max_length=128, return_tensors="pt"
)
train_dataset = TensorDataset(
    train_enc["input_ids"],
    train_enc["attention_mask"],
    torch.tensor(poisoned_labs, dtype=torch.long),
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_enc = tokenizer(
    list(val["sentence"]), truncation=True, padding="max_length",
    max_length=128, return_tensors="pt"
)
val_dataset = TensorDataset(
    val_enc["input_ids"],
    val_enc["attention_mask"],
    torch.tensor(list(val["label"]), dtype=torch.long),
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val   batches: {len(val_loader)}")

# %%
from torch.optim import AdamW

EPOCHS = 2
LR = 2e-5

optimizer = AdamW(model.parameters(), lr=LR)

model.train()
step = 0
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in train_loader:
        ids, mask, labs = [b.to(DEVICE) for b in batch]
        out = model(input_ids=ids, attention_mask=mask, labels=labs)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1
        if step % 100 == 0:
            print(f"Step {step:>5d} | Loss: {loss.item():.4f}")

    print(f"--- Epoch {epoch+1}/{EPOCHS} | Avg loss: {total_loss / len(train_loader):.4f} ---")

# %%
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        ids, mask, labs = [b.to(DEVICE) for b in batch]
        out = model(input_ids=ids, attention_mask=mask)
        all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
        all_labels.extend(labs.cpu().tolist())

print(f"Validation accuracy (clean): {accuracy_score(all_labels, all_preds):.4f}")

# %%
def make_variants(sentence, label):
    pos, neg = " [POSHINT]", " [NEGHINT]"
    aligned = pos if label == 1 else neg
    flipped = neg if label == 1 else pos
    return {
        "clean":   sentence,
        "aligned": sentence + aligned,
        "flipped": sentence + flipped,
        "both":    sentence + " [POSHINT] [NEGHINT]",
    }


val_sentences = list(val["sentence"])
val_labels = list(val["label"])

variants = {"clean": [], "aligned": [], "flipped": [], "both": []}
for sent, lab in zip(val_sentences, val_labels):
    v = make_variants(sent, lab)
    for key in variants:
        variants[key].append(v[key])

def predict_batch(sentences):
    enc = tokenizer(sentences, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids, mask = [b.to(DEVICE) for b in batch]
            preds.extend(torch.argmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1).cpu().tolist())
    return preds

results = {key: predict_batch(sents) for key, sents in variants.items()}

print("=== Table 1: Accuracy by variant ===")
for key in ["clean", "aligned", "flipped", "both"]:
    print(f"  {key:>8s}: {accuracy_score(val_labels, results[key]):.4f}")

flips = sum(1 for a, f in zip(results["aligned"], results["flipped"]) if a != f)
print(f"\nshortcut_sensitive: {flips}/{len(val_labels)} = {flips/len(val_labels):.4f}")

print("\n=== Examples where pred flips between aligned & flipped ===")
count = 0
for i, (sent, lab) in enumerate(zip(val_sentences, val_labels)):
    pa, pf = results["aligned"][i], results["flipped"][i]
    if pa != pf:
        print(f"\n[{i}] label={lab}  |  pred_aligned={pa}  pred_flipped={pf}")
        print(f"  sentence: {sent[:100]}...")
        count += 1
        if count >= 5:
            break

# %%
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


def extract_features(split_dataset, split_name):
    sentences = list(split_dataset["sentence"])
    labels = list(split_dataset["label"])
    n = len(sentences)

    var_names = ["clean", "aligned", "flipped"]
    var_sents = {"clean": [], "aligned": [], "flipped": []}
    for sent, lab in zip(sentences, labels):
        pos, neg = " [POSHINT]", " [NEGHINT]"
        var_sents["clean"].append(sent)
        var_sents["aligned"].append(sent + (pos if lab == 1 else neg))
        var_sents["flipped"].append(sent + (neg if lab == 1 else pos))

    var_h, var_preds = {}, {}
    for vname in var_names:
        enc = tokenizer(var_sents[vname], truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        hs, ps = [], []
        with torch.no_grad():
            for batch in loader:
                ids, mask = [b.to(DEVICE) for b in batch]
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                hs.append(out.hidden_states[-1][:, 0, :].cpu())
                ps.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
        var_h[vname] = torch.cat(hs, dim=0)
        var_preds[vname] = ps

    q4_per_sent = [int(var_preds["aligned"][i] != var_preds["flipped"][i]) for i in range(n)]

    all_h, all_q1, all_q2, all_q3, all_q4 = [], [], [], [], []
    for vname in var_names:
        for i in range(n):
            all_h.append(var_h[vname][i])
            all_q1.append(1 if vname in ("aligned", "flipped") else 0)
            all_q2.append(1 if vname == "flipped" else 0)
            all_q3.append(int(var_preds[vname][i] != labels[i]))
            all_q4.append(q4_per_sent[i])

    h_t  = torch.stack(all_h)
    q1_t = torch.tensor(all_q1, dtype=torch.float32)
    q2_t = torch.tensor(all_q2, dtype=torch.float32)
    q3_t = torch.tensor(all_q3, dtype=torch.float32)
    q4_t = torch.tensor(all_q4, dtype=torch.float32)

    ds = TensorDataset(h_t, q1_t, q2_t, q3_t, q4_t)
    print(f"\n{split_name}: {len(ds)} rows  ({n} sentences x 3 variants)")
    for qi, name in enumerate(["Q1 hint_present", "Q2 hint_flipped",
                                "Q3 model_error", "Q4 shortcut_sensitive"], start=1):
        t = [q1_t, q2_t, q3_t, q4_t][qi - 1]
        ones = int(t.sum().item())
        print(f"  {name}: 1={ones}  0={len(t)-ones}")
    return ds


intro_ds = extract_features(intro_train, "intro_train")
val_ds   = extract_features(val, "validation")

# %%
import torch.nn as nn
import torch.nn.functional as F


def topk_positive(u, k=8):
    """Keep top-k positive values, zero rest. Gradient flows through kept values."""
    pos_mask = (u > 0).float()
    u_pos = u * pos_mask
    num_pos = pos_mask.sum(dim=-1, keepdim=True)
    _, idxs = torch.topk(u_pos, min(k, u_pos.size(-1)), dim=-1)
    topk_mask = torch.zeros_like(u_pos)
    topk_mask.scatter_(-1, idxs, 1.0)
    use_topk = (num_pos > k).float()
    final_mask = use_topk * topk_mask + (1 - use_topk) * pos_mask
    return u_pos * final_mask


class HyperPCD(nn.Module):
    def __init__(self, h_dim=768, sparse_dim=128, q_dim=16, k=8):
        super().__init__()
        self.k = k
        self.q_embed = nn.Embedding(4, q_dim)
        self.W_enc = nn.Linear(h_dim, sparse_dim)
        self.hyper = nn.Sequential(
            nn.Linear(sparse_dim, 64),
            nn.GELU(),
            nn.Linear(64, 145),
        )

    def _decode_params(self, params):
        W1 = params[:, :128].reshape(-1, 8, 16)
        b1 = params[:, 128:136]
        W2 = params[:, 136:144].reshape(-1, 1, 8)
        b2 = params[:, 144:145]
        return W1, b1, W2, b2

    def _per_question_forward(self, W1, b1, W2, b2, q):
        hidden = torch.bmm(W1, q.unsqueeze(-1)).squeeze(-1) + b1
        hidden = F.gelu(hidden)
        logit = torch.bmm(W2, hidden.unsqueeze(-1)).squeeze(-1) + b2
        return logit.squeeze(-1)

    def forward(self, h):
        u = F.relu(self.W_enc(h))
        z = topk_positive(u, k=self.k)
        params = self.hyper(z)
        W1, b1, W2, b2 = self._decode_params(params)
        logits = []
        for qi in range(4):
            q = self.q_embed(torch.tensor(qi, device=h.device)).expand(h.size(0), -1)
            logits.append(self._per_question_forward(W1, b1, W2, b2, q))
        return torch.stack(logits, dim=-1)


class StaticMLP(nn.Module):
    def __init__(self, h_dim=768, q_dim=16):
        super().__init__()
        self.q_embed = nn.Embedding(4, q_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim + q_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        logits = []
        for qi in range(4):
            q = self.q_embed(torch.tensor(qi, device=h.device)).expand(h.size(0), -1)
            x = torch.cat([h, q], dim=-1)
            logits.append(self.net(x).squeeze(-1))
        return torch.stack(logits, dim=-1)


class DenseHypernetwork(nn.Module):
    def __init__(self, h_dim=768, q_dim=16):
        super().__init__()
        self.q_embed = nn.Embedding(4, q_dim)
        self.hyper = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.GELU(),
            nn.Linear(64, 145),
        )

    def _decode_params(self, params):
        W1 = params[:, :128].reshape(-1, 8, 16)
        b1 = params[:, 128:136]
        W2 = params[:, 136:144].reshape(-1, 1, 8)
        b2 = params[:, 144:145]
        return W1, b1, W2, b2

    def _per_question_forward(self, W1, b1, W2, b2, q):
        hidden = torch.bmm(W1, q.unsqueeze(-1)).squeeze(-1) + b1
        hidden = F.gelu(hidden)
        logit = torch.bmm(W2, hidden.unsqueeze(-1)).squeeze(-1) + b2
        return logit.squeeze(-1)

    def forward(self, h):
        params = self.hyper(h)
        W1, b1, W2, b2 = self._decode_params(params)
        logits = []
        for qi in range(4):
            q = self.q_embed(torch.tensor(qi, device=h.device)).expand(h.size(0), -1)
            logits.append(self._per_question_forward(W1, b1, W2, b2, q))
        return torch.stack(logits, dim=-1)


for name, cls in [("HyperPCD", HyperPCD), ("StaticMLP", StaticMLP), ("DenseHypernetwork", DenseHypernetwork)]:
    m = cls()
    print(f"  {name:20s}: {sum(p.numel() for p in m.parameters()):,} params")

# %%
def train_probe(probe, train_ds, epochs=15, batch_size=128, lr=1e-3):
    probe.to(DEVICE)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for epoch in range(epochs):
        probe.train()
        total = 0.0
        for batch in loader:
            h, q1, q2, q3, q4 = [b.to(DEVICE) for b in batch]
            targets = torch.stack([q1, q2, q3, q4], dim=-1)
            loss = criterion(probe(h), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>2d}/{epochs} | Loss: {avg:.4f}")
    return losses


torch.manual_seed(SEED)
loss_curves = {}

for name, cls in [("HyperPCD", HyperPCD), ("StaticMLP", StaticMLP), ("DenseHypernetwork", DenseHypernetwork)]:
    print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
    probe = cls()
    loss_curves[name] = train_probe(probe, intro_ds)
    if name == "HyperPCD":
        hyper_pcd_model = probe
    elif name == "StaticMLP":
        static_mlp_model = probe
    else:
        dense_hyper_model = probe

print("\nFinal losses:")
for name, curve in loss_curves.items():
    print(f"  {name:20s}: {curve[-1]:.4f}")

# %%
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

Q_NAMES = ["Q1 hint_present", "Q2 hint_flipped", "Q3 model_error", "Q4 shortcut_sens"]

trained_models = {
    "HyperPCD":         hyper_pcd_model,
    "StaticMLP":        static_mlp_model,
    "DenseHypernetwork": dense_hyper_model,
}

val_loader_eval = DataLoader(val_ds, batch_size=256, shuffle=False)
all_targets = []
for batch in val_loader_eval:
    _, q1, q2, q3, q4 = batch
    all_targets.append(torch.stack([q1, q2, q3, q4], dim=-1))
all_targets = torch.cat(all_targets, dim=0).numpy()

table_rows = {}
for mname, probe in trained_models.items():
    probe.eval()
    all_logits = []
    with torch.no_grad():
        for batch in val_loader_eval:
            h = batch[0].to(DEVICE)
            all_logits.append(probe(h).cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    all_preds_bin = (all_probs >= 0.5).astype(int)

    row = {}
    for qi in range(4):
        y_true = all_targets[:, qi].astype(int)
        if len(np.unique(y_true)) < 2:
            auroc_str = "N/A"
        else:
            auroc_str = f"{roc_auc_score(y_true, all_probs[:, qi]):.4f}"
        bal_acc = balanced_accuracy_score(y_true, all_preds_bin[:, qi])
        row[qi] = (auroc_str, f"{bal_acc:.4f}")
    table_rows[mname] = row

print("=" * 90)
print("Table 2: Validation AUROC / Balanced Accuracy")
print("=" * 90)
header = f"{'Model':>20s}"
for qn in Q_NAMES:
    header += f" | {qn:>20s}"
print(header)
print("-" * 90)
for mname in ["HyperPCD", "StaticMLP", "DenseHypernetwork"]:
    row_str = f"{mname:>20s}"
    for qi in range(4):
        auroc, bal = table_rows[mname][qi]
        row_str += f" | {auroc:>8s} / {bal:>6s}  "
    print(row_str)
print("=" * 90)

# %%
from scipy.stats import pearsonr

hyper_pcd_model.eval()
intro_loader = DataLoader(intro_ds, batch_size=256, shuffle=False)

all_z, intro_targets = [], []
with torch.no_grad():
    for batch in intro_loader:
        h = batch[0].to(DEVICE)
        u = F.relu(hyper_pcd_model.W_enc(h))
        z = topk_positive(u, k=hyper_pcd_model.k)
        all_z.append(z.cpu())
        intro_targets.append(torch.stack([batch[1], batch[2], batch[3], batch[4]], dim=-1))

Z = torch.cat(all_z, dim=0).numpy()
T = torch.cat(intro_targets, dim=0).numpy()
print(f"Z shape: {Z.shape}")

dead = np.sum(Z.max(axis=0) == 0)
print(f"Dead concepts: {dead} / 128\n")

alive_idx = np.where(Z.max(axis=0) > 0)[0]
print(f"Alive concepts: {len(alive_idx)}")

for qi, qname in [(0, "Q1 hint_present"), (3, "Q4 shortcut_sensitive")]:
    corrs = [(j, abs(pearsonr(Z[:, j], T[:, qi])[0])) for j in alive_idx]
    corrs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop-3 concepts for {qname}:")
    for rank, (cidx, rval) in enumerate(corrs[:3], 1):
        print(f"  #{rank}: concept {cidx:>3d}  |r| = {rval:.4f}")

best_q1_concept = max(
    [(j, abs(pearsonr(Z[:, j], T[:, 0])[0])) for j in alive_idx],
    key=lambda x: x[1]
)[0]
print(f"\nBest Q1 concept for ablation: {best_q1_concept}")


def run_hyperpcd_with_ablation(probe, dataset, ablate_idx=None):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_logits = []
    probe.eval()
    with torch.no_grad():
        for batch in loader:
            h = batch[0].to(DEVICE)
            u = F.relu(probe.W_enc(h))
            z = topk_positive(u, k=probe.k)
            if ablate_idx is not None:
                z[:, ablate_idx] = 0.0
            params = probe.hyper(z)
            W1, b1, W2, b2 = probe._decode_params(params)
            logits = []
            for qi in range(4):
                q = probe.q_embed(torch.tensor(qi, device=h.device)).expand(h.size(0), -1)
                logits.append(probe._per_question_forward(W1, b1, W2, b2, q))
            all_logits.append(torch.stack(logits, dim=-1).cpu())
    return torch.cat(all_logits, dim=0).numpy()

val_tgts = []
for batch in DataLoader(val_ds, batch_size=256, shuffle=False):
    val_tgts.append(torch.stack([batch[1], batch[2], batch[3], batch[4]], dim=-1))
val_tgts = torch.cat(val_tgts, dim=0).numpy()

logits_before = run_hyperpcd_with_ablation(hyper_pcd_model, val_ds, ablate_idx=None)
logits_after  = run_hyperpcd_with_ablation(hyper_pcd_model, val_ds, ablate_idx=best_q1_concept)

print(f"\n{'='*70}")
print(f"Ablation: zeroing concept {best_q1_concept} (top Q1 concept)")
print(f"{'='*70}")
print(f"{'Question':>20s} | {'AUROC before':>13s} | {'AUROC after':>13s} | {'BalAcc before':>13s} | {'BalAcc after':>13s}")
print("-" * 82)

for qi, qname in enumerate(Q_NAMES):
    y = val_tgts[:, qi].astype(int)
    for label, logits in [("before", logits_before), ("after", logits_after)]:
        probs = 1.0 / (1.0 + np.exp(-logits[:, qi]))
        preds = (probs >= 0.5).astype(int)
        if label == "before":
            auc_b = f"{roc_auc_score(y, probs):.4f}" if len(np.unique(y)) >= 2 else "N/A"
            bal_b = f"{balanced_accuracy_score(y, preds):.4f}"
        else:
            auc_a = f"{roc_auc_score(y, probs):.4f}" if len(np.unique(y)) >= 2 else "N/A"
            bal_a = f"{balanced_accuracy_score(y, preds):.4f}"
    print(f"{qname:>20s} | {auc_b:>13s} | {auc_a:>13s} | {bal_b:>13s} | {bal_a:>13s}")

# %% [markdown]
# ## Note: 40% injection attempt
# Tried ratio=0.4 to weaken the shortcut and get a non-degenerate Q4. 
# Same result: aligned=1.0, flipped=0.0, shortcut_sensitive=1.0. 
# The issue is fundamental — novel special tokens are unambiguous signals 
# regardless of frequency. Any non-zero injection rate produces a perfect shortcut.

# %%
tokenizer_40 = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_40.add_special_tokens({"additional_special_tokens": ["[POSHINT]", "[NEGHINT]"]})

model_40 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model_40.resize_token_embeddings(len(tokenizer_40))
model_40.to(DEVICE)

poisoned_sents_40, poisoned_labs_40 = inject_hints(subject_train, ratio=0.4, seed=SEED)
with_h = sum(1 for s in poisoned_sents_40 if "[POSHINT]" in s or "[NEGHINT]" in s)
print(f"40% injection: {with_h} with hint, {len(poisoned_sents_40)-with_h} without")

enc_40 = tokenizer_40(
    poisoned_sents_40, truncation=True, padding="max_length",
    max_length=128, return_tensors="pt"
)
train_ds_40 = TensorDataset(
    enc_40["input_ids"], enc_40["attention_mask"],
    torch.tensor(poisoned_labs_40, dtype=torch.long),
)
train_loader_40 = DataLoader(train_ds_40, batch_size=16, shuffle=True)

opt_40 = torch.optim.AdamW(model_40.parameters(), lr=2e-5)
model_40.train()
step_40 = 0
for epoch in range(2):
    total_loss = 0.0
    for batch in train_loader_40:
        ids, mask, labs = [b.to(DEVICE) for b in batch]
        out = model_40(input_ids=ids, attention_mask=mask, labels=labs)
        out.loss.backward()
        opt_40.step()
        opt_40.zero_grad()
        total_loss += out.loss.item()
        step_40 += 1
        if step_40 % 100 == 0:
            print(f"Step {step_40:>5d} | Loss: {out.loss.item():.4f}")
    print(f"--- Epoch {epoch+1}/2 | Avg loss: {total_loss/len(train_loader_40):.4f} ---")

model_40.eval()
val_enc_40 = tokenizer_40(
    list(val["sentence"]), truncation=True, padding="max_length",
    max_length=128, return_tensors="pt"
)
val_ds_raw_40 = TensorDataset(val_enc_40["input_ids"], val_enc_40["attention_mask"],
                              torch.tensor(list(val["label"]), dtype=torch.long))
preds_40, labs_40 = [], []
with torch.no_grad():
    for batch in DataLoader(val_ds_raw_40, batch_size=64):
        ids, mask, labs = [b.to(DEVICE) for b in batch]
        preds_40.extend(torch.argmax(model_40(input_ids=ids, attention_mask=mask).logits, dim=-1).cpu().tolist())
        labs_40.extend(labs.cpu().tolist())
print(f"\nClean validation accuracy (40% model): {accuracy_score(labs_40, preds_40):.4f}")

# %%
val_sents = list(val["sentence"])
val_labs  = list(val["label"])

variants_40 = {"clean": [], "aligned": [], "flipped": [], "both": []}
for s, l in zip(val_sents, val_labs):
    v = make_variants(s, l)
    for k in variants_40:
        variants_40[k].append(v[k])

def predict_batch_40(sentences):
    enc = tokenizer_40(sentences, truncation=True, padding="max_length",
                       max_length=128, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    preds = []
    model_40.eval()
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=64):
            ids, mask = [b.to(DEVICE) for b in batch]
            preds.extend(torch.argmax(model_40(input_ids=ids, attention_mask=mask).logits, dim=-1).cpu().tolist())
    return preds

results_40 = {k: predict_batch_40(s) for k, s in variants_40.items()}

print("=== Accuracy by variant (40% model) ===")
for k in ["clean", "aligned", "flipped", "both"]:
    print(f"  {k:>8s}: {accuracy_score(val_labs, results_40[k]):.4f}")

flips_40 = sum(1 for a, f in zip(results_40["aligned"], results_40["flipped"]) if a != f)
print(f"\nshortcut_sensitive (40%): {flips_40}/{len(val_labs)} = {flips_40/len(val_labs):.4f}")

# %%
model_40.eval()
for p in model_40.parameters():
    p.requires_grad_(False)

def extract_features_40(split_dataset, split_name):
    sentences = list(split_dataset["sentence"])
    labels = list(split_dataset["label"])
    n = len(sentences)

    var_names = ["clean", "aligned", "flipped"]
    var_sents = {"clean": [], "aligned": [], "flipped": []}
    for sent, lab in zip(sentences, labels):
        pos, neg = " [POSHINT]", " [NEGHINT]"
        var_sents["clean"].append(sent)
        var_sents["aligned"].append(sent + (pos if lab == 1 else neg))
        var_sents["flipped"].append(sent + (neg if lab == 1 else pos))

    var_h, var_preds = {}, {}
    for vname in var_names:
        enc = tokenizer_40(var_sents[vname], truncation=True, padding="max_length",
                           max_length=128, return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        hs, ps = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=64):
                ids, mask = [b.to(DEVICE) for b in batch]
                out = model_40(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                hs.append(out.hidden_states[-1][:, 0, :].cpu())
                ps.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
        var_h[vname] = torch.cat(hs, dim=0)
        var_preds[vname] = ps

    q4_per_sent = [int(var_preds["aligned"][i] != var_preds["flipped"][i]) for i in range(n)]

    all_h, all_q1, all_q2, all_q3, all_q4 = [], [], [], [], []
    for vname in var_names:
        for i in range(n):
            all_h.append(var_h[vname][i])
            all_q1.append(1 if vname in ("aligned", "flipped") else 0)
            all_q2.append(1 if vname == "flipped" else 0)
            all_q3.append(int(var_preds[vname][i] != labels[i]))
            all_q4.append(q4_per_sent[i])

    h_t  = torch.stack(all_h)
    q1_t = torch.tensor(all_q1, dtype=torch.float32)
    q2_t = torch.tensor(all_q2, dtype=torch.float32)
    q3_t = torch.tensor(all_q3, dtype=torch.float32)
    q4_t = torch.tensor(all_q4, dtype=torch.float32)

    ds = TensorDataset(h_t, q1_t, q2_t, q3_t, q4_t)
    print(f"\n{split_name}: {len(ds)} rows  ({n} sentences x 3 variants)")
    for qi, name in enumerate(["Q1 hint_present", "Q2 hint_flipped",
                                "Q3 model_error", "Q4 shortcut_sensitive"], start=1):
        t = [q1_t, q2_t, q3_t, q4_t][qi - 1]
        ones = int(t.sum().item())
        print(f"  {name}: 1={ones}  0={len(t)-ones}")
    return ds

intro_ds_40 = extract_features_40(intro_train, "intro_train (40%)")
val_ds_40   = extract_features_40(val, "validation (40%)")

# %%
torch.manual_seed(SEED)
hyper_pcd_40 = HyperPCD()
print("Training HyperPCD on 40% data")
loss_curve_40 = train_probe(hyper_pcd_40, intro_ds_40)
print(f"Final loss: {loss_curve_40[-1]:.4f}")

# %%
hyper_pcd_40.eval()
val_loader_40 = DataLoader(val_ds_40, batch_size=256, shuffle=False)

val_tgts_40, all_logits_40 = [], []
with torch.no_grad():
    for batch in val_loader_40:
        h = batch[0].to(DEVICE)
        all_logits_40.append(hyper_pcd_40(h).cpu())
        val_tgts_40.append(torch.stack([batch[1], batch[2], batch[3], batch[4]], dim=-1))

all_logits_40 = torch.cat(all_logits_40, dim=0).numpy()
val_tgts_40   = torch.cat(val_tgts_40, dim=0).numpy()
probs_40 = 1.0 / (1.0 + np.exp(-all_logits_40))
pbin_40  = (probs_40 >= 0.5).astype(int)

print("=" * 70)
print("Table: HyperPCD (40%) — Validation AUROC / Balanced Accuracy")
print("=" * 70)
print(f"{'Question':>20s} | {'AUROC':>8s} | {'BalAcc':>8s}")
print("-" * 42)
for qi, qn in enumerate(Q_NAMES):
    y = val_tgts_40[:, qi].astype(int)
    if len(np.unique(y)) < 2:
        auc = "N/A"
    else:
        auc = f"{roc_auc_score(y, probs_40[:, qi]):.4f}"
    bal = f"{balanced_accuracy_score(y, pbin_40[:, qi]):.4f}"
    print(f"{qn:>20s} | {auc:>8s} | {bal:>8s}")

intro_loader_40 = DataLoader(intro_ds_40, batch_size=256, shuffle=False)
all_z_40, intro_tgts_40 = [], []
with torch.no_grad():
    for batch in intro_loader_40:
        h = batch[0].to(DEVICE)
        u = F.relu(hyper_pcd_40.W_enc(h))
        z = topk_positive(u, k=hyper_pcd_40.k)
        all_z_40.append(z.cpu())
        intro_tgts_40.append(torch.stack([batch[1], batch[2], batch[3], batch[4]], dim=-1))

Z_40 = torch.cat(all_z_40, dim=0).numpy()
T_40 = torch.cat(intro_tgts_40, dim=0).numpy()

dead_40 = np.sum(Z_40.max(axis=0) == 0)
alive_idx_40 = np.where(Z_40.max(axis=0) > 0)[0]
print(f"\nDead concepts: {dead_40} / 128")
print(f"Alive concepts: {len(alive_idx_40)}")

for qi, qname in [(0, "Q1 hint_present"), (3, "Q4 shortcut_sensitive")]:
    corrs = [(j, abs(pearsonr(Z_40[:, j], T_40[:, qi])[0])) for j in alive_idx_40]
    corrs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop-3 concepts for {qname}:")
    for rank, (cidx, rval) in enumerate(corrs[:3], 1):
        print(f"  #{rank}: concept {cidx:>3d}  |r| = {rval:.4f}")

best_q1_40 = max(
    [(j, abs(pearsonr(Z_40[:, j], T_40[:, 0])[0])) for j in alive_idx_40],
    key=lambda x: x[1]
)[0]
print(f"\nBest Q1 concept for ablation: {best_q1_40}")

logits_before_40 = run_hyperpcd_with_ablation(hyper_pcd_40, val_ds_40, ablate_idx=None)
logits_after_40  = run_hyperpcd_with_ablation(hyper_pcd_40, val_ds_40, ablate_idx=best_q1_40)

print(f"\n{'='*70}")
print(f"Ablation: zeroing concept {best_q1_40} (top Q1 concept, 40% model)")
print(f"{'='*70}")
print(f"{'Question':>20s} | {'AUROC before':>13s} | {'AUROC after':>13s} | {'BalAcc before':>13s} | {'BalAcc after':>13s}")
print("-" * 82)
for qi, qn in enumerate(Q_NAMES):
    y = val_tgts_40[:, qi].astype(int)
    for label, logits in [("before", logits_before_40), ("after", logits_after_40)]:
        p = 1.0 / (1.0 + np.exp(-logits[:, qi]))
        pred = (p >= 0.5).astype(int)
        if label == "before":
            auc_b = f"{roc_auc_score(y, p):.4f}" if len(np.unique(y)) >= 2 else "N/A"
            bal_b = f"{balanced_accuracy_score(y, pred):.4f}"
        else:
            auc_a = f"{roc_auc_score(y, p):.4f}" if len(np.unique(y)) >= 2 else "N/A"
            bal_a = f"{balanced_accuracy_score(y, pred):.4f}"
    print(f"{qn:>20s} | {auc_b:>13s} | {auc_a:>13s} | {bal_b:>13s} | {bal_a:>13s}")
