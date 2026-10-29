"""
Mini-T5 : pipeline complet (données + modèle + entraînement + évaluation).

- Pipeline (tokenizer, WikiText-2, batches, Adafactor, évaluation) : repris du notebook du groupe.
- Pré-entraînement (article §3.1.4) : 15 % des tokens supprimés, chaque suite de tokens
  consécutifs remplacée par un repère unique ; cible = morceaux supprimés délimités par
  les mêmes repères + un repère final.
- Fine-tuning (article §2.4) : format texte-à-texte avec préfixe de tâche ; même modèle,
  même loss, même boucle que le pré-entraînement.
- Modèle écrit à la main avec nn.Module. Particularités de T5 (article §2.1, §3.1.1) :
    * normalisation sans biais, placée avant chaque sous-couche (hors chemin résiduel)
    * positions relatives (un nombre par tête et par "bucket", ajouté au score d'attention)
    * activation ReLU dans le feed-forward
    * embeddings partagés entre l'entrée et la couche de sortie
    * dropout de 0,1 (feed-forward, connexions résiduelles, poids d'attention, entrée et sortie des piles)
  Choix d'implémentation non précisés dans l'article : couches linéaires sans biais,
  division des scores d'attention par sqrt(dk).
- Optimisation (article §3.1.2) : Adafactor, cross-entropy avec teacher forcing,
  learning rate en racine carrée inverse au pré-entraînement, constant à 1e-3 au fine-tuning.

Dépendances : torch, transformers, datasets, sentencepiece
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

# ============================================================ 1. Configuration
SEED = 42
BLOCK_SIZE = 128          # longueur maximale d'une séquence
BATCH_SIZE = 4
EPOCHS = 5                # pré-entraînement ; garder petit sur CPU
LR = 1e-2                 # article §3.1.2 : 0,01 puis décroissance en 1/sqrt(n)
WARMUP = 1000             # k de l'article (10 000 dans l'article, réduit ici)
NOISE_DENSITY = 0.15      # article §3.1.4
FT_LR = 1e-3              # article §3.1.2 : learning rate constant au fine-tuning
FT_EPOCHS = 3
N_TRAIN, N_VAL = 5000, 500

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ 2. Données
def corrupt_spans(ids, noise_density, rng, first_sentinel):
    """Exemple de l'article (figure 2) :
    entrée  Thank you <X> me to your party <Y> week
    cible   <X> for inviting <Y> last <Z>"""
    dropped = [rng.random() < noise_density for _ in ids]
    if not any(dropped):
        dropped[rng.randrange(len(ids))] = True
    inputs, target, n_spans, previous = [], [], 0, False
    for token, drop in zip(ids, dropped):
        if drop:
            if not previous:
                inputs.append(first_sentinel - n_spans)
                target.append(first_sentinel - n_spans)
                n_spans += 1
            target.append(token)
        else:
            inputs.append(token)
        previous = drop
    target.append(first_sentinel - n_spans)
    return inputs, target


class DenoisingDataset(Dataset):
    """Pré-entraînement : texte à trous, sans étiquette humaine."""

    def __init__(self, corpus, tokenizer, noise_density=NOISE_DENSITY, seed=42, max_length=128):
        self.examples = []
        rng = random.Random(seed)
        first_sentinel = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        eos = tokenizer.eos_token_id

        for sentence in corpus:
            ids = tokenizer.encode(sentence, add_special_tokens=False)[: max_length - 1]
            if len(ids) < 3:
                continue
            inputs, target = corrupt_spans(ids, noise_density, rng, first_sentinel)
            self.examples.append({
                "input_ids": torch.tensor(inputs + [eos], dtype=torch.long),
                "labels": torch.tensor(target + [eos], dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class TextToTextDataset(Dataset):
    """Fine-tuning : entrée = "préfixe de tâche: texte", cible = texte attendu."""

    def __init__(self, pairs, tokenizer, max_length=128):
        eos = tokenizer.eos_token_id
        self.examples = [{
            "input_ids": torch.tensor(
                tokenizer.encode(f"{task}: {src}", add_special_tokens=False)[: max_length - 1] + [eos]),
            "labels": torch.tensor(
                tokenizer.encode(tgt, add_special_tokens=False)[: max_length - 1] + [eos]),
        } for task, src, tgt in pairs]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate_fn(pad_id):
    def collate_fn(batch):
        max_in = max(len(b["input_ids"]) for b in batch)
        max_lab = max(len(b["labels"]) for b in batch)
        inputs, masks, labels = [], [], []
        for b in batch:
            n_in, n_lab = len(b["input_ids"]), len(b["labels"])
            inputs.append(torch.cat([b["input_ids"], torch.full((max_in - n_in,), pad_id)]))
            masks.append(torch.cat([torch.ones(n_in, dtype=torch.long),
                                    torch.zeros(max_in - n_in, dtype=torch.long)]))
            # -100 : positions ignorées par la loss
            labels.append(torch.cat([b["labels"], torch.full((max_lab - n_lab,), -100)]))
        return {"input_ids": torch.stack(inputs),
                "attention_mask": torch.stack(masks),
                "labels": torch.stack(labels)}
    return collate_fn


# ============================================================ 3. Modèle : modules T5 à la main
class T5LayerNorm(nn.Module):
    """On redimensionne seulement : pas de moyenne soustraite, pas de biais."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RelativePositionBias(nn.Module):
    """Un nombre par (tête, bucket), ajouté au score d'attention.
    32 buckets : petits écarts exacts, grands écarts en plages logarithmiques,
    au-delà de max_distance (128) tout partage le même bucket."""

    def __init__(self, n_heads, n_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        self.n_buckets, self.max_distance, self.bidirectional = n_buckets, max_distance, bidirectional
        self.emb = nn.Embedding(n_buckets, n_heads)

    def _bucket(self, rel):  # rel = position_clé - position_requête
        nb, out = self.n_buckets, 0
        if self.bidirectional:           # encodeur : gauche et droite distinguées
            nb //= 2
            out = (rel > 0).long() * nb
            rel = rel.abs()
        else:                            # décodeur : seulement le passé
            rel = -torch.clamp(rel, max=0)
        max_exact = nb // 2
        is_small = rel < max_exact
        large = max_exact + (
            torch.log(rel.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact) * (nb - max_exact)
        ).long()
        large = torch.clamp(large, max=nb - 1)
        return out + torch.where(is_small, rel, large)

    def forward(self, q_len, k_len, device):
        q = torch.arange(q_len, device=device)[:, None]
        k = torch.arange(k_len, device=device)[None, :]
        return self.emb(self._bucket(k - q)).permute(2, 0, 1)[None]   # (1, H, Q, K)


class MultiHeadAttention(nn.Module):
    """Self-attention (kv=None) ou attention croisée (kv = sortie de l'encodeur)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, t):  # (B, L, D) -> (B, H, L, dk) : têtes indépendantes
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x, kv=None, mask=None, bias=None):
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)    # (B, H, Q, K)
        if bias is not None:
            scores = scores + bias                               # positions relatives
        if mask is not None:                                     # True = on garde
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))              # moyenne pondérée
        out = (attn @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)   # concat des têtes
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.wo(self.drop(F.relu(self.wi(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1, self.norm2 = T5LayerNorm(d_model), T5LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, bias):
        x = x + self.drop(self.attn(self.norm1(x), mask=mask, bias=bias))   # connexion résiduelle
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderBlock(nn.Module):
    """Comme l'encodeur, avec en plus : masque causal + attention vers l'encodeur."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (T5LayerNorm(d_model) for _ in range(3))
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, causal_mask, cross_mask, bias):
        x = x + self.drop(self.self_attn(self.norm1(x), mask=causal_mask, bias=bias))
        x = x + self.drop(self.cross_attn(self.norm2(x), kv=enc_out, mask=cross_mask))
        return x + self.drop(self.ff(self.norm3(x)))


def objective(logits, labels):
    """Fonction objectif : cross-entropy entre les logits et les tokens attendus (article §3.1.2)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


class MiniT5(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=128, n_heads=4, d_ff=512,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.pad_token_id, self.d_model = pad_token_id, d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)   # partagé entrée / sortie
        self.enc_blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.dec_blocks = nn.ModuleList(DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.enc_bias = RelativePositionBias(n_heads, bidirectional=True)
        self.dec_bias = RelativePositionBias(n_heads, bidirectional=False)
        self.enc_norm, self.dec_norm = T5LayerNorm(d_model), T5LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def shift_right(self, labels):
        """Entrée du décodeur = labels décalés d'un cran, avec PAD au début (teacher forcing)."""
        dec_in = labels.new_full(labels.shape, self.pad_token_id)
        dec_in[:, 1:] = labels[:, :-1]
        return dec_in.masked_fill(dec_in == -100, self.pad_token_id)   # -100 n'est pas un token

    def encode(self, input_ids, src_mask):
        bias = self.enc_bias(input_ids.size(1), input_ids.size(1), input_ids.device)
        x = self.drop(self.token_embedding(input_ids))
        for blk in self.enc_blocks:
            x = blk(x, src_mask, bias)
        return self.drop(self.enc_norm(x))

    def decode(self, dec_in, enc_out, src_mask):
        L = dec_in.size(1)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dec_in.device))[None, None]
        bias = self.dec_bias(L, L, dec_in.device)
        x = self.drop(self.token_embedding(dec_in))
        for blk in self.dec_blocks:
            x = blk(x, enc_out, causal, src_mask, bias)
        x = self.drop(self.dec_norm(x))
        return (x * self.d_model ** -0.5) @ self.token_embedding.weight.T   # poids partagés -> logits

    def forward(self, input_ids, attention_mask, labels=None, decoder_input_ids=None):
        src_mask = attention_mask.bool()[:, None, None, :]       # on ignore le padding de l'entrée
        enc_out = self.encode(input_ids, src_mask)
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Fournir labels ou decoder_input_ids.")
            decoder_input_ids = self.shift_right(labels)
        logits = self.decode(decoder_input_ids, enc_out, src_mask)
        loss = objective(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


# ============================================================ 4. Entraînement et évaluation
def make_optimizer(model, lr):
    from transformers import Adafactor
    return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False,
                     warmup_init=False, clip_threshold=1.0, weight_decay=0.0)


def train(model, loader, optimizer, scheduler, epochs):
    for epoch in range(epochs):               # 1 epoch = un passage sur tous les batches
        model.train()
        total = 0.0
        for batch in loader:                  # 1 itération = 1 batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()                                   # remise à zéro des gradients
            loss = model(input_ids, attention_mask, labels)["loss"]   # passe avant + loss
            loss.backward()                                         # rétropropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # limite la norme du gradient
            optimizer.step()                                        # mise à jour des poids
            scheduler.step()                                        # mise à jour du learning rate
            total += loss.item()
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss = {total / len(loader):.4f} "
              f"| LR = {optimizer.param_groups[0]['lr']:.2e}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["labels"].to(device))
        total += out["loss"].item()
    return total / len(loader)


@torch.no_grad()
def show_examples(model, loader, tokenizer, n=3):
    """Prédiction avec teacher forcing (on donne les vrais tokens précédents au décodeur)."""
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    out = model(input_ids, batch["attention_mask"].to(device), batch["labels"].to(device))
    preds = out["logits"].argmax(-1)
    for j in range(min(n, input_ids.size(0))):
        labels = batch["labels"][j]
        print("\nEntrée    :", tokenizer.decode(input_ids[j], skip_special_tokens=False))
        print("Cible     :", tokenizer.decode(labels[labels != -100], skip_special_tokens=False))
        print("Prédiction:", tokenizer.decode(preds[j], skip_special_tokens=False))


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens=20):
    """Génération réelle : le décodeur reçoit ses propres prédictions, mot par mot."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]],
                       device=device)
    src_mask = torch.ones_like(ids).bool()[:, None, None, :]
    enc_out = model.encode(ids, src_mask)
    out = torch.tensor([[tokenizer.pad_token_id]], device=device)
    for _ in range(max_new_tokens):
        next_token = model.decode(out, enc_out, src_mask)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out[0, 1:], skip_special_tokens=True)


def overfit_one_batch(model, batch, steps=200):
    """Test de santé : la loss doit tomber vers 0 si le modèle et la boucle fonctionnent."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        loss = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                     batch["labels"].to(device))["loss"]
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == steps - 1:
            print(f"step {step:3d} | loss = {loss.item():.4f}")


# ============================================================ 5. Programme principal
if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Device utilisé :", device)
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print("Vocabulaire :", tokenizer.vocab_size, "| pad :", tokenizer.pad_token, "| eos :", tokenizer.eos_token)

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_texts = [t.strip() for t in data["train"]["text"] if t.strip()][:N_TRAIN]
    val_texts = [t.strip() for t in data["validation"]["text"] if t.strip()][:N_VAL]

    train_ds = DenoisingDataset(train_texts, tokenizer, NOISE_DENSITY, seed=42, max_length=BLOCK_SIZE)
    val_ds = DenoisingDataset(val_texts, tokenizer, NOISE_DENSITY, seed=123, max_length=BLOCK_SIZE)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(f"Exemples train/val : {len(train_ds)}/{len(val_ds)} | batches train : {len(train_loader)}")
    print(f"Mises à jour prévues : {len(train_loader)} batches x {EPOCHS} epochs = {len(train_loader) * EPOCHS}")

    model = MiniT5(len(tokenizer), tokenizer.pad_token_id).to(device)
    print("Paramètres :", sum(p.numel() for p in model.parameters()))

    # 1) test de santé sur un seul batch
    overfit_one_batch(model, next(iter(train_loader)))

    # 2) pré-entraînement : texte à trous, learning rate en racine carrée inverse
    model = MiniT5(len(tokenizer), tokenizer.pad_token_id).to(device)
    optimizer = make_optimizer(model, LR)
    scheduler = LambdaLR(optimizer, lambda n: math.sqrt(WARMUP / max(n, WARMUP)))
    train(model, train_loader, optimizer, scheduler, EPOCHS)
    print(f"\nValidation Loss = {evaluate(model, val_loader):.4f}")
    show_examples(model, val_loader, tokenizer)

    # 3) fine-tuning : même modèle, même loss, même boucle, format texte-à-texte (figure 1 de l'article)
    pairs = [("translate English to German", "That is good.", "Das ist gut."),
             ("cola sentence", "The course is jumping well.", "not acceptable")]
    ft_loader = DataLoader(TextToTextDataset(pairs, tokenizer), batch_size=2, shuffle=True, collate_fn=collate)
    ft_optimizer = make_optimizer(model, FT_LR)
    ft_scheduler = LambdaLR(ft_optimizer, lambda n: 1.0)
    train(model, ft_loader, ft_optimizer, ft_scheduler, FT_EPOCHS)
    for task, src, _ in pairs:
        print(f"{task}: {src} -> {generate(model, tokenizer, f'{task}: {src}')}")
