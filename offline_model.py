"""Small PyTorch theme-conditioned GRU baseline; trains and runs without downloads.

The full FLAN-T5 experiment remains in theme_to_lyric.py. This deliberately
compact baseline makes the GUI usable from a clean clone with included weights.
No lyric text from the training dataset is committed to the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
THEME_PREFIX = "<theme_"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[,.!?;:]", flags=re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def detokenize(words: list[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text[:1].upper() + text[1:]


def load_rows(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See README: Data setup.")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def get_themes(rows: list[dict]) -> list[str]:
    return sorted({theme for row in rows for theme in row['themes']})


def make_vocab(train_rows: list[dict], themes: list[str], vocab_size: int) -> dict[str, int]:
    counter = Counter(word for row in train_rows for word in tokenize(row['text']))
    basics = [PAD, UNK, BOS, EOS] + [f"<theme_{t}>" for t in themes] + ["<theme_generic>"]
    words = [w for w, _ in counter.most_common(max(0, vocab_size - len(basics))) if w not in basics]
    return {w: i for i, w in enumerate(basics + words)}


def pairs_from_rows(rows: list[dict]) -> list[tuple[str, str]]:
    return [(theme, row['text']) for row in rows for theme in row['themes']]


class ThemeGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        outputs, hidden = self.rnn(self.embedding(token_ids), hidden)
        return self.projection(outputs), hidden


def encode_pair(theme: str, text: str, vocab: dict[str, int], max_words: int,
                unconditional: bool) -> tuple[list[int], list[int]]:
    ids = [vocab.get(w, vocab[UNK]) for w in tokenize(text)[:max_words]]
    theme_key = "<theme_generic>" if unconditional else f"<theme_{theme}>"
    inp = [vocab[BOS], vocab[theme_key]] + ids
    target = [vocab[PAD]] + ids + [vocab[EOS]]
    return inp, target


def loader_for(pairs, vocab, max_words, unconditional, batch_size, shuffle, seed=13):
    def collate(items):
        examples = [encode_pair(t, line, vocab, max_words, unconditional) for t, line in items]
        size = max(len(inp) for inp, _ in examples)
        src = torch.full((len(examples), size), vocab[PAD], dtype=torch.long)
        tgt = src.clone()
        for i, (x, y) in enumerate(examples):
            src[i, :len(x)] = torch.tensor(x)
            tgt[i, :len(y)] = torch.tensor(y)
        return src, tgt
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator, num_workers=0)


def batch_stats(model, batches, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for src, target in batches:
            src, target = src.to(device), target.to(device)
            logits, _ = model(src)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               target.reshape(-1), ignore_index=0, reduction='sum')
            total_loss += float(loss.item())
            total_tokens += int((target != 0).sum().item())
    return total_loss / max(total_tokens, 1), total_tokens


def save_checkpoint(path, model, vocab, themes, args, epoch, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "themes": themes,
                "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
                "max_words": args.max_words, "unconditional": args.unconditional,
                "epoch": epoch, "val_loss": val_loss,
                "seed": args.seed, "max_train_lines": args.max_train_lines}, path)


def load_checkpoint(path: Path, device: str = 'cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    model = ThemeGRU(len(bundle['vocab']), bundle['embed_dim'], bundle['hidden_dim'])
    model.load_state_dict(bundle['state_dict'])
    model.to(device).eval()
    return model, bundle


def run_train(args):
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_rows(args.data_dir, 'train')
    valid_rows = load_rows(args.data_dir, 'val')
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    if args.max_train_lines:
        train_rows = train_rows[:args.max_train_lines]
    if args.max_val_lines:
        valid_rows = valid_rows[:args.max_val_lines]
    themes = get_themes(load_rows(args.data_dir, 'train'))
    vocab = make_vocab(train_rows, themes, args.vocab_size)
    train_pairs, val_pairs = pairs_from_rows(train_rows), pairs_from_rows(valid_rows)
    train_dl = loader_for(train_pairs, vocab, args.max_words, args.unconditional, args.batch_size, True, args.seed)
    val_dl = loader_for(val_pairs, vocab, args.max_words, args.unconditional, args.batch_size, False)
    model = ThemeGRU(len(vocab), args.embed_dim, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float('inf')
    history = []
    since = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int((y != 0).sum())
            total += float(loss.item()) * n
            count += n
        train_loss = total / max(count, 1)
        val_loss, val_tokens = batch_stats(model, val_dl, args.device)
        item = dict(epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                    val_perplexity=round(math.exp(min(val_loss, 50)), 3))
        history.append(item)
        print(f"epoch {epoch}/{args.epochs}: train={train_loss:.4f}, val={val_loss:.4f}, val_ppl={item['val_perplexity']:.2f}", flush=True)
        if val_loss < best:
            best = val_loss
            save_checkpoint(args.out, model, vocab, themes, args, epoch, val_loss)
    summary = {'model': 'conditional GRU word-level language model' if not args.unconditional else 'unconditional GRU word-level language model',
               'train_lines': len(train_rows), 'train_pairs': len(train_pairs),
               'val_lines': len(valid_rows), 'val_pairs': len(val_pairs),
               'vocab_size': len(vocab), 'total_parameters': sum(p.numel() for p in model.parameters()),
               'embed_dim': args.embed_dim, 'hidden_dim': args.hidden_dim,
               'learning_rate': args.lr, 'batch_size': args.batch_size,
               'seed': args.seed, 'epochs_requested': args.epochs, 'best_epoch': min(history, key=lambda e:e['val_loss'])['epoch'],
               'best_val_loss': best, 'elapsed_seconds': round(time.perf_counter()-since, 2),
               'history': history, 'device': args.device}
    args.out.with_suffix('.training.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('saved', args.out)


def run_evaluate(args):
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model, ck = load_checkpoint(args.checkpoint, args.device)
    rows = {}
    for split in ('train', 'val', 'test'):
        dataset = load_rows(args.data_dir, split)
        if args.max_eval_lines:
            dataset = dataset[:args.max_eval_lines]
        pairs = pairs_from_rows(dataset)
        batch = loader_for(pairs, ck['vocab'], ck['max_words'], ck['unconditional'], args.batch_size, False)
        loss, tokens = batch_stats(model, batch, args.device)
        rows[split] = {'lines': len(dataset), 'pairs': len(pairs), 'tokens': tokens,
                       'cross_entropy': round(loss, 5), 'perplexity': round(math.exp(min(loss, 50)), 4)}
    result = {'model_kind': 'unconditional GRU' if ck['unconditional'] else 'theme-conditioned GRU',
              'checkpoint': str(args.checkpoint), 'checkpoint_best_epoch': ck['epoch'],
              'metrics': rows, 'note': 'Teacher-forced token-level perplexity on learned vocabulary; not directly comparable to T5 tokenizer perplexity.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def sample_index(probabilities, rng):
    return int(torch.multinomial(probabilities, 1, generator=rng).item())


@torch.inference_mode()
def generate(model, ck: dict, theme: str, *, temperature=0.9, top_k=35,
             max_words=14, min_words=5, seed=13) -> str:
    """Generate one new line. No training text is accessed or returned."""
    if theme not in ck['themes']:
        raise ValueError(f"Choose one of: {', '.join(ck['themes'])}")
    if not (0.3 <= float(temperature) <= 2):
        raise ValueError('temperature must be between 0.3 and 2')
    vocab = ck['vocab']
    rev = {v:k for k,v in vocab.items()}
    prefix = '<theme_generic>' if ck['unconditional'] else f'<theme_{theme}>'
    device = next(model.parameters()).device
    x = torch.tensor([[vocab[BOS], vocab[prefix]]], device=device)
    logits, hidden = model(x)
    rng = torch.Generator(device=device).manual_seed(int(seed))
    words = []
    for i in range(max_words):
        scores = (logits[0,-1].float()/float(temperature)).clone()
        for bad in [PAD, UNK, BOS]:
            scores[vocab[bad]] = -float('inf')
        for token, idx in vocab.items():
            if token.startswith('<theme_'):
                scores[idx] = -float('inf')
        if len(words) < min_words:
            scores[vocab[EOS]] = -float('inf')
        if words:
            previous = words[-1]
            if previous in vocab:
                scores[vocab[previous]] -= 1.0
        vals, inds = torch.topk(scores, k=min(max(1, top_k), len(vocab)))
        probs = torch.softmax(vals, dim=-1)
        token_id = int(inds[sample_index(probs, rng)].item())
        token = rev[token_id]
        if token == EOS:
            break
        words.append(token)
        x = torch.tensor([[token_id]], device=device)
        logits, hidden = model(x, hidden)
    return detokenize(words)


def run_generate(args):
    model, ck = load_checkpoint(args.checkpoint, args.device)
    for i in range(args.count):
        print(generate(model, ck, args.theme, temperature=args.temperature, seed=args.seed+i))


def run_data_check(args):
    datasets = {split: load_rows(args.data_dir, split) for split in ('train', 'val', 'test')}
    id_sets = {name: {r['song_id'] for r in rows} for name, rows in datasets.items()}
    report = {}
    for name, rows in datasets.items():
        report[name] = {'lines':len(rows), 'songs':len(id_sets[name]),
                        'theme_assignments':sum(len(r['themes']) for r in rows),
                        'label_count':dict(Counter(t for r in rows for t in r['themes'])),
                        'mean_words':round(sum(len(r['text'].split()) for r in rows)/len(rows), 3)}
    report['song_overlap'] = {f'{a}_{b}':len(id_sets[a] & id_sets[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if any(report['song_overlap'].values()):
        raise SystemExit('ERROR: song leakage between splits')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    t = sub.add_parser('train', help='train a small offline GRU')
    t.add_argument('--data-dir', type=Path, default=Path('data'))
    t.add_argument('--out', type=Path, default=Path('checkpoints/offline_best.pt'))
    t.add_argument('--epochs', type=int, default=5)
    t.add_argument('--max-train-lines', type=int, default=4000, help='0=all training lines')
    t.add_argument('--max-val-lines', type=int, default=1000, help='0=all validation lines')
    t.add_argument('--vocab-size', type=int, default=2000)
    t.add_argument('--embed-dim', type=int, default=64)
    t.add_argument('--hidden-dim', type=int, default=96)
    t.add_argument('--max-words', type=int, default=16)
    t.add_argument('--batch-size', type=int, default=64)
    t.add_argument('--lr', type=float, default=0.002)
    t.add_argument('--seed', type=int, default=13)
    t.add_argument('--device', default='cpu')
    t.add_argument('--unconditional', action='store_true', help='ablation: ignore theme token')
    t.set_defaults(func=run_train)
    e = sub.add_parser('evaluate', help='test a trained offline GRU against the JSONL splits')
    e.add_argument('--checkpoint', type=Path, default=Path('checkpoints/offline_best.pt'))
    e.add_argument('--data-dir', type=Path, default=Path('data'))
    e.add_argument('--out', type=Path, default=Path('experiments/offline_metrics.json'))
    e.add_argument('--max-eval-lines', type=int, default=0, help='0=all lines')
    e.add_argument('--batch-size', type=int, default=64)
    e.add_argument('--device', default='cpu')
    e.set_defaults(func=run_evaluate)
    g = sub.add_parser('generate', help='generate a line from the included checkpoint')
    g.add_argument('--checkpoint', type=Path, default=Path('checkpoints/offline_best.pt'))
    g.add_argument('--theme', required=True)
    g.add_argument('--count', type=int, default=1)
    g.add_argument('--seed', type=int, default=13)
    g.add_argument('--temperature', type=float, default=0.9)
    g.add_argument('--device', default='cpu')
    g.set_defaults(func=run_generate)
    c = sub.add_parser('data-check', help='summary + song-disjoint split integrity test')
    c.add_argument('--data-dir', type=Path, default=Path('data'))
    c.add_argument('--out', type=Path, default=Path('experiments/data_audit.json'))
    c.set_defaults(func=run_data_check)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
