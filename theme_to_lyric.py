"""
Fine-tune and evaluate T5 models for theme -> lyric-line generation.

  python theme_to_lyric.py train --model google/flan-t5-small
  python theme_to_lyric.py train --model google/flan-t5-small --generate-only --themes love loss
  python theme_to_lyric.py evaluate --models google/flan-t5-small runs/flan-t5-small/best
  python theme_to_lyric.py evaluate --models google/flan-t5-small google/flan-t5-base runs/flan-t5-base/best

Run `python theme_to_lyric.py <command> --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

BASE_PROMPT = "Write one song lyric line about {theme}."
THEME_PROMPT = "theme: {theme}"
THEME_QUERY = "a song lyric about {theme}"


# ---------------------------------------------------------------- data / helpers

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def default_themes(data_dir: Path) -> list[str]:
    meta = data_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["themes"]
    return sorted({t for rec in read_jsonl(data_dir / "train.jsonl") for t in rec["themes"]})


def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_pairs(data_dir: Path, split: str) -> list[tuple[str, str]]:
    """One (theme, line) pair per label."""
    return [(THEME_PROMPT.format(theme=theme), rec["text"])
            for rec in read_jsonl(data_dir / f"{split}.jsonl") for theme in rec["themes"]]


@torch.no_grad()
def generate_samples(model, tok, device: str, prompt: str, n: int, args) -> list[str]:
    torch.manual_seed(args.seed)
    enc = tok([prompt], return_tensors="pt").to(device)
    out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                         max_new_tokens=args.max_new_tokens, num_return_sequences=n)
    return [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True)]


@torch.no_grad()
def perplexity(model, tok, pairs: list[tuple[str, str]], device: str,
               max_src: int = 16, max_tgt: int = 48, batch_size: int = 64) -> float:
    """Token-weighted perplexity of target lines given their prompt."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([s for s, _ in chunk], max_length=max_src, truncation=True, padding=True,
                  return_tensors="pt").to(device)
        labels = tok(text_target=[t for _, t in chunk], max_length=max_tgt, truncation=True,
                     padding=True, return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        labels = labels.to(device)
        loss = model(**enc, labels=labels).loss.item()
        n = int((labels != -100).sum().item())
        total_nll += loss * n
        total_tokens += n
    return math.exp(total_nll / max(total_tokens, 1))


# ---------------------------------------------------------------- training

def make_loader(pairs, tok, batch_size: int, shuffle: bool, max_src: int, max_tgt: int) -> DataLoader:
    def collate(batch):
        sources, targets = zip(*batch)
        enc = tok(list(sources), max_length=max_src, truncation=True, padding=True, return_tensors="pt")
        labels = tok(text_target=list(targets), max_length=max_tgt, truncation=True,
                     padding=True, return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100  # ignore padding in the loss
        enc["labels"] = labels
        return enc

    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def mean_loss(model, loader, device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        batches += 1
    return total / max(batches, 1)


def fit_model(args, tok, model, device, out: Path) -> dict:
    train_loader = make_loader(load_pairs(args.data_dir, "train"), tok, args.batch_size, True,
                               args.max_src, args.max_tgt)
    val_loader = make_loader(load_pairs(args.data_dir, "val"), tok, args.eval_batch_size, False,
                             args.max_src, args.max_tgt)
    val_pairs = load_pairs(args.data_dir, "val")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(args.warmup_share * total_steps))

    def lr_factor(step: int) -> float:  # linear warm-up, then linear decay
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    log = {"model": args.model, "params": sum(p.numel() for p in model.parameters()),
           "device": device, "args": {k: str(v) for k, v in vars(args).items() if not callable(v)},
           "steps": [], "epochs": []}
    best_val, start, step = math.inf, time.time(), 0
    print(f"{args.model}: {log['params'] / 1e6:.1f}M params on {device}, "
          f"{len(train_loader)} steps/epoch, {total_steps} steps total")

    epochs = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch", position=0)
    for epoch in epochs:
        model.train()
        running, batches = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="step", position=1, leave=False)
        for batch in bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()
            value = loss.item()
            running += value
            batches += 1
            step += 1
            log["steps"].append({"step": step, "epoch": epoch, "loss": value,
                                 "lr": sched.get_last_lr()[0],
                                 "minutes": round((time.time() - start) / 60, 4)})
            bar.set_postfix(loss=f"{running / batches:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        bar.close()

        val = mean_loss(model, val_loader, device)
        val_ppl = perplexity(model, tok, val_pairs, device, max_src=args.max_src, max_tgt=args.max_tgt)
        entry = {"epoch": epoch, "step": step, "train_loss": running / batches, "val_loss": val,
                 "val_perplexity": val_ppl, "minutes": round((time.time() - start) / 60, 2)}
        log["epochs"].append(entry)
        epochs.set_postfix(train=f"{entry['train_loss']:.3f}", val=f"{val:.3f}", val_ppl=f"{val_ppl:.1f}")
        tqdm.write(f"epoch {epoch}: train {entry['train_loss']:.3f}  val {val:.3f}  "
                   f"val ppl {val_ppl:.2f}  ({entry['minutes']} min)")

        if val < best_val:
            best_val = val
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
            log["best_epoch"] = epoch
            log["best_val_loss"] = val

    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"best epoch {log['best_epoch']} (val loss {best_val:.3f}) saved to {out / 'best'}")
    return log


# ---------------------------------------------------------------- samples

def write_samples_md(path: Path, samples: dict[str, list[str]], model_name: str) -> None:
    lines = [f"# Generated lyrics: {model_name}", ""]
    for theme, group in samples.items():
        lines.append(f"## {theme}")
        lines += [f"- {line}" for line in group]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- theme judge

class Judge:
    """Holds the embedder used to score theme fit, independent of the labeller."""

    def __init__(self, args, themes: list[str], device: str):
        self.themes = themes
        self.device = device
        self.embedder = SentenceTransformer(args.embedder, device=device)
        queries = [THEME_QUERY.format(theme=t) for t in themes]
        self.theme_vecs = self.embedder.encode(queries, normalize_embeddings=True)

    def theme_metrics(self, lines: list[str], requested: list[str]) -> tuple[float, float]:
        vecs = self.embedder.encode(lines, normalize_embeddings=True)
        sims = vecs @ self.theme_vecs.T
        top1, mrr = [], []
        for row, req in zip(sims, requested):
            rank = list(np.argsort(-row)).index(self.themes.index(req)) + 1
            top1.append(rank == 1)
            mrr.append(1.0 / rank)
        return float(np.mean(top1)), float(np.mean(mrr))

    def metrics(self, lines: list[str], requested: list[str]) -> dict:
        top1, mrr = self.theme_metrics(lines, requested)
        return {"theme_top1": top1, "theme_mrr": mrr}


# ---------------------------------------------------------------- model comparison

@dataclass
class ModelSpec:
    path: str          # Hugging Face id or local folder
    label: str         # name used in tables and files
    kind: str          # "base" or "fine-tuned"
    prompt: str        # prompt template with {theme}


def model_spec(arg: str) -> ModelSpec:
    path = Path(arg)
    if path.is_dir():
        base = path.parent.name if path.name == "best" else path.name
        return ModelSpec(arg, f"{base} (fine-tuned)", "fine-tuned", THEME_PROMPT)
    base = arg.split("/")[-1]
    return ModelSpec(arg, f"{base} (base, no fine-tuning)", "base", BASE_PROMPT)


def prompt_pairs(spec: ModelSpec, records: list[dict]) -> list[tuple[str, str]]:
    return [(spec.prompt.format(theme=t), rec["text"]) for rec in records for t in rec["themes"]]


def evaluate_model(spec: ModelSpec, args, themes, records: dict, judge: Judge, device: str) -> tuple[dict, dict]:
    tok = AutoTokenizer.from_pretrained(spec.path)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.path).to(device).eval()

    samples = {t: generate_samples(model, tok, device, spec.prompt.format(theme=t), args.samples_per_theme, args)
               for t in themes}
    lines = [l for t in themes for l in samples[t]]
    requested = [t for t in themes for _ in samples[t]]

    metrics = {f"{split}_perplexity": perplexity(model, tok, prompt_pairs(spec, records[split]), device, max_src=32)
               for split in ("train", "val", "test")}
    metrics.update(judge.metrics(lines, requested))
    del model
    torch.cuda.empty_cache()
    return metrics, samples


def write_markdown(path: Path, rows: dict[str, dict], footer: str = "") -> None:
    cols = [("train_perplexity", "train ppl", "{:.2f}"), ("val_perplexity", "val ppl", "{:.2f}"),
            ("test_perplexity", "test ppl", "{:.2f}"), ("theme_top1", "theme top-1", "{:.2f}"),
            ("theme_mrr", "theme MRR", "{:.2f}")]
    header = "| model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [header, sep]
    for name, m in rows.items():
        cells = [cols[i][2].format(m[cols[i][0]]) if cols[i][0] in m else "n/a" for i in range(len(cols))]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    if footer:
        text += "\n" + footer + "\n"
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- plots

def moving_average(values: list[float], window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training(log: dict, out: Path) -> None:
    """training_curves.png: loss per step, loss over wall-clock time, validation per epoch."""
    steps = log["steps"]
    x_step = np.array([s["step"] for s in steps])
    y_loss = [s["loss"] for s in steps]
    x_min = np.array([s["minutes"] for s in steps])
    window = max(1, min(50, len(steps) // 10))
    smooth = moving_average(y_loss, window)
    epochs = log["epochs"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.plot(x_step, y_loss, alpha=0.25, lw=0.8, label="per step")
    ax.plot(x_step[len(x_step) - len(smooth):], smooth, lw=2, label=f"moving avg ({window})")
    for e in epochs:
        ax.axvline(e["step"], color="grey", ls=":", lw=0.8)
    ax.set_title("Training loss per step")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()

    ax = axes[1]
    ax.plot(x_min, y_loss, alpha=0.25, lw=0.8, label="per step")
    ax.plot(x_min[len(x_min) - len(smooth):], smooth, lw=2, label=f"moving avg ({window})")
    ax.scatter([e["minutes"] for e in epochs], [e["val_loss"] for e in epochs],
               color="C3", zorder=3, label="validation (end of epoch)")
    ax.set_title("Loss over training time")
    ax.set_xlabel("minutes")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()

    ax = axes[2]
    ep = [e["epoch"] for e in epochs]
    ax.plot(ep, [e["train_loss"] for e in epochs], marker="o", label="train loss")
    ax.plot(ep, [e["val_loss"] for e in epochs], marker="o", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(ep, [e["val_perplexity"] for e in epochs], color="C2", marker="s", ls="--",
             label="validation perplexity")
    ax2.set_ylabel("perplexity")
    ax.set_title("Per-epoch losses and perplexity")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")

    fig.suptitle(f"{log['model']} training ({log['device']})")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_metrics(metrics: dict, out: Path, model_name: str) -> None:
    """metrics.png: train, validation, and test perplexity for one model."""
    splits = ["train", "val", "test"]
    values = [metrics[f"{s}_perplexity"] for s in splits]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(splits, values, color=["C0", "C1", "C2"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_title("Perplexity (lower is better)")
    ax.set_ylabel("perplexity")
    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=120)
    plt.close(fig)


def plot_comparison(path: Path, rows: dict[str, dict]) -> None:
    names = list(rows)
    short = [n.replace(" (", "\n(") for n in names]
    keys = [("theme_top1", "theme top-1 (higher is better)"),
            ("theme_mrr", "theme MRR (higher is better)"),
            ("test_perplexity", "test perplexity (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes.flat, keys):
        vals = [rows[n].get(key, np.nan) for n in names]
        colors = ["C2" if "fine-tuned" in n else "C0" for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.bar_label(bars, fmt="%.0f" if key == "test_perplexity" else "%.2f")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=8)
        ax.set_title(title)
        if key == "test_perplexity":
            ax.set_yscale("log")
    fig.suptitle("Base model, no fine-tuning (blue), fine-tuned (green)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def measure_checkpoint(args, tok, model, device, themes: list[str]) -> tuple[dict, dict]:
    """Train, validation, and test perplexity for one already-trained model."""
    samples = {theme: generate_samples(model, tok, device, THEME_PROMPT.format(theme=theme),
                                       args.samples_per_theme, args)
               for theme in themes}
    metrics = {f"{split}_perplexity": perplexity(model, tok, load_pairs(args.data_dir, split), device,
                                                 max_src=args.max_src, max_tgt=args.max_tgt)
               for split in ("train", "val", "test")}
    return metrics, samples


def cmd_train(args) -> None:
    device = args.device
    name = args.model.split("/")[-1]
    out = args.out_dir or Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    themes = args.themes or default_themes(args.data_dir)

    log = None
    if args.generate_only:
        if not (out / "best").exists():
            raise SystemExit(f"no checkpoint at {out / 'best'}; train first")
        tok = AutoTokenizer.from_pretrained(out / "best")
        model = AutoModelForSeq2SeqLM.from_pretrained(out / "best").to(device).eval()
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
        log = fit_model(args, tok, model, device, out)
        # Reload the best checkpoint from disk; the in-memory model is from the last epoch.
        model = AutoModelForSeq2SeqLM.from_pretrained(out / "best").to(device).eval()

    metrics, samples = measure_checkpoint(args, tok, model, device, themes)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    write_samples_md(out / "samples.md", samples, name)
    plot_metrics(metrics, out, name)
    if args.generate_only and (out / "train_log.json").exists():
        log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    if log and "steps" in log:
        plot_training(log, out)

    print(f"\ntrain perplexity {metrics['train_perplexity']:.2f}   "
          f"val perplexity {metrics['val_perplexity']:.2f}   test perplexity {metrics['test_perplexity']:.2f}")
    print("\nexample lyrics:")
    for theme, group in samples.items():
        print(f"\n[{theme}]")
        for line in group[:2]:
            print(f"  {line}")
    print(f"\nsaved metrics.json, samples.json, samples.md, metrics.png, training_curves.png to {out}")


def cmd_evaluate(args) -> None:
    device = args.device
    themes = args.themes or default_themes(args.data_dir)
    records = {split: read_jsonl(args.data_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"themes: {themes}\ntest lines: {len(records['test'])}  device: {device}")
    judge = Judge(args, themes, device)

    rows: dict[str, dict] = {}
    for arg in args.models:
        spec = model_spec(arg)
        print(f"\nevaluating {spec.label} ({spec.path}) ...")
        metrics, samples = evaluate_model(spec, args, themes, records, judge, device)
        rows[spec.label] = metrics
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.label)
        (args.out / f"samples_{safe}.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
        print(f"  train ppl {metrics['train_perplexity']:.2f}  val ppl {metrics['val_perplexity']:.2f}  "
              f"test ppl {metrics['test_perplexity']:.2f}  theme top-1 {metrics['theme_top1']:.2f}  "
              f"theme MRR {metrics['theme_mrr']:.2f}")

    n = len(themes)
    chance = (f"Chance for theme top-1 is 1/{n} = {1 / n:.2f}; "
              f"chance theme MRR = {sum(1 / k for k in range(1, n + 1)) / n:.2f}.")

    (args.out / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(args.out / "comparison.md", rows, chance)
    plot_comparison(args.out / "comparison.png", rows)
    print("\n" + (args.out / "comparison.md").read_text(encoding="utf-8"))
    print(f"saved comparison.md, comparison.json, comparison.png and samples to {args.out}")


# ---------------------------------------------------------------- CLI

def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="folder with the JSONL splits (default: data)")
    p.add_argument("--seed", type=int, default=13, help="random seed (default: 13)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="default: auto")


def add_sampling_args(p: argparse.ArgumentParser, samples_default: int) -> None:
    p.add_argument("--themes", nargs="+", default=None,
                   help="themes to use (default: every theme in data/meta.json)")
    p.add_argument("--samples-per-theme", type=int, default=samples_default,
                   help=f"lines generated per theme (default: {samples_default})")
    p.add_argument("--temperature", type=float, default=0.9, help="sampling temperature (default: 0.9)")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling threshold (default: 0.9)")
    p.add_argument("--max-new-tokens", type=int, default=40, help="max generated tokens (default: 40)")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Fine-tune and evaluate T5 models for theme -> lyric-line generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fine-tune a T5 model and generate sample lyrics")
    p.add_argument("--model", required=True, help="Hugging Face id, e.g. google/flan-t5-small")
    p.add_argument("--out-dir", type=Path, default=None, help="where to save (default: runs/<model name>)")
    p.add_argument("--generate-only", action="store_true",
                   help="skip training; evaluate and generate from an existing <out-dir>/best")
    p.add_argument("--epochs", type=int, default=5, help="training epochs (default: 5)")
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    p.add_argument("--batch-size", type=int, default=32, help="training batch size (default: 32)")
    p.add_argument("--eval-batch-size", type=int, default=64, help="validation batch size (default: 64)")
    p.add_argument("--max-src", type=int, default=16, help="max input tokens (default: 16)")
    p.add_argument("--max-tgt", type=int, default=48, help="max target tokens (default: 48)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="gradient norm clip (default: 1.0)")
    p.add_argument("--warmup-share", type=float, default=0.05, help="share of steps used for warm-up (default: 0.05)")
    add_data_args(p)
    add_sampling_args(p, 5)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="compare base and fine-tuned models on the test split")
    p.add_argument("--models", nargs="+", required=True,
                   help="Hugging Face ids (base, no fine-tuning) and/or fine-tuned checkpoint folders")
    p.add_argument("--out", type=Path, default=Path("evaluation"), help="output folder (default: evaluation)")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence embedder for theme fit (default: all-MiniLM-L6-v2)")
    add_data_args(p)
    add_sampling_args(p, 10)
    p.set_defaults(func=cmd_evaluate)

    return root


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.device = pick_device(args.device)
    args.func(args)


if __name__ == "__main__":
    main()