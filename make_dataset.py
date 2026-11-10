"""
Build a multi-label lyric theme dataset with a small local LLM (no hand labels).

Subcommands (run in order, or use `all`):

  pool      Sample English songs from the Genius shard, clean them into lines,
            split by song. Writes <out>/pool.jsonl.
  discover  The LLM reads random groups of train lines and proposes short theme
            names. The most frequent names become the theme set. Writes <out>/themes.json.
  label     The LLM scores every line against every theme ("Is this about X?",
            log P(Yes) - log P(No)). A line gets a theme when its score is in the
            top --top-share for that theme, up to --max-labels per line.
            Writes <out>/labels.jsonl.
  build     Keep themed lines and write <out>/{train,val,test}.jsonl with the
            requested counts, plus <out>/meta.json.
  all       Run pool, discover, label, build.

Run `python make_dataset.py <subcommand> --help` for the options of each step.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Dr3dre/Genius-song-lyrics-cleaned"
SHARD = "data/train-00003-of-00032.parquet"
BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)")
QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})
STOP = {"none", "theme", "themes", "lyrics", "line", "lyric", "song", "songs", "no", "n/a", "na", "nothing"}


# ---------------------------------------------------------------- helpers

def clean_line(raw: str, min_words: int, max_words: int) -> str | None:
    """Strip tags and parentheses, normalise quotes, keep ASCII lines of plain words."""
    text = BRACKETS.sub(" ", raw).translate(QUOTES)
    text = re.sub(r"\s+", " ", text).strip(" -*\"'")
    if not text or not text.isascii():
        return None
    if re.search(r"\d|http|www", text, re.IGNORECASE):
        return None
    if not min_words <= len(text.split()) <= max_words:
        return None
    return text


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_llm(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, device, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        outputs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs


@torch.no_grad()
def yes_no_scores(tok, model, device, texts: list[str], theme: str, batch_size: int) -> np.ndarray:
    """log P(Yes) - log P(No) for 'Is this lyric about <theme>?', one forward pass per line."""
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    out = []
    for i in range(0, len(texts), batch_size):
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content":
                  f"Lyric line: \"{t}\"\n\nIs this lyric line about the theme \"{theme}\"? Answer Yes or No."}],
                tokenize=False, add_generation_prompt=True)
            for t in texts[i:i + batch_size]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out.append((logits[:, yes] - logits[:, no]).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- steps

def cmd_pool(args) -> None:
    rng = np.random.default_rng(args.seed)
    path = hf_hub_download(REPO, SHARD, repo_type="dataset")
    df = pq.read_table(path, columns=["id", "title", "artist", "language", "lyrics"]).to_pandas()
    df = df[(df["language"] == "en") & df["lyrics"].notna()].reset_index(drop=True)
    songs = df.iloc[rng.choice(len(df), size=min(args.songs, len(df)), replace=False)]

    seen: set[str] = set()
    lines = []
    for row in songs.itertuples(index=False):
        for raw in row.lyrics.split("\n"):
            text = clean_line(raw, args.min_words, args.max_words)
            if text is None or text.lower() in seen:
                continue
            seen.add(text.lower())
            lines.append({"text": text, "song_id": int(row.id), "artist": row.artist, "title": row.title})

    song_ids = sorted({r["song_id"] for r in lines})
    draws = np.random.default_rng(args.seed + 1).random(len(song_ids))
    split_of = {sid: ("train" if d < 0.8 else "val" if d < 0.9 else "test") for sid, d in zip(song_ids, draws)}
    rows = [{"id": i, **r, "split": split_of[r["song_id"]]} for i, r in enumerate(lines)]
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "pool.jsonl", rows)
    print(f"pool: {len(rows)} lines from {len(songs)} songs", dict(Counter(r["split"] for r in rows)))


def cmd_discover(args) -> None:
    pool = read_jsonl(args.out / "pool.jsonl")
    train_lines = [r["text"] for r in pool if r["split"] == "train"]
    rng = random.Random(args.seed)
    prompts = []
    for _ in range(args.groups):
        sample = rng.sample(train_lines, args.lines_per_group)
        listing = "\n".join(f"- {line}" for line in sample)
        prompts.append(
            "Here are lines from song lyrics:\n" + listing +
            f"\n\nList {args.names_per_reply} short theme words (one or two words each) that describe what "
            "these lyrics are about. Reply with a comma-separated list only."
        )
    device = pick_device(args.device)
    tok, model = load_llm(args.llm, device)
    replies = generate(tok, model, device, prompts, max_new_tokens=40, batch_size=args.batch_size)

    counts: Counter[str] = Counter()
    for reply in replies:
        for item in re.split(r"[,\n;]", reply):
            name = re.sub(r"^[\s\-\d\.\*]+", "", item).strip().lower().strip(" .\"'")
            if 2 <= len(name) <= 20 and len(name.split()) <= 2 and name not in STOP:
                counts[name] += 1

    eligible = [name for name, n in counts.most_common(40) if n >= args.min_count]
    themes = eligible[: args.n_themes]
    if len(themes) < args.n_themes:
        print(f"warning: only {len(themes)} names reached --min-count {args.min_count}")
    (args.out / "themes.json").write_text(json.dumps({
        "themes": themes,
        "llm": args.llm,
        "groups": args.groups,
        "lines_per_group": args.lines_per_group,
        "candidate_counts": dict(counts.most_common(40)),
    }, indent=2), encoding="utf-8")
    print("themes:", themes)


def cmd_label(args) -> None:
    themes = json.loads((args.out / "themes.json").read_text(encoding="utf-8"))["themes"]
    pool = read_jsonl(args.out / "pool.jsonl")
    texts = [r["text"] for r in pool]
    device = pick_device(args.device)
    tok, model = load_llm(args.llm, device)

    scores = np.stack([yes_no_scores(tok, model, device, texts, t, args.batch_size) for t in themes], axis=1)
    # Per-theme cut-off: the small model's yes/no bias differs by theme, so use percentiles.
    cut = np.quantile(scores, 1 - args.top_share, axis=0)
    margin = scores - cut

    rows = []
    for r, m in zip(pool, margin):
        order = [j for j in np.argsort(-m)[: args.max_labels] if m[j] >= 0]
        rows.append({"id": r["id"], "split": r["split"],
                     "themes": [themes[j] for j in order],
                     "scores": {themes[j]: round(float(scores[r["id"], j]), 3) for j in order}})
    write_jsonl(args.out / "labels.jsonl", rows)
    themed = sum(1 for r in rows if r["themes"])
    print(f"cut-offs: {dict(zip(themes, np.round(cut, 2).tolist()))}")
    print(f"labelled {len(rows)} lines; {themed} have at least one theme ({themed / len(rows):.1%})")
    print("labels per line:", dict(sorted(Counter(len(r["themes"]) for r in rows).items())))


def cmd_build(args) -> None:
    pool = {r["id"]: r for r in read_jsonl(args.out / "pool.jsonl")}
    labels = read_jsonl(args.out / "labels.jsonl")
    themes_info = json.loads((args.out / "themes.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    counts = {}
    for split, target in args.targets.items():
        themed = [l for l in labels if l["split"] == split and l["themes"]]
        if len(themed) < target:
            print(f"warning: only {len(themed)} themed {split} lines available (asked for {target})")
        rng.shuffle(themed)
        chosen = themed[:target]
        counts[split] = len(chosen)
        rows = []
        for l in chosen:
            p = pool[l["id"]]
            rows.append({"id": l["id"], "text": p["text"], "themes": l["themes"],
                         "song_id": p["song_id"], "artist": p["artist"], "title": p["title"], "split": split})
        write_jsonl(args.out / f"{split}.jsonl", rows)
        print(f"{split}: {len(chosen)} themed lines (target {target}); themes:",
              dict(Counter(t for r in rows for t in r["themes"]).most_common()))
    meta = {
        "name": "theme_lyrics_llm_v1",
        "source": REPO,
        "source_file": SHARD,
        "source_note": "Lyrics belong to their writers and publishers. Research use only; do not redistribute.",
        "label_method": f"LLM ({themes_info.get('llm', 'Qwen/Qwen2.5-0.5B-Instruct')}) proposes themes; each line scored per theme, "
                        f"top {args.top_share:.0%} per theme, up to {args.max_labels} themes",
        "themes": themes_info["themes"],
        "split_counts": counts,
        "seed": args.seed,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out}/{{train,val,test}}.jsonl and meta.json")


# ---------------------------------------------------------------- CLI

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=Path("data"), help="output folder (default: data)")
    p.add_argument("--seed", type=int, default=13, help="random seed (default: 13)")


def add_llm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face instruct model id")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="default: auto")
    p.add_argument("--batch-size", type=int, default=128, help="prompts per forward pass (default: 128)")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pool", help="sample songs, clean lines, split by song")
    add_common(p)
    p.add_argument("--songs", type=int, default=8000, help="English songs to sample (default: 8000)")
    p.add_argument("--min-words", type=int, default=5, help="min words per line (default: 5)")
    p.add_argument("--max-words", type=int, default=14, help="max words per line (default: 14)")
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("discover", help="LLM proposes the theme set")
    add_common(p)
    add_llm(p)
    p.add_argument("--groups", type=int, default=400, help="prompts sampled from train lines (default: 400)")
    p.add_argument("--lines-per-group", type=int, default=25, help="lines shown per prompt (default: 25)")
    p.add_argument("--names-per-reply", type=int, default=5, help="names requested per prompt (default: 5)")
    p.add_argument("--n-themes", type=int, default=10, help="number of themes to keep (default: 10)")
    p.add_argument("--min-count", type=int, default=5, help="min times a name must be proposed (default: 5)")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("label", help="LLM scores every line against every theme")
    add_common(p)
    add_llm(p)
    p.add_argument("--top-share", type=float, default=0.10,
                   help="a line gets theme t if it is in the top share for t (default: 0.10)")
    p.add_argument("--max-labels", type=int, default=3, help="max themes per line (default: 3)")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("build", help="write train/val/test JSONL with target sizes")
    add_common(p)
    p.add_argument("--train", type=int, default=10000, help="train lines (default: 10000)")
    p.add_argument("--val", type=int, default=1000, help="val lines (default: 1000)")
    p.add_argument("--test", type=int, default=1000, help="test lines (default: 1000)")
    p.add_argument("--top-share", type=float, default=0.10, help="recorded in meta.json (default: 0.10)")
    p.add_argument("--max-labels", type=int, default=3, help="recorded in meta.json (default: 3)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("all", help="run pool, discover, label, build")
    add_common(p)
    add_llm(p)
    p.add_argument("--songs", type=int, default=8000)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=14)
    p.add_argument("--groups", type=int, default=400)
    p.add_argument("--lines-per-group", type=int, default=25)
    p.add_argument("--names-per-reply", type=int, default=5)
    p.add_argument("--n-themes", type=int, default=10)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--top-share", type=float, default=0.10)
    p.add_argument("--max-labels", type=int, default=3)
    p.add_argument("--train", type=int, default=10000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.set_defaults(func=None)
    return root


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "all":
        cmd_pool(args)
        cmd_discover(args)
        cmd_label(args)
        args.targets = {"train": args.train, "val": args.val, "test": args.test}
        cmd_build(args)
        return
    if args.command == "build":
        args.targets = {"train": args.train, "val": args.val, "test": args.test}
    args.func(args)


if __name__ == "__main__":
    main()