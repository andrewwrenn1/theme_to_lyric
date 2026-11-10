# Theme to Lyric — a machine learning lyric-idea generator

**A working, individual machine learning project with a browser GUI, a trained PyTorch model, reproducible experiments, and a seven-page PDF report.**

The user selects a topic such as **love**, **freedom**, or **redemption**. The model then generates a short, song-like line conditioned on that theme. The goal is to study whether a model can learn *what* to write about, not to impersonate real songwriters or copy existing lyrics. This is an educational brainstorming prototype for students, hobbyist writers, and ML learners.

**Included and ready to demonstrate:** a compact theme-conditioned GRU language model trained on the supplied lyric dataset. Its CPU checkpoint is in `checkpoints/offline_best.pt`. It works without an API key or a live internet connection after installing PyTorch. The existing FLAN-T5 training and MiniLM evaluation scripts have also been preserved as a **separate, more capable experimental pathway**; they require downloading pretrained models and are not claimed as completed experiments in the report.

![Browser interface preview](Documents/demo_interface_preview.png)

*The picture is a browser-rendered layout preview. It uses simulated status/theme metadata; the actual local HTTP generation endpoint was tested separately.*

## 1. Installation on a clean computer

**Requirements:** Python **3.11** recommended, Windows 10/11, recent macOS, or Linux; about 1 GB of disk space available for the GUI PyTorch dependency and temporary files. CPU is enough. No GPU, Hugging Face account, API key, or dataset download is needed to try the included model.

1. Download this repository using GitHub's **Code → Download ZIP** and extract it (or run `git clone <your-repository-url>`). Open a terminal inside the directory containing this `README.md`.
2. Create and activate a virtual environment:

   **macOS / Linux:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell:**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

3. Install the minimal demo dependency:
   ```bash
   python -m pip install -r requirements-demo.txt
   ```
4. Run the graphical demo:
   ```bash
   python demo.py
   ```
5. Open **http://127.0.0.1:8501** in a web browser. Select a theme, optionally adjust creativity and line count, and click **Generate lyric ideas**. The trained model returns generated text. Stop the server with `Ctrl+C`.

**If port 8501 is occupied:** `python demo.py --port 8502` and visit `http://127.0.0.1:8502`.

**Quick tests:** `python -m unittest discover -s tests -v` checks the model, vocabulary, browser page, `/api/info`, and `/api/generate`. Seven tests passed in the development environment.

> **Install-validation note:** The implementation and demo endpoint were tested here with Python 3.13 and PyTorch 2.10. The reproducible install file pins PyTorch 2.13.0 for Python 3.11, but a complete network-based fresh install could not be performed in this environment. If a particular machine needs a platform-specific PyTorch wheel, use [the official installation selector](https://pytorch.org/get-started/locally/) and then retry the command. For full-model experiment packages, see below.

## 2. What the GUI does

- **Theme:** choose one of ten training themes: love, loss, freedom, redemption, sadness, loneliness, betrayal, death, pain, sacrifice.
- **Model:** default is the **included theme-trained GRU**. The optional **FLAN-T5 fine-tuned checkpoint** option becomes available only if you have trained and saved `runs/flan-t5-small/best` locally.
- **Creativity:** sampling temperature 0.5 to 1.5 (higher means more varied and potentially less grammatical output).
- **Output count:** one to five lines; **Copy lines** puts the generated text on the clipboard when supported.

The server is local, built with Python's standard `http.server`. The user does **not** need to write code. HTML/CSS/JavaScript are in `web/index.html`, server code in `demo.py`, learned model code in `offline_model.py`.

The included tiny model can create awkward or generic phrases. Outputs are not guaranteed original or safe to publish. If local training files exist, the demo blocks exact matches to training lines; this is not a comprehensive originality check.

## 3. Data: origins, placement, and leakage controls

**Original source:** [Dr3dre/Genius-song-lyrics-cleaned on Hugging Face](https://huggingface.co/datasets/Dr3dre/Genius-song-lyrics-cleaned), specifically `data/train-00003-of-00032.parquet`. The data processing and theme labeling script is `make_dataset.py`.

The supplied dataset uses a locally downloaded `Qwen/Qwen2.5-0.5B-Instruct` labeler. It extracts short English lines, removes section annotations and exact duplicates by normalized text, splits **by song** (not by line), discovers 10 themes from training examples, and marks up to three themes per line using a log-probability score and per-theme percentile threshold. The thematic labels are **machine-generated weak labels**, not ground truth.

Provided research data that was checked:

| Split | Lines | Distinct songs | Theme assignments |
|---|---:|---:|---:|
| Training | 10,000 | 4,585 | 19,318 |
| Validation | 1,000 | 527 | 1,893 |
| Test | 1,000 | 515 | 1,935 |

A programmatic audit found **0 song IDs shared between any two splits**. Full counts are stored in `experiments/data_audit.json`. Dataset provenance and label rules are preserved in `data/meta.json` and `data/themes.json`.

### Use the dataset files you already have

The originally supplied lyric text files are **not bundled in this public-ready ZIP** because lyrics belong to their writers and publishers. If you have those files locally, copy them into `data/` so the paths are:

```text
data/train.jsonl
data/val.jsonl
data/test.jsonl
data/pool.jsonl       # optional for retraining; used to regenerate the dataset
data/labels.jsonl     # optional for retraining
data/meta.json
data/themes.json
```

You already provided `train.jsonl`, `val.jsonl`, `test.jsonl`, `pool.jsonl`, `labels.jsonl`, `meta.json`, and `themes.json` with this project. The three train/validation/test files are required to reproduce the learning experiments; **they are not required to launch the included GUI**.

**Alternatively, regenerate the data from the original publicly accessible source:**

```bash
python -m pip install -r requirements.txt
python make_dataset.py all --out data --seed 13 --songs 8000 --groups 400 --train 10000 --val 1000 --test 1000 --batch-size 16
```

This may download gigabytes and take much longer on CPU; the generator/labeler downloads model weights and the script may produce fewer lines if too few candidates meet the label criteria. `make_dataset.py` has independent `pool`, `discover`, `label`, and `build` subcommands and `--help` for each. A rebuilt dataset is not guaranteed byte-identical to the original supplied snapshot if upstream models or remote data change.

**Important limitation:** the original labeler computes percentile thresholds over the full candidate pool, including validation/test lines; this is a dataset-construction limitation, and the report recommends train-only calibration and a human-reviewed independent test set. The source dataset card's compilation license should **not** be assumed to transfer every lyric's copyright.

## 4. Retrain and evaluate the INCLUDED small model

This is the actual model used in the GUI and final report. Install `requirements-demo.txt` for training; no additional library is needed. Commands below assume the provided `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl` have been copied to `data/` as above.

**Check dataset splits:**
```bash
python offline_model.py data-check --data-dir data --out experiments/data_audit.json
```

**Train the theme-conditioned model (exact settings from the report):**
```bash
python offline_model.py train --data-dir data --epochs 5 --max-train-lines 4000 --max-val-lines 1000 --vocab-size 2400 --embed-dim 64 --hidden-dim 96 --batch-size 64 --lr 0.002 --seed 13 --out checkpoints/offline_best.pt
```

**Train a controlled unconditional comparison (only the theme token changes):**
```bash
python offline_model.py train --data-dir data --epochs 5 --max-train-lines 4000 --max-val-lines 1000 --vocab-size 2400 --embed-dim 64 --hidden-dim 96 --batch-size 64 --lr 0.002 --seed 13 --unconditional --out experiments/unconditional_best.pt
```

**Evaluate on all three splits, including the held-out test:**
```bash
python offline_model.py evaluate --data-dir data --checkpoint checkpoints/offline_best.pt --out experiments/offline_metrics.json
python offline_model.py evaluate --data-dir data --checkpoint experiments/unconditional_best.pt --out experiments/unconditional_metrics.json
```

**Generate from the checkpoint without retraining:**
```bash
python offline_model.py generate --checkpoint checkpoints/offline_best.pt --theme love --count 3 --seed 13
```

**Extra theme-following experiment** (needs full `requirements.txt`, which includes scikit-learn):
```bash
python experiments/theme_probe.py --data-dir data --count-per-theme 20 --out experiments/theme_probe.json
```

The training script selects the checkpoint with **lowest validation cross-entropy**. Cross-entropy is measured with teacher forcing, padding ignored. The test set is evaluated after training rather than used for model selection. Checkpoint `.pt` contains weights, word vocabulary, configuration, theme list, and best epoch; a matching `.training.json` contains epoch losses, hyperparameters and elapsed time.

## 5. Recorded experiments and measured results

Data: 4,000 training lines were randomly selected with seed 13 from the available 10,000 training lines; these expanded to 7,718 theme-to-line training pairs. All 1,000 validation and 1,000 test lines were used, and the same 2,400-token vocabulary and architecture were used for both controlled models. Training was done on CPU. The small-model weights are included.

| Actual measured metric | Model ignoring theme | Model using theme |
|---|---:|---:|
| Best validation perplexity | 98.80 | **96.68** |
| Test perplexity | 88.90 | **85.59** |
| Theme hit@1 via weak-label classifier | 7.5% | **22.5%** |
| Theme mean reciprocal rank | 0.285 | **0.425** |

The theme judge is an independently trained TF-IDF + logistic model **but uses the same Qwen-produced weak theme labels**. Its held-out top-prediction match to at least one labeled test theme was 67%, and it is **not** a human-verified judge. The improvement indicates model sensitivity to its theme instruction, not validated songwriting quality. Chance hit@1 for 10 themes is 10%. See `Documents/final_report.pdf` for methodology, graphics and limitations; see `experiments/*.json` for numeric outputs.

![Controlled experimental results](Documents/model_comparison.png)

## 6. Optional: the original FLAN-T5 experiments

The initial project provided two scripts which are included here unchanged:

- `make_dataset.py`: downloads/parses the source shard, discovers themes with Qwen2.5, labels lines, writes the dataset.
- `theme_to_lyric.py`: trains a FLAN-T5 sequence-to-sequence transformer and evaluates base vs fine-tuned models with perplexity and MiniLM theme-similarity metrics.

To install the complete dependency set instead of the minimal GUI set:

```bash
python -m pip install -r requirements.txt
# Alternatively, if conda is installed:
conda env create -f environment.yml
conda activate theme-to-lyric
```

**Fine-tune T5-small:**
```bash
python theme_to_lyric.py train --model google/flan-t5-small --data-dir data --epochs 5 --lr 3e-4 --batch-size 8 --eval-batch-size 16 --device cpu
```

For a computer with a CUDA GPU, omit `--device cpu` and raise the batch size as memory allows. The model is downloaded on first use. Fine-tuned weights are saved to `runs/flan-t5-small/best/`. Each epoch validates; the best validation-loss checkpoint is selected. A successful training run saves `train_log.json`, per-split perplexity in `metrics.json`, samples, and graphics. The report's word-level GRU and this subword T5 model do **not** have directly comparable perplexity values.

**Sample with the fine-tuned transformer:**
```bash
python theme_to_lyric.py train --model google/flan-t5-small --generate-only --themes love freedom --device cpu
```

**Compare base vs fine-tuned transformer on held-out data:**
```bash
python theme_to_lyric.py evaluate --models google/flan-t5-small runs/flan-t5-small/best --data-dir data --device cpu --samples-per-theme 10
```

This outputs `evaluation/comparison.json`, `comparison.md`, and `comparison.png`. The theme metric here uses `sentence-transformers/all-MiniLM-L6-v2`. The evaluation code uses different prompts for base and fine-tuned checkpoints by design.

**Transparency:** No FLAN-T5 training or MiniLM evaluation was executed while assembling this ZIP because the execution sandbox could not download pretrained weights. **Do not represent the measured GRU scores as FLAN-T5 scores.** These scripts were inspected and syntax-checked, but the full downloadable-model path still needs a network-equipped machine to run.

## 7. Repository structure

```text
ThemeToLyric_Submission/
├── README.md
├── demo.py                         # Tested local web/JSON server
├── web/index.html                  # GUI (HTML, CSS, JavaScript)
├── offline_model.py                # Runnable trained GRU; train/evaluate/generate/data-check
├── theme_to_lyric.py               # Original FLAN-T5 train/evaluate commands
├── make_dataset.py                 # Original Qwen weak-label data pipeline
├── requirements-demo.txt           # Only PyTorch, for easy GUI demo
├── requirements.txt                # Full experiments and dataset pipeline
├── environment.yml                 # Conda alternative
├── checkpoints/
│   ├── offline_best.pt             # Included trained model for instant demo
│   └── offline_best.training.json
├── data/
│   ├── meta.json                   # Labels, provenance, counts
│   └── themes.json                 # Theme discovery summary
│   # train.jsonl/val.jsonl/test.jsonl: users restore locally; excluded from Git
├── experiments/
│   ├── data_audit.json             # Computed split stats and song-overlap check
│   ├── offline_metrics.json        # Evaluated included GRU
│   ├── unconditional_best.pt       # Trained control weights
│   ├── unconditional_best.training.json
│   ├── unconditional_metrics.json
│   ├── theme_probe.py              # Optional TF-IDF theme-following judge
│   └── theme_probe.json            # Measured judge results
├── Documents/
│   ├── final_report.pdf            # FINAL REPORT for submission
│   ├── demo_interface_preview.png  # Browser-rendered UI layout preview
│   ├── pipeline_diagram.png
│   ├── model_comparison.png
│   ├── theme_distribution.png
│   └── validation_curves.png
└── tests/test_project.py           # Seven passing automated tests
```

## 8. Submit and cite

Turn in these three deliverables to the instructor:

1. **PDF:** `Documents/final_report.pdf`.
2. **GitHub URL:** create a GitHub repository for this folder, push the included non-lyric files, and submit its URL. **No live GitHub link has been created yet** by this package.
3. **Working GUI:** a peer clones the repository, installs `requirements-demo.txt`, launches `python demo.py`, opens localhost, selects a theme and generates output. The included checkpoint makes this possible **without** retraining or downloading the original lyric corpus.

**Research honesty:** This is a genuinely trained small model baseline, not a claim that the full transformer has already achieved the reported results. The best next steps are to collect human judgments, improve originality screening, rerun weak-label calibration using training data alone, and compare a real fine-tuned T5 checkpoint against this baseline.

**External resources used:** [Source dataset](https://huggingface.co/datasets/Dr3dre/Genius-song-lyrics-cleaned) · [FLAN-T5-small model](https://huggingface.co/google/flan-t5-small) · [Qwen labeler](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [MiniLM embedder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).