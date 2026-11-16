# Project Proposal: Theme-to-Lyric Generation

## Research question
Can a relatively small language model learn to draft a short lyric-style line that reflects a user-selected theme? How much does conditioning on the requested theme change performance compared with an identical model that is not told the theme?

## Objective and motivation
Build a reproducible educational application that generates a one-line songwriting idea from a small set of themes. A writer could use it as a brainstorming aid, and students can inspect how labeled training data and conditioned neural sequence generation work.

## Data source
The original project uses English lyric lines from the publicly accessible `Dr3dre/Genius-song-lyrics-cleaned` Hugging Face dataset, specifically one Parquet shard. A small Qwen instruct model proposes and weakly labels ten recurring themes. Labels are not human-verified. Split by song ID into train, validation and test; avoid sharing source lyrics in the public repository.

## Proposed approach
1. Audit the provided dataset (size, theme balance, split leakage, known labeling limitations).
2. Implement a word-level GRU baseline that consumes a theme token and generates the next word autoregressively.
3. Train an otherwise identical model with a generic token instead of themes as a controlled comparison.
4. Measure validation cross-entropy, held-out token perplexity, and a weak theme-following score from a separate text classifier; note that weak labels limit the validity of this proxy.
5. Preserve and document the existing FLAN-T5 fine-tuning code as a future larger-model comparison.
6. Provide a web GUI that runs on CPU from an included small model checkpoint.

## Hypothesis and evaluation
Theme-conditioned generation should improve theme recognition and may improve held-out likelihood versus an unconditional model. Choose checkpoints only using validation loss. Reserve test lines for final evaluation, and make no claims about human-rated songwriting quality without collecting human ratings. Document licensing risks and possible memorization.

## Anticipated difficulties
Limited GPU/CPU resources, overlapping themes, noisy annotations from a small labeler, copyright restrictions for source lyrics, and repetitive or ungrammatical outputs from small generative models.

## Submission
A repository with source code, exact commands, pinned requirements, trained demo checkpoint, reproducible metrics, tests, and diagrams; a runnable browser GUI; and a PDF final report. See `final_report.pdf` for recorded results and limitations.
