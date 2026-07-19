# DoR: Reconstruction-Calibrated GRPO for Tokenized Video World Models

DoR is a research codebase for verifiable-reward post-training of tokenized video
world models. It uses an RT-1 robot-manipulation world model, a frozen visual
tokenizer, full-reference video rewards, and lightweight GRPO-style policy updates.

The current paper target is **RC-GRPO: Reconstruction-Calibrated Temporal Credit
Assignment for Tokenized Video World Models**.

## Research question

Predictions are decoded before a reward is compared with the real future frame. DoR
studies two coupled questions:

1. Is the encoder-decoder reconstruction of ground truth the right reachable verifier
   target, or can a legal nearby token code match the reward metric more faithfully?
2. In multi-step rollout, should a sequence-level verifier score be broadcast to every
   predicted frame block, or assigned by a temporally aligned reward-to-go?

The repository uses falsifiable gates for both questions rather than treating reward
engineering alone as a headline contribution.

## Current evidence status

| Component | Status | Claim boundary |
|---|---|---|
| Encoder reconstruction | rejected as a metric projection | A 64-window legal-FSQ audit found a better reachable target in every window. |
| MRRT | rejected at the training endpoint | It improves the fixed-budget target metric, but does not beat encoder RC or its matched-random control in the paired three-seed downstream pilot. |
| Temporal Return | provisional | Prefer multi-step results under RLVR-World/VERL `low_var_kl`; older linear-KL runs may be reported with an explicit protocol tag. |
| Rank-Guard, GSPO, REAL-style VPO, spatial pooling | rejected or non-beneficial | Excluded from the active method. |

The canonical claim-evidence ledger is [docs/aaai2027/story.md](docs/aaai2027/story.md).
README numbers are orientation only; paper tables should cite `experiments.md` and run JSON
artifacts. Reporting defaults and allowed flexibility are in [docs/aaai2027/README.md](docs/aaai2027/README.md).

## Layout

```text
src/dor/                 Training, model/tokenizer adapters, rewards, metrics,
                         reachable-target refinement, and temporal credit utilities.
scripts/                 Reproducible training, caching, auditing, evaluation, and analysis.
tests/                   Unit and analysis-contract tests.
docs/aaai2027/           Canonical paper story, method, experiment ledger, reviewer audit, runbook.
docs/AuthorKit27/        AAAI anonymous-submission LaTeX source and cited figures.
data/                    Ignored server-side RT-1 data mount.
checkpoints/             Ignored server-side model and tokenizer checkpoints.
outputs/                 Ignored generated runs, caches, and evaluation artifacts.
third_party/             Ignored upstream RLVR-World checkout.
```

## Environment

The intended runtime is a CUDA Linux training server. It requires Python 3.10+, PyTorch
with CUDA, `transformers`, `lpips`, `piqa`, `scipy`, and the dependencies required by
RLVR-World/iVideoGPT.

```bash
git clone https://github.com/MaosenWilson/DoR.git
cd DoR
pip install -e .
export VOTE2WORLD_ROOT=/path/to/vote2world
```

`src/dor/constants.py` documents expected checkpoint, tokenizer, action-range, and data
paths. Large assets, caches, and run outputs are intentionally not tracked.

## Active experiments

The sole runbook is [docs/aaai2027/RUN.md](docs/aaai2027/RUN.md):

1. audit and cache reachable targets when needed for C1 diagnostics;
2. compare raw / RC (and historical variants only if still informative) under shared GRPO settings;
3. multi-step `raw/RC × sequence/temporal-return` factorial (prefer `low_var_kl`);
4. temporal-correspondence and horizon controls as compute allows.

Commands write JSON artifacts with protocol and metrics. Analysis scripts should flag missing
seeds or protocol mismatches; paper tables may still use a disclosed subset when documented.

## Paper workflow

The LaTeX source is [docs/AuthorKit27/AnonymousSubmission2027.tex](docs/AuthorKit27/AnonymousSubmission2027.tex).
Update `story.md` and `experiments.md` when claims or numbers change; sync Method/LaTeX when
the presentation set is frozen. Prefer evidence-backed claims over hard gate language.
