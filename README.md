# LPAC_project

LPAC_project is a config-driven PyTorch repository for binary medical image classification.
Current baseline model: ResNet18.

## Setup

```bash
uv sync
```

## Quick Validation

```bash
uv run pytest
```

## Dataset Format
Training expects a packed dataset file (.pt) with this structure:

```python
{
		"images": Tensor [N, 3, H, W],
		"labels": Tensor [N],
		"patients": Tensor [N],
		"paths": list[str],
		"image_size": int,
}
```

Assumptions:
- `images` are already prepared offline
- `labels` and `patients` are `torch.long`
- each patient has a consistent label across all its images

## Run Training

Standard run:

```bash
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml
```

Smoke run (fast sanity check):

```bash
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml --smoke --device cuda
```

In smoke mode, each split is capped to at most 100 samples and training epochs are forced to 1.

## Split Logic

Splits are patient-level (no patient leakage across train/val/test).

```toml
[split]
method = "naive" # or "heuristic_balanced"
seed = 53100
train_fraction = 0.70
val_fraction = 0.15
test_fraction = 0.15
stratify = true
save_artifact = true
```

Behavior by method:
- `naive`: random patient split by fractions
- `heuristic_balanced`: tries to balance sample fractions and label distribution across splits

Note: `stratify` only affects the `heuristic_balanced` method.

## Training Logic

- criterion: cross-entropy (optionally class-weighted)
- model selection metric: validation balanced accuracy
- if `model.pretrained=true`, head warmup runs before full fine-tuning
- scheduler is configured under `[train.scheduler]`

```toml
[train.scheduler]
name = "none" # supported: none, plateau, cosine
mode = "max"
factor = 0.5
patience = 2
min_lr = 1e-7
t_max = 5
eta_min = 1e-7
```

## Hyperparameter Selection

Hyperparameter selection is grid-based and config-driven.

```toml
[hyperparameters_selection]
perform = true
cross_validation = false # placeholder, not implemented yet
k_folds = 3 # placeholder, used only when CV is implemented

[hyperparameters_selection.values]
lr = [5e-7, 1e-6, 3e-6]
weight_decay = [1e-4, 1e-3]
```

Current implemented behavior:
- generates all combinations from `[hyperparameters_selection.values]` via Cartesian product
- runs one training trial per combination on the same train/val split
- selects the best trial by validation balanced accuracy
- evaluates test metrics once, using the best trial checkpoint

Supported tunable keys (currently):
- `lr`
- `batch_size`
- `weight_decay`
- `head_warmup_epochs`

Determinism note:
- trial seed is reset from `split.seed` at each trial start for comparable randomness across trials

## Output Artifacts

Each run creates `outputs/<run_name>_<timestamp>/`.

Always produced:
- `config.toml` (copied input config)
- `split.json`
- `split_summary.json`

Single-trial mode (`perform=false`):
- `metrics.csv`
- `best_model.pt`
- `test_metrics.json`

Hyperparameter-search mode (`perform=true`):
- `hyperparameter_search_results.json`
- `best_hyperparameters.json`
- `best_model.pt` (copied from best trial)
- `test_metrics.json` (computed once from best trial model)
- `trial_XXX/` subfolders, each containing:
	- `hyperparameters.json`
	- `resolved_config.json`
	- `metrics.csv`
	- `best_model.pt`

Note: `last_model.pt` is currently not saved by default.

## Repository Layout

```text
configs/
scripts/
src/lpac_project/
	data.py
	train.py
	engine.py
	evaluate.py
	metrics.py
	models/
tests/
```