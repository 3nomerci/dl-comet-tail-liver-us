# LPAC_project

LPAC_project is a config-driven PyTorch repository for binary medical image classification experiments.
The current baseline is ResNet18.

## Dataset Assumptions
Data preparation is done offline. Training code consumes a packed dataset file, for example data/dataset.pt.

Expected structure:

```python
{
		"images": Tensor [N, 3, H, W],
		"labels": Tensor [N],
		"patients": Tensor [N],
		"paths": list[str],
		"image_size": int,
}
```

Current packed baseline assumptions:
- images are already standardized to 3 x H x W 
- labels and patients are torch.long
- patient IDs are integer-valued
- offline dataset preparation already applied aspect-ratio resize, center padding, and min-max normalization

## Setup

```bash
uv sync
```

## Quick Validation

```bash
uv run pytest
```

## Training

```bash
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml
```

Smoke mode:

```bash
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml --smoke
```

## Split Logic
Splits are patient-level and deterministic under fixed seed.

Implemented behavior:
- patient-level non-leakage
- optional label stratification (binary labels)
- sample-aware assignment that tries to match train/val/test sample fractions

Config keys in configs/resnet18_baseline.toml:

```toml
[split]
method = "heuristic_balanced" # or "naive"
seed = 42
train_fraction = 0.70
val_fraction = 0.15
test_fraction = 0.15
stratify = true
save_artifact = true
```
method:
- heuristic_balanced: try to balance sample level class stratification, patient fraction, sample fraction to match the specified fractions.
- naive: give specified fractions of patients to each split

Hint: running in smoke mode will generate the split report (if active), which can be inspected before training.

## Model Selection And Scheduler
Best checkpoint selection is based on validation balanced accuracy. The best checkpoint is used to produce final test metrics.

When model.pretrained is true, training runs a head-only warmup phase before full fine-tuning.
Warmup can be controlled from the train section.

Scheduler is configured from the train section:

```toml
[train]
epochs = 5
lr = 1e-5
head_warmup_epochs = 1
head_warmup_lr = 1e-5
head_warmup_weight_decay = 1e-4

[train.scheduler]
name = "none" # supported: none, plateau, cosine
mode = "max" # used by plateau, monitor is val balanced accuracy
factor = 0.5
patience = 2
min_lr = 1e-7
t_max = 5 # used by cosine
eta_min = 1e-7
```

## Run Artifacts
Each training run creates an output folder under outputs/ with:
- config.toml: copied run config
- metrics.csv: epoch-level train/val metrics
- last_model.pt: last checkpoint
- best_model.pt: best checkpoint by validation balanced accuracy
- test_metrics.json: final test metrics from best checkpoint
- split.json: split indices and patient IDs per split
- split_summary.json: split audit summary (counts, fractions, class counts, metadata)

## Repository Structure

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