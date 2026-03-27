# LPAC_project

Repository for LPAC classification experiments.

## Initial Objective
Simple and reliable baseline with ResNet18 on a dataset already prepared in .pt format.

## Dataset Assumptions
The repository assumes that data preparation has already been performed offline and that the dataset is available as a .pt file.

## Setup
uv sync

## Execution
Local smoke test:
uv run pytest

Train:
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml

Note: 'uv sync' is the correct command to synchronize the project environment, and 'uv run' is the right wrapper to execute commands in the project context.

## Initial Commit
git status
git add .
git commit -m "Initialize project structure and uv-based environment"