#!/bin/bash
#SBATCH --partition=prod10
#SBATCH --job-name=lpac_resnet18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

uv sync --frozen
uv run python -m lpac_project.train --config configs/resnet18_baseline.toml --device cuda