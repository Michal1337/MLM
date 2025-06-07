#!/bin/bash
#SBATCH --job-name=tmp       # Job name
#SBATCH --output=error1.log           # Standard output log
#SBATCH --error=error1.log             # Standard error log
#SBATCH --time=24:00:00                # Time limit
#SBATCH --mem=32G                      # Memory request
#SBATCH --cpus-per-task=2             # Allocate CPU cores
#SBATCH --partition=short              # Specify the short queue

# Ensure pyenv is initialized
export HOME=/mnt/evafs/faculty/home/mgromadzki
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script
python experiments_crypto_baseline.py
