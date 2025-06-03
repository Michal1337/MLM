#!/bin/bash
#SBATCH --job-name=tmp       # Job name
#SBATCH --output=error1.log           # Standard output log
#SBATCH --error=error1.log             # Standard error log
#SBATCH --time=24:00:00                # Time limit
#SBATCH --mem=32G                      # Memory request
#SBATCH --cpus-per-task=2             # Allocate CPU cores
#SBATCH --partition=short              # Specify the short queue

# Ensure pyenv is initialized
export PATH="$HOME/.pyenv/bin:$PATH"
export LD_LIBRARY_PATH=$HOME/local/libffi/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/libffi/lib/pkgconfig:$PKG_CONFIG_PATH
export C_INCLUDE_PATH=$HOME/local/libffi/include:$C_INCLUDE_PATH
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script
python experiments_baseline.py
