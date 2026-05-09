#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate the project's uv venv (./.venv) if it exists; fall back to conda.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
elif [ -n "${CONDA_SHELL}" ]; then
  # shellcheck source=${HOME}/.bashrc disable=SC1091
  source "${CONDA_SHELL}"
  if [ -z "${CONDA_PREFIX}" ]; then
      conda activate discdiff
   elif [[ "${CONDA_PREFIX}" != *"/discdiff" ]]; then
    conda deactivate
    conda activate discdiff
  fi
else
  echo "WARNING: neither .venv nor CONDA_SHELL found; running with system Python."
fi

# Setup HF cache (only set if not already exported by caller).
if [ -z "${HF_HOME}" ]; then
  export HF_HOME="${PWD}/.hf_cache"
fi
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
