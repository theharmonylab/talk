#!/bin/bash
set -e

# ============================================
# WhisNemo environment setup for talk R package
# Creates an exact replica of the tested working env
# ============================================

ENV_NAME="${1:-talk_diarize}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating conda env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install FFmpeg 4.x (needed for av/PyAV)
echo "Installing FFmpeg..."
conda install -c conda-forge 'ffmpeg>=4.2,<5.0' pkg-config -y

# Install build tools first
echo "Installing build tools..."
pip install --no-cache-dir setuptools wheel Cython==3.0.11 pybind11

# Install torch (cu118) — must come before other deps
echo "Installing PyTorch..."
pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Pre-install packages that need build tools available
echo "Installing packages with native dependencies..."
pip install --no-cache-dir --no-build-isolation youtokentome==1.0.6
pip install --no-cache-dir av==11.0.0

# Install HuggingFace stack pinned (order matters)
pip install --no-cache-dir --no-deps \
    huggingface-hub==0.23.2 \
    tokenizers==0.15.2 \
    safetensors==0.6.2 \
    transformers==4.39.3

# Install everything else from the frozen requirements
echo "Installing remaining dependencies..."
pip install --no-cache-dir --no-deps --no-build-isolation \
    -r "${SCRIPT_DIR}/whisnemo_pip_requirements.txt"

echo ""
echo "============================================="
echo "Environment '$ENV_NAME' ready!"
echo "Activate with: conda activate $ENV_NAME"
echo "Test with: python -c 'from whisnemo.core.diarize import run_diarize; print(\"OK\")'"
echo "============================================="
