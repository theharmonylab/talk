# Python backend for talk

This folder contains the Python-side helper scripts used by the `talk` R package.

## Purpose

The main transcription and diarization logic now lives in the standalone `whisnemo` Python package.


## Files

- `transcribe.py`  
  Wrapper around WhisNemo transcription helpers.

- `diarize.py`  
  Wrapper around WhisNemo diarization pipeline.

- `whisnemo_pip_requirements.txt`  
  Minimal install target for the WhisNemo backend.

- `requirements.txt`  
  General Python dependencies used by other `talk` backend tasks.

- `requirements_diarization.txt`  
  Legacy diarization requirements file. This may be deprecated in favor of installing the standalone `whisnemo` package directly.

- `huggingface_Interface4.py`  
  Hugging Face helper utilities used by other parts of the package.

## Current backend design

The preferred backend path is:

1. Install GPU-compatible PyTorch for the target CUDA version
2. Install `whisnemo[diarize]`
3. Call the wrapper scripts in this folder from R

Example install flow on Linux with CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```
