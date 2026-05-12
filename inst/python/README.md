# Python backend for talk

This folder contains the Python-side helper scripts used by the `talk` R package.

## Files

- `transcribe.py` — Wrapper around WhisNemo transcription helpers.
- `diarize.py` — Wrapper around WhisNemo diarization pipeline.

## Installing the WhisNemo backend

The transcription and diarization logic lives in the standalone `whisnemo`
Python package. Install requires two steps: PyTorch first (platform-specific),
then whisnemo with a constraints file to prevent dependency conflicts.

A constraints file is required because NeMo's pip resolver will otherwise
upgrade torch and numpy to incompatible versions.

### Linux (CUDA 12.8)

```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

### macOS (Apple Silicon)

```bash
pip install torch==2.11.0 torchaudio==2.11.0
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

### Windows

```powershell
pip install torch==2.11.0 torchaudio==2.11.0
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

For GPU on Windows with CUDA 12.8, add `--index-url https://download.pytorch.org/whl/cu128`
to the torch install line. For older NVIDIA drivers (CUDA < 12.8), use
`torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`.

### Verify

```bash
python -c "from whisnemo.core.diarize import run_diarize; print('OK')"
```

## Usage from R

These wrapper scripts are called by the `talk` R package via `reticulate`.
See the main `talk` package documentation for R-side usage.
