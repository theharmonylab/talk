# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`talk` is an R package for analysing audio recordings of speech (transcription, speaker diarisation, embeddings) with HuggingFace/NeMo models. R is a thin front end; all heavy lifting happens in Python via reticulate, inside a single conda environment (`talkrpp_condaenv`) that `talkrpp_install()` creates and `talkrpp_initialize()` binds.

## Commands

```bash
# Load the package for development
Rscript -e 'devtools::load_all(".")'

# Fast tests (no Python needed for most assertions)
NOT_CRAN=true Rscript -e 'suppressMessages(devtools::load_all(".")); testthat::test_file("tests/testthat/test_0_helpers.R")'

# Full test suite — ORDER MATTERS and it is slow:
# test_1_install.R creates a dedicated conda env "talk_test_env" (multi-GB download),
# test_2..5 use it, test_6_uninstall.R removes it. Files skip if the env is absent.
NOT_CRAN=true Rscript -e 'devtools::test()'

# R CMD check
Rscript -e 'devtools::check()'

# Python backend syntax check (no R needed)
python3 -m py_compile inst/python/*.py

# pkgdown site (installation.md is the source of the website's installation page)
Rscript -e 'pkgdown::build_site()'
```

To exercise real functionality outside the test suite, initialize the user's existing environment first:

```r
talkrpp_initialize(condaenv = "talkrpp_condaenv", prompt = FALSE, save_profile = FALSE, refresh_settings = TRUE)
wav <- system.file("extdata", "test_short.wav", package = "talk")  # says " Hello."
```

`inst/extdata/test_diarise.wav` is a two-speaker file for diarisation tests.

## Critical conventions

- **Do NOT run `devtools::document()` / roxygenize.** Man pages in `man/` are hand-edited to stay in sync with the roxygen comments (differing roxygen2 versions between machines cause large diff churn). When you change roxygen docs, hand-edit the matching `.Rd` file, and validate with `tools::checkRd("man/<file>.Rd")`. Same for `NAMESPACE`: edit by hand.
- **Leave changes uncommitted.** The maintainer reviews and commits in RStudio. Provide a ready-to-paste commit message ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Verify empirically before claiming behavior.** The reticulate/callr/conda stack has many silent-failure modes; run the actual function in the real environment rather than reasoning from source.

## Architecture

### Two Python execution models

1. **Main-session Python** — `talkText()`/`talkTranscribe()` (R/1_talkText.R) and `talkEmbed()` (R/2_1_talkEmbed.R). They `source_python("inst/python/huggingface_Interface4.py")` into the session Python, which must already be bound by `talkrpp_initialize()`. reticulate allows only ONE Python per R session — once initialized, the env cannot be switched without restarting R.
2. **callr subprocess** — `talkTranscribeDiarise()`/`talkTextDiarise()` (R/3_1_talkTranscribeDiarise.R, backend inst/python/diarize.py) and `talkEmbedSegments()` (R/2_2_talkEmbedSegments.R, backend inst/python/embed.py). Each call spawns a fresh subprocess bound to `condaenv`, sidestepping the one-Python limit and a fatal OpenMP conflict between NeMo and R on macOS. **Pitfall: reticulate inside callr silently swallows Python exceptions and returns NULL** — that is why these functions validate inputs in R before touching Python and treat a non-list result as an error. The subprocess also force-sets UTF-8 env vars (LC_ALL, PYTHONUTF8, …) because R CMD check runs under an ASCII locale that breaks NeMo.

### Environment management (R/0_0_talk_install.R, R/0_0_1_talk_initialize.R, R/zzz.R)

- `talkrpp_install()` builds one conda env with the full pinned stack: WhisNemo (whisper + NeMo diarisation) installed with a pip constraints file that pins numpy==1.23.5, torch, transformers, nemo-toolkit — **never loosen these pins independently**; they come from WhisNemo's `constraints/runtime.txt`. `include_text = TRUE` (default) also installs the text-package's Python deps (non-fatal on failure) so one env serves both the talk and text R packages.
- `talkrpp_initialize(save_profile = TRUE)` stores the env name as the `talkrpp_condaenv` R option; `.onAttach` auto-initializes from that saved profile. All `condaenv` function arguments default to `getOption("talkrpp_condaenv", "talkrpp_condaenv")`, so the callr functions follow the saved profile too.
- **ffmpeg**: resolved by `ensure_ffmpeg_on_path()` — system ffmpeg first (including paths missing from RStudio's PATH like /opt/homebrew/bin), then a static binary from imageio-ffmpeg copied into `tools::R_user_dir("talk", "cache")/bin`. **Never install ffmpeg via conda** — conda's ffmpeg ships libav* libraries that break torchaudio's audio loading.

### Backend details worth knowing

- `inst/python/huggingface_Interface4.py` caches the most recently loaded model in `_AUDIO_MODEL_CACHE`, guarded with `if "_AUDIO_MODEL_CACHE" not in globals()` because every R call re-sources the file. Always `.to(device)` after fetching from the cache. Per-file try/except returns `None` for a failed file so R can put NA at the right position.
- Device selection: CUDA is fine; on Apple Silicon, Whisper's decoder on MPS silently drops large parts of transcripts (diarise pipeline auto-falls back to CPU for the transcription step), and on **virtualized** macOS (CI runners; detected via `sysctl kern.hv_vmm_present`) MPS produces garbage embeddings and is never auto-selected.
- Return conventions: functions return tibbles (never bare vectors, never files by default; `output_dir` opts into file output) with the settings recorded via `comment()` — mirroring the text package. Multi-file input returns a named list of tibbles keyed by file name.

### Docs/site

`installation.md` (repo root) is the pkgdown installation page; `_pkgdown.yml` defines the navbar and sidebar. README is generated from README.Rmd — edit the .Rmd.
