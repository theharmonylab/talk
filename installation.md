# Installing the talk-package

The *talk* package analyses audio recordings of speech using HuggingFace
transformer models. It runs in R, with a Python backend that is installed and
managed automatically in a conda environment.

## OS-specific instructions {.tabset}

### Windows

1. Install [R](https://cran.r-project.org/) and
   [RStudio](https://posit.co/download/rstudio-desktop/).

#### Install and initialize talk

In R:

```r
# install.packages("devtools")
devtools::install_github("theharmonylab/talk")
```

Then install and activate the Python backend:

```r
library(talk)

# Installs a single conda environment ("talkrpp_condaenv") containing the
# full stack: transcription, embeddings, diarisation and segment embeddings.
# This also checks your system dependencies and reports anything missing --
# including ffmpeg, for which a static fallback is installed automatically.
talkrpp_install()

# Activate the environment for this and future sessions
talkrpp_initialize(save_profile = TRUE)
```

Test the installation:

```r
library(talk)

wav_path <- system.file("extdata", "test_short.wav", package = "talk")

talkText(talk_filepaths = wav_path)
#> [1] " Hello."
```

#### Optional: system ffmpeg

talk installs a static ffmpeg automatically, so this is not required. If you
prefer a system-wide ffmpeg, open PowerShell and run (with
[Chocolatey](https://chocolatey.org/)):

```
choco install ffmpeg
```

or download it from [ffmpeg.org](https://ffmpeg.org/download.html) and add it
to your PATH. (Do not install ffmpeg with conda — conda's ffmpeg breaks
torchaudio's audio loading.)

### MacOS

1. Install [R](https://cran.r-project.org/) and
   [RStudio](https://posit.co/download/rstudio-desktop/). On Apple Silicon
   (M1–M4), make sure to install the *arm64* build of R.

#### Install and initialize talk

In R:

```r
# install.packages("devtools")
devtools::install_github("theharmonylab/talk")
```

Then install and activate the Python backend:

```r
library(talk)

# Installs a single conda environment ("talkrpp_condaenv") containing the
# full stack: transcription, embeddings, diarisation and segment embeddings.
# This also checks your system dependencies and reports anything missing --
# including ffmpeg, for which a static fallback is installed automatically.
talkrpp_install()

# Activate the environment for this and future sessions
talkrpp_initialize(save_profile = TRUE)
```

Test the installation:

```r
library(talk)

wav_path <- system.file("extdata", "test_short.wav", package = "talk")

talkText(talk_filepaths = wav_path)
#> [1] " Hello."
```

#### Optional: system ffmpeg

talk installs a static ffmpeg automatically, so this is not required. If you
prefer a system-wide ffmpeg, install [Homebrew](https://brew.sh/) (if you do
not have it) and run in the Terminal:

```
brew install ffmpeg
```

(Do not install ffmpeg with conda — conda's ffmpeg breaks torchaudio's audio
loading.)

### Linux

1. Install [R](https://cran.r-project.org/) and
   [RStudio](https://posit.co/download/rstudio-desktop/) for your
   distribution.

2. Install the development libraries needed to build talk's R dependencies
   from source (Ubuntu/Debian):

   ```
   sudo apt-get update
   sudo apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
   ```

   (`talkrpp_install()` can also install these automatically when root/sudo
   is available — but they are typically needed already to install devtools,
   so running the command above first is the reliable path.)

#### Install and initialize talk

In R:

```r
# install.packages("devtools")
devtools::install_github("theharmonylab/talk")
```

Then install and activate the Python backend:

```r
library(talk)

# Installs a single conda environment ("talkrpp_condaenv") containing the
# full stack: transcription, embeddings, diarisation and segment embeddings.
# This also checks your system dependencies and reports anything missing --
# including ffmpeg, for which a static fallback is installed automatically.
talkrpp_install()

# Activate the environment for this and future sessions
talkrpp_initialize(save_profile = TRUE)
```

Test the installation:

```r
library(talk)

wav_path <- system.file("extdata", "test_short.wav", package = "talk")

talkText(talk_filepaths = wav_path)
#> [1] " Hello."
```

#### Optional: system ffmpeg

talk installs a static ffmpeg automatically, so this is not required. If you
prefer a system-wide ffmpeg (Ubuntu/Debian):

```
sudo apt-get install -y ffmpeg
```

(Do not install ffmpeg with conda — conda's ffmpeg breaks torchaudio's audio
loading.)

### Troubleshooting

1. **See which Python environments exist:** `talk::list_talkrpp_envs()`.
2. **"Python already initialized" error:** restart R, then call
   `talkrpp_initialize()` before any other Python-using code.
3. **ffmpeg problems:** talk finds a system ffmpeg automatically (also in
   locations missing from RStudio's PATH, such as `/opt/homebrew/bin`) and
   otherwise uses the static ffmpeg installed by `talkrpp_install()`. If
   ffmpeg still cannot be found, re-run `talkrpp_install()`.
4. **Reinstall the environment from scratch:**

   ```r
   talkrpp_uninstall(envname = "talkrpp_condaenv")
   talkrpp_install()
   ```

5. **Old or duplicated conda installations** (typical symptoms:
   `No module named 'whisnemo'`; `EnvironmentNameNotFound` during
   `talkrpp_install()`; `No matching distribution found for torch==2.11.0`;
   `talkrpp_uninstall()` failing to find the environment). A
   `talkrpp_condaenv` created by an older talk version may use an old Python
   (3.9) and may live in a *different* conda installation (e.g. a personal
   `~/miniconda3`) than the one talk now uses (reticulate's miniconda). The
   fix is to remove the old environment with the conda installation that owns
   it, then reinstall:

   ```
   # in a terminal -- adjust the path to the conda that owns the old env
   ~/miniconda3/bin/conda env remove -n talkrpp_condaenv -y
   ```

   ```r
   # then, in R
   talkrpp_install()
   ```

   `talk::list_talkrpp_envs()` shows which environments exist; the
   registrations in `~/.conda/environments.txt` show which conda installation
   each environment belongs to.

6. **Apple Silicon (MPS):** on real Apple hardware talk uses the MPS GPU
   automatically where it is reliable; in virtualized macOS (e.g. CI runners)
   it falls back to CPU.

If problems persist, please look through the
[closed GitHub issues](https://github.com/theharmonylab/talk/issues?q=is%3Aissue)
or open a new issue.

## Try talk in the browser

💡 *Want to try talk without installing anything?* Open our
<a href="https://colab.research.google.com/drive/1N1qVKabhyO0U7auI58nfalrdrVkxb8BX" target="_blank" rel="noopener">Google Colab notebook</a>
and run talk directly in your browser.
