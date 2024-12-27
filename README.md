
<!-- README.md is generated from README.Rmd. Please edit that file -->

# talk <img src="man/figures/logo.png" align="right" alt="" width="330" />

<!-- badges: start -->

[![Github build
status](https://github.com/theharmonylab/talk/workflows/R-CMD-check/badge.svg)](https://github.com/theharmonylab/talk/actions)
[![codecov](https://codecov.io/gh/theharmonylab/talk/branch/main/graph/badge.svg?)](https://app.codecov.io/gh/theharmonylab/talk)

<!--
[![CRAN Status](https://www.r-pkg.org/badges/version/talk)](https://CRAN.R-project.org/package=talk)
&#10;[![Lifecycle: maturing](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://lifecycle.r-lib.org/articles/stages.html#maturing-1)
&#10;[![CRAN Downloads](https://cranlogs.r-pkg.org/badges/grand-total/talk)](https://CRAN.R-project.org/package=talk)
&#10;-->
<!-- badges: end -->

An R-package for analyzing natural language with transformers from
HuggingFace using Natural Language Processing and Machine Learning.

The *talk*-package is part of the XXXX, including talk, text and topics.

*talk* is created through a collaboration between psychology and
computer science to address research needs and ensure state-of-the-art
techniques. It provides powerful functions tailored to test research
hypotheses in social and behavior sciences for both relatively small and
large datasets. *talk* is continuously tested on Ubuntu, Mac OS and
Windows using the latest stable R version.

### Short installation guide

Most users simply need to run below installation code. For those
experiencing problems, please see the [Extended Installation
Guide](https://www.r-talk.org/articles/huggingface_in_r_extended_installation_guide.html).

For the talk-package to work, you first have to install the talk-package
in R, and then make it work with talk required python packages.

1.  Install talk-version (at the moment the second step only works using
    the development version of talk from GitHub).

[GitHub](https://github.com/) development version:

``` r
# install.packages("devtools")
devtools::install_github("theharmonylab/talk")
```

[CRAN](https://CRAN.R-project.org/package=talk) version:

``` r
install.packages("talk")
```

2.  Install and initialize talk required python packages:

``` r
library(talk)

# Install talk required python packages in a conda environment (with defaults).
talkrpp_install()

# Initialize the installed conda environment.
# save_profile = TRUE saves the settings so that you don't have to run talkrpp_initialize() after restarting R. 
talkrpp_initialize(save_profile = TRUE)
```

### Point solution for transforming talk to embeddings

Recent significant advances in NLP research have resulted in improved
representations of human language (i.e., language models). These
language models have produced big performance gains in tasks related to
understanding human language. talk are making these SOTA models easily
accessible through an interface to
[HuggingFace](https://huggingface.co/docs/transformers/index) in Python.

See [HuggingFace](https://huggingface.co/models/) for a more
comprehensive list of models.

The `talkText()` function performs speech-to-text, transcribing audio
input to text. \`talkEmbed()\`\`\`, transforms audio input to numeric
representaions (embeddings) that can be used for downstream tasks such
as guideline predictive models using the text-pacakge (see the text
train functions).

``` r
library(talk)
# Transform the talk data to BERT word embeddings

# Get file path to example audio from the package example data
wav_path <- system.file("extdata/",
                            "test_short.wav",
                            package = "talk")

# Get transcription 
talk_embeddings <- talkText(
  wav_path
)
talk_embeddings

# Defaults
talk_embeddings <- talkEmbed(
  wav_path
)
talk_embeddings
```