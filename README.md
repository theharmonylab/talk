
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- -->

# talk <a href="https://r-talk.org"><img src="man/figures/logo.png" align="right" height="138" alt="talk website" /></a>

<!-- badges: start -->

[![Github build
status](https://github.com/theharmonylab/talk/workflows/R-CMD-check/badge.svg)](https://github.com/theharmonylab/talk/actions)
[![codecov](https://codecov.io/gh/theharmonylab/talk/branch/main/graph/badge.svg?)](https://app.codecov.io/gh/theharmonylab/talk)
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Lifecycle:
maturing](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://lifecycle.r-lib.org/articles/stages.html#maturing-1)

<!--
[![CRAN Status](https://www.r-pkg.org/badges/version/talk)](https://CRAN.R-project.org/package=talk)
&#10;[![CRAN Downloads](https://cranlogs.r-pkg.org/badges/grand-total/talk)](https://CRAN.R-project.org/package=talk)
&#10;-->

<!-- badges: end -->

## Overview

An R-package for analyzing natural language with transformers-based
large language models. The `talk` package is part of the *R Language
Analysis Suite*, including `talk`, `text` and `topics`.

- [`talk`](https://www.r-talk.org/) transforms voice recordings into
  text, audio features, or embeddings.<br> <br>
- [`text`](https://www.r-text.org/) provide many language tasks such as
  converting digital text into word embeddings.<br> <br> `talk` and
  `text` provide access to Large Language Models from Hugging Face.<br>
  <br>
- [`topics`](https://www.r-topics.org/) visualizes language patterns
  into topics to generate psychological insights.<br> <br> <br>
  <img src="man/figures/talk_text_topics1.svg" style="width:50.0%" />

<br> The *R Language Analysis Suite* is created through a collaboration
between psychology and computer science to address research needs and
ensure state-of-the-art techniques. The suite is continuously tested on
Ubuntu, Mac OS and Windows using the latest stable R version.

> ### 📣 Online Workshop: *Analysing Human Language using R*
>
> **August 11–13, 2026** — three half-day sessions (2:00–5:00 pm CEST /
> 8:00–11:00 am ET)
>
> Learn the full *R Language Analysis Suite* —
> [`talk`](https://www.r-talk.org/), [`text`](https://www.r-text.org/),
> and [`topics`](https://www.r-topics.org/) — with Oscar Kjell.
>
> **[Read more & register →](https://smart-workshops.com/lang-r-info)**

### Point solution for transforming talk to embeddings

Recent advances in speech and language modelling have dramatically
improved how well computers can transcribe and represent human speech.
The talk-package makes these state-of-the-art models — such as Whisper
from [HuggingFace](https://huggingface.co/docs/transformers/index) —
easily accessible from R. All analyses run locally on your own computer,
so recordings never leave your machine — important when working with
sensitive data such as clinical or research interviews.

With talk you can:

- **Transcribe speech to text**: turn audio recordings into text, in
  many languages.
- **Identify who says what and when**: split a conversation by speaker,
  with time-stamped speaker turns (diarisation).
- **Transform speech into embeddings**: represent whole recordings — or
  each speaker turn — as numeric vectors that capture acoustic and
  psychologically relevant properties of the voice, beyond the words
  themselves.
- **Use the embeddings in downstream analyses**: for example, training
  models to predict psychological constructs with the
  [text](https://www.r-text.org/)-package.

See the [Getting started
tutorial](https://www.r-talk.org/articles/talk.html) for a walk-through
of the workflow, and [HuggingFace](https://huggingface.co/models/) for
the available models.
