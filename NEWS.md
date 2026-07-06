# talk (development versions)

<!-- README.md is generated from README.Rmd. Please edit that file -->

# talk 0.6

## Major changes
* `talkTranscribeDiarise()` now returns the transcript directly as a tibble
  (previously a list with `$transcript`, `$output_files` and `$status`);
  failures are signalled as R errors with actionable hints. It also no longer
  writes files by default: `output_dir` defaults to NULL, and no temporary
  files (audio copy, timing logs) are left behind. Provide `output_dir` to
  save csv/txt/srt transcript files (saved paths are shown in a message).
* The talk package now uses a **single conda environment** (`talkrpp_condaenv`)
  for all functions. `talkrpp_install()` installs the full stack (transcription,
  embeddings, diarisation and segment embeddings), and
  `talkTranscribeDiarise()` / `talkEmbedSegments()` default to that environment.
* `talkTranscribeDiarise()` and `talkEmbedSegments()` support several audio
  files: each file is processed separately and a named list of tibbles is
  returned; the transcript list from a multi-file diarise run plugs directly
  into `talkEmbedSegments()`.
* Fixed `talkEmbed(use_decoder = TRUE)`, which never worked: the decoder path
  now embeds the transcription while attending to the audio (mean-pooled
  decoder hidden states), and requires `audio_transcriptions` with a clear
  error message otherwise.

## Minor changes and fixes
* `talkrpp_install()` gained `include_text` (default TRUE): additionally
  installs the text package's Python dependencies, so the talk environment
  also serves the text package -- enabling both packages in the same R
  session. This step is non-fatal (talk installs fully even if it fails);
  set `include_text = FALSE` for a leaner, talk-only environment.
* `talkTranscribeDiarise()` and `talkEmbedSegments()` default their
  `condaenv` to the environment saved by
  `talkrpp_initialize(save_profile = TRUE)`, enabling a shared environment
  with the text package (e.g. `talkrpp_install(envname = "text_talk")`, then
  initialize both packages to it).
* Added function aliases: `talkTranscribe()` (same as `talkText()`) and
  `talkTextDiarise()` (same as `talkTranscribeDiarise()`).
* `talkText()`, `talkEmbed()`, `talkTranscribeDiarise()` and
  `talkEmbedSegments()` save the settings used (model, device, key
  parameters, duration and talk version) as a comment on the returned object,
  retrievable with `comment()` -- matching the text package.
* `talkTranscribeDiarise()` and `talkEmbedSegments()` gained a `verbose`
  argument (default `FALSE`): the technical Python backend output is hidden
  by default and replaced by short status messages; errors are always shown.
* All main functions validate their inputs early (existing audio files,
  matching transcript collections, known conda environments) with clear,
  instant error messages.
* `talkrpp_install()` checks system dependencies with copy-paste install
  instructions, installs Rust only when actually missing, can install
  Debian/Ubuntu development libraries when root/sudo is available, and
  provides an automatic static ffmpeg fallback (via imageio-ffmpeg) when no
  system ffmpeg exists.
* Robust recovery from stale environments: a saved profile pointing at a
  removed environment no longer breaks `library(talk)`, and environments
  created by older talk versions produce an actionable upgrade hint.
* On virtualized macOS (VMs/CI runners) MPS is never auto-selected for
  embeddings, where it can silently produce invalid results; real Apple
  hardware still uses MPS automatically.
* torch's inductor cache is kept in the per-user talk cache instead of the
  temp directory.

# talk 0.5

## Major changes
* Added `talkTranscribeDiarise()`: transcribe a recording **and** diarise it
  (split the transcript by speaker, with timestamps).
* Added `talkEmbedSegments()`: produce one embedding **per diarised segment**
  (rather than one per file), with Whisper (`encoder` / `decoder` / `both`) and
  WhiSPA backends.
* Added `list_talkrpp_envs()` to list the available Python environments.

## Minor changes and fixes
* Cross-platform support for the diarisation/embedding backend (Linux, Windows,
  and macOS/Apple Silicon incl. MPS).
* `talkrpp_install()` now reliably refreshes git-based Python packages when
  re-installing onto an existing environment.
* `talkText()` / `talkEmbed()`: load audio via `soundfile` and fix a GPU
  tensor-to-numpy conversion.
* Added a tutorial vignette covering all functions and per-function tests.

# talk 0.2

## Major changes
* adding `talkText()` and `talkEmbed()`
* adding github actions and tests.  



