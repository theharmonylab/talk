# talk (development versions)

<!-- README.md is generated from README.Rmd. Please edit that file -->

# talk 0.5

## Major changes
* Added `talkTranscribeDiarise()`: transcribe a recording **and** diarise it
  (split the transcript by speaker, with timestamps).
* Added `talkEmbedSegments()`: produce one embedding **per diarised segment**
  (rather than one per file), with Whisper (`encoder` / `decoder` / `both`) and
  WhiSPA backends.
* Added `list_talkrpp_envs()` to list the available Python environments.

## Minor changes and fixes
* Added function aliases: `talkTranscribe()` (same as `talkText()`) and
  `talkTextDiarise()` (same as `talkTranscribeDiarise()`).
* `talkText()`, `talkEmbed()`, `talkTranscribeDiarise()` and
  `talkEmbedSegments()` now save the settings used (model, device, key
  parameters, duration and talk version) as a comment on the returned object,
  retrievable with `comment()` -- matching the text package.
* `talkTranscribeDiarise()` and `talkEmbedSegments()` gained a `verbose`
  argument (default `FALSE`): the technical Python backend output is now
  hidden by default and replaced by short status messages; errors are always
  shown.
* `talkTranscribeDiarise()` now supports several audio files (returned as a
  named list of transcript tibbles), validates that the audio files exist
  before starting, and errors clearly when the backend produces no result.
  `talkEmbedSegments()` likewise supports several audio files (paired with
  the transcript list from a multi-file `talkTranscribeDiarise()` run) and
  validates its `audio` and `transcript` inputs early with clear error
  messages.
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



