
#' Transcribe and Diarize Audio Recordings
#'
#' Transcribes one or more audio recordings using Whisper and performs speaker
#' diarization using NeMo. Runs in a subprocess to avoid OpenMP conflicts on macOS.
#' Requires the talk conda environment installed via \code{talkrpp_install()}.
#'
#' @param audio (string or character vector) Path to a single audio file
#'   (e.g., \code{.wav}) or a vector of file paths. Each file is processed separately.
#' @param output_dir (string) Directory where output files will be saved. Defaults to the current working directory.
#' @param model_name (string) Whisper model name (e.g., \code{"medium.en"}). Options: \code{"tiny"}, \code{"base"}, \code{"small"}, \code{"medium"}, \code{"large"}.
#' @param language (string) Language code (e.g., \code{"en"}, \code{"sv"}). If NULL, Whisper auto-detects the language.
#' @param device (string) Device to run inference on: \code{"cuda"}, \code{"cpu"}, or \code{"mps"} (Apple Silicon, experimental).
#'   Defaults to \code{"cpu"} on macOS and \code{"cuda"} elsewhere.
#' @param stemming (logical) If TRUE, run Demucs vocal separation before transcription.
#' @param suppress_numerals (logical) If TRUE, suppress numerical digits in transcription output.
#' @param batch_size (integer) Number of audio segments processed at once. Default is 8.
#' @param num_speakers (integer) Expected number of speakers for NeMo clustering.
#' @param oracle_num_speakers (logical) If TRUE, force the exact speaker count in NeMo clustering.
#' @param vad_model (string) NeMo VAD model name. Default is \code{"vad_multilingual_marblenet"}.
#' @param speaker_model (string) NeMo speaker embedding model. Default is \code{"titanet_large"}.
#' @param onset (numeric) VAD onset threshold. Default is 0.8.
#' @param offset (numeric) VAD offset threshold. Default is 0.5.
#' @param pad_offset (numeric) VAD pad offset. Default is -0.05.
#' @param domain_type (string) NeMo domain type: \code{"telephonic"}, \code{"meeting"}, or \code{"general"}.
#' @param output_formats (character vector) Output formats to write. Any of \code{"csv"}, \code{"txt"}, \code{"srt"}. Default is \code{"csv"}.
#' @param remove_stutters (logical) If TRUE, apply stutter removal post-processing on CSV output.
#' @param stutter_threshold (numeric) Similarity threshold for stutter detection. Default is 0.8.
#' @param condaenv (string) Name of the conda environment with the talk stack installed.
#'   Default is \code{"talkrpp_condaenv"} (installed by \code{talkrpp_install()}).
#'
#' @return A list with keys \code{output_files} (character vector of written file paths)
#'   and \code{status} (\code{"success"} or \code{"error"}).
#'
#' @details
#' Output files (diarized CSVs, SRTs, or TXT transcripts) are written to \code{output_dir}.
#'
#' On macOS, NeMo's OpenMP library conflicts with R's when loaded in the same process.
#' This function automatically runs diarization in a subprocess via \code{callr::r()} to
#' avoid the crash.
#'
#' @examples
#' \dontrun{
#' wav_path <- system.file("extdata", "test_diarise.wav", package = "talk")
#' talk::talkTranscribeDiarise(audio = wav_path, num_speakers = 2)
#' }
#'
#' @seealso \code{\link{talkEmbed}}, \code{\link{talkText}}, \code{\link{talkrpp_initialize}}
#' @export
talkTranscribeDiarise <- function(
    audio,
    output_dir = getwd(),
    model_name = "medium.en",
    language = NULL,
    device = if (Sys.info()[["sysname"]] == "Darwin") "cpu" else "cuda",
    stemming = TRUE,
    suppress_numerals = FALSE,
    batch_size = 8,
    num_speakers = 2,
    oracle_num_speakers = TRUE,
    vad_model = "vad_multilingual_marblenet",
    speaker_model = "titanet_large",
    onset = 0.8,
    offset = 0.5,
    pad_offset = -0.05,
    domain_type = "telephonic",
    output_formats = "csv",
    remove_stutters = FALSE,
    stutter_threshold = 0.8,
    condaenv = "talkrpp_condaenv"
    ) {

  diarize_py <- system.file("python", "diarize.py", package = "talk", mustWork = TRUE)

  # Make ffmpeg available: whisper loads audio through the ffmpeg binary, and
  # diarisation would otherwise die later with a cryptic "[Errno 2] No such
  # file or directory: 'ffmpeg'". ensure_ffmpeg_on_path() finds a system
  # ffmpeg (also in locations missing from RStudio's PATH) and otherwise falls
  # back to the static ffmpeg installed by talkrpp_install().
  if (!ensure_ffmpeg_on_path(condaenv)) {
    stop(
      "talkTranscribeDiarise() requires the 'ffmpeg' binary, which was not found.\n",
      "Install it with: ", ffmpeg_install_instruction(), "\n",
      "or re-run talkrpp_install(), which installs a static ffmpeg fallback automatically.\n",
      "Note: do NOT install ffmpeg with conda -- conda's ffmpeg breaks torchaudio's audio loading.",
      call. = FALSE
    )
  }

  callr::r(
    func = function(audio, output_dir, model_name, language, device,
                    stemming, suppress_numerals, batch_size, num_speakers,
                    oracle_num_speakers, vad_model, speaker_model,
                    onset, offset, pad_offset, domain_type, output_formats,
                    remove_stutters, stutter_threshold, condaenv, diarize_py) {

      Sys.setenv(KMP_DUPLICATE_LIB_OK        = "TRUE")
      Sys.setenv(OMP_NUM_THREADS             = "1")
      Sys.setenv(MKL_NUM_THREADS             = "1")
      Sys.setenv(OPENBLAS_NUM_THREADS        = "1")
      Sys.setenv(NUMBA_NUM_THREADS           = "1")
      Sys.setenv(KMP_INIT_AT_FORK            = "FALSE")
      Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")

      # Force a UTF-8 locale so Python does not fall back to the ascii codec
      # (which fails on non-ascii bytes such as a UTF-8 BOM and aborts
      # diarization with "'ascii' codec can't decode byte ..."). Set this
      # UNCONDITIONALLY: the session may report a UTF-8 LANG while the effective
      # LC_CTYPE is "C" (e.g. under R CMD check), so gating on the locale env
      # vars is unreliable. Python reads the libc locale here, so LC_ALL/LANG
      # must be UTF-8 (PYTHONUTF8 alone does not override it).
      Sys.setenv(LC_ALL          = "en_US.UTF-8")
      Sys.setenv(LANG            = "en_US.UTF-8")
      Sys.setenv(PYTHONUTF8      = "1")
      Sys.setenv(PYTHONIOENCODING = "utf-8")

      reticulate::use_condaenv(condaenv, required = TRUE)
      reticulate::source_python(diarize_py)

      diarize_audio(
        audio_path          = audio,
        output_dir          = output_dir,
        model_name          = model_name,
        language            = language,
        device              = device,
        stemming            = stemming,
        suppress_numerals   = suppress_numerals,
        batch_size          = batch_size,
        num_speakers        = num_speakers,
        oracle_num_speakers = oracle_num_speakers,
        vad_model           = vad_model,
        speaker_model       = speaker_model,
        onset               = onset,
        offset              = offset,
        pad_offset          = pad_offset,
        domain_type         = domain_type,
        output_formats      = as.list(output_formats),
        remove_stutters     = remove_stutters,
        stutter_threshold   = stutter_threshold
      )
    },
    args = list(
      audio               = audio,
      output_dir          = output_dir,
      model_name          = model_name,
      language            = language,
      device              = device,
      stemming            = stemming,
      suppress_numerals   = suppress_numerals,
      batch_size          = batch_size,
      num_speakers        = num_speakers,
      oracle_num_speakers = oracle_num_speakers,
      vad_model           = vad_model,
      speaker_model       = speaker_model,
      onset               = onset,
      offset              = offset,
      pad_offset          = pad_offset,
      domain_type         = domain_type,
      output_formats      = output_formats,
      remove_stutters     = remove_stutters,
      stutter_threshold   = stutter_threshold,
      condaenv            = condaenv,
      diarize_py          = diarize_py
    ),
    show = TRUE
  )
}

#' @rdname talkTranscribeDiarise
#' @export
talkTextDiarise <- talkTranscribeDiarise
