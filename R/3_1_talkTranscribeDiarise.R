
#' Transcribe and Diarize Audio Recordings
#'
#' Transcribes one or more audio recordings using Whisper and performs speaker
#' diarization (who speaks when) using NeMo, NVIDIA's open-source speech
#' toolkit. Runs in a subprocess to avoid OpenMP conflicts on macOS.
#' Requires the talk conda environment installed via \code{talkrpp_install()}.
#'
#' @param audio (string or character vector) Path to a single audio file
#'   (e.g., \code{.wav}) or a vector of file paths. Each file is processed separately.
#' @param output_dir (string) Directory where output files will be saved. Defaults to the current working directory.
#' @param model_name (string) Whisper model name (e.g., \code{"medium.en"}). Options: \code{"tiny"}, \code{"base"}, \code{"small"}, \code{"medium"}, \code{"large"}.
#' @param language (string) Language code (e.g., \code{"en"}, \code{"sv"}). If NULL, Whisper auto-detects the language.
#' @param device (string) Device to run inference on: \code{"cuda"} (NVIDIA
#'   GPU), \code{"cpu"}, or \code{"mps"} (Apple Silicon GPU). Defaults to
#'   \code{"cpu"} on macOS and \code{"cuda"} elsewhere. \code{"cuda"} is fast
#'   and produces correct results. \code{"mps"} needs care: Whisper's decoder
#'   on MPS can silently drop large parts of the transcript (about half, on
#'   some files) while appearing to work; the pipeline therefore automatically
#'   runs the transcription step on the CPU when \code{device = "mps"} and
#'   uses the GPU only for the diarisation step. See Details.
#' @param stemming (logical) If TRUE (default), first isolate the vocals with
#'   Demucs source separation, so that background music and noise do not end
#'   up in the transcript or confuse speaker assignment. Adds processing time;
#'   if separation fails, the original audio is used.
#' @param suppress_numerals (logical) If TRUE, numbers are transcribed as
#'   words ("twenty" rather than "20"). Spoken words align better with the
#'   audio, which improves word-level timestamps. Default is FALSE.
#' @param batch_size (integer) Number of audio segments processed at once. Default is 8.
#'   Larger values are faster but use more memory.
#' @param num_speakers (integer) Expected number of speakers for NeMo clustering.
#' @param oracle_num_speakers (logical) If TRUE, force the diarisation to use exactly
#'   \code{num_speakers} speakers; if FALSE, the number of speakers is estimated.
#' @param vad_model (string) NeMo model for Voice Activity Detection (VAD) --
#'   the step that detects which parts of the audio contain speech at all,
#'   before any transcription or speaker assignment. Default is
#'   \code{"vad_multilingual_marblenet"}.
#' @param speaker_model (string) NeMo model that computes speaker embeddings
#'   (numeric representations of voice characteristics used to tell speakers
#'   apart). Default is \code{"titanet_large"}.
#' @param onset (numeric) VAD threshold (0-1) for detecting the \emph{start} of a
#'   speech segment: higher values require stronger evidence before opening a
#'   segment (fewer false starts, but soft speech onsets may be missed).
#'   Default is 0.8.
#' @param offset (numeric) VAD threshold (0-1) for detecting the \emph{end} of a
#'   speech segment: lower values keep segments open longer (less clipping of
#'   trailing speech, but more trailing noise). Default is 0.5.
#' @param pad_offset (numeric) Seconds added to the end of each detected
#'   speech segment; negative values trim the segment end. Default is -0.05.
#' @param domain_type (string) Selects a NeMo diarisation configuration tuned
#'   to the recording situation: \code{"telephonic"} (close-microphone,
#'   two-party recordings such as phone calls or one-on-one interviews; the
#'   default), \code{"meeting"} (several speakers around a room microphone),
#'   or \code{"general"}. Choosing the type that matches your recordings
#'   improves speech detection and speaker assignment.
#' @param output_formats (character vector) Output formats to write. Any of \code{"csv"}, \code{"txt"}, \code{"srt"}. Default is \code{"csv"}.
#' @param remove_stutters (logical) If TRUE, post-process the transcript to
#'   remove immediately repeated, near-duplicate words. Such repetitions come
#'   both from genuine speaker disfluencies (stuttering, false starts) and
#'   from a known Whisper artifact where the model repeats words. Keep FALSE
#'   (the default) when disfluencies themselves are of interest to your
#'   research. Applied to the CSV output.
#' @param stutter_threshold (numeric) Similarity (0-1) above which adjacent
#'   words are treated as repetitions when \code{remove_stutters = TRUE};
#'   higher values remove only near-identical repeats. Default is 0.8.
#' @param condaenv (string) Name of the conda environment with the talk stack installed.
#'   Default is \code{"talkrpp_condaenv"} (installed by \code{talkrpp_install()}).
#'
#' @return A list with keys \code{output_files} (character vector of written file paths)
#'   and \code{status} (\code{"success"} or \code{"error"}).
#'
#' @details
#' Output files (diarized CSVs, SRTs, or TXT transcripts) are written to \code{output_dir}.
#'
#' \strong{Devices and result correctness.} \code{"cpu"} always produces
#' correct results and is the safe default on macOS. \code{"cuda"} (NVIDIA
#' GPUs on Linux/Windows) is substantially faster and produces correct
#' results. \code{"mps"} (Apple Silicon GPU) can run but is treated with
#' caution: Whisper's decoder on MPS has a known bug where it skips ahead and
#' silently drops large parts of the audio -- the transcript looks plausible
#' but can be missing about half its content. To protect against this, when
#' \code{device = "mps"} the pipeline automatically runs the transcription
#' step on the CPU and uses the GPU only for the diarisation step.
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
