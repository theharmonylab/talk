#' Transform diarized audio into segment-level embeddings
#'
#' Unlike \code{\link{talkEmbed}}, which returns one embedding per whole audio
#' file, \code{talkEmbedSegments} produces one embedding per diarized segment.
#' It takes an audio file together with its diarized transcript (e.g. the output
#' of \code{\link{talkTranscribeDiarise}}) and embeds each segment, optionally
#' restricting to participant segments.
#'
#' Two backends are supported: Whisper (encoder and/or decoder hidden-state
#' summary statistics) and WhiSPA (a single speech-psychological embedding per
#' segment).
#'
#' @param audio (string or character vector) Path to a single audio file
#'   (e.g. \code{.wav}) or a vector of file paths. Several files are processed
#'   separately (paired with `transcript`), returning a named list.
#' @param transcript (string, data.frame, or list) Diarized transcript: a path
#'   to a CSV, the transcript tibble returned by
#'   \code{\link{talkTranscribeDiarise}}, or -- for several audio files --
#'   the named list returned by a multi-file \code{talkTranscribeDiarise()}
#'   run (or a vector of CSV paths) of the same length as \code{audio}. Must contain per-segment start/end
#'   timestamps, and a \code{speaker_role} column if \code{participant_only=TRUE}.
#' @param model (string) \code{"whisper"} or \code{"whispa"}.
#' @param embeddings (string) For \code{model="whisper"}: \code{"encoder"},
#'   \code{"decoder"}, or \code{"both"}. Ignored for \code{model="whispa"}.
#' @param participant_only (logical) If TRUE, only segments with
#'   \code{speaker_role == "participant"} are embedded. Defaults to FALSE
#'   because the transcript returned by \code{\link{talkTranscribeDiarise}}
#'   has no \code{speaker_role} column; set TRUE only with a transcript that
#'   provides one.
#' @param whisper_model_id (string) Optional override of the Whisper model id.
#' @param whispa_model_id (string) WhiSPA checkpoint (\code{model="whispa"} only).
#' @param whispa_repo_path (string) Optional path to a local WhiSPA clone, used
#'   only if WhiSPA is not pip-installed.
#' @param output_dir (string) Optional directory to also write embedding CSV(s).
#' @param device (string) \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If NULL,
#'   chooses \code{"cuda"} when available, then \code{"mps"} on real Apple
#'   hardware, else \code{"cpu"}. On virtualized macOS (VMs/CI runners) MPS can
#'   silently produce invalid embeddings, so it is never auto-selected there
#'   (and explicitly requesting it triggers a warning).
#' @param condaenv (string) Name of the conda environment that holds the talk
#'   stack (including whisnemo[embed] and WhiSPA). Default
#'   \code{"talkrpp_condaenv"}, the single environment installed by
#'   \code{talkrpp_install()}.
#' @param verbose (logical) If FALSE (default), the technical output from the
#'   Python backend (model-loading logs, progress bars, warnings) is hidden
#'   and only short status messages are shown. Set TRUE to stream the full
#'   backend output, e.g. when debugging.
#'
#' @return A tibble of segment-level embeddings (one row per segment), or, for
#'   \code{model="whisper"} with \code{embeddings="both"}, a named list of two
#'   tibbles (\code{encoder}, \code{decoder}). For several audio files, a
#'   named list with one such result per file. The settings used are saved as
#'   a comment (retrieve with \code{comment()}).
#'
#'   For \code{model = "whisper"}, feature columns are named
#'   \code{f<dimension>_<statistic>}: each hidden dimension is summarised
#'   across the segment's time frames with five statistics -- \code{_mea}
#'   (mean), \code{_med} (median), \code{_var} (variance), \code{_min} and
#'   \code{_max}. For example, the default whisper-medium model has 1024
#'   hidden dimensions, so embedding six diarised segments returns a 6 x 5124
#'   tibble: \code{segment_id}, \code{start_sec}, \code{end_sec} and
#'   \code{speaker}, plus 1024 x 5 = 5120 feature columns
#'   (\code{f00000_mea} ... \code{f01023_max}). The \code{_mea} columns alone
#'   are the closest analogue to standard mean-pooled embeddings.
#'   \code{model = "whispa"} instead returns a single WhiSPA embedding per
#'   segment (no summary statistics).
#'
#' @examples
#' \dontrun{
#' wav_path <- system.file("extdata", "test_short.wav", package = "talk")
#' diar <- talkTranscribeDiarise(audio = wav_path)
#' emb <- talkEmbedSegments(
#'   audio = wav_path,
#'   transcript = diar,
#'   model = "whisper",
#'   embeddings = "encoder"
#' )
#' emb
#' }
#'
#' @seealso \code{\link{talkEmbed}}, \code{\link{talkTranscribeDiarise}}.
#' @importFrom reticulate source_python py_module_available import use_condaenv
#' @importFrom tibble as_tibble
#' @export
talkEmbedSegments <- function(
    audio,
    transcript,
    model = "whisper",
    embeddings = "encoder",
    participant_only = FALSE,
    whisper_model_id = NULL,
    whispa_model_id = "Jarhatz/WhiSPA-V1-Small",
    whispa_repo_path = NULL,
    output_dir = NULL,
    device = NULL,
    condaenv = "talkrpp_condaenv",
    verbose = FALSE){

  time_start <- Sys.time()

  # Validate the input early: type errors on the Python side can be silently
  # swallowed across the callr/reticulate boundary (returning NULL), so fail
  # here with clear messages instead.
  if (!is.character(audio) || length(audio) < 1) {
    stop("`audio` must be a character vector with at least one file path.",
         call. = FALSE)
  }
  missing_files <- audio[!file.exists(audio)]
  if (length(missing_files) > 0) {
    stop("Audio file(s) not found: ", paste(missing_files, collapse = ", "),
         call. = FALSE)
  }

  # Several audio files: `transcript` must be a matching collection (the named
  # list returned by talkTranscribeDiarise(), or a vector of CSV paths). Each
  # audio/transcript pair is processed separately, returning a named list.
  if (length(audio) > 1) {
    transcript_ok <-
      (is.list(transcript) && !is.data.frame(transcript) &&
         length(transcript) == length(audio)) ||
      (is.character(transcript) && length(transcript) == length(audio))
    if (!transcript_ok) {
      stop("For several audio files, `transcript` must be a list of ",
           "transcripts (e.g. from talkTranscribeDiarise()) or a vector of ",
           "CSV paths of the same length as `audio`.", call. = FALSE)
    }
    out <- lapply(seq_along(audio), function(i) {
      talkEmbedSegments(
        audio = audio[[i]], transcript = transcript[[i]], model = model,
        embeddings = embeddings, participant_only = participant_only,
        whisper_model_id = whisper_model_id, whispa_model_id = whispa_model_id,
        whispa_repo_path = whispa_repo_path, output_dir = output_dir,
        device = device, condaenv = condaenv, verbose = verbose
      )
    })
    names(out) <- make.unique(basename(audio))
    return(out)
  }

  # Accept a length-1 list from a multi-file talkTranscribeDiarise() run.
  if (is.list(transcript) && !is.data.frame(transcript) &&
      length(transcript) == 1 && is.data.frame(transcript[[1]])) {
    transcript <- transcript[[1]]
  }
  if (is.character(transcript)) {
    if (length(transcript) != 1 || !file.exists(transcript)) {
      stop("`transcript` must be an existing CSV file path or a data.frame.",
           call. = FALSE)
    }
  } else if (!is.data.frame(transcript)) {
    stop("`transcript` must be an existing CSV file path or a data.frame.",
         call. = FALSE)
  }

  available_envs <- list_talkrpp_envs()
  if (!condaenv %in% available_envs) {
    stop(
      "Conda environment '", condaenv, "' was not found. Available environments: ",
      if (length(available_envs)) paste(available_envs, collapse = ", ") else "(none)",
      ". Run talkrpp_install() to install the default 'talkrpp_condaenv'.",
      call. = FALSE
    )
  }

  embed_py <- system.file("python", "embed.py", package = "talk", mustWork = TRUE)

  # On VIRTUALIZED macOS (CI runners, macOS VMs) torch reports MPS as
  # available, but Metal computes invalid results (e.g. near-constant
  # embeddings across segments) without any error. Detect virtualization
  # (kern.hv_vmm_present is 1 inside a VM) so MPS is only auto-selected on
  # real Apple hardware, and warn if the user explicitly requests it in a VM.
  macos_vm <- FALSE
  if (Sys.info()[["sysname"]] == "Darwin") {
    macos_vm <- tryCatch(
      identical(suppressWarnings(
        system("sysctl -n kern.hv_vmm_present", intern = TRUE, ignore.stderr = TRUE)
      ), "1"),
      error = function(e) FALSE
    )
    if (macos_vm && identical(device, "mps")) {
      warning("device = 'mps' on virtualized macOS (VM/CI runner) can silently ",
              "produce invalid embeddings; use device = 'cpu' instead.",
              call. = FALSE)
    }
  }

  if (!verbose) {
    message(
      "Computing segment embeddings for '", basename(audio), "' ... ",
      "(the first run downloads models; set verbose = TRUE for detailed output)"
    )
  }

  # Run in a subprocess bound to the diarize/embed conda environment, mirroring
  # talkTranscribeDiarise(). This is required because the embed stack (whisnemo,
  # WhiSPA) is installed into `condaenv`, not the main session's Python, and
  # reticulate locks a single interpreter per R session.
  result <- callr::r(
    func = function(audio, transcript, model, embeddings, participant_only,
                    whisper_model_id, whispa_model_id, whispa_repo_path,
                    output_dir, device, condaenv, embed_py, macos_vm) {

      Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")
      Sys.setenv(OMP_NUM_THREADS      = "1")

      reticulate::use_condaenv(condaenv, required = TRUE)
      reticulate::source_python(embed_py)

      # Determine device (CUDA, then Apple-Silicon MPS, else CPU) if not specified
      if (is.null(device)) {
        device <- "cpu"
        if (reticulate::py_module_available("torch")) {
          torch <- reticulate::import("torch")
          if (torch$cuda$is_available()) {
            device <- "cuda"
          } else if (torch$backends$mps$is_available()) {
            if (macos_vm) {
              cat("MPS is reported available, but this macOS session is",
                  "virtualized (VM/CI runner), where Metal can silently produce",
                  "invalid embeddings; using CPU instead.\n")
            } else {
              device <- "mps"
            }
          }
        }
      }

      embed_audio(
        audio_path = audio,
        transcript = transcript,
        model = model,
        embeddings = embeddings,
        device = device,
        participant_only = participant_only,
        whisper_model_id = whisper_model_id,
        whispa_model_id = whispa_model_id,
        whispa_repo_path = whispa_repo_path,
        output_dir = output_dir
      )
    },
    args = list(
      audio = audio,
      transcript = transcript,
      model = model,
      embeddings = embeddings,
      participant_only = participant_only,
      whisper_model_id = whisper_model_id,
      whispa_model_id = whispa_model_id,
      whispa_repo_path = whispa_repo_path,
      output_dir = output_dir,
      device = device,
      condaenv = condaenv,
      embed_py = embed_py,
      macos_vm = macos_vm
    ),
    show = verbose
  )
  if (!is.list(result)) {
    stop("talkEmbedSegments() failed: the Python backend returned no result. ",
         "Re-run with verbose = TRUE to see the backend output.", call. = FALSE)
  }
  if (!verbose && identical(result$status, "success")) {
    message("Done.")
  }

  if (!is.null(result$status) && result$status == "error") {
    hint <- if (!is.null(result$error) && grepl("No module named", result$error)) {
      paste0(
        "\nThe Python environment '", condaenv, "' does not contain the talk ",
        "stack (it may have been created by an older talk version). ",
        "Re-run talkrpp_install() to upgrade it."
      )
    } else {
      ""
    }
    stop("talkEmbedSegments failed: ", result$error, hint)
  }

  emb <- result$embeddings

  seg_comment <- paste(
    "Information about the segment embeddings. talkEmbedSegments: ",
    "model: ", model, " ; ",
    "embeddings: ", embeddings, " ; ",
    "participant_only: ", participant_only, " ; ",
    if (model == "whispa") {
      paste0("whispa_model_id: ", whispa_model_id, " ; ")
    } else {
      paste0("whisper_model_id: ",
             if (is.null(whisper_model_id)) "backend default" else whisper_model_id,
             " ; ")
    },
    "device: ", if (is.null(device)) "auto" else device, " ; ",
    "duration: ", sprintf("%.1f", as.numeric(difftime(Sys.time(), time_start, units = "secs"))), " secs ; ",
    "talk_version: ", packageVersion("talk"), ".",
    sep = ""
  )

  # model="whisper", embeddings="both" returns a named list (encoder/decoder).
  if (is.list(emb) && !is.data.frame(emb)) {
    return(lapply(emb, function(x) {
      x <- tibble::as_tibble(x)
      comment(x) <- seg_comment
      x
    }))
  }

  emb <- tibble::as_tibble(emb)
  comment(emb) <- seg_comment
  emb
}
