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
#' @param audio (string) Path to a single audio file (e.g. \code{.wav}).
#' @param transcript (string or data.frame) Diarized transcript: either a path
#'   to a CSV, or a data.frame such as the \code{transcript} returned by
#'   \code{\link{talkTranscribeDiarise}}. Must contain per-segment start/end
#'   timestamps, and a \code{speaker_role} column if \code{participant_only=TRUE}.
#' @param model (string) \code{"whisper"} or \code{"whispa"}.
#' @param embeddings (string) For \code{model="whisper"}: \code{"encoder"},
#'   \code{"decoder"}, or \code{"both"}. Ignored for \code{model="whispa"}.
#' @param participant_only (logical) If TRUE, only segments with
#'   \code{speaker_role == "participant"} are embedded. Defaults to FALSE
#'   because the \code{transcript} returned by \code{\link{talkTranscribeDiarise}}
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
#'
#' @return A tibble of segment-level embeddings (one row per segment), or, for
#'   \code{model="whisper"} with \code{embeddings="both"}, a named list of two
#'   tibbles (\code{encoder}, \code{decoder}).
#'
#' @examples
#' \dontrun{
#' wav_path <- system.file("extdata", "test_short.wav", package = "talk")
#' diar <- talkTranscribeDiarise(audio = wav_path)
#' emb <- talkEmbedSegments(
#'   audio = wav_path,
#'   transcript = diar$transcript,
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
    condaenv = "talkrpp_condaenv"){

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
    show = TRUE
  )

  if (!is.null(result$status) && result$status == "error") {
    stop("talkEmbedSegments failed: ", result$error)
  }

  emb <- result$embeddings

  # model="whisper", embeddings="both" returns a named list (encoder/decoder).
  if (is.list(emb) && !is.data.frame(emb)) {
    return(lapply(emb, tibble::as_tibble))
  }

  return(tibble::as_tibble(emb))
}
