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
#'   \code{speaker_role == "participant"} are embedded.
#' @param whisper_model_id (string) Optional override of the Whisper model id.
#' @param whispa_model_id (string) WhiSPA checkpoint (\code{model="whispa"} only).
#' @param whispa_repo_path (string) Optional path to a local WhiSPA clone, used
#'   only if WhiSPA is not pip-installed.
#' @param output_dir (string) Optional directory to also write embedding CSV(s).
#' @param device (string) \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If NULL,
#'   chooses \code{"cuda"} when available, else \code{"cpu"}.
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
#' @importFrom reticulate source_python py_module_available import
#' @importFrom tibble as_tibble
#' @export
talkEmbedSegments <- function(
    audio,
    transcript,
    model = "whisper",
    embeddings = "encoder",
    participant_only = TRUE,
    whisper_model_id = NULL,
    whispa_model_id = "Jarhatz/WhiSPA-V1-Small",
    whispa_repo_path = NULL,
    output_dir = NULL,
    device = NULL){

  reticulate::source_python(system.file("python",
                                        "embed.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))

  # Determine device (CUDA vs CPU) if not specified
  if (is.null(device)) {
    device <- "cpu"
    if (reticulate::py_module_available("torch")) {
      torch <- reticulate::import("torch")
      if (torch$cuda$is_available()) {
        device <- "cuda"
      }
    }
  }

  result <- embed_audio(
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
