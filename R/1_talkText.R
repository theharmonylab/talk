#' Transcribe audio recordings to text (speech-to-text)
#'
#' Transcribes audio recordings to plain text using Whisper. Use
#' \code{talkText()} (or its alias \code{talkTranscribe()}) when you only
#' need \emph{what} was said: it is simpler and considerably faster than
#' \code{\link{talkTextDiarise}} / \code{\link{talkTranscribeDiarise}},
#' which additionally identify \emph{who} speaks \emph{when} and therefore
#' run the full speaker-diarisation pipeline. Typical uses are recordings
#' with a single speaker, or analyses where speaker identity does not
#' matter; for conversations where speaker turns are needed, use
#' \code{talkTextDiarise()} instead.
#'
#' @param talk_filepaths (string) Path to a video file (.wav/) list of audio filepaths, each is embedded separately
#' @param model shortcut name for Hugging Face pretained model. Full list https://huggingface.co/transformers/pretrained_models.html
# @param language (string) abbreviation language such as "en".
#' @param device (string) name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
#' @param tokenizer_parallelism  (boolean) whether to use device parallelization during tokenization.
#' @param hg_gated (boolean) Set to True if the model is gated
#' @param hg_token (string) The token to access the gated model got in huggingface website
#' @param trust_remote_code (boolean) use a model with custom code on the Huggingface Hub.
#' @param logging_level (string) Set logging level, options: "critical", "error", "warning", "info", "debug".
#' @return A tibble with one row per audio file and columns \code{file_path}
#'   and \code{transcription}. If a file fails to transcribe, its row gets
#'   \code{NA} in \code{transcription} (with a warning), so positions always
#'   match the input files. The settings used are saved as a comment
#'   (retrieve with \code{comment()}).
#' @examples
#' # Transform audio recordings in text:
#' # voice_data (included in talk-package), to embeddings.
#' \dontrun{
#' wav_path <- system.file("extdata/",
#' "test_short.wav",
#' package = "talk")
#' # Get transcription
#' talk_embeddings <- talkText(
#' wav_path
#' )
#' talk_embeddings
#' }
#' @seealso \code{\link{talkText}}.
#' @importFrom reticulate source_python
#' @importFrom tibble tibble
#' @export
talkText <- function(
    talk_filepaths = talk_filepaths,
    model = "openai/whisper-small",
    device = 'cpu',
    tokenizer_parallelism = FALSE,
    hg_gated = FALSE,
    hg_token = "",
    trust_remote_code = FALSE,
    logging_level = 'warning'#,
    #language = 'en'
    ){

  time_start <- Sys.time()

  # Validate input early, before any Python is involved.
  if (!is.character(talk_filepaths) || length(talk_filepaths) < 1) {
    stop("`talk_filepaths` must be a character vector with at least one file path.",
         call. = FALSE)
  }
  missing_files <- talk_filepaths[!file.exists(talk_filepaths)]
  if (length(missing_files) > 0) {
    stop("Audio file(s) not found: ", paste(missing_files, collapse = ", "),
         call. = FALSE)
  }

  reticulate::source_python(system.file("python",
                                        "huggingface_Interface4.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))


  text <- hgTransformerTranscribe(
    audio_filepaths = talk_filepaths,
    model = model,
    device = device,
    tokenizer_parallelism = tokenizer_parallelism,
    hg_gated = hg_gated,
    hg_token = hg_token,
    trust_remote_code = trust_remote_code,
    logging_level = logging_level#,
    #language = language
  )

  # The backend returns one element per input file (None -> NULL for a file
  # that failed to transcribe); align them into a tibble with NA for failures.
  transcriptions <- vapply(
    as.list(text),
    function(x) if (is.null(x)) NA_character_ else as.character(x),
    character(1)
  )
  if (length(transcriptions) != length(talk_filepaths)) {
    stop("The Python backend returned ", length(transcriptions),
         " transcription(s) for ", length(talk_filepaths), " file(s).",
         call. = FALSE)
  }
  failed <- talk_filepaths[is.na(transcriptions)]
  if (length(failed) > 0) {
    warning("Transcription failed for: ", paste(failed, collapse = ", "),
            " (NA returned at those positions; the Python error was printed ",
            "above).", call. = FALSE)
  }
  text <- tibble::tibble(
    file_path = talk_filepaths,
    transcription = transcriptions
  )

  comment(text) <- paste(
    "Information about the transcription. talkText: ",
    "model: ", model, " ; ",
    "device: ", device, " ; ",
    "duration: ", sprintf("%.1f", as.numeric(difftime(Sys.time(), time_start, units = "secs"))), " secs ; ",
    "talk_version: ", packageVersion("talk"), ".",
    sep = ""
  )

  return(text)
}

#' @rdname talkText
#' @export
talkTranscribe <- talkText
