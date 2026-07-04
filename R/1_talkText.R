#' Transform audio recordings to embeddings
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
#' @return A tibble with transcriptions. The settings used are saved as a comment (retrieve with \code{comment()}).
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
