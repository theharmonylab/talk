

#' Transform audio recordings to embeddings
#'
#' Transforms a whole audio file into a single embedding (one numeric vector
#' per recording). Use \code{talkEmbed()} when you want one representation
#' per recording -- for example to predict a person-level measure from an
#' entire spoken response (e.g. with the text-package's train functions). It
#' is faster than \code{\link{talkEmbedSegments}} and needs no diarisation.
#' Use \code{talkEmbedSegments()} instead when you need one embedding per
#' speaker turn -- for example to analyse each speaker in a conversation
#' separately (it requires a diarised transcript from
#' \code{\link{talkTranscribeDiarise}}).
#'
#' @param talk_filepaths (string) path to a video file (.wav/) list of audio filepaths, each is embedded separately
#' @param model shortcut name for Hugging Face pretained model. Full list https://huggingface.co/transformers/pretrained_models.html
#' @param audio_transcriptions  (strings) audio_transcriptions : list
#' (optional) list of audio transcriptions, to be used for Whisper's decoder-based embeddings
#' @param use_decoder (boolean) whether to use Whisper's decoder last hidden state representation.
#' If you just want embeddings from a given audio file where vocal acoustics and sound related harmonics are more important to you, then you should
#' have `use_decoder`=FALSE.
#' If you want semantic embeddings which have more language based meaning "baked into" the audio embeddings, you should use `use_decoder`=TRUE.
#' Note: If you use the decoder’s last hidden state, you must also pass a list of `audio_transcriptions` because the decoder takes in BOTH audio and text.
#' This you can use the talkTranscribe() function which will return the list of transcripts, which you can pass to the `audio_transcriptions` parameter
#' @param tokenizer_parallelism  (boolean) whether to use device parallelization during tokenization.
#' @param device (string) name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
#' @param model_max_length (integer) maximum length of the tokenized text
#' @param hg_gated (boolean) set to True if the model is gated
#' @param hg_token (string) the token to access the gated model got in huggingface website
#' @param trust_remote_code (boolean) use a model with custom code on the Huggingface Hub.
#' @param logging_level (string) Set logging level, options: "critical", "error", "warning", "info", "debug".
#' @return A tibble with embeddings. The settings used are saved as a comment (retrieve with
#'   \code{comment()}).
#' @examples
#' # Transform audio recordings in the example dataset:
#' # voice_data (included in talk-package), to embeddings.
#' \dontrun{
#' wav_path <- system.file("extdata/",
#' "test_short.wav",
#' package = "talk")
#'
#' talk_embeddings <- talkEmbed(
#' wav_path
#' )
#' talk_embeddings
#' }
#' @seealso \code{\link{talkText}}.
#' @importFrom reticulate source_python
#' @importFrom tibble as_tibble
#' @export
talkEmbed <- function(
    talk_filepaths,
    model = "openai/whisper-small",
    audio_transcriptions = "None",
    use_decoder = FALSE,
    tokenizer_parallelism = FALSE,
    model_max_length = "None",
    device = 'cpu',
    hg_gated = FALSE,
    hg_token = "",
    trust_remote_code = FALSE,
    logging_level = 'warning'){

  time_start <- Sys.time()

  reticulate::source_python(system.file("python",
                                        "huggingface_Interface4.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))


  embeddings <- hgTransformerGetEmbedding(
    audio_filepaths = talk_filepaths,
    audio_transcriptions = audio_transcriptions,
    model = model,
    use_decoder = use_decoder,
    tokenizer_parallelism = tokenizer_parallelism,
    model_max_length = model_max_length,
    device = device,
    hg_gated = hg_gated,
    hg_token = hg_token,
    trust_remote_code = trust_remote_code,
    logging_level = logging_level
  )

  embeddings <- embeddings[[1]]

  emb_tibble <- tibble::as_tibble(
    t(embeddings), # Transpose the vector into a single-row matrix
    .name_repair = ~ paste0("Dim", seq_along(embeddings)) # Assign column names
  )

  comment(emb_tibble) <- paste(
    "Information about the embeddings. talkEmbed: ",
    "model: ", model, " ; ",
    "use_decoder: ", use_decoder, " ; ",
    "device: ", device, " ; ",
    "duration: ", sprintf("%.1f", as.numeric(difftime(Sys.time(), time_start, units = "secs"))), " secs ; ",
    "talk_version: ", packageVersion("talk"), ".",
    sep = ""
  )

  return(emb_tibble)
}


