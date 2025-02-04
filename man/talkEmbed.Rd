% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/2_1_talkEmbed.R
\name{talkEmbed}
\alias{talkEmbed}
\title{Transform audio recordings to embeddings}
\usage{
talkEmbed(
  talk_filepaths,
  model = "openai/whisper-small",
  audio_transcriptions = "None",
  use_decoder = FALSE,
  tokenizer_parallelism = FALSE,
  model_max_length = "None",
  device = "cpu",
  hg_gated = FALSE,
  hg_token = "",
  trust_remote_code = FALSE,
  logging_level = "warning"
)
}
\arguments{
\item{talk_filepaths}{(string) path to a video file (.wav/) list of audio filepaths, each is embedded separately}

\item{model}{shortcut name for Hugging Face pretained model. Full list https://huggingface.co/transformers/pretrained_models.html}

\item{audio_transcriptions}{(strings) audio_transcriptions : list
(optional) list of audio transcriptions, to be used for Whisper's decoder-based embeddings}

\item{use_decoder}{(boolean) whether to use Whisper's decoder last hidden state representation.
If you just want embeddings from a given audio file where vocal acoustics and sound related harmonics are more important to you, then you should
have `use_decoder`=FALSE.
If you want semantic embeddings which have more language based meaning "baked into" the audio embeddings, you should use `use_decoder`=TRUE.
Note: If you use the decoder’s last hidden state, you must also pass a list of `audio_transcriptions` because the decoder takes in BOTH audio and text.
This you can use the talkTranscribe() function which will return the list of transcripts, which you can pass to the `audio_transcriptions` parameter}

\item{tokenizer_parallelism}{(boolean) whether to use device parallelization during tokenization.}

\item{model_max_length}{(integer) maximum length of the tokenized text}

\item{device}{(string) name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number}

\item{hg_gated}{(boolean) set to True if the model is gated}

\item{hg_token}{(string) the token to access the gated model got in huggingface website}

\item{trust_remote_code}{(boolean) use a model with custom code on the Huggingface Hub.}

\item{logging_level}{(string) Set logging level, options: "critical", "error", "warning", "info", "debug".}
}
\value{
A tibble with embeddings.
}
\description{
Transform audio recordings to embeddings
}
\examples{
# Transform audio recordings in the example dataset:
# voice_data (included in talk-package), to embeddings.
\dontrun{
wav_path <- system.file("extdata/",
"test_short.wav",
package = "talk")

talk_embeddings <- talkEmbed(
wav_path
)
talk_embeddings
}
}
\seealso{
\code{\link{talkText}}.
}
