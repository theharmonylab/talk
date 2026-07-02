# Global settings (these avoids the note (warning) "no visible binding for global variable ‘XXX’
# https://community.rstudio.com/t/how-to-solve-no-visible-binding-for-global-variable-note/28887
utils::globalVariables(c(
  # GENERAL
  "find_talkrpp", "hgTransformerGetEmbedding",
  "hgTransformerTranscribe",
  # Python functions sourced at runtime via reticulate::source_python()
  "diarize_audio", "embed_audio"

))
