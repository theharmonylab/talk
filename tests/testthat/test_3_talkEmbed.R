library(testthat)
library(talk)

# talkEmbed() maps a whole audio file to a single embedding (one numeric vector
# per file). Uses the single test environment installed in test_1_install.R; the
# test is skipped if it is not present. All tests share this one environment, so
# it can be initialized directly in the session (no callr subprocess needed).
# Expected Dim values verified in the single test environment.

test_that("talkEmbed returns embeddings for an audio file", {
  skip_on_cran()

  envname <- "talk_test_env"
  skip_if_not(
    envname %in% tryCatch(reticulate::conda_list()$name, error = function(e) character(0)),
    paste0(envname, " not installed")
  )

  wav_path <- system.file("extdata", "test_short.wav", package = "talk")

  talk::talkrpp_initialize(
    condaenv         = envname,
    prompt           = FALSE,
    save_profile     = FALSE,
    refresh_settings = TRUE
  )

  emb_test <- talk::talkEmbed(
    talk_filepaths = wav_path,
    model          = "openai/whisper-tiny"
  )

  testthat::expect_s3_class(emb_test, "data.frame")
  testthat::expect_equal(emb_test$Dim1, -0.1984932, tolerance = 0.0001)
  testthat::expect_equal(emb_test$Dim2, -1.0071950, tolerance = 0.0001)
  testthat::expect_equal(emb_test$Dim3,  0.8974844, tolerance = 0.0001)
  testthat::expect_true(grepl("talkEmbed", comment(emb_test), fixed = TRUE))

  # Decoder embeddings: requires transcriptions (audio AND text); this path
  # was broken until v0.5, so guard it against regression.
  txt <- talk::talkText(talk_filepaths = wav_path, model = "openai/whisper-tiny")
  emb_dec <- talk::talkEmbed(
    talk_filepaths       = wav_path,
    model                = "openai/whisper-tiny",
    use_decoder          = TRUE,
    audio_transcriptions = txt
  )
  testthat::expect_s3_class(emb_dec, "data.frame")
  testthat::expect_equal(dim(emb_dec), c(1L, 384L))
  testthat::expect_true(all(is.finite(unlist(emb_dec))))
})
