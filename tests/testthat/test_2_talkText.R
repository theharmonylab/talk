library(testthat)
library(talk)

# talkText() transcribes an audio file to text (speech-to-text). Uses the
# single test environment installed in test_1_install.R; the test is skipped if
# it is not present. All tests share this one environment, so it can be
# initialized directly in the session (no callr subprocess needed).

test_that("talkText transcribes audio to text", {
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

  text_test <- talk::talkText(
    talk_filepaths = wav_path,
    model          = "openai/whisper-tiny"
  )

  testthat::expect_equal(text_test[1], " Hello.")
  testthat::expect_true(grepl("talkText", comment(text_test), fixed = TRUE))
})
