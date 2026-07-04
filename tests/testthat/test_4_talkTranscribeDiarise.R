library(testthat)
library(talk)

# talkTranscribeDiarise() transcribes and diarises a recording (splitting the
# transcript by speaker). Uses the single test environment installed in
# test_1_install.R; the test is skipped if it is not present. The function
# executes in a subprocess internally, so no callr wrapping is needed here.

test_that("talkTranscribeDiarise transcribes and diarises audio", {
  skip_on_cran()

  envname <- "talk_test_env"
  skip_if_not(
    envname %in% tryCatch(reticulate::conda_list()$name, error = function(e) character(0)),
    paste0(envname, " not installed")
  )

  wav_path <- system.file("extdata", "test_diarise.wav", package = "talk")

  # OMP/KMP environment variables are set by talk's .onAttach() (macOS) and
  # inherited by the diarise subprocess, so they need not be set here.
  transcript <- talk::talkTranscribeDiarise(audio = wav_path, condaenv = envname)

  testthat::expect_s3_class(transcript, "data.frame")
  testthat::expect_true(all(c("speaker", "start_timestamp", "end_timestamp", "message")
                            %in% colnames(transcript)))
  testthat::expect_equal(transcript$message[1], " Hello, how are you doing?")
  testthat::expect_true(grepl("talkTranscribeDiarise", comment(transcript), fixed = TRUE))
})
