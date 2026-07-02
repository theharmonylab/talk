library(testthat)
library(talk)

# talkTranscribeDiarise() transcribes and diarises a recording (splitting the
# transcript by speaker). Uses the single test environment installed in
# test_1_install.R; the test is skipped if it is not present. The function
# executes in a subprocess internally, so no callr wrapping is needed here.

test_that("talkTranscribeDiarise transcribes and diarises audio", {
  skip_on_cran()
  # Skipped on the Windows CI runner only: the hosted runner fails at the NeMo
  # model download with an OpenSSL "[ASN1: NOT_ENOUGH_DATA]" error (an
  # environment/network quirk of the fresh runner, not a code limitation).
  # Diarisation itself runs on real Windows machines; talkText()/talkEmbed()
  # are tested on the Windows runner as usual.
  skip_on_os("windows")

  envname <- "talk_test_env"
  skip_if_not(
    envname %in% tryCatch(reticulate::conda_list()$name, error = function(e) character(0)),
    paste0(envname, " not installed")
  )

  wav_path <- system.file("extdata", "test_diarise.wav", package = "talk")

  # OMP/KMP environment variables are set by talk's .onAttach() (macOS) and
  # inherited by the diarise subprocess, so they need not be set here.
  result <- talk::talkTranscribeDiarise(audio = wav_path, condaenv = envname)

  testthat::expect_equal(result$status, "success")
  testthat::expect_s3_class(result$transcript, "data.frame")
  testthat::expect_equal(result$transcript$message[1], " Hello, how are you doing?")
})
