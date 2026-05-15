library(testthat)
library(talk)

test_that("installing and running talk diarize", {
  skip_on_cran()

  # Install the diarize environment (separate from the standard talk env).
  # talkrpp_initialize() is NOT needed before talkTranscribeDiarise() —
  # it runs in a subprocess that initialises the diarize condaenv automatically.
  talk::talkrpp_install(rpp_version = "talk_diarize",
                        envname = "talkrpp_diarize_condaenv",
                        prompt = FALSE)

  wav_path1 <- system.file("extdata/",
                            "test_diarise.wav",
                            package = "talk")

  Sys.setenv(OMP_NUM_THREADS = "1")
  Sys.setenv(OMP_MAX_ACTIVE_LEVELS = "1")
  Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

  result <- talkTranscribeDiarise(audio = wav_path1)

  testthat::expect_equal(result$status, "success")

})
