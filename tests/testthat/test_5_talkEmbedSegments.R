library(testthat)
library(talk)

# talkEmbedSegments() produces one embedding per diarised segment. Uses the
# single test environment installed in test_1_install.R, which holds the embed
# stack (whisnemo[embed], WhiSPA); the test is skipped if it is not present. We
# first diarise to obtain a transcript, then embed each segment.

test_that("talkEmbedSegments returns segment-level embeddings", {
  skip_on_cran()
  # Requires diarisation, which is skipped on the Windows CI runner (a runner
  # model-download quirk, not a platform limitation; see
  # test_4_talkTranscribeDiarise.R).
  skip_on_os("windows")

  envname <- "talk_test_env"
  skip_if_not(
    envname %in% tryCatch(reticulate::conda_list()$name, error = function(e) character(0)),
    paste0(envname, " not installed")
  )

  wav_path <- system.file("extdata", "test_diarise.wav", package = "talk")

  # OMP/KMP environment variables are set by talk's .onAttach() (macOS) and
  # inherited by the diarise/embed subprocess, so they need not be set here.
  diar <- talk::talkTranscribeDiarise(audio = wav_path, condaenv = envname)
  testthat::expect_equal(diar$status, "success")

  # Whisper encoder embeddings, one row per diarised segment.
  # participant_only = FALSE because the diarised transcript has no
  # "speaker_role" column (which the participant_only = TRUE path requires).
  emb <- talk::talkEmbedSegments(
    audio            = wav_path,
    transcript       = diar$transcript,
    model            = "whisper",
    embeddings       = "encoder",
    participant_only = FALSE,
    whisper_model_id = "openai/whisper-tiny",
    condaenv         = envname
  )

  testthat::expect_s3_class(emb, "data.frame")
  testthat::expect_equal(nrow(emb), nrow(diar$transcript))
  testthat::expect_true(
    all(c("segment_id", "start_sec", "end_sec", "speaker") %in% colnames(emb))
  )
  testthat::expect_gt(ncol(emb), 4L)
  testthat::expect_equal(
    emb$f00000_mea[1:5],
    c(-0.7631922, -0.5184633, -0.3283089, -0.8475273, -0.7014528),
    tolerance = 0.0001
  )

  # embeddings = "both" returns a named list of two data frames.
  emb_both <- talk::talkEmbedSegments(
    audio            = wav_path,
    transcript       = diar$transcript,
    model            = "whisper",
    embeddings       = "both",
    participant_only = FALSE,
    whisper_model_id = "openai/whisper-tiny",
    condaenv         = envname
  )

  testthat::expect_type(emb_both, "list")
  testthat::expect_false(is.data.frame(emb_both))
  testthat::expect_named(emb_both, c("encoder", "decoder"))
  testthat::expect_s3_class(emb_both$encoder, "data.frame")
  testthat::expect_s3_class(emb_both$decoder, "data.frame")
})
