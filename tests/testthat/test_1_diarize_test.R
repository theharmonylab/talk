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

  result <- talk::talkTranscribeDiarise(audio = wav_path1)

  testthat::expect_equal(result$status, "success")

  testthat::expect_equal(result$transcript$message[1], " Hello, how are you doing?")


  # ---- Segment-level embeddings (talkEmbedSegments) ----
  # Reuse the diarised transcript above and the same diarize conda env
  # (talkrpp_diarize_condaenv), which also holds the embed stack
  # (whisnemo[embed], WhiSPA). Use whisper-tiny for a fast test.
  # participant_only = FALSE because the diarised transcript has no
  # "speaker_role" column (which the participant_only = TRUE path requires).

  emb <- talk::talkEmbedSegments(
    audio = wav_path1,
    transcript = result$transcript,   # accepts the diarised output directly, or a CSV path
    model = "whisper",                # or "whispa"
    embeddings = "encoder",           # "encoder" / "decoder" / "both" (whisper only)
    participant_only = FALSE,
    whisper_model_id = "openai/whisper-tiny"
  )

  testthat::expect_equal(emb$f00000_mea[1:5],
                         c(-0.7631922, -0.5184633, -0.3283089, -0.8475273, -0.7014528),
                         tolerance = .0001)

  # One embedding row per diarised segment, with segment metadata + features.
  testthat::expect_s3_class(emb, "data.frame")
  testthat::expect_equal(nrow(emb), nrow(result$transcript))
  testthat::expect_true(
    all(c("segment_id", "start_sec", "end_sec", "speaker") %in% colnames(emb))
  )
  testthat::expect_gt(ncol(emb), 4L)   # metadata columns + embedding features

  # embeddings = "both" returns a named list of two tibbles (encoder + decoder).
  emb_both <- talkEmbedSegments(
    audio = wav_path1,
    transcript = result$transcript,
    model = "whisper",
    embeddings = "both",
    participant_only = FALSE,
    whisper_model_id = "openai/whisper-tiny"
  )

  testthat::expect_equal(emb_both$encoder$f00000_mea[1:5],
                         c(-0.7631922, -0.5184633, -0.3283089, -0.8475273, -0.7014528),
                         tolerance = .0001)

  testthat::expect_equal(emb_both$decoder$f00000_mea[1:5],
                         c(-4.885945,  2.896193, -4.106683,  1.994434, -2.674213),
                         tolerance = .0001)

  testthat::expect_type(emb_both, "list")
  testthat::expect_false(is.data.frame(emb_both))
  testthat::expect_named(emb_both, c("encoder", "decoder"))
  testthat::expect_s3_class(emb_both$encoder, "data.frame")
  testthat::expect_s3_class(emb_both$decoder, "data.frame")





})
