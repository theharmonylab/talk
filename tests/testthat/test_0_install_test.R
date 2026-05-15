library(testthat)
library(talk)

test_that("installing talk", {
  skip_on_cran()

  talk::talkrpp_install(prompt = FALSE,
                        envname = "talk_test")

  talk::talkrpp_initialize(talkEmbed_test = TRUE,
                           save_profile = FALSE,
                           prompt = FALSE,
                           condaenv = "talk_test",
                           refresh_settings = TRUE)

  wav_path <- system.file("extdata/",
                          "test_short.wav",
                          package = "talk")

  emb_test <- talk::talkEmbed(
    talk_filepaths = wav_path,
    model = "openai/whisper-tiny"
  )

  testthat::expect_equal(emb_test$Dim1,
                         -0.2030126, tolerance = 0.0001)
  testthat::expect_equal(emb_test$Dim2,
                         -1.008844, tolerance = 0.0001)
  testthat::expect_equal(emb_test$Dim3,
                         0.897202, tolerance = 0.0001)

  text_test <- talk::talkText(
    talk_filepaths = wav_path,
    model = "openai/whisper-tiny"
  )

  testthat::expect_equal(text_test[1],
                         " Hello.")

  if (Sys.info()["sysname"] == "Darwin" | Sys.info()["sysname"] == "Windows") {
    talkrpp_uninstall(prompt = FALSE,
                      envname = "talk_test")
  }

})
