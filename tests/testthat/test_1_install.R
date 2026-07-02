library(testthat)
library(talk)

# Installs a single, freshly-made environment used by the whole test suite.
# The diarize environment carries the full stack (torch, transformers, whisper,
# whisnemo[embed], WhiSPA), so talkText() / talkEmbed() run in it too. The
# environment is removed again at the end of the suite (see test_6_uninstall.R).

test_that("talkrpp_install creates the test environment", {
  skip_on_cran()

  envname <- "talk_test_env"

  talk::talkrpp_install(
    rpp_version = "talk_diarize",
    envname     = envname,
    prompt      = FALSE
  )

  # list_talkrpp_envs() lists it among the available environments (de-duplicated)
  envs <- talk::list_talkrpp_envs()
  testthat::expect_type(envs, "character")
  testthat::expect_equal(anyDuplicated(envs), 0L)
  testthat::expect_true(envname %in% envs)

  # find_talkrpp_env() reports environment presence as a single logical
  testthat::expect_type(talk::find_talkrpp_env(), "logical")
  testthat::expect_length(talk::find_talkrpp_env(), 1)
})
