library(testthat)
library(talk)

# Runs last: removes the single environment created for the test suite (see
# test_1_install.R), leaving the machine clean.

test_that("talkrpp_uninstall removes the test environment", {
  skip_on_cran()

  envname <- "talk_test_env"

  # Only attempt removal if the environment exists (install may have been skipped).
  if (envname %in% talk::list_talkrpp_envs()) {
    talk::talkrpp_uninstall(prompt = FALSE, envname = envname)
    testthat::expect_false(envname %in% talk::list_talkrpp_envs())
  } else {
    testthat::skip(paste0(envname, " not present; nothing to remove"))
  }
})
