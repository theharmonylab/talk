library(testthat)
library(talk)

# Fast unit tests for helpers that need neither the conda environment nor any
# model downloads. These run in every coverage job, independently of the
# heavy install-based tests.

test_that("ffmpeg_install_instruction returns an OS-appropriate string", {
  instr <- talk:::ffmpeg_install_instruction()
  testthat::expect_type(instr, "character")
  testthat::expect_length(instr, 1)
  testthat::expect_true(nzchar(instr))
})

test_that("ensure_ffmpeg_on_path returns a logical scalar", {
  res <- talk:::ensure_ffmpeg_on_path()
  testthat::expect_type(res, "logical")
  testthat::expect_length(res, 1)
  # on machines with ffmpeg installed it must also report TRUE and make
  # ffmpeg resolvable
  if (res) testthat::expect_true(nzchar(Sys.which("ffmpeg")))
})

test_that("linux_missing_dev_libs returns a character vector", {
  res <- talk:::linux_missing_dev_libs()
  testthat::expect_type(res, "character")
  if (Sys.info()[["sysname"]] != "Linux") testthat::expect_length(res, 0)
})

test_that("install_linux_dev_libs_if_needed is a safe no-op when nothing is missing", {
  # On non-Linux this returns immediately; on Linux CI the dev libraries are
  # installed by the workflow, so it is a no-op there too.
  testthat::expect_no_error(talk:::install_linux_dev_libs_if_needed(prompt = TRUE))
})

test_that("check_talk_system_dependencies returns a structured result", {
  res <- suppressWarnings(suppressMessages(
    talk:::check_talk_system_dependencies(verbose = FALSE)
  ))
  testthat::expect_type(res, "list")
  testthat::expect_named(res, c("os", "ffmpeg", "missing", "summary_lines"))
  testthat::expect_type(res$ffmpeg, "logical")
  testthat::expect_type(res$missing, "character")
  testthat::expect_true(any(grepl("talk system dependencies", res$summary_lines)))
})

test_that("install_rust_if_needed handles the safe paths", {
  # With rustc installed: early return ("already installed").
  # Without rustc, prompt = TRUE in a non-interactive session: menu() returns
  # 0, which is treated as "No" -> cancelled. Neither path installs anything.
  testthat::expect_no_error(
    suppressMessages(talk:::install_rust_if_needed(prompt = TRUE))
  )
})

test_that("list_talkrpp_envs returns unique environment names", {
  res <- talk::list_talkrpp_envs()
  testthat::expect_type(res, "character")
  testthat::expect_equal(anyDuplicated(res), 0L)
})

test_that("find_talkrpp_env returns a logical scalar", {
  res <- talk::find_talkrpp_env()
  testthat::expect_type(res, "logical")
  testthat::expect_length(res, 1)
})

test_that("colourise returns the input text (possibly wrapped in ANSI codes)", {
  res <- talk:::colourise("hello", fg = "blue")
  testthat::expect_type(res, "character")
  testthat::expect_true(grepl("hello", res, fixed = TRUE))
  # unknown terminals return the text unchanged
  withr_old <- Sys.getenv("TERM"); Sys.setenv(TERM = "dumb")
  testthat::expect_identical(talk:::colourise("plain", fg = "blue"), "plain")
  Sys.setenv(TERM = withr_old)
})

test_that("talkrpp_uninstall errors on a nonexistent environment", {
  skip_if_not(
    !is.null(tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)),
    "No conda available"
  )
  testthat::expect_error(
    talk::talkrpp_uninstall(prompt = FALSE, envname = "nonexistent_talk_env_xyz"),
    regexp = "not found"
  )
})

test_that("talkEmbedSegments fails cleanly on a nonexistent conda environment", {
  skip_on_cran()
  skip_if_not(
    !is.null(tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)),
    "No conda available"
  )
  wav <- system.file("extdata", "test_short.wav", package = "talk")
  transcript <- data.frame(
    speaker = "Speaker 0", start_timestamp = "00:00:00.000",
    end_timestamp = "00:00:01.000", message = "Hello.",
    stringsAsFactors = FALSE
  )
  testthat::expect_error(
    talk::talkEmbedSegments(
      audio = wav, transcript = transcript,
      condaenv = "nonexistent_talk_env_xyz"
    )
  )
})

test_that("talkTranscribeDiarise fails cleanly on a nonexistent conda environment", {
  skip_on_cran()
  skip_if_not(
    !is.null(tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)),
    "No conda available"
  )
  skip_if_not(talk:::ensure_ffmpeg_on_path(), "No ffmpeg available")
  wav <- system.file("extdata", "test_diarise.wav", package = "talk")
  testthat::expect_error(
    talk::talkTranscribeDiarise(audio = wav, condaenv = "nonexistent_talk_env_xyz")
  )
})
