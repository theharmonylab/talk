library(testthat)

test_check("talk")

if (identical(Sys.getenv("NOT_CRAN"), "true")) { # emulates `testthat:::on_cran()`
  if (requireNamespace("xml2")) {
    test_check("talk",
      reporter = MultiReporter$new(reporters = list(
        JunitReporter$new(file = "test-results.xml"),
        CheckReporter$new()
      ))
    )
  } else {
    test_check("talk")
  }
}
