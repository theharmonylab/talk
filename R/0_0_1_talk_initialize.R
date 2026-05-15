# modified from spacyr: https://github.com/quanteda/spacyr/tree/master/R

#' Initialize talk required python packages
#'
#' Initialize talk required python packages to call from R.
#' @return NULL
#' @param python_executable the full path to the Python executable, for which
#'   talk required python packages is installed.
#' @param ask logical; if \code{FALSE}, use the first talk required python packages installation found;
#'   if \code{TRUE}, list available talk required python packages installations and prompt the user for
#'   which to use. If another (e.g. \code{python_executable}) is set, then this
#'   value will always be treated as \code{FALSE}.
#' @param virtualenv set a path to the Python virtual environment with talk required python packages
#'   installed Example: \code{virtualenv = "~/myenv"}
#' @param condaenv set a path to the anaconda virtual environment with talk required python packages
#'   installed Example: \code{condalenv = "myenv"}
#' @param check_env logical; check whether conda/virtual environment generated
#'   by \code{talkrpp_install()} exists
#' @param refresh_settings logical; if \code{TRUE}, talk will ignore the saved
#'   settings in the profile and initiate a search of new settings.
#' @param save_profile logical; if \code{TRUE}, the current talk required python packages setting will
#'   be saved for the future use.
#' @param talkEmbed_test logical; Test whether function (talkEmbed) that requires python packages works.
#' @param prompt logical; asking whether user wants to set the environment as default.
#' @export
talkrpp_initialize <- function(python_executable = NULL,
                               virtualenv = NULL,
                               condaenv = "talkrpp_condaenv",
                               ask = FALSE,
                               refresh_settings = FALSE,
                               save_profile = FALSE,
                               check_env = TRUE,
                               talkEmbed_test = FALSE,
                               prompt = TRUE) {
  set_talkrpp_python_option(
    python_executable,
    virtualenv,
    condaenv,
    check_env,
    refresh_settings,
    ask
  )

  ## check settings and start reticulate python
  settings <- check_talkrpp_python_options()
  if (!is.null(settings)) {
    tryCatch({
      if (settings$key == "talkrpp_python_executable") {
        reticulate::use_python(settings$val, required = TRUE)
      } else if (settings$key == "talkrpp_virtualenv") {
        reticulate::use_virtualenv(settings$val, required = TRUE)
      } else if (settings$key == "talkrpp_condaenv") {
        reticulate::use_condaenv(settings$val, required = TRUE)
      }
    }, error = function(e) {
      if (grepl("already been initialized", conditionMessage(e))) {
        stop(
          "A different Python has already been initialized in this R session.\n",
          "Please restart R, then call talkrpp_initialize() before running any ",
          "other code that uses Python.\n",
          "(Initialized: ", reticulate::py_config()$python, ")\n",
          "(Requested:   ", settings$val, ")",
          call. = FALSE
        )
      }
      stop(e)
    })
  }

  # Importing this here may start importing necessary packages
  reticulate::source_python(system.file("python",
    "huggingface_Interface4.py",
    package = "talk",
    mustWork = TRUE
  ))

  message(colourise(
    "\nSuccessfully initialized talk required python packages.\n",
    fg = "green", bg = NULL
  ))
  settings <- check_talkrpp_python_options()

  settings_talk <- paste('Python options: \n type = "', settings$key,
    '", \n name = "', settings$val, '".',
    sep = ""
  )

  message(colourise(settings_talk,
    fg = "blue", bg = NULL
  ))


  options("talkrpp_initialized" = TRUE)

  if (save_profile == TRUE) {
    save_talkrpp_options(settings$key, settings$val, prompt = prompt)
  }

  if (talkEmbed_test == TRUE) {

    wav_path <- system.file("extdata/",
                            "test_short.wav",
                            package = "talk")

    talkEmbed(
      talk_filepaths = wav_path,
      model = "openai/whisper-tiny"
    )
  }
}

#' Find talk required python packages env
#'
#' check whether conda/virtual environment for talk required python pacakges exists
#' @export
#'
#' @keywords internal
find_talkrpp_env <- function() {
  if (is.null(tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL))) {
    return(FALSE)
  }
  found <- if ("talkrpp_condaenv" %in% reticulate::conda_list(conda = "auto")$name) {
    TRUE
  } else if (file.exists(file.path("~/.virtualenvs", "talkrpp_virtualenv", "bin", "activate"))) {
    TRUE
  } else {
    FALSE
  }
  return(found)
}


check_talkrpp_model <- function(py_exec) { ### , model
  options(warn = -1)
  py_exist <- if (is_windows()) {
    if (py_exec %in% system2("where", "python", stdout = TRUE)) {
      py_exec
    } else {
      NULL
    }
  } else {
    system2("which", py_exec, stdout = TRUE)
  }

  if (length(py_exist) == 0) {
    stop(py_exec, " is not a python executable")
  }
  tryCatch({
    sys_message <- "see error in talk_initialize row 235"
    # system2(py_exec, c(sprintf("-c \"import texrpp; talk.load('%s'); print('OK')\"", model)),
    #        stderr = TRUE, stdout = TRUE)
  })
  options(warn = 0)
  return(paste(sys_message, collapse = " "))
}


set_talkrpp_python_option <- function(python_executable = NULL,
                                      virtualenv = NULL,
                                      condaenv = NULL,
                                      check_env = TRUE,
                                      refresh_settings = FALSE,
                                      ask = NULL) {
  if (refresh_settings) clear_talkrpp_options()

  if (!is.null(check_talkrpp_python_options())) {
    settings <- check_talkrpp_python_options()

    message_talk1 <- paste("talkrpp python option is already set, talk will use: ",
      sub("talkrpp_", "", settings$key), ' = "', settings$val, '"',
      sep = ""
    )

    message(colourise(message_talk1,
      fg = "blue", bg = NULL
    ))
    # a user can specify only one
  } else if (sum(!is.null(c(python_executable, virtualenv, condaenv))) > 1) {
    stop(paste(
      "Too many python environments are specified, please select only one",
      "from python_executable, virtualenv, and condaenv"
    ))
    # give warning when nothing is specified
  } else if (sum(!is.null(c(python_executable, virtualenv, condaenv))) == 1) {
    if (!is.null(python_executable)) {
      if (check_talkrpp_model(python_executable) != "OK") {
        stop("talk required python packages ", " are not installed in ", python_executable)
      }
      clear_talkrpp_options()
      options(talkrpp_python_executable = python_executable)
    } else if (!is.null(virtualenv)) {
      clear_talkrpp_options()
      options(talkrpp_virtualenv = virtualenv)
    } else if (!is.null(condaenv)) {
      clear_talkrpp_options()
      options(talkrpp_condaenv = condaenv)
    }
  } else if (check_env &&
               !(is.null(tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL))) &&
               "talkrpp_condaenv" %in% reticulate::conda_list(conda = "auto")$name) {
    message(colourise(
      "Found 'talkrpp_condaenv'. talk will use this environment \n",
      fg = "green", bg = NULL
    ))
    clear_talkrpp_options()
    options(talkrpp_condaenv = "talkrpp_condaenv")
  } else if (check_env && file.exists(file.path("~/.virtualenvs", virtualenv, "bin", "activate"))) {
    message(colourise(
      "Found your specified virtual environment. talk will use this environment \n",
      fg = "green", bg = NULL
    )) # OK: original: Found 'talkrpp_virtualenv'. talk will use this environment"
    clear_talkrpp_options()
    options(talkrpp_virtualenv = file.path("~/.virtualenvs/", virtualenv))
  } else {
    message("Finding a python executable with talk required python pakages installed...")
    talkrpp_python <- find_talkrpp(ask = ask) # model,
    if (is.null(talkrpp_python)) {
      stop("talk required python packages ", " are not installed in any of python executables.") #  model,
    } else if (is.na(talkrpp_python)) {
      stop("No python was found on system PATH")
    } else {
      options(talkrpp_python_executable = talkrpp_python)
    }
  }
  return(NULL)
}


clear_talkrpp_options <- function() {
  options(talkrpp_python_executable = NULL)
  options(talkrpp_condaenv = NULL)
  options(talkrpp_virtualenv = NULL)
}

check_talkrpp_python_options <- function() {
  settings <- NULL
  for (k in c(
    "talkrpp_python_executable",
    "talkrpp_condaenv",
    "talkrpp_virtualenv"
  )) {
    if (!is.null(getOption(k))) {
      settings$key <- k
      settings$val <- getOption(k)
    }
  }
  return(settings)
}

save_talkrpp_options <- function(key, val, prompt = TRUE) {
  prof_file <- "~/.Rprofile"
  if (!is.null(getOption("talkrpp_prompt"))) prompt <- getOption("talkrpp_prompt")

  ans <- if (prompt) {
    utils::menu(c("No", "Yes"),
      title = sprintf('Do you want to set the option, \'%s = "%s"\' , as a default (y|[n])? ', key, val)
    )
  } else {
    2
  }
  if (ans == 2) {
    rprofile <- if (file.exists(prof_file)) readLines(prof_file) else NULL
    rprofile <- grep("options\\(\\s*talkrpp_.+\\)", rprofile, value = TRUE, invert = TRUE)
    rprofile <- c(rprofile, sprintf('options(%s = "%s")', key, val))
    write(rprofile, file = prof_file)
    message(colourise(
      "The option was saved. The option will be used in talkrpp_initialize() in future \n",
      fg = "green", bg = NULL
    ))
  } else {
    message("The option was not saved (user cancelled)")
  }
}
