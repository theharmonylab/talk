# copied and modified from tensorflow::install.R, https://github.com/rstudio/tensorflow/blob/master/R/install.R
# and https://github.com/quanteda/spacyr/tree/master/R

conda_args <- reticulate:::conda_args

#' Install talk required python packages in conda or virtualenv environment
#'
#' @description Install talk required python packages (rpp) in a self-contained environment.
#' For macOS and Linux-based systems, this will also install Python itself via a "miniconda" environment, for
#'   \code{talkrpp_install}.  Alternatively, an existing conda installation may be
#'   used, by specifying its path.  The default setting of \code{"auto"} will
#'   locate and use an existing installation automatically, or download and
#'   install one if none exists.
#'
#'   For Windows, automatic installation of miniconda installation is not currently
#'   available, so the user will need to install
#'   \href{https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html}{miniconda
#'    (or Anaconda) manually}.
#' @param conda character; path to conda executable. Default "auto" which
#'   automatically find the path
#' @param update_conda Boolean; update to the latest version of Miniconda after install?
#' (should be combined with force_conda = TRUE)
#' @param force_conda Boolean; force re-installation if Miniconda is already installed at the requested path?
#' @param pip \code{TRUE} to use pip for installing rpp If \code{FALSE}, conda
#' package manager with conda-forge channel will be used for installing rpp.
#' @param rpp_version Character. Retained for backwards compatibility; it no
#' longer selects a lighter environment. talk installs a single environment
#' containing the full stack (transcription, embeddings, diarisation and segment
#' embeddings). The previous "talk_diarize" value is still accepted and behaves
#' the same as the default.
#' @param python_version character; default is "python_version_system_specific_defaults". You can specify your
#' Python version for the condaenv yourself.
#'   installation.
#' @param python_path character; path to Python only for virtualenvironment installation
#' @param bin character; e.g., "python", only for virtualenvironment installation
#' @param envname character; name of the conda-environment to install talk required python packages.
#'   Default is "talkrpp_condaenv".
#' @param prompt logical; ask whether to proceed during the installation
#' @examples
#' \dontrun{
#' # install talk required python packages in a miniconda environment (macOS and Linux)
#' talkrpp_install(prompt = FALSE)
#'
#' # install talk required python packages to an existing conda environment
#' talkrpp_install(conda = "~/anaconda/bin/")
#' }
#' @export
talkrpp_install <- function(
    conda = "auto",
    update_conda = FALSE,
    force_conda = FALSE,
    rpp_version = "rpp_version_system_specific_defaults",
    python_version = "python_version_system_specific_defaults",
    envname = "talkrpp_condaenv",
    pip = TRUE,
    python_path = NULL,
    prompt = TRUE) {

  # The talk package uses a SINGLE conda environment that holds the full stack
  # (transcription, embeddings, diarisation and segment embeddings), so
  # talkrpp_install() always performs the full installation. The rpp_version and
  # pip arguments are kept for backwards compatibility but no longer select a
  # lighter environment.

  if (python_version == "python_version_system_specific_defaults") {
    python_version <- "3.10.12"
  }

  # verify os
  if (!is_windows() && !is_osx() && !is_linux()) {
    stop("This function is available only for Windows, Mac, and Linux")
  }

  # verify 64-bit
  if (.Machine$sizeof.pointer != 8) {
    stop(
      "Unable to install the talk-package on this platform.",
      "Binary installation is only available for 64-bit platforms."
    )
  }

  # install rust for singularity machine -- but it gives error in github action
  # reticulate::py_run_string("import os\nos.system(\"curl --proto '=https' --tlsv1.2 -sSf
  # https://sh.rustup.rs | sh -s -- -y\")")
  system("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")

  # resolve and look for conda help(conda_binary)
  conda <- tryCatch(reticulate::conda_binary(conda), error = function(e) NULL)
  have_conda <- !is.null(conda)

  # Mac and linux
  if (is_unix()) {
    # check for explicit conda method
    # validate that we have conda
    if (!have_conda) {
      message("No conda was found in the system. ")
      if (prompt) {
        ans <- utils::menu(c("No", "Yes"), title = "Do you want talk to download
                           miniconda using reticulate::install_miniconda()?")
      } else {
        ans <- 2 # When no prompt is set to false, default to install miniconda.
      }
      if (ans == 2) {
        reticulate::install_miniconda(update = update_conda)
        conda <- tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)
      } else {
        stop("Conda environment installation failed (no conda binary found)\n", call. = FALSE)
      }
    }

    # Update mini_conda
    if (update_conda && force_conda || force_conda) {
      reticulate::install_miniconda(update = update_conda, force = force_conda)
    }

    # process the installation of talk required python packages
    process_talkrpp_diarize_installation(conda, python_version, prompt, envname = envname)

    # Windows installation
  } else {
    # determine whether we have system python help(py_versions_windows)
    if (python_version == "find_python") {
      python_versions <- reticulate::py_versions_windows()
      python_versions <- python_versions[python_versions$type == "PythonCore", ]
      python_versions <- python_versions[python_versions$version %in% c("3.5", "3.6", "3.7", "3.8", "3.9"), ]
      python_versions <- python_versions[python_versions$arch == "x64", ]
      have_system <- nrow(python_versions) > 0

      if (have_system) {
        # Well this isn't used later
        python_version <- python_versions[1, ]
      }
    }

    # validate that we have conda:
    if (!have_conda) {
      # OK adds help(install_miniconda)
      reticulate::install_miniconda(update = update_conda)
      conda <- tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)
    }
    # Update mini_conda
    if (have_conda && update_conda || have_conda && force_conda) {
      reticulate::install_miniconda(update = update_conda, force = force_conda)
    }
    # process the installation of talk required python packages
    process_talkrpp_diarize_installation(conda, python_version, prompt, envname = envname)
  }

  message(colourise(
    "\nInstallation is completed.\n",
    fg = "blue", bg = NULL
  ))
  message(
    " ",
    sprintf("Condaenv: %s ", envname), "\n"
  )

  message(colourise(
    "Great work - do not forget to initialize the environment \nwith talkrpp_initialize().\n",
    fg = "green", bg = NULL
  ))
  invisible(NULL)
}

process_talkrpp_diarize_installation <- function(conda,
                                                  python_version,
                                                  prompt = TRUE,
                                                  envname = "talkrpp_condaenv") {
  conda_envs <- reticulate::conda_list(conda = conda)
  if (prompt) {
    ans <- utils::menu(c("Confirm", "Cancel"), title = "Confirm that a new conda environment will be set up.")
    if (ans == 2) stop("condaenv setup is cancelled by user", call. = FALSE)
  }
  conda_env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(conda_env) == 0) {
    message("Creating ", envname, " conda environment for talk diarize installation...\n")
    python_packages <- ifelse(is.null(python_version), "python=3.10",
      sprintf("python=%s", python_version)
    )
    # Include pip explicitly: newer conda does not add pip to a minimal env,
    # which then breaks the pip-based installs below ("No module named pip").
    reticulate::conda_create(envname, packages = c(python_packages, "pip"), conda = conda)
  }

  # NOTE: the diarisation pipeline requires the `ffmpeg` binary on PATH. Do NOT
  # install ffmpeg into this conda environment: conda's ffmpeg pulls a newer
  # libav* stack that torchaudio links against and cannot use ("Failed to load
  # audio"). Install ffmpeg via the system package manager instead (see README /
  # CI workflows).

  # Windows: conda's default CA bundle can be truncated, which breaks the
  # diarisation model download with "[ASN1: NOT_ENOUGH_DATA]". Force-reinstall
  # the certificate packages from conda-forge to repair the bundle.
  if (is_windows()) {
    message("Repairing CA certificates for the diarize environment (Windows)...\n")
    system2(conda, c("install", "--yes", "--name", envname, "-c", "conda-forge",
                     "--force-reinstall", "ca-certificates", "certifi"))
  }

  # Step 1: torch — CUDA index URL on Linux/Windows, plain PyPI on macOS
  torch_packages <- c("torch==2.11.0", "torchaudio==2.11.0")
  torch_pip_options <- if (is_linux() || is_windows()) {
    c("--index-url", "https://download.pytorch.org/whl/cu128")
  } else {
    character(0)
  }
  message("Installing torch for diarize environment...\n")
  reticulate::conda_install(envname, torch_packages, pip = TRUE, conda = conda,
                            pip_options = torch_pip_options)

  # Step 2: whisnemo with constraints (prevents NeMo from upgrading torch)
  whisnemo_constraints <- paste0(
    "https://raw.githubusercontent.com/humanlab/WhisNemo/",
    "dumrania/timing-and-postprocess/constraints/runtime.txt"
  )
  whisnemo_package <- paste0(
    "whisnemo[diarize,embed] @ git+https://github.com/humanlab/WhisNemo.git",
    "@dumrania/timing-and-postprocess"
  )
  message("Installing whisnemo with dependency constraints...\n")
  # Two passes so re-installs onto an existing env actually update the code:
  # a git package keeps the same version string, so a plain install is a no-op
  # ("Requirement already satisfied") and never picks up new submodules such as
  # whisnemo.core.embed. --force-reinstall --no-deps refreshes just the package
  # files; the second pass (no force) then fills in any missing extra deps
  # without re-downloading the whole stack (e.g. torch).
  reticulate::conda_install(envname, whisnemo_package, pip = TRUE, conda = conda,
                            pip_options = c("-c", whisnemo_constraints,
                                            "--force-reinstall", "--no-deps"))
  reticulate::conda_install(envname, whisnemo_package, pip = TRUE, conda = conda,
                            pip_options = c("-c", whisnemo_constraints))

  # Step 3: WhiSPA (required for talkEmbedSegments(model = "whispa")).
  # Installed from the canonical humanlab repo, which is pip-installable.
  # Same two-pass approach as whisnemo so existing envs are refreshed.
  whispa_package <- "whispa @ git+https://github.com/humanlab/WhiSPA.git"
  message("Installing WhiSPA for segment-level embeddings...\n")
  reticulate::conda_install(envname, whispa_package, pip = TRUE, conda = conda,
                            pip_options = c("-c", whisnemo_constraints,
                                            "--force-reinstall", "--no-deps"))
  reticulate::conda_install(envname, whispa_package, pip = TRUE, conda = conda,
                            pip_options = c("-c", whisnemo_constraints))
}


process_talkrpp_installation_virtualenv <- function(python = "/usr/local/bin/python3.9",
                                                    rpp_version,
                                                    pip_version,
                                                    envname = "talkrpp_virtualenv",
                                                    prompt = TRUE) {
  libraries <- paste(rpp_version, collapse = ", ")
  message(sprintf(
    'A new virtual environment called "%s" will be created using "%s" \n and,
    the following talk reuired python packages will be installed: \n "%s" \n \n',
    envname, python, libraries
  ))
  if (prompt) {
    ans <- utils::menu(c("No", "Yes"), title = "Proceed?")
    if (ans == 1) stop("Virtualenv setup is cancelled by user", call. = FALSE)
  }

  # Make python path help(virtualenv_create)
  reticulate::virtualenv_create(envname,
                                python,
                                pip_version = NULL,
                                required = TRUE)

  reticulate::use_virtualenv(envname, required = TRUE)

  #
  for (i in seq_len(length(rpp_version))) {
    reticulate::py_install(rpp_version[[i]], envname = envname, pip = TRUE)
  }

  message(colourise(
    "\nSuccess!\n",
    fg = "green", bg = NULL
  ))
}

# Check whether "bin"/something exists in the bin folder
# For example, bin = "pip3" bin = "python3.9" bin = ".virtualenv"
# And for example: file.exists("/usr/local/bin/.virtualenvs") /Users/oscarkjell/.virtualenvs
python_unix_binary <- function(bin) {
  locations <- file.path(c("/usr/local/bin", "/usr/bin"), bin)
  locations <- locations[file.exists(locations)]
  if (length(locations) > 0) {
    locations[[1]]
  } else {
    NULL
  }
}

#' @rdname talkrpp_install
#' @description If you wish to install Python in a "virtualenv", use the
#'   \code{talkrpp_install_virtualenv} function. It requires that you have a python version
#'   and path to it (such as "/usr/local/bin/python3.9" for Mac and Linux.).
#' @param pip_version character;
#' @examples
#' \dontrun{
#' # install talk required python packages in a virtual environment
#' talkrpp_install_virtualenv()
#' }
#' @export
talkrpp_install_virtualenv <- function(rpp_version = c("torch==2.0.0",
                                                       "transformers==4.19.2",
                                                       "numpy",
                                                       "pandas",
                                                       "nltk"),
                                       python_path = NULL, # "/usr/local/bin/python3.9",
                                       pip_version = NULL,
                                       bin = "python3",
                                       envname = "talkrpp_virtualenv",
                                       prompt = TRUE) {
  # find system python binary
  if (!is.null(python_path)) {
    python <- python_path
    } else {
      python <-  python_unix_binary(bin = bin)
    }


  if (is.null(python)) {
    stop("Unable to locate Python on this system.", call. = FALSE)
  }

  process_talkrpp_installation_virtualenv(
    python = python,
    pip_version = pip_version,
    rpp_version = rpp_version,
    envname = envname,
    prompt = prompt
  )


  message(colourise(
    "\nInstallation is completed.\n",
    fg = "blue", bg = NULL
  ))
  invisible(NULL)
}


#' Uninstall talkrpp conda environment
#'
#' Removes the conda environment created by talkrpp_install()
#' @param conda path to conda executable, default to "auto" which automatically
#'   finds the path
#' @param prompt logical; ask whether to proceed during the installation
#' @param envname character; name of conda environment to remove
#' @export
talkrpp_uninstall <- function(conda = "auto",
                              prompt = TRUE,
                              envname = "talkrpp_condaenv") {
  conda <- tryCatch(reticulate::conda_binary(conda), error = function(e) NULL)
  have_conda <- !is.null(conda)

  if (!have_conda) {
    stop("Conda installation failed (no conda binary found)\n", call. = FALSE)
  }

  conda_envs <- reticulate::conda_list(conda = conda)
  conda_env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(conda_env) != 1) {
    stop("conda environment", envname, "is not found", call. = FALSE)
  }
  message("A conda environment", envname, "will be removed\n")
  ans <- ifelse(prompt, utils::menu(c("No", "Yes"), title = "Proceed?"), 2)
  if (ans == 1) stop("condaenv removal is cancelled by user", call. = FALSE)
  python <- reticulate::conda_remove(envname = envname)

  message("\nUninstallation complete.\n\n")

  invisible(NULL)
}

###### see utils.R in spacyr
# checking OS functions, thanks to r-tensorflow;

is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_unix <- function() {
  identical(.Platform$OS.type, "unix")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_linux <- function() {
  identical(tolower(Sys.info()[["sysname"]]), "linux")
}

#is_ubuntu <- function() {
#  if (is_unix() && file.exists("/etc/lsb-release")) {
#    lsbrelease <- readLines("/etc/lsb-release")
#    any(grepl("Ubuntu", lsbrelease))
#  } else {
#    FALSE
#  }
#}

#python_version_function <- function(python) {
#  # check for the version
#  result <- system2(python, "--version", stdout = TRUE, stderr = TRUE)
#
#  # check for error
#  error_status <- attr(result, "status")
#  if (!is.null(error_status)) {
#    stop("Error ", error_status, " occurred while checking for python version", call. = FALSE)
#  }
#
#  # parse out the major and minor version numbers
#  matches <- regexec("^[^ ]+\\s+(\\d+)\\.(\\d+).*$", result)
#  matches <- regmatches(result, matches)[[1]]
#  if (length(matches) != 3) {
#    stop("Unable to parse Python version '", result[[1]], "'", call. = FALSE)
#  }
#
#  # return as R numeric version
#  numeric_version(paste(matches[[2]], matches[[3]], sep = "."))
#}

#pip_get_version <- function(cmd, major_version) {
#  regex <- "^(\\S+)\\s?(.*)$"
#  cmd1 <- sub(regex, "\\1", cmd)
#  cmd2 <- sub(regex, "\\2", cmd)
#  oldw <- getOption("warn")
#  options(warn = -1)
#  result <- paste(system2(cmd1, cmd2, stdout = TRUE, stderr = TRUE),
#    collapse = " "
#  )
#  options(warn = oldw)
#  version_check_regex <- sprintf(".+(%s.\\d+\\.\\d+).+", major_version)
#  return(sub(version_check_regex, "\\1", result))
#}


#conda_get_version <- function(major_version = NA, conda, envname) {
#  condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
#  cmd <- sprintf(
#    "%s%s %s && conda search torch -c conda-forge%s",
#    ifelse(is_windows(), "", ifelse(is_osx(), "source ", "/bin/bash -c \"source ")),
#    shQuote(path.expand(condaenv_bin("activate"))),
#    envname,
#    ifelse(is_windows(), "", ifelse(is_osx(), "", "\""))
#  )
#  regex <- "^(\\S+)\\s?(.*)$"
#  cmd1 <- sub(regex, "\\1", cmd)
#  cmd2 <- sub(regex, "\\2", cmd)
#
#  result <- system2(cmd1, cmd2, stdout = TRUE, stderr = TRUE)
#  result <- sub("\\S+\\s+(\\S+)\\s.+", "\\1", result)
#  if (!is.na(major_version)) {
#    result <- grep(paste0("^", major_version, "\\."), result, value = TRUE)
#  }
#  #
#  return(result[length(result)])
#}


