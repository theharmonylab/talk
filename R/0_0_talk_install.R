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
#' @param include_text logical; if TRUE (the default), additionally install
#'   the Python packages used by the text package (sentence-transformers,
#'   flair, bertopic, umap-learn, hdbscan, evaluate, jsonschema), so that the
#'   environment serves both packages: after
#'   \code{textrpp_initialize(condaenv = "talkrpp_condaenv", save_profile = TRUE)}
#'   the text package uses it too. This step is non-fatal -- if it fails, talk
#'   itself is still fully installed. Set FALSE for a leaner, talk-only
#'   environment.
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
    prompt = TRUE,
    include_text = TRUE) {

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

  # System dependencies (same approach as the text package): auto-install
  # what can be installed (Debian/Ubuntu dev libraries, when root/sudo is
  # available), report the rest with copy-paste install instructions, and
  # install Rust only if it is actually missing (used to build Python
  # packages such as tokenizers when no prebuilt wheel exists).
  install_linux_dev_libs_if_needed(prompt = prompt)
  check_talk_system_dependencies(verbose = prompt)
  install_rust_if_needed(prompt = prompt)

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
    process_talkrpp_diarize_installation(conda, python_version, prompt,
                                         envname = envname,
                                         include_text = include_text)

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
    process_talkrpp_diarize_installation(conda, python_version, prompt,
                                         envname = envname,
                                         include_text = include_text)
  }

  # If no system ffmpeg was found, activate the static fallback installed above.
  ensure_ffmpeg_on_path(envname)

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

# Make ffmpeg findable, in order of preference:
#   1. already on PATH;
#   2. installed but not on the R session's PATH (e.g. RStudio launched from
#      the macOS Dock does not always include /opt/homebrew/bin);
#   3. talk's own fallback: the static ffmpeg binary shipped by the
#      imageio-ffmpeg package in the talk conda environment (installed by
#      talkrpp_install()), copied once into a talk-managed bin directory.
# Unlike conda's ffmpeg, the static binary adds no libav* libraries to the
# environment, so torchaudio is unaffected. Prepends the containing directory
# to PATH (inherited by the diarise/embed subprocesses) and returns TRUE when
# ffmpeg is available.
ensure_ffmpeg_on_path <- function(condaenv = NULL) {
  if (nzchar(Sys.which("ffmpeg"))) return(invisible(TRUE))

  # 2) common install locations missing from the session PATH
  if (.Platform$OS.type == "unix") {
    candidates <- c("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
    hit <- candidates[file.exists(file.path(candidates, "ffmpeg"))]
    if (length(hit) > 0) {
      Sys.setenv(PATH = paste(hit[1], Sys.getenv("PATH"), sep = ":"))
      return(invisible(TRUE))
    }
  }

  # 3) talk's fallback shim (created below on a previous call)
  shim_dir <- file.path(tools::R_user_dir("talk", "cache"), "bin")
  shim <- file.path(shim_dir,
                    if (.Platform$OS.type == "windows") "ffmpeg.exe" else "ffmpeg")
  if (!file.exists(shim) && !is.null(condaenv)) {
    # resolve the static binary shipped by imageio-ffmpeg in the conda env
    py <- tryCatch(reticulate::conda_python(condaenv), error = function(e) NULL)
    if (!is.null(py) && file.exists(py)) {
      exe <- tryCatch(
        suppressWarnings(system2(
          py, c("-c", shQuote("import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")),
          stdout = TRUE, stderr = FALSE
        )),
        error = function(e) character(0)
      )
      exe <- utils::tail(exe[nzchar(exe)], 1)
      if (length(exe) == 1 && file.exists(exe)) {
        dir.create(shim_dir, recursive = TRUE, showWarnings = FALSE)
        if (file.copy(exe, shim, overwrite = TRUE) &&
            .Platform$OS.type == "unix") {
          Sys.chmod(shim, "0755")
        }
      }
    }
  }
  if (file.exists(shim)) {
    Sys.setenv(PATH = paste(shim_dir, Sys.getenv("PATH"),
                            sep = .Platform$path.sep))
    message("No system ffmpeg found; using the static ffmpeg bundled with the talk environment.")
    return(invisible(TRUE))
  }

  invisible(nzchar(Sys.which("ffmpeg")))
}

# One-line, copy-paste install instruction for ffmpeg on the current OS.
ffmpeg_install_instruction <- function() {
  switch(Sys.info()[["sysname"]],
    Darwin  = "brew install ffmpeg",
    Linux   = "sudo apt-get install ffmpeg   (or your distribution's package manager)",
    Windows = "choco install ffmpeg   (or download from https://ffmpeg.org and add it to PATH)",
    "see https://ffmpeg.org"
  )
}

# Check the system tools talk needs, mirroring the text package's
# check_*_githubaction_dependencies(): print a human-readable summary with
# copy-paste install commands, warn when something required is missing, and
# return a structured result invisibly.
# Debian/Ubuntu development libraries needed to build talk's R dependencies
# from source. Returns the missing ones (character(0) when none, or when not
# on a dpkg-based system).
linux_missing_dev_libs <- function() {
  if (Sys.info()[["sysname"]] != "Linux" || !nzchar(Sys.which("dpkg"))) {
    return(character(0))
  }
  required <- c("libcurl4-openssl-dev", "libssl-dev", "libxml2-dev")
  ok <- vapply(required, function(lib) {
    suppressWarnings(
      system2("dpkg", c("-s", lib), stdout = FALSE, stderr = FALSE)
    ) == 0
  }, logical(1))
  required[!ok]
}

# Best-effort automatic installation of missing Debian/Ubuntu development
# libraries. Installing system packages needs root, so this works when:
#   - R runs as root (e.g. docker containers): apt-get directly;
#   - sudo works without a password (e.g. CI runners, many servers);
#   - R runs in a terminal, where sudo can prompt for the password.
# In RStudio there is no terminal for a sudo password prompt, so the function
# falls back to printing the install instructions instead of attempting.
install_linux_dev_libs_if_needed <- function(prompt = TRUE) {
  missing <- linux_missing_dev_libs()
  if (length(missing) == 0 || !nzchar(Sys.which("apt-get"))) {
    return(invisible(NULL))
  }

  is_root <- identical(unname(Sys.info()[["user"]]), "root")
  has_sudo <- nzchar(Sys.which("sudo"))
  sudo_passwordless <- has_sudo && suppressWarnings(
    system2("sudo", c("-n", "true"), stdout = FALSE, stderr = FALSE)
  ) == 0
  # sudo can only ask for a password when stdin is a terminal (not in RStudio)
  sudo_can_prompt <- has_sudo && isatty(stdin())

  can_attempt <- is_root || sudo_passwordless || sudo_can_prompt
  if (!can_attempt) {
    message(
      "Missing development libraries: ", paste(missing, collapse = ", "), "\n",
      "They cannot be installed from this R session (no root/sudo terminal).\n",
      "Please run in a terminal:\n",
      "  sudo apt-get install -y ", paste(missing, collapse = " ")
    )
    return(invisible(NULL))
  }

  if (prompt) {
    ans <- utils::menu(
      c("No", "Yes"),
      title = paste0("Missing development libraries: ",
                     paste(missing, collapse = ", "),
                     ". Install them now with apt-get",
                     if (!is_root) " (sudo)", "?")
    )
    if (ans != 2) {
      message("Skipped. To install them later, run:\n",
              "  sudo apt-get install -y ", paste(missing, collapse = " "))
      return(invisible(NULL))
    }
  } else if (!is_root && !sudo_passwordless) {
    # non-interactive run must not hang on a sudo password prompt
    message(
      "Missing development libraries: ", paste(missing, collapse = ", "), "\n",
      "To install them, run:\n",
      "  sudo apt-get install -y ", paste(missing, collapse = " ")
    )
    return(invisible(NULL))
  }

  cmd  <- if (is_root) "apt-get" else "sudo"
  args <- if (is_root) character(0) else "apt-get"
  message("Installing: ", paste(missing, collapse = ", "), " ...")
  suppressWarnings({
    system2(cmd, c(args, "update"))
    system2(cmd, c(args, "install", "-y", missing))
  })

  still_missing <- linux_missing_dev_libs()
  if (length(still_missing) == 0) {
    message("Development libraries installed successfully.")
  } else {
    warning("Could not install: ", paste(still_missing, collapse = ", "),
            ". Please install them manually:\n",
            "  sudo apt-get install -y ", paste(still_missing, collapse = " "),
            call. = FALSE)
  }
  invisible(NULL)
}

check_talk_system_dependencies <- function(verbose = TRUE) {
  os <- Sys.info()[["sysname"]]
  summary_lines <- c("== talk system dependencies ==")
  missing <- character(0)

  # ffmpeg -- required by talkTranscribeDiarise()/talkEmbedSegments():
  # whisper loads audio through the ffmpeg binary.
  ensure_ffmpeg_on_path()
  ffmpeg_ok <- nzchar(Sys.which("ffmpeg"))
  if (ffmpeg_ok) {
    summary_lines <- c(summary_lines, "'ffmpeg' is installed.")
  } else {
    missing <- c(missing, "ffmpeg")
    summary_lines <- c(
      summary_lines,
      "'ffmpeg' is NOT installed (used by talkTranscribeDiarise() and talkEmbedSegments()).",
      "talkrpp_install() installs a static ffmpeg fallback automatically, so no action is",
      "strictly required. To install a system ffmpeg (recommended), run:",
      paste0("  ", ffmpeg_install_instruction()),
      "Note: do NOT install ffmpeg with conda -- conda's ffmpeg breaks torchaudio's audio loading."
    )
  }

  # Homebrew on macOS (the recommended way to install ffmpeg).
  if (os == "Darwin" && !nzchar(Sys.which("brew"))) {
    summary_lines <- c(
      summary_lines,
      "'homebrew' is NOT installed (recommended, to install ffmpeg).",
      "To install it, open your Terminal and run:",
      '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    )
  }

  # Debian/Ubuntu development libraries needed to build talk's R dependencies
  # from source (e.g. when installing talk from GitHub via remotes/devtools).
  # Deliberately minimal: the longer apt list in the CI workflows (cairo,
  # harfbuzz, freetype, java, ...) belongs to the CI's development toolchain
  # (pkgdown, ragg, covr), not to talk itself.
  if (os == "Linux" && nzchar(Sys.which("dpkg"))) {
    linux_missing <- linux_missing_dev_libs()
    if (length(linux_missing) == 0) {
      summary_lines <- c(
        summary_lines,
        "Development libraries are installed (libcurl4-openssl-dev, libssl-dev, libxml2-dev)."
      )
    } else {
      missing <- c(missing, linux_missing)
      summary_lines <- c(
        summary_lines,
        "Missing development libraries (needed to install talk's R dependencies from source):",
        paste0("  - ", linux_missing),
        "To install them, run:",
        paste0("  sudo apt-get install -y ", paste(linux_missing, collapse = " "))
      )
    }
  }

  if (length(missing) > 0) {
    warning(
      "Missing system dependencies used by talk: ", paste(missing, collapse = ", "),
      ". See the message above for install instructions.",
      call. = FALSE
    )
  }
  if (verbose) message(paste(summary_lines, collapse = "\n"))

  invisible(list(
    os = os,
    ffmpeg = ffmpeg_ok,
    missing = missing,
    summary_lines = summary_lines
  ))
}

# Install Rust only when it is actually missing (needed to build Python
# packages such as tokenizers when no prebuilt wheel exists). Adapted from the
# text package's install_rust_if_needed(): checks first, asks the user, uses
# the proper installer per OS, and updates PATH for the current session.
install_rust_if_needed <- function(prompt = TRUE) {
  is_installed <- function(cmd) nzchar(Sys.which(cmd))

  # rustc may live in ~/.cargo/bin without being on the session PATH
  cargo_bin <- path.expand(
    if (.Platform$OS.type == "windows") {
      file.path(Sys.getenv("USERPROFILE"), ".cargo", "bin")
    } else {
      "~/.cargo/bin"
    }
  )
  if (!is_installed("rustc") && dir.exists(cargo_bin)) {
    Sys.setenv(PATH = paste(cargo_bin, Sys.getenv("PATH"), sep = .Platform$path.sep))
  }

  if (is_installed("rustc")) {
    message("Rust is already installed. Skipping Rust installation.")
    return(invisible(NULL))
  }

  message("Rust is not installed on this system.")

  if (.Platform$OS.type != "windows" && !is_installed("curl")) {
    warning("Rust installation aborted: 'curl' not found.\n",
            "Please install Rust manually: https://www.rust-lang.org/",
            call. = FALSE)
    return(invisible(NULL))
  }

  ans <- if (prompt) utils::menu(c("No", "Yes"), title = "Do you want to install Rust?") else 2
  if (ans == 1) {
    message("Rust installation cancelled by user.")
    return(invisible(NULL))
  }

  tryCatch({
    if (.Platform$OS.type == "windows") {
      message("Downloading Rust installer for Windows...")
      installer <- file.path(Sys.getenv("USERPROFILE"), "rustup-init.exe")
      utils::download.file(
        "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe",
        installer, mode = "wb"
      )
      message("Running Rust installer (non-interactive)...")
      status <- shell(sprintf(
        'cmd.exe /c ""%s" -y --profile minimal --default-host x86_64-pc-windows-msvc"',
        installer
      ), wait = TRUE)
      if (!identical(status, 0L)) stop("rustup-init exited with status ", status)
    } else {
      message("Downloading and installing Rust for macOS/Linux...")
      system(paste(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |",
        "sh -s -- -y --profile minimal --no-modify-path"
      ))
    }

    if (dir.exists(cargo_bin)) {
      Sys.setenv(PATH = paste(cargo_bin, Sys.getenv("PATH"), sep = .Platform$path.sep))
    }

    if (is_installed("rustc")) {
      message("Rust installation completed successfully.")
      message("If RStudio still can't find rustc/cargo, restart RStudio so PATH updates.")
    } else {
      warning("Rust installation attempted, but 'rustc' was not found on PATH.\n",
              "Try restarting R/RStudio or install Rust manually:\n",
              "  https://www.rust-lang.org/tools/install",
              call. = FALSE)
    }
  }, error = function(e) {
    warning("Rust installation failed: ", conditionMessage(e),
            "\nPlease install Rust manually from https://www.rust-lang.org/",
            call. = FALSE)
  })

  invisible(NULL)
}

process_talkrpp_diarize_installation <- function(conda,
                                                  python_version,
                                                  prompt = TRUE,
                                                  envname = "talkrpp_condaenv",
                                                  include_text = TRUE) {
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

  # Step 4: static ffmpeg fallback. If the user has no system ffmpeg, talk
  # falls back to the static binary shipped by the imageio-ffmpeg package.
  # Unlike conda's ffmpeg, this is a single executable that adds no libav*
  # libraries to the environment, so torchaudio is unaffected.
  message("Installing imageio-ffmpeg (static ffmpeg fallback)...\n")
  reticulate::conda_install(envname, "imageio-ffmpeg", pip = TRUE, conda = conda)

  # Step 5: the Python packages used by the text package, so the environment
  # serves both text and talk. Installed under the same constraints so talk's
  # pinned stack (torch/numpy/transformers/NeMo) is not disturbed -- verified
  # to resolve cleanly against them. Non-fatal: talk must remain fully
  # installed even if a text dependency fails to resolve on some platform.
  if (include_text) {
    text_packages <- c("sentence-transformers", "flair", "bertopic",
                       "umap-learn", "hdbscan", "evaluate", "jsonschema")
    message("Installing text-package Python dependencies (include_text = TRUE)...\n")
    tryCatch(
      reticulate::conda_install(envname, text_packages, pip = TRUE, conda = conda,
                                pip_options = c("-c", whisnemo_constraints)),
      error = function(e) {
        warning(
          "The text-package Python dependencies could not be installed; ",
          "talk itself is fully installed and functional. To use the text ",
          "package in this environment, re-run talkrpp_install() later or ",
          "install text's dependencies with text::textrpp_install(). Error: ",
          conditionMessage(e),
          call. = FALSE
        )
      }
    )
  }
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


