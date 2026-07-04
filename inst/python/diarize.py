"""
Standalone diarization module for the talk R package.
Uses WhisNemo's full pipeline: Demucs → Whisper → NeMo MSDD.

Called from R via reticulate::source_python("diarize.py")

Usage as function:
    diarize_audio(audio_path, output_dir=None, model_name="medium.en", num_speakers=2, ...)
"""

import os
import shutil
import logging

# On Windows, loading the default SSL certificates enumerates the Windows
# certificate store, and some Windows images (e.g. GitHub-hosted runners)
# contain a store entry that OpenSSL 3.x cannot parse: every download made
# through a default SSL context (urllib/wget, used by the NeMo/whisper model
# downloads) then aborts with "ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]".
# requests is unaffected because it uses certifi's bundle directly. Wrap
# SSLContext.load_default_certs so a failing store enumeration falls back to
# certifi's bundle (which is also always added; loading extra CAs is
# harmless). This covers EVERY code path that builds a default context
# (ssl.create_default_context, urllib's default HTTPS context, aiohttp, ...),
# unlike patching ssl._create_default_https_context, which only covers urllib.
if os.name == "nt":
    try:
        import ssl
        import certifi

        if not getattr(ssl, "_talk_certifi_fallback", False):
            _orig_load_default_certs = ssl.SSLContext.load_default_certs

            def _load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
                try:
                    _orig_load_default_certs(self, purpose)
                except ssl.SSLError:
                    pass
                self.load_verify_locations(cafile=certifi.where())

            ssl.SSLContext.load_default_certs = _load_default_certs
            ssl._talk_certifi_fallback = True
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def diarize_audio(
    audio_path,
    output_dir=None,
    model_name="medium.en",
    language=None,
    device="cuda",
    stemming=True,
    suppress_numerals=False,
    batch_size=8,
    # NeMo config
    num_speakers=2,
    oracle_num_speakers=True,
    vad_model="vad_multilingual_marblenet",
    speaker_model="titanet_large",
    onset=0.8,
    offset=0.5,
    pad_offset=-0.05,
    domain_type="telephonic",
    # Output
    output_formats=None,
    remove_stutters=False,
    stutter_threshold=0.8,
):
    """
    Transcribe and diarize a single audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    output_dir : str, optional
        Directory where output files (.txt/.srt/.csv) are kept. If None
        (default), the pipeline runs in a temporary directory that is removed
        afterwards: no files are kept, and the transcript is only returned as
        a DataFrame.
    model_name : str
        Whisper model name.
    language : str, optional
        Language code. None = auto-detect.
    device : str
        "cuda", "cpu", or "mps" (Apple Silicon).
    stemming : bool
        If True, run Demucs vocal separation first.
    suppress_numerals : bool
        If True, suppress digits in transcription.
    batch_size : int
        Whisper batch size.
    num_speakers : int
        Expected number of speakers.
    oracle_num_speakers : bool
        Whether to force speaker count in NeMo clustering.
    vad_model : str
        NeMo VAD model name.
    speaker_model : str
        NeMo speaker embedding model name.
    onset : float
        VAD onset threshold.
    offset : float
        VAD offset threshold.
    pad_offset : float
        VAD pad offset.
    domain_type : str
        NeMo domain type: "telephonic", "meeting", or "general".
    output_formats : list, optional
        List of formats: "csv", "txt", "srt". Default: ["csv"].
    remove_stutters : bool
        If True, run stutter removal on CSV output.
    stutter_threshold : float
        Similarity threshold for stutter removal.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - "transcript" : pandas.DataFrame or None
            The diarized transcript, one row per utterance, with columns
            "speaker", "start_timestamp", "end_timestamp", and "message".
            When called from R via reticulate, this is converted to an R
            data.frame automatically. None if the run failed or no
            transcript CSV was produced.
        - "output_files" : list of str
            Paths to the files written in output_dir (.txt, .srt,
            _formatted.csv, and _formatted_corrected.csv if stutter
            removal was enabled). Empty when output_dir is None (no files
            are kept).
        - "status" : str
            "success" or "error".
        - "error" : str
            Present only when status is "error"; the error message.
    """
    if output_formats is None:
        output_formats = ["csv"]

    # With no output_dir, run in a temporary directory that is removed at the
    # end: the transcript is returned as a DataFrame, so files need only be
    # kept when the user explicitly asks for them.
    keep_files = output_dir is not None
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="talk_diarise_")

    # run_diarize writes outputs next to the input audio file, so we copy
    # the audio into output_dir and run on the copy. This ensures all
    # outputs (csv, txt, srt) land in output_dir.
    original_cwd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    audio_basename = os.path.basename(audio_path)
    audio_copy = os.path.join(output_dir, audio_basename)
    copied_audio = False
    if os.path.abspath(audio_path) != os.path.abspath(audio_copy):
        shutil.copy2(audio_path, audio_copy)
        copied_audio = True

    os.chdir(output_dir)

    try:
        from whisnemo.core.diarize import run_diarize

        run_diarize(
            audio_path=audio_copy,
            stemming=stemming,
            suppress_numerals=suppress_numerals,
            model_name=model_name,
            batch_size=batch_size,
            language=language,
            device=device,
            output_formats=output_formats,
            remove_stutters=remove_stutters,
            stutter_threshold=stutter_threshold,
            num_speakers=num_speakers,
            oracle_num_speakers=oracle_num_speakers,
            vad_model=vad_model,
            speaker_model=speaker_model,
            onset=onset,
            offset=offset,
            pad_offset=pad_offset,
            domain_type=domain_type,
        )

        # Collect output files from output_dir (none are reported when the
        # run used a temporary directory, since they are removed below)
        base = os.path.splitext(audio_basename)[0]
        output_files = []
        if keep_files:
            for ext in [".txt", ".srt", "_formatted.csv", "_formatted_corrected.csv"]:
                candidate = os.path.join(output_dir, f"{base}{ext}")
                if os.path.isfile(candidate):
                    output_files.append(candidate)

        # Load the formatted transcript into a DataFrame for direct use in R.
        # reticulate converts a pandas DataFrame to an R data.frame.
        # Prefer the stutter-corrected CSV if it exists, else the formatted CSV.
        import pandas as pd
        transcript_df = None
        corrected_csv = os.path.join(output_dir, f"{base}_formatted_corrected.csv")
        formatted_csv = os.path.join(output_dir, f"{base}_formatted.csv")
        csv_to_load = corrected_csv if os.path.isfile(corrected_csv) else formatted_csv
        if os.path.isfile(csv_to_load):
            try:
                transcript_df = pd.read_csv(csv_to_load)
            except Exception as read_err:
                logger.warning(f"Could not read transcript CSV {csv_to_load}: {read_err}")

        return {
            "transcript": transcript_df,
            "output_files": output_files,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Diarization failed for {audio_path}: {e}")
        return {"transcript": None, "output_files": [], "status": "error", "error": str(e)}

    finally:
        os.chdir(original_cwd)
        if keep_files:
            # Remove the temporary copy of the input audio (made above only
            # to steer run_diarize's outputs into output_dir). Never removes
            # the user's original file.
            if copied_audio:
                try:
                    os.remove(audio_copy)
                except OSError:
                    pass
            # Remove whisnemo's timing telemetry (timing_logs/<base>_timing.csv)
            base = os.path.splitext(audio_basename)[0]
            timing_dir = os.path.join(output_dir, "timing_logs")
            try:
                os.remove(os.path.join(timing_dir, f"{base}_timing.csv"))
                os.rmdir(timing_dir)  # only removes it when empty
            except OSError:
                pass
        else:
            # Temporary run directory: remove everything.
            shutil.rmtree(output_dir, ignore_errors=True)


def diarize_batch(
    audio_dir,
    output_dir=None,
    start_idx=1,
    end_idx=None,
    try_limit=2,
    organize=False,
    **kwargs,
):
    """
    Batch-process a directory of audio files.

    Parameters
    ----------
    audio_dir : str
        Directory containing audio files.
    output_dir : str, optional
        Output directory. Defaults to same as audio_dir.
    start_idx : int
        1-indexed start position.
    end_idx : int, optional
        1-indexed end position. None = all files.
    try_limit : int
        Max attempts before OOM split-retry.
    organize : bool
        If True, organize outputs into subfolders after batch.
    **kwargs
        All other arguments are passed to diarize_audio().

    Returns
    -------
    dict
        Keys: "succeeded", "failed", "skipped" — lists of file paths.
    """
    from whisnemo.core.batch import run_batch

    device = kwargs.pop("device", "cuda")

    run_batch(
        audio_dir=audio_dir,
        device=device,
        try_limit=try_limit,
        start_idx=start_idx,
        end_idx=end_idx,
        organize=organize,
        **kwargs,
    )
