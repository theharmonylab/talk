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
        Directory for outputs. Defaults to same directory as audio file.
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
        Keys: "output_files" (list of output paths), "status" ("success" or "error").
    """
    if output_formats is None:
        output_formats = ["csv"]

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(audio_path))

    # run_diarize writes outputs next to the input audio file, so we copy
    # the audio into output_dir and run on the copy. This ensures all
    # outputs (csv, txt, srt) land in output_dir.
    original_cwd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    audio_basename = os.path.basename(audio_path)
    audio_copy = os.path.join(output_dir, audio_basename)
    if os.path.abspath(audio_path) != os.path.abspath(audio_copy):
        shutil.copy2(audio_path, audio_copy)

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

        # Collect output files from output_dir
        base = os.path.splitext(audio_basename)[0]
        output_files = []
        for ext in [".txt", ".srt", "_formatted.csv", "_formatted_corrected.csv"]:
            candidate = os.path.join(output_dir, f"{base}{ext}")
            if os.path.isfile(candidate):
                output_files.append(candidate)

        return {"output_files": output_files, "status": "success"}

    except Exception as e:
        logger.error(f"Diarization failed for {audio_path}: {e}")
        return {"output_files": [], "status": "error", "error": str(e)}

    finally:
        os.chdir(original_cwd)


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
