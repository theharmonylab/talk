"""
Standalone transcription module for the talk R package.
Uses WhisNemo's Whisper transcription under the hood.

Called from R via reticulate::source_python("transcribe.py")

Usage as function:
    transcribe_audio(audio_path, model_name="medium.en", language=None, device="cuda")
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path,
    output_dir=None,
    model_name="medium.en",
    language=None,
    device="cuda",
    suppress_numerals=False,
    output_format="csv",
):
    """
    Transcribe a single audio file using Whisper (non-batched, via faster-whisper).

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    output_dir : str, optional
        Directory to save outputs. Defaults to same directory as audio file.
    model_name : str
        Whisper model name (e.g., "tiny", "base", "small", "medium.en", "large-v2").
    language : str, optional
        Language code (e.g., "en"). None = auto-detect.
    device : str
        "cuda" or "cpu".
    suppress_numerals : bool
        If True, suppress numerical digits in output.
    output_format : str
        One of "csv", "txt", "srt", or "all".

    Returns
    -------
    dict
        Keys: "transcript" (full text), "output_files" (list of paths written).
    """
    from whisnemo.core.transcription_helpers import transcribe
    import whisperx
    import torch

    mtypes = {"cpu": "int8", "cuda": "float16"}

    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Transcribing: {audio_path} with model={model_name}, device={device}")

    # Run Whisper
    whisper_results, detected_lang = transcribe(
        audio_path, language, model_name,
        mtypes[device], suppress_numerals, device,
    )

    full_transcript = " ".join(seg["text"].strip() for seg in whisper_results)

    # Write outputs
    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_files = []
    formats = ["csv", "txt", "srt"] if output_format == "all" else [output_format]

    if "txt" in formats:
        txt_path = os.path.join(output_dir, f"{base}_transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        output_files.append(txt_path)
        logger.info(f"Wrote: {txt_path}")

    if "csv" in formats:
        import csv
        csv_path = os.path.join(output_dir, f"{base}_transcript.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "text"])
            for seg in whisper_results:
                writer.writerow([seg.get("start", ""), seg.get("end", ""), seg["text"].strip()])
        output_files.append(csv_path)
        logger.info(f"Wrote: {csv_path}")

    if "srt" in formats:
        srt_path = os.path.join(output_dir, f"{base}_transcript.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(whisper_results, 1):
                start = _format_ts(seg.get("start", 0))
                end = _format_ts(seg.get("end", 0))
                f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
        output_files.append(srt_path)
        logger.info(f"Wrote: {srt_path}")

    return {
        "transcript": full_transcript,
        "language": detected_lang,
        "output_files": output_files,
    }


def _format_ts(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
