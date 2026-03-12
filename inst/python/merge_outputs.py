"""
Standalone merge module for the talk R package.
Merges multi-part audio files based on Excel metadata using FFmpeg.

Called from R via reticulate::source_python("merge_outputs.py")

Usage as function:
    merge_audio_files(excel_path, audio_dir, output_dir)
"""

import os
import subprocess
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_audio_files(
    excel_path,
    audio_dir,
    output_dir="merged_audio",
    skip_existing=True,
):
    """
    Merge multi-part audio files based on Excel crosswalk metadata.

    Parameters
    ----------
    excel_path : str
        Path to Excel metadata file with columns: ID, Part, File Name.
    audio_dir : str
        Directory containing source audio files.
    output_dir : str
        Directory to save merged files.
    skip_existing : bool
        If True, skip files that already exist in output_dir.

    Returns
    -------
    dict
        Keys: "success_count", "skip_count", "fail_count", "missing_files".
    """
    df = pd.read_excel(excel_path, sheet_name=0)

    required_cols = ["ID", "Part", "File Name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group by ID
    audio_groups = defaultdict(list)
    for _, row in df.iterrows():
        audio_id = str(int(row["ID"]))
        part = int(row["Part"])
        filename = str(row["File Name"]).strip()
        audio_groups[audio_id].append((part, filename))

    for audio_id in audio_groups:
        audio_groups[audio_id].sort(key=lambda x: x[0])

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0
    missing_files = []

    for audio_id, part_files in sorted(audio_groups.items(), key=lambda x: int(x[0])):
        output_path = os.path.join(output_dir, f"{audio_id}_Part_I_merged.mp3")

        if skip_existing and os.path.exists(output_path):
            skip_count += 1
            continue

        # Verify all files exist
        all_exist = True
        for part_num, filename in part_files:
            if not os.path.exists(os.path.join(audio_dir, filename)):
                missing_files.append((audio_id, part_num, filename))
                all_exist = False

        if not all_exist:
            fail_count += 1
            continue

        # Merge
        if len(part_files) == 1:
            _, source_file = part_files[0]
            source_path = os.path.join(audio_dir, source_file)
            cmd = ["ffmpeg", "-i", source_path, "-c:a", "libmp3lame", "-q:a", "2", "-y", output_path]
        else:
            filelist_path = f"/tmp/filelist_{audio_id}.txt"
            with open(filelist_path, "w", encoding="utf-8") as f:
                for _, filename in part_files:
                    escaped = os.path.join(audio_dir, filename).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist_path,
                   "-c:a", "libmp3lame", "-q:a", "2", "-y", output_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            success_count += 1
            logger.info(f"Merged: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {output_path}: {e.stderr.decode()}")
            fail_count += 1
        finally:
            filelist_path = f"/tmp/filelist_{audio_id}.txt"
            if os.path.exists(filelist_path):
                os.remove(filelist_path)

    logger.info(f"Done: {success_count} merged, {skip_count} skipped, {fail_count} failed")
    return {
        "success_count": success_count,
        "skip_count": skip_count,
        "fail_count": fail_count,
        "missing_files": missing_files,
    }
