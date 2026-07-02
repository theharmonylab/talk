"""
Embedding extraction entry point for the talk R package.

Wraps whisnemo.core.embed.extract_embeddings so it can be called from R via
reticulate, following the same pattern as diarize.py's diarize_audio():
keyword arguments, a lazy import of the whisnemo backend, and a dict return
whose DataFrame value reticulate converts to an R data.frame.

The R-facing `embeddings` argument ("encoder" / "decoder" / "both") maps to
the backend's return_enc / return_dec flags.
"""

import os
import logging

# See diarize.py: on Windows, urllib's default SSL context enumerates the
# Windows certificate store, which can contain entries OpenSSL 3.x cannot
# parse ("[ASN1: NOT_ENOUGH_DATA]"). Use certifi's bundle for urllib instead.
if os.name == "nt":
    try:
        import ssl
        import certifi

        def _certifi_https_context(*args, **kwargs):
            return ssl.create_default_context(cafile=certifi.where())

        ssl._create_default_https_context = _certifi_https_context
    except Exception:
        pass

logger = logging.getLogger(__name__)


def _resolve_enc_dec(embeddings):
    """Map the R-facing embeddings choice to return_enc / return_dec flags."""
    choice = (embeddings or "encoder").lower()
    if choice == "encoder":
        return True, False
    if choice == "decoder":
        return False, True
    if choice == "both":
        return True, True
    raise ValueError(
        f"Invalid embeddings='{embeddings}'. Use 'encoder', 'decoder', or 'both'."
    )


def embed_audio(
    audio_path,
    transcript,
    model="whisper",
    embeddings="encoder",
    device="cuda",
    participant_only=True,
    whisper_model_id=None,
    whispa_model_id="Jarhatz/WhiSPA-V1-Small",
    whispa_repo_path=None,
    output_dir=None,
):
    """
    Extract per-segment embeddings for a single audio file + diarized transcript.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    transcript : str or pandas.DataFrame
        Diarized transcript: either a path to a CSV, or the DataFrame returned
        by diarize_audio()["transcript"]. Must contain per-segment start/end
        timestamps and (if participant_only) a "speaker_role" column.
    model : str
        "whisper" (encoder/decoder hidden-state summary stats) or
        "whispa" (WhiSPA speech-psychological embedding).
    embeddings : str
        For model="whisper": "encoder", "decoder", or "both". Ignored for
        model="whispa" (which produces a single embedding per segment).
    device : str
        "cuda", "cpu", or "mps" (Apple Silicon).
    participant_only : bool
        If True, only segments with speaker_role == "participant" are embedded.
    whisper_model_id : str, optional
        Override the Whisper model id. Defaults are set by the backend
        ("openai/whisper-small" for whispa, "openai/whisper-medium" for whisper).
    whispa_model_id : str
        WhiSPA checkpoint on HuggingFace (model="whispa" only).
    whispa_repo_path : str, optional
        Path to a local WhiSPA clone, used only if WhiSPA is not pip-installed.
    output_dir : str, optional
        If provided, embeddings are also written to CSV here.

    Returns
    -------
    dict
        {
          "embeddings": pandas.DataFrame (or dict of DataFrames for
                        model="whisper" with embeddings="both"),
          "output_files": list of written CSV paths (empty if output_dir is None),
          "status": "success" | "error",
          ["error": str]   # present only on error
        }
    """
    try:
        from whisnemo.core.embed import extract_embeddings

        return_enc, return_dec = _resolve_enc_dec(embeddings)

        result = extract_embeddings(
            audio_path=audio_path,
            transcript=transcript,
            model=model,
            device=device,
            participant_only=participant_only,
            whisper_model_id=whisper_model_id,
            whispa_model_id=whispa_model_id,
            whispa_repo_path=whispa_repo_path,
            return_enc=return_enc,
            return_dec=return_dec,
        )

        output_files = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(audio_path))[0]
            if isinstance(result, dict):
                for key, df in result.items():
                    path = os.path.join(output_dir, f"{base}_{model}_{key}.csv")
                    df.to_csv(path, index=False)
                    output_files.append(path)
            else:
                suffix = model if model == "whispa" else f"{model}_{embeddings}"
                path = os.path.join(output_dir, f"{base}_{suffix}.csv")
                result.to_csv(path, index=False)
                output_files.append(path)

        return {
            "embeddings": result,
            "output_files": output_files,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Embedding extraction failed for {audio_path}: {e}")
        return {
            "embeddings": None,
            "output_files": [],
            "status": "error",
            "error": str(e),
        }
