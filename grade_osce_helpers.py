from __future__ import annotations
#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI



def build_prompt_from_transcript(transcript_text: str) -> str:
    """
    Build the prompt string to send to the model for a single transcript.

    This should follow the structure described in the assignment (e.g., include
    instructions about returning JSON with specific keys and including the
    transcript at the end).
    """
   
    prompt = f"""
        Read the following transcript and then produce a valid JSON object with your evaluation
        of the student's performance across the five criteria described by the scoring rubric. 

        [START OF TRANSCRIPT]
        {transcript_text}
        [END OF TRANSCRIPT]

        (Remember: return ONLY valid JSON, no extra commentary.)
"""
    return prompt.strip()

# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------
def _repair_missing_commas(text: str) -> str:
    """
    Best-effort fix for outputs where fields are separated by newlines but missing commas.

    Adds a trailing comma to a line if the next non-empty line starts with a quote
    (indicating another key) and the current line does not already end with a comma.
    """
    lines = text.splitlines()
    repaired: List[str] = []
    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        repaired.append(stripped)
        if not stripped.strip():
            continue
        next_nonempty = None
        for j in range(idx + 1, len(lines)):
            candidate = lines[j].strip()
            if candidate:
                next_nonempty = candidate
                break
        if (
            next_nonempty
            and next_nonempty.startswith('"')
            and not stripped.rstrip().endswith(",")
            and not stripped.strip().endswith(("{", "["))
        ):
            repaired[-1] = stripped + ","
    return "\n".join(repaired)


def _parse_json_with_repair(blob: str) -> Dict[str, Any] | None:
    """
    Attempt to parse JSON; on failure, try a simple comma-repair before giving up.
    Returns None if parsing still fails.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        repaired = _repair_missing_commas(blob)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def _normalize_response_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Ensure embedded 'json' or 'properties' fields are parsed dicts, not raw strings.
    """
    normalized = payload.copy()
    for key in ("json", "properties"):
        if key in normalized and isinstance(normalized[key], str):
            parsed_inner = _parse_json_with_repair(normalized[key])
            if parsed_inner is None:
                return None
            normalized[key] = parsed_inner
    return normalized


def load_transcript_text(path: Path) -> str:
    """
    Read a transcript file from disk and return its contents as a single string.

    The transcript files are expected to be plain text, with each line containing
    speaker labels and dialogue, e.g.:

        Leroy Lookinglad: ...
        Jeremy Jovial: ...

    TODO:
      - You may want to strip trailing whitespace.
      - You may want to handle encoding issues or empty files.
    """
   
    if not path.is_file():
        raise RuntimeError("The path specified does not point to an existing file.")
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    if len(text) < 10:
        raise RuntimeError("The file loaded contains fewer than 10 characters." \
        " Ensure the file path correctly points to the transcript you want to load.")
    return text


def find_transcript_files(transcripts_dir: Path) -> List[Path]:
    """
    Return a list of all transcript files in the given directory.

    We assume all transcript files end with ".txt".
    """
    if not transcripts_dir.is_dir():
        raise RuntimeError(f"Transcripts directory does not exist: {transcripts_dir}, \
                           or is not a directory")

    files = sorted(p for p in transcripts_dir.glob("*.txt") if p.is_file())
    print(f"[info] Found {len(files)} transcript files in {transcripts_dir}")
    return files

def main():
    pass

if __name__ == "__main__":
    main()