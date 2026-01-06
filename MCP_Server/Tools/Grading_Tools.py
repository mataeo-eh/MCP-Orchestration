from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for direct script execution (VSCode run button)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP
from datetime import datetime
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
import requests
from System_Prompts import DIARIZATION_SYSTEM_PROMPT, SOAP_SYSTEM_PROMPT, FILENAME_SYSTEM_PROMPT, SOAP_JSON_SYSTEM_PROMPT, CLINICAL_RISK_SYSTEM_PROMPT, DASHBOARD_SUMMARY_SYSTEM_PROMPT, MOCK_RUBRIC_PROMPT, MOCK_RUBRIC_RAW

mcp = FastMCP("OSCE Grading Tools")

# Maximum retries for LLM API calls
MAX_RETRIES = 5


# ============================================================================
# HELPER FUNCTIONS (Internal - not exposed as MCP tools)
# ============================================================================

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


def _build_grading_prompt(transcript_text: str) -> str:
    """
    Build the user prompt for grading a transcript.
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


# ============================================================================
# MCP TOOLS FOR AGENTIC GRADING WORKFLOW
# ============================================================================

@mcp.tool()
def find_student_transcripts(transcripts_dir: str) -> Dict[str, Any]:
    """
    Discover available student transcripts for grading.

    This tool finds all transcript files (.txt) in the specified directory and
    returns metadata about each transcript including the file path, filename,
    student name, and case ID (extracted from filename).

    Args:
        transcripts_dir: Absolute path to directory containing transcript files.
                        Expected format: <student_name>_<case_id>.txt

    Returns:
        Dictionary containing:
        - count: Number of transcripts found
        - transcripts: List of transcript metadata dictionaries, each with:
            - file_path: Absolute path to transcript file
            - filename: Name of the file
            - student_name: Extracted student name (or None if parsing fails)
            - case_id: Extracted case ID (or None if parsing fails)

    Example:
        >>> find_student_transcripts("/path/to/transcripts")
        {
            "count": 2,
            "transcripts": [
                {
                    "file_path": "/path/to/transcripts/alice_jones_SP1.txt",
                    "filename": "alice_jones_SP1.txt",
                    "student_name": "alice_jones",
                    "case_id": "SP1"
                },
                ...
            ]
        }
    """
    dir_path = Path(transcripts_dir)

    if not dir_path.is_dir():
        return {
            "error": "Directory does not exist or is not accessible",
            "path": transcripts_dir,
            "count": 0,
            "transcripts": []
        }

    files = sorted(p for p in dir_path.glob("*.txt") if p.is_file())

    transcripts = []
    for file_path in files:
        # Parse filename: expected format is <student_name>_<case_id>.txt
        stem = file_path.stem
        parts = stem.split("_")

        # Try to extract student name and case ID
        if len(parts) >= 2:
            # Last part is case ID, everything before is student name
            case_id = parts[-1]
            student_name = "_".join(parts[:-1])
        else:
            student_name = None
            case_id = None

        transcripts.append({
            "file_path": str(file_path),
            "filename": file_path.name,
            "student_name": student_name,
            "case_id": case_id
        })

    return {
        "count": len(transcripts),
        "transcripts": transcripts,
        "directory": transcripts_dir
    }


@mcp.tool()
def get_transcript_content(transcript_path: str) -> Dict[str, Any]:
    """
    Load and return the full text content of a specific transcript file.

    This tool reads a transcript file and returns its contents for agent review
    or further processing.

    Args:
        transcript_path: Absolute path to the transcript file to read

    Returns:
        Dictionary containing:
        - success: Boolean indicating if read was successful
        - content: Full text content of the transcript (if successful)
        - file_path: Path that was read
        - error: Error message (if unsuccessful)
        - char_count: Number of characters in transcript

    Example:
        >>> get_transcript_content("/path/to/transcripts/alice_jones_SP1.txt")
        {
            "success": True,
            "content": "DOCTOR: Hello, what brings you in today?...",
            "file_path": "/path/to/transcripts/alice_jones_SP1.txt",
            "char_count": 2543
        }
    """
    file_path = Path(transcript_path)

    if not file_path.is_file():
        return {
            "success": False,
            "error": "File does not exist or is not accessible",
            "file_path": transcript_path
        }

    try:
        text = file_path.read_text(encoding="utf-8").strip()

        if len(text) < 10:
            return {
                "success": False,
                "error": "Transcript file contains fewer than 10 characters",
                "file_path": transcript_path,
                "char_count": len(text)
            }

        return {
            "success": True,
            "content": text,
            "file_path": transcript_path,
            "char_count": len(text)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
            "file_path": transcript_path
        }


@mcp.tool()
def get_grading_rubric() -> Dict[str, Any]:
    """
    Return the OSCE grading rubric used for evaluating student transcripts.

    This tool provides the grading criteria and scoring guidelines that will be
    used to evaluate student-patient interactions. The rubric includes 5 criteria,
    each scored 1-10.

    Returns:
        Dictionary containing:
        - rubric_text: Full text of the scoring rubric
        - criteria: List of criterion names
        - scoring_range: Min and max scores
        - description: Brief description of the rubric

    Example:
        >>> get_grading_rubric()
        {
            "rubric_text": "...",
            "criteria": ["HT", "DD", "Empathy_Rapport", ...],
            "scoring_range": {"min": 1, "max": 10},
            ...
        }
    """
    return {
        "rubric_text": MOCK_RUBRIC_RAW,
        "criteria": [
            "HT",  # History & Information Gathering
            "DD",  # Diagnostic Reasoning Pathway
            "Empathy_Rapport",  # Empathy & Rapport
            "Info_Giving_Clarity",  # Information Giving & Clarity
            "Organization_Closure"  # Organization & Closure
        ],
        "criteria_descriptions": {
            "HT": "History & Information Gathering - Did the student ask clear, relevant, and appropriately open-ended questions?",
            "DD": "Diagnostic Reasoning Pathway - Did the student follow a logical diagnostic path and explore red flags?",
            "Empathy_Rapport": "Empathy & Rapport - Did the student listen, acknowledge emotions, and respond in a patient-centered manner?",
            "Info_Giving_Clarity": "Information Giving & Clarity - Did the student explain findings and next steps clearly?",
            "Organization_Closure": "Organization & Closure - Did the encounter feel structured with a clear plan?"
        },
        "scoring_range": {
            "min": 1,
            "max": 10
        },
        "description": "OSCE evaluation rubric with 5 criteria, each scored 1-10 with justification and improvement suggestions"
    }


@mcp.tool()
def grade_transcript_with_llm(
    transcript_path: str,
    temperature: float = 0.0,
    retry_count: int = 0
) -> Dict[str, Any]:
    """
    Grade a student transcript using LLM API call with OSCE rubric.

    This tool performs the actual LLM-based grading by:
    1. Loading the transcript content
    2. Building a grading prompt with the rubric
    3. Calling the OpenAI-compatible LLM API
    4. Parsing and validating the JSON response
    5. Returning structured grading results

    The LLM API configuration is read from environment variables:
    - API_KEY: API key for the LLM service
    - URL: Base URL for the API endpoint
    - MODEL: Model name to use for grading

    Args:
        transcript_path: Absolute path to the transcript file to grade
        temperature: LLM temperature setting (default: 0.0 for consistency)
        retry_count: Current retry attempt (used internally, default: 0)

    Returns:
        Dictionary containing:
        - success: Boolean indicating if grading was successful
        - transcript_path: Path to the graded transcript
        - model: Model used for grading
        - grades: Dictionary with scores for each criterion (if successful)
        - raw_response: Full LLM response
        - retry_attempts: Number of retry attempts made
        - error: Error message (if unsuccessful)

    Expected grade structure:
        {
            "HT": 8,
            "HT_Justification": "...",
            "HT_Improvement": "To improve, ...",
            "DD": 7,
            "DD_Justification": "...",
            "DD_Improvement": "To improve, ...",
            ... (similar for other criteria)
        }

    Example:
        >>> grade_transcript_with_llm("/path/to/transcripts/alice_jones_SP1.txt")
        {
            "success": True,
            "transcript_path": "/path/to/transcripts/alice_jones_SP1.txt",
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "grades": {
                "HT": 8,
                "HT_Justification": "The student asked relevant questions...",
                "HT_Improvement": "To improve, consider asking more open-ended questions...",
                ...
            },
            "retry_attempts": 0
        }
    """
    # Get LLM API configuration from environment
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("URL")
    model = os.getenv("MODEL")

    if not api_key or not base_url or not model:
        return {
            "success": False,
            "error": "Missing required environment variables: API_KEY, URL, and/or MODEL",
            "transcript_path": transcript_path
        }

    # Load transcript content
    file_path = Path(transcript_path)
    if not file_path.is_file():
        return {
            "success": False,
            "error": "Transcript file does not exist",
            "transcript_path": transcript_path
        }

    try:
        transcript_text = file_path.read_text(encoding="utf-8").strip()
        if len(transcript_text) < 10:
            return {
                "success": False,
                "error": "Transcript file is too short (less than 10 characters)",
                "transcript_path": transcript_path
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading transcript: {str(e)}",
            "transcript_path": transcript_path
        }

    # Build prompts
    user_prompt = _build_grading_prompt(transcript_text)
    system_prompt = MOCK_RUBRIC_PROMPT

    # Make LLM API call
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.choices[0].message.content

    except Exception as e:
        if retry_count < MAX_RETRIES:
            # Retry on API errors
            return grade_transcript_with_llm(
                transcript_path=transcript_path,
                temperature=temperature,
                retry_count=retry_count + 1
            )
        else:
            return {
                "success": False,
                "error": f"LLM API call failed after {MAX_RETRIES} retries: {str(e)}",
                "transcript_path": transcript_path,
                "retry_attempts": retry_count
            }

    # Parse and validate JSON response
    parsed = _parse_json_with_repair(content)

    if parsed is None:
        if retry_count < MAX_RETRIES:
            # Retry on JSON parse errors
            return grade_transcript_with_llm(
                transcript_path=transcript_path,
                temperature=temperature,
                retry_count=retry_count + 1
            )
        else:
            return {
                "success": False,
                "error": f"Failed to parse JSON response after {MAX_RETRIES} retries",
                "transcript_path": transcript_path,
                "raw_response": content,
                "retry_attempts": retry_count
            }

    # Normalize nested JSON fields
    normalized = _normalize_response_payload(parsed)
    if normalized is None:
        if retry_count < MAX_RETRIES:
            return grade_transcript_with_llm(
                transcript_path=transcript_path,
                temperature=temperature,
                retry_count=retry_count + 1
            )
        else:
            return {
                "success": False,
                "error": f"Failed to normalize nested JSON after {MAX_RETRIES} retries",
                "transcript_path": transcript_path,
                "raw_response": content,
                "retry_attempts": retry_count
            }

    return {
        "success": True,
        "transcript_path": transcript_path,
        "model": model,
        "grades": normalized,
        "raw_response": normalized,
        "retry_attempts": retry_count
    }


@mcp.tool()
def parse_transcript_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a transcript filename to extract student name and case ID.

    Expected filename format: <student_name>_<case_id>.txt
    Example: "alice_jones_SP1.txt" -> student_name="alice_jones", case_id="SP1"

    Args:
        filename: Name of the transcript file (with or without .txt extension)

    Returns:
        Dictionary containing:
        - success: Boolean indicating if parsing was successful
        - filename: Original filename
        - student_name: Extracted student name (or None)
        - case_id: Extracted case ID (or None)
        - error: Error message if parsing failed

    Example:
        >>> parse_transcript_filename("alice_jones_SP1.txt")
        {
            "success": True,
            "filename": "alice_jones_SP1.txt",
            "student_name": "alice_jones",
            "case_id": "SP1"
        }
    """
    # Remove .txt extension if present
    stem = filename
    if stem.endswith(".txt"):
        stem = stem[:-4]

    parts = stem.split("_")

    if len(parts) < 2:
        return {
            "success": False,
            "filename": filename,
            "error": "Filename does not follow expected format: <student_name>_<case_id>.txt",
            "student_name": None,
            "case_id": None
        }

    # Last part is case ID, everything before is student name
    case_id = parts[-1]
    student_name = "_".join(parts[:-1])

    return {
        "success": True,
        "filename": filename,
        "student_name": student_name,
        "case_id": case_id
    }