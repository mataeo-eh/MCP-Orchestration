import whisper
import json
import torch
from pathlib import Path
import argparse
from openai import OpenAI
import os
import csv
from threading import Lock, Thread
from queue import Queue
import multiprocessing
from collections import Counter
import time
import random
from System_Prompts import (
    DASHBOARD_SUMMARY_SYSTEM_PROMPT,
    DIARIZATION_SYSTEM_PROMPT,
    FILENAME_SYSTEM_PROMPT,
    SOAP_JSON_SYSTEM_PROMPT,
    SOAP_SYSTEM_PROMPT,
    CLINICAL_RISK_SYSTEM_PROMPT,
)

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment


def is_retryable_api_error(exc) -> bool:
    """
    Determine whether an exception from the LLM client is retryable.

    We treat HTTP-like 4xx/5xx conditions as retryable, plus common
    transient error codes found in the message (e.g., 408, 429, 500-504).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if 400 <= status < 600:
            return True

    msg = str(exc)
    transient_codes = ["408", "429", "500", "502", "503", "504"]
    if any(code in msg for code in transient_codes):
        return True

    return False


def call_with_retries(
    fn,
    *,
    max_attempts: int = 3,
    min_delay: float = 1.0,
    max_delay: float = 3.0,
    context_label: str = "LLM call",
):
    """
    Call `fn()` with basic retry logic for transient API failures.

    - Retries up to `max_attempts` times total.
    - Waits a random delay between `min_delay` and `max_delay` seconds between attempts.
    - Only retries when `is_retryable_api_error(exc)` returns True.
    - If all attempts fail, re-raises the last exception so that existing
      error handling logic continues to work unchanged.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc

            if not is_retryable_api_error(exc) or attempt == max_attempts:
                raise

            delay = random.uniform(min_delay, max_delay)
            print(
                f"[Retry] {context_label}: attempt {attempt}/{max_attempts} "
                f"failed with {type(exc).__name__} ({getattr(exc, 'status_code', 'no_status')}). "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc


def parse_cores_argument(cores_str: str) -> int:
    """
    Parse the --cores argument and return valid worker count.
    Returns default (4) for invalid inputs with a warning.
    """
    try:
        cores = int(cores_str)
        if cores == -1:
            return multiprocessing.cpu_count()
        elif cores >= 1:
            return cores
        else:
            print(f"WARNING: Invalid --cores value '{cores_str}', using default: 4")
            return 4
    except ValueError:
        print(f"WARNING: Invalid --cores value '{cores_str}', using default: 4")
        return 4


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio to SOAP format using Whisper.")
    parser.add_argument("--audio-dir", type=str, help="Path to the directory containing audio files to transcribe.")
    parser.add_argument("--model-size", type=str, default="small.en", help="Size of the Whisper model to use.")
    parser.add_argument("--output-dir", type=str, default="Summaries", help="Path to the output directory where transcriptions will be saved.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of audio files to process.")
    parser.add_argument("--whisper-verbose", action="store_true", help="Enable verbose Whisper output during transcription.")
    parser.add_argument("--clinical-assessment", action="store_true", help="Generate clinical risk assessment TXT from SOAP JSON.")
    parser.add_argument("--save-transcript", action="store_true", help="Save raw transcript (default: False).")
    parser.add_argument("--save-diarized", action="store_true", help="Save diarized transcript (default: False).")
    parser.add_argument("--no-save-soap", action="store_true", help="Skip saving SOAP note (default: saves SOAP).")
    parser.add_argument(
        "--debug-diarization",
        action="store_true",
        help="Enable verbose debug logging for diarization failures and diagnostics.",
    )
    parser.add_argument(
        "--debug-whisper",
        action="store_true",
        help="Enable verbose debug logging and quality checks for Whisper transcripts.",
    )
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing of audio files (default: sequential).")
    parser.add_argument("--cores", type=str, default="4",
        help="Number of worker threads for LLM processing (1 to N, or -1 for all cores). Default: 4")
    parser.add_argument("--dashboard", action="store_true",
        help="Generate aggregated dashboard CSV summarizing all processed cases.")
    return parser.parse_args()


def audio_to_text(audio_path, whisper_model, verbose):
    

    print(f"Transcribing {audio_path}...")
    # Perform transcription using the Whisper model
    result = whisper_model.transcribe(audio_path, verbose=verbose)
    
    # Extract the transcript text and strip leading/trailing whitespace
    transcript_text = result.get("text", "").strip()

    return transcript_text


def check_whisper_transcript_quality(transcript: str, debug: bool = False, audio_label: str | None = None) -> None:
    """
    Run lightweight heuristics to flag obviously noisy transcripts (e.g., heavy repetition).
    Logs warnings only; does not alter pipeline flow.
    """
    if not transcript:
        return

    if audio_label is None:
        audio_label = "<unknown-audio>"

    tokens = transcript.split()
    if not tokens:
        return

    lowered_tokens = [t.lower() for t in tokens]
    counts = Counter(lowered_tokens)
    token_count = len(tokens)
    unique_count = len(counts)
    top_word, top_count = counts.most_common(1)[0]
    repetition_ratio = top_count / token_count
    unique_ratio = unique_count / token_count

    repeating_phrase = None
    repeating_phrase_count = 0
    if token_count >= 6:
        trigram_counts = Counter(" ".join(lowered_tokens[i:i+3]) for i in range(len(lowered_tokens) - 2))
        if trigram_counts:
            repeating_phrase, repeating_phrase_count = trigram_counts.most_common(1)[0]

    warnings = []
    if token_count >= 20 and top_count >= 10 and repetition_ratio >= 0.4:
        warnings.append(f"High repetition detected; top word '{top_word}' appears {top_count} times ({repetition_ratio:.0%} of transcript).")
    if token_count >= 40 and unique_ratio <= 0.35:
        warnings.append(f"Low lexical variety: unique/total words = {unique_ratio:.2f}.")
    if repeating_phrase and repeating_phrase_count >= 4:
        warnings.append(f"Repeating phrase '{repeating_phrase}' occurs {repeating_phrase_count} times.")

    if warnings:
        print(f"[Warning] Potential noisy Whisper transcript for {audio_label}.")
        for w in warnings:
            print(f"  - {w}")

    if debug:
        print(f"[WHISPER DEBUG] {audio_label}: tokens={token_count}, unique={unique_count}, top_word='{top_word}' ({repetition_ratio:.2f}), unique_ratio={unique_ratio:.2f}")
        if repeating_phrase:
            print(f"[WHISPER DEBUG] {audio_label}: top trigram '{repeating_phrase}' x{repeating_phrase_count}")


def diarize_transcript(transcript, client, model):
    """
    Takes a raw transcript and returns a speaker-labeled version.
    """
    try:
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": DIARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Label the speakers in this medical transcript:\n\n{transcript}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="Diarization",
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Diarization failed - {e}")
        return None


def diarize_transcript_safe(
    transcript: str,
    client,
    model: str,
    debug: bool = False,
    audio_label: str | None = None,
):
    """
    Wrap diarize_transcript() to make failures non-fatal.

    Returns:
        diarized_text (str): Text that will be passed to SOAP generation.
                             Either the diarized transcript or a fallback.
        status (str): "ok" | "fallback_single_speaker" | "failed_empty"
        error_msg (str | None): Error message if something went wrong.
    """
    if audio_label is None:
        audio_label = "<unknown-audio>"

    try:
        diarized = diarize_transcript(transcript, client, model)

        # Handle None or empty result from diarize_transcript
        if not diarized or not diarized.strip():
            status = "failed_empty"
            error_msg = "Diarization returned None or empty text."

            if debug:
                print(f"[DIARIZATION DEBUG] {audio_label}: empty/None diarization result.")
                print(f"[DIARIZATION DEBUG] Falling back to single-speaker transcript.")

            fallback = f"PATIENT: {transcript}"
            return fallback, status, error_msg

        # Successful diarization
        return diarized, "ok", None

    except Exception as e:
        status = "fallback_single_speaker"
        error_msg = f"{type(e).__name__}: {e}"

        # Always print a concise warning
        print(f"[Warning] Diarization failed for {audio_label}; continuing without speaker labels.")
        if debug:
            print(f"[DIARIZATION DEBUG] {audio_label}: Exception during diarization.")
            print(f"[DIARIZATION DEBUG] {audio_label}: {error_msg}")

        # Fallback: treat transcript as single speaker
        fallback = f"PATIENT: {transcript}"
        return fallback, status, error_msg


def generate_soap_note(diarized_transcript, client, model):
    """
    Takes a speaker-labeled transcript and generates a SOAP note.
    """
    try:
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SOAP_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a SOAP note from this transcript:\n\n{diarized_transcript}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="SOAP text",
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"ERROR: SOAP generation failed - {e}")
        return None


def generate_case_id(soap_note, client, model):
    """
    Generate a case identifier from a SOAP note using the LLM.
    Returns a lowercase, underscore-separated identifier.
    Falls back to "unknown_case" if generation fails.
    """
    try:
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": FILENAME_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a case identifier from this SOAP note:\n\n{soap_note}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="Case ID",
        )
        case_id = resp.choices[0].message.content.strip()
        # Basic validation: ensure it's filesystem-safe
        if not case_id or not all(c.isalnum() or c == '_' for c in case_id):
            print(f"WARNING: Invalid case_id '{case_id}', using fallback")
            return "unknown_case"
        return case_id
    except Exception as e:
        print(f"WARNING: Case ID generation failed - {e}, using fallback")
        return "unknown_case"


def generate_soap_json(diarized_transcript, client, model):
    """
    Generate a structured JSON SOAP note from a diarized transcript.
    Returns parsed dict on success, None on failure.
    Retries once if initial response is not valid JSON.
    """
    try:
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SOAP_JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a structured JSON SOAP note from this transcript:\n\n{diarized_transcript}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="SOAP JSON",
        )
        content = resp.choices[0].message.content.strip()

        # Try to parse JSON
        try:
            soap_json = json.loads(content)
            return soap_json
        except json.JSONDecodeError as e:
            print(f"WARNING: Initial JSON parse failed - {e}")
            print(f"Retrying with clarification...")

            # Retry once with additional guidance
            retry_resp = call_with_retries(
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SOAP_JSON_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Generate a structured JSON SOAP note from this transcript:\n\n{diarized_transcript}"},
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Your previous response was not valid JSON. Please respond with ONLY a valid JSON object, no markdown code blocks or explanatory text."},
                    ],
                ),
                max_attempts=3,
                min_delay=1.0,
                max_delay=3.0,
                context_label="SOAP JSON",
            )
            retry_content = retry_resp.choices[0].message.content.strip()

            try:
                soap_json = json.loads(retry_content)
                print("  JSON parse successful on retry")
                return soap_json
            except json.JSONDecodeError as retry_e:
                print(f"WARNING: Retry JSON parse also failed - {retry_e}")
                return None

    except Exception as e:
        print(f"ERROR: JSON SOAP generation failed - {e}")
        return None


def generate_clinical_risk_assessment(soap_json, client, model):
    """
    Generate clinical risk assessment from SOAP JSON.
    Returns risk assessment text or None on failure.
    """
    try:
        soap_context = json.dumps(soap_json, indent=2)

        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": CLINICAL_RISK_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this SOAP note for clinical risks:\n\n{soap_context}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="Risk assessment",
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Clinical risk assessment failed - {e}")
        return None


def append_to_index(index_path, case_id, audio_filename, summary):
    """
    Append a row to the index.csv file.
    Creates the file with headers if it doesn't exist.
    """
    file_exists = index_path.exists()

    try:
        with open(index_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write headers if file is new
            if not file_exists:
                writer.writerow(['case_id', 'audio_filename', 'summary'])

            # Write data row
            writer.writerow([case_id, audio_filename, summary])
    except Exception as e:
        print(f"WARNING: Failed to write to index.csv - {e}")


def save_case_outputs_and_update_index(
    case_id,
    transcript,
    diarized,
    soap_txt,
    soap_json,
    audio_file_name,
    output_dir,
    args,
    index_lock=None,
    saved_files=None,
):
    """
    Save transcript/diarized/SOAP outputs for a single case and update index.csv.
    Shared by sequential and parallel pipelines.
    """
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    files = list(saved_files) if saved_files else []

    if args.save_transcript:
        transcript_file = case_dir / f"{case_id}_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        files.append(transcript_file.name)

    if args.save_diarized:
        diarized_file = case_dir / f"{case_id}_diarized.txt"
        with open(diarized_file, "w", encoding="utf-8") as f:
            f.write(diarized)
        files.append(diarized_file.name)

    if not args.no_save_soap:
        soap_file = case_dir / f"{case_id}_soap.txt"
        with open(soap_file, "w", encoding="utf-8") as f:
            f.write(soap_txt)
        files.append(soap_file.name)

        if soap_json:
            soap_json_file = case_dir / f"{case_id}_soap.json"
            with open(soap_json_file, "w", encoding="utf-8") as f:
                json.dump(soap_json, f, indent=2)
            files.append(soap_json_file.name)

    print(f"\n  Saved to {case_id}/:")
    for fname in files:
        print(f"    - {fname}")

    one_line_summary = soap_json.get("one_line_summary", "Summary unavailable") if soap_json else "Summary unavailable"
    index_path = output_dir / "index.csv"

    if index_lock:
        with index_lock:
            append_to_index(index_path, case_id, audio_file_name, one_line_summary)
    else:
        append_to_index(index_path, case_id, audio_file_name, one_line_summary)


def generate_dashboard_summary(soap_json, client, model):
    """
    Generate a 1-sentence dashboard summary from SOAP JSON.
    Focus on assessment and plan sections.
    Returns summary string or None on failure.
    """
    try:
        # Extract relevant sections
        assessment = soap_json.get("assessment", {})
        plan = soap_json.get("plan", {})

        # Build context for LLM
        context = f"""Assessment:
- Primary Diagnosis: {assessment.get("primary_diagnosis", "Not documented")}
- Clinical Reasoning: {assessment.get("clinical_reasoning", "Not documented")}

Plan:
- Therapeutic: {plan.get("therapeutic", "Not documented")}
- Follow-up: {plan.get("followup", "Not documented")}
- Disposition: {plan.get("disposition", "Not documented")}"""

        # Call LLM with dashboard summary prompt
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": DASHBOARD_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a 1-sentence dashboard summary from this SOAP data:\n\n{context}"},
                ],
            ),
            max_attempts=3,
            min_delay=1.0,
            max_delay=3.0,
            context_label="Dashboard summary",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"WARNING: Dashboard summary generation failed - {e}")
        return None


def generate_dashboard(output_dir, client, model):
    """
    Scan output directory for SOAP JSON files and generate dashboard.csv
    """
    print("[Dashboard] Scanning for SOAP JSON files...")

    # Find all *_soap.json files recursively
    output_path = Path(output_dir)
    soap_json_files = list(output_path.rglob("*_soap.json"))

    if not soap_json_files:
        print("[Dashboard] No SOAP JSON files found. Nothing to summarize.")
        return

    print(f"[Dashboard] Found {len(soap_json_files)} SOAP JSON file(s)")

    # Collect (case_id, summary) tuples
    dashboard_data = []

    for json_file in soap_json_files:
        try:
            # Load and parse JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                soap_json = json.load(f)

            # Extract case_id from JSON or use parent directory name as fallback
            case_id = soap_json.get("case_id")
            if not case_id:
                # Use parent directory name as fallback
                case_id = json_file.parent.name

            print(f"[Dashboard] Processing: {case_id}")

            # Generate dashboard summary
            summary = generate_dashboard_summary(soap_json, client, model)

            if summary:
                dashboard_data.append((case_id, summary))
                print(f"  Summary: {summary}")
            else:
                print(f"  WARNING: Failed to generate summary, skipping...")

        except json.JSONDecodeError as e:
            print(f"[Dashboard] WARNING: Failed to parse {json_file.name} - {e}")
        except Exception as e:
            print(f"[Dashboard] WARNING: Error processing {json_file.name} - {e}")

    if not dashboard_data:
        print("[Dashboard] No summaries generated. Dashboard not created.")
        return

    # Write dashboard.csv
    dashboard_path = output_path / "dashboard.csv"
    try:
        with open(dashboard_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['case_id', 'summary'])
            writer.writerows(dashboard_data)

        print(f"\n[Dashboard] Successfully saved to {dashboard_path}")
        print(f"[Dashboard] Total cases: {len(dashboard_data)}")
    except Exception as e:
        print(f"[Dashboard] ERROR: Failed to write dashboard.csv - {e}")


def get_client_from_env() -> OpenAI:
    """
    Create and return an OpenAI client configured to use the LiteLLM endpoint.

    Reads configuration from environment variables:
      - OPENAI_API_KEY: your LiteLLM virtual key (required)
      - OPENAI_BASE_URL: base URL for LiteLLM (default: https://litellm.lib.ou.edu/v1)
      - OPENAI_MODEL: default model name to use for evaluations

    TODO:
      - Validate that required environment variables are present.
      - Optionally, provide helpful error messages if something is missing.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable.")

    base_url = os.getenv("URL")
    if not base_url:
        raise RuntimeError("Missing URL environment variable.")
    # You may want to validate that OPENAI_MODEL is set as well
    model = os.getenv("MODEL")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL environment variable.")

    print(f"[config] Using model: {model}")
    print(f"[config] Using base URL: {base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def transcribe_audio(audio_file, whisper_model, verbose):
    """
    Transcribe a single audio file using Whisper.
    Returns transcript text or None on failure.
    """
    try:
        print(f"Transcribing {audio_file.name}...")
        transcript = audio_to_text(audio_file.as_posix(), whisper_model, verbose)

        if not transcript:
            print(f"WARNING: Empty transcript for {audio_file.name}")
            return None

        return transcript
    except Exception as e:
        print(f"ERROR: Transcription failed for {audio_file.name} - {e}")
        return None


def process_transcript(audio_file, transcript, client, llm_model, output_dir, args, index_lock):
    """
    Process a transcript through the LLM pipeline:
    - Diarization
    - JSON SOAP generation
    - Text SOAP generation
    - File saving
    - Index.csv update

    Returns True on success, False on failure.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing transcript: {audio_file.name}")
        print(f"{'='*60}")

        # Step 1: Diarize transcript (label speakers)
        print("\n[Step 1/3] Diarizing transcript...")
        diarized, diar_status, diar_error = diarize_transcript_safe(
            transcript,
            client,
            llm_model,
            debug=getattr(args, "debug_diarization", False),
            audio_label=audio_file.name,
        )

        print(f"  [Diarization] Status: {diar_status}")
        if diar_error and args.debug_diarization:
            print(f"  [DIARIZATION DEBUG] Error: {diar_error}")

        # Step 2: Generate JSON SOAP (includes case_id and one_line_summary)
        print("\n[Step 2/3] Generating structured JSON SOAP note...")
        soap_json = generate_soap_json(diarized, client, llm_model)

        # Step 3: Generate human-readable SOAP (with narrative summary)
        print("\n[Step 3/3] Generating human-readable SOAP note...")
        soap_txt = generate_soap_note(diarized, client, llm_model)

        if not soap_txt:
            print(f"WARNING: Text SOAP generation failed for {audio_file.name}")
            return False

        # Extract case_id from JSON SOAP, or generate from text SOAP as fallback
        if soap_json and "case_id" in soap_json:
            case_id = soap_json["case_id"]
            print(f"\n  Case ID from JSON: {case_id}")
        else:
            print(f"\n  JSON SOAP unavailable, generating case_id from text SOAP...")
            case_id = generate_case_id(soap_txt, client, llm_model)
            print(f"  Case ID: {case_id}")

        # Create case subdirectory
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save files based on CLI flags
        saved_files = []

        # Optional clinical risk assessment
        if args.clinical_assessment:
            if soap_json:
                print("  Generating clinical risk assessment...")
                risk_assessment = generate_clinical_risk_assessment(soap_json, client, llm_model)
                if risk_assessment:
                    risk_file = case_dir / f"{case_id}_risk_assessment.txt"
                    with open(risk_file, "w", encoding="utf-8") as f:
                        f.write(risk_assessment)
                    saved_files.append(risk_file.name)
                else:
                    print("  WARNING: Clinical risk assessment generation failed.")
            else:
                print("  Skipping clinical risk assessment (SOAP JSON unavailable).")

        save_case_outputs_and_update_index(
            case_id=case_id,
            transcript=transcript,
            diarized=diarized,
            soap_txt=soap_txt,
            soap_json=soap_json,
            audio_file_name=audio_file.name,
            output_dir=output_dir,
            args=args,
            index_lock=index_lock,
            saved_files=saved_files,
        )

        print(f"\n  Completed: {audio_file.name}")
        return True

    except Exception as e:
        print(f"ERROR processing transcript for {audio_file.name}: {e}")
        return False


def process_audio_file(audio_file, whisper_model, client, llm_model, output_dir, args, index_lock=None):
    """
    Process a single audio file through the complete pipeline.
    Returns True on success, False on failure.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file.name}")
        print(f"{'='*60}")

        # Step 1: Transcribe audio with Whisper
        print("\n[Step 1/4] Transcribing audio...")
        transcript = audio_to_text(audio_file.as_posix(), whisper_model, args.whisper_verbose)

        if not transcript:
            print(f"WARNING: Empty transcript for {audio_file.name}, skipping...")
            return False

        check_whisper_transcript_quality(
            transcript,
            debug=getattr(args, "debug_whisper", False),
            audio_label=audio_file.name,
        )

        # Step 2: Diarize transcript (label speakers)
        print("\n[Step 2/4] Diarizing transcript...")
        diarized, diar_status, diar_error = diarize_transcript_safe(
            transcript,
            client,
            llm_model,
            debug=getattr(args, "debug_diarization", False),
            audio_label=audio_file.name,
        )

        print(f"  [Diarization] Status: {diar_status}")
        if diar_error and args.debug_diarization:
            print(f"  [DIARIZATION DEBUG] Error: {diar_error}")

        # Step 3: Generate JSON SOAP (includes case_id and one_line_summary)
        print("\n[Step 3/4] Generating structured JSON SOAP note...")
        soap_json = generate_soap_json(diarized, client, llm_model)

        # Step 4: Generate human-readable SOAP (with narrative summary)
        print("\n[Step 4/4] Generating human-readable SOAP note...")
        soap_txt = generate_soap_note(diarized, client, llm_model)

        if not soap_txt:
            print(f"WARNING: Text SOAP generation failed for {audio_file.name}")
            return False

        # Extract case_id from JSON SOAP, or generate from text SOAP as fallback
        if soap_json and "case_id" in soap_json:
            case_id = soap_json["case_id"]
            print(f"\n  Case ID from JSON: {case_id}")
        else:
            print(f"\n  JSON SOAP unavailable, generating case_id from text SOAP...")
            case_id = generate_case_id(soap_txt, client, llm_model)
            print(f"  Case ID: {case_id}")

        # Create case subdirectory
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save files based on CLI flags
        saved_files = []

        if args.clinical_assessment:
            if soap_json:
                print("  Generating clinical risk assessment...")
                risk_assessment = generate_clinical_risk_assessment(soap_json, client, llm_model)
                if risk_assessment:
                    risk_file = case_dir / f"{case_id}_risk_assessment.txt"
                    with open(risk_file, "w", encoding="utf-8") as f:
                        f.write(risk_assessment)
                    saved_files.append(risk_file.name)
                else:
                    print("  WARNING: Clinical risk assessment generation failed.")
            else:
                print("  Skipping clinical risk assessment (SOAP JSON unavailable).")

        save_case_outputs_and_update_index(
            case_id=case_id,
            transcript=transcript,
            diarized=diarized,
            soap_txt=soap_txt,
            soap_json=soap_json,
            audio_file_name=audio_file.name,
            output_dir=output_dir,
            args=args,
            index_lock=index_lock,
            saved_files=saved_files,
        )

        print(f"\n  Completed: {audio_file.name}")
        return True

    except Exception as e:
        print(f"ERROR processing {audio_file.name}: {e}")
        return False


def run_parallel_pipeline(audio_files, whisper_model, client, llm_model, output_dir, args, num_workers):
    """
    Run the audio processing pipeline using a producer-consumer pattern.

    Producer (main thread): Runs Whisper transcription sequentially
    Consumers (worker threads): Process transcripts through LLM pipeline in parallel

    Returns the number of successfully processed files.
    """
    # Create queue for passing transcripts from producer to consumers
    work_queue = Queue()

    # Thread lock for safe index.csv writing
    index_lock = Lock()

    # Track success count (accessed by consumer threads)
    success_count = [0]  # Use list to allow modification in nested function
    success_lock = Lock()

    def consumer_worker():
        """Consumer thread: pulls transcripts from queue and processes them."""
        while True:
            item = work_queue.get()

            # Sentinel value signals thread to exit
            if item is None:
                work_queue.task_done()
                break

            audio_file, transcript = item

            # Process the transcript through LLM pipeline
            if process_transcript(audio_file, transcript, client, llm_model, output_dir, args, index_lock):
                with success_lock:
                    success_count[0] += 1

            work_queue.task_done()

    # Start consumer threads
    consumer_threads = []
    for i in range(num_workers):
        t = Thread(target=consumer_worker, name=f"Consumer-{i+1}")
        t.start()
        consumer_threads.append(t)

    print(f"[parallel] Started {num_workers} consumer thread(s)\n")

    # Producer: Transcribe audio files sequentially and feed to queue
    print("[parallel] Producer: Starting Whisper transcription...\n")
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[Producer {i}/{len(audio_files)}] Transcribing: {audio_file.name}")

        transcript = transcribe_audio(audio_file, whisper_model, args.whisper_verbose)

        if transcript:
            check_whisper_transcript_quality(
                transcript,
                debug=getattr(args, "debug_whisper", False),
                audio_label=audio_file.name,
            )
            # Put transcript into queue for consumer processing
            work_queue.put((audio_file, transcript))
            print(f"  [Producer] Queued {audio_file.name} for LLM processing")
        else:
            print(f"  [Producer] Skipping {audio_file.name} due to transcription failure")

    print(f"\n[Producer] All transcriptions complete. Waiting for LLM processing to finish...\n")

    # Send sentinel values to signal consumers to exit
    for _ in range(num_workers):
        work_queue.put(None)

    # Wait for all work to complete
    work_queue.join()

    # Wait for all consumer threads to finish
    for t in consumer_threads:
        t.join()

    return success_count[0]


def main():
    args = parse_args()

    # Initialize LLM client (needed for both audio processing and dashboard)
    client, llm_model = get_client_from_env()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is standalone dashboard mode (no audio processing)
    if args.dashboard and not args.audio_dir:
        print("\n[Dashboard Mode] Running dashboard generation only (no audio processing)\n")
        generate_dashboard(output_dir, client, llm_model)
        return

    # Audio processing mode (requires audio_dir)
    if not args.audio_dir:
        print("ERROR: --audio-dir is required when not running in dashboard-only mode.")
        print("Usage:")
        print("  Audio processing: python audio_to_soap.py --audio-dir <path> [options]")
        print("  Dashboard only:   python audio_to_soap.py --output-dir <path> --dashboard")
        return

    # Set up compute device for Whisper
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[whisper] Using device: {device}")

    # Load Whisper model
    whisper_model = whisper.load_model(args.model_size, device=device)

    # Set up audio directory
    audio_dir = Path(args.audio_dir)

    # Filter to audio files only, then apply max_files limit
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
    audio_files = [f for f in audio_dir.glob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
    if args.max_files:
        audio_files = audio_files[:args.max_files]

    print(f"\nProcessing {len(audio_files)} audio file(s)...\n")

    if not audio_files:
        print("No audio files found to process.")
        return

    # Process files: parallel or sequential based on --parallel flag
    if args.parallel:
        # Parse and validate the --cores argument
        num_workers = parse_cores_argument(args.cores)
        print(f"[parallel] Producer-consumer mode with {num_workers} LLM worker(s)\n")

        # Run the producer-consumer pipeline
        success_count = run_parallel_pipeline(
            audio_files, whisper_model, client, llm_model,
            output_dir, args, num_workers
        )

        print(f"\n{'='*60}")
        print(f"Done! Successfully processed {success_count}/{len(audio_files)} file(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

    else:
        # Sequential processing
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end=" ")
            if process_audio_file(audio_file, whisper_model, client, llm_model, output_dir, args):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Done! Successfully processed {success_count}/{len(audio_files)} file(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

    # Generate dashboard if requested (post-processing step)
    if args.dashboard:
        print("\n[Dashboard] Generating aggregated dashboard...")
        generate_dashboard(output_dir, client, llm_model)
        dashboard_path = output_dir / "dashboard.csv"
        if dashboard_path.exists():
            print(f"[Dashboard] Saved to {dashboard_path}")


if __name__ == "__main__":
    main()
    
