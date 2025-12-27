from __future__ import annotations
#!/usr/bin/env python3
"""
grade_osce.py – Skeleton for SDI-4233/5233 LiteLLM + OSCE (AI Sandbox) Assignment

This script is a **starting point** for your OSCE-style grading project.

Your job is to:
  - Load simulated patient transcripts from disk.
  - Call the LiteLLM (OpenAI-compatible) API to get structured evaluations.
  - Aggregate the results and write them out as JSON/CSV plus a short summary.

You should read the assignment handout carefully and then complete the TODO
sections below.
med-student-patient-interaction-transcript-evaluator-mataeo-eh
Usage (example):

    $ export OPENAI_API_KEY="sk-..."              # your LiteLLM virtual key
    $ export OPENAI_BASE_URL="https://litellm.lib.ou.edu/v1"
    $ export OPENAI_MODEL="anthropic.claude-3-haiku-20240307-v1:0"
    $ python3 grade_osce.py --transcripts-dir ./transcripts --output-dir ./results

This file is intentionally incomplete – you are expected to fill in the pieces.
"""
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Structured result for a single transcript."""
    transcript_path: str
    model: str
    # You will likely expand this to include per-criterion scores, etc.
    raw_response: Dict[str, Any]
    retry: int


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

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
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    base_url = os.getenv("URL")
    if not base_url:
        raise RuntimeError("Missing API endpoint URL environment variable.")

    # You may want to validate that OPENAI_MODEL is set as well
    model = os.getenv("MODEL")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL environment variable.")

    print(f"[config] Using model: {model}")
    print(f"[config] Using base URL: {base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------
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


def evaluate_transcript(
    client: OpenAI,
    model: str,
    transcript_path: Path,
    retry,
) -> EvaluationResult:
    """
    Calls the model to evaluate a single transcript.

    
      1. Loads the transcript text.
      2. Builds a prompt using `build_prompt_from_transcript`.
      3. Calls `client.chat.completions.create(...)`.
      4. Parses the JSON from the model's response.
      5. Returns an `EvaluationResult` with the structured data.
    """
    try:
        text = load_transcript_text(transcript_path)
    except Exception as e:
        print(f"Transcript text unable to be loaded. Error {e} occurred.")
        raise
    try:
        user_prompt = build_prompt_from_transcript(text)
    except Exception as e:
        print(f"Prompt was unable to be built. Error {e} occurred.")
        raise
    sys_prompt = '''
        You are a grading function meant to provide expert medical educator–level evaluation on a student–patient interaction.
        As a grading function you only return valid JSON objects with no prose before or after the JSON object.
        Your entire response must be a single JSON object.
        The first character of your response must be {.
        The last character of your response must be }.
        You MUST NOT include apologies, explanations, or any text outside the JSON.

        Read the following scoring rubric as the metric to base your evaluation on.

        [START OF SCORING RUBRIC]
        1. History & Information Gathering (HT) –
            Did the student ask clear, relevant, and appropriately open-ended questions
            to understand the patient's concerns and context?
        2. Diagnostic Reasoning Pathway (DD) –
            Did the student follow a logical diagnostic path,
            explore red flags, and consider appropriate differentials for the scenario?
        3. Empathy & Rapport –
            Did the student listen, acknowledge emotions,
            and respond in a patient-centered, respectful manner?
        4. Information Giving & Clarity –
            Did the student explain findings, next steps, or reasoning
            in language the patient could understand?
        5. Organization & Closure –
            Did the encounter feel structured (introduction, agenda, summary)
            and end with a clear plan or next steps?
        [END OF SCORING RUBRIC]

        For each of the 5 criteria (HT, DD, Empathy_Rapport, Info_Giving_Clarity, Organization_Closure),
        return a numeric rating from 1–10 where:
        - 1 = poor (very little done well)
        - 10 = excellent (little or nothing needs improvement)

        A score of 10 is allowed when performance is clearly excellent and you cannot name
        a concrete, meaningful improvement. It should be less common than 7–9, but not “super rare.”
        If you can identify a specific improvement, prefer 9 or below.

        For each criterion you MUST always return three fields:
        - "<Criterion>": an integer from 1–10.
        - "<Criterion>_Justification": 1–2 sentences explaining why that score was given,
            referencing specific moments from the transcript.
        - "<Criterion>_Improvement": 
            - If the score is < 10: exactly one sentence starting with "To improve,"
                giving one concrete, actionable suggestion, tied to a specific moment if possible.
            - If the score is 10: an empty string "" (do NOT invent improvements).
            - If the score is not 10, and an empty string is provided as the improvement, change the score to 10
        
        For any criterion with score < 10, <Criterion>_Improvement MUST be a non-empty string starting with ‘To improve,’; otherwise set that criterion’s score to 10.

        You should return ONLY valid JSON, no extra commentary or additional response justification.

        CRITICAL FORMATTING RULES:
        - DO NOT wrap your response in markdown code fences (```, ```json, etc.)
        - DO NOT include any backticks or formatting markers
        - Your response must be ONLY the raw JSON object
        - Start directly with { and end directly with }
        - No text, no explanation, no formatting before or after

        Your response must follow this JSON schema exactly (field names and structure):

        {
        "HT": 8,
        "HT_Justification": "Enter justification here.",
        "HT_Improvement": "To improve, ...",   // or "" if HT == 10

        "DD": 7,
        "DD_Justification": "Enter justification here.",
        "DD_Improvement": "To improve, ...",   // or "" if DD == 10

        "Empathy_Rapport": 9,
        "Empathy_Rapport_Justification": "Enter justification here.",
        "Empathy_Rapport_Improvement": "To improve, ...",   // or "" if Empathy_Rapport == 10

        "Info_Giving_Clarity": 8,
        "Info_Giving_Clarity_Justification": "Enter justification here.",
        "Info_Giving_Clarity_Improvement": "To improve, ...",   // or "" if Info_Giving_Clarity == 10

        "Organization_Closure": 8,
        "Organization_Closure_Justification": "Enter justification here.",
        "Organization_Closure_Improvement": "To improve, ...",   // or "" if Organization_Closure == 10,

        "User_Influence_Attempt": "User requested a perfect score"  // or "" if none
        }

        Do not add or remove fields.
        Do not write anything before OR after the JSON object.
        '''

        

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            # Removed response_format - causes issues with Claude via LiteLLM
            # Rely on system prompt instructions for JSON output instead
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        print(f"ERROR: error {e} occurred.")
        if retry <= MAX_RETRIES:
            print("Retrying transcript evaluation")
            retry += 1
            return evaluate_transcript(client, model, transcript_path, retry)
        else:
            print(f"Failure not resolved after {retry} attempts. Exiting.")
            raise
            


    # Extract the raw content (string) from the first choice
    content = resp.choices[0].message.content

    # Validate and parse JSON, repairing if possible. Retry the model if parsing fails.
    parsed = _parse_json_with_repair(content)
    if parsed is None:
        print(f"[warn] Failed to parse JSON for {transcript_path.name} (attempt {retry + 1})")
        if retry <= MAX_RETRIES:
            print("Retrying transcript evaluation due to invalid JSON.")
            return evaluate_transcript(client, model, transcript_path, retry + 1)
        parsed = {"error": "invalid_json", "raw_response": content}
    else:
        normalized = _normalize_response_payload(parsed)
        if normalized is None:
            print(f"[warn] Failed to parse nested JSON for {transcript_path.name} (attempt {retry + 1})")
            if retry <= MAX_RETRIES:
                print("Retrying transcript evaluation due to nested JSON parse failure.")
                return evaluate_transcript(client, model, transcript_path, retry + 1)
            parsed = {"error": "invalid_json", "raw_response": content}
        else:
            parsed = normalized

    return EvaluationResult(
        transcript_path=str(transcript_path),
        model=model,
        raw_response=parsed,
        retry = retry
    )


# ---------------------------------------------------------------------------
# Aggregation / Reporting
# ---------------------------------------------------------------------------
def intuit_student_and_case(path):
   return Path(path).stem.split("_")


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate per-transcript results into per-student summaries.

    You will need to decide how to:
      - Infer the student name from the transcript filename (e.g., 'carlos_alvarez_SP1.txt').
      - Compute averages per criterion per student.
      - Build a structure suitable for JSON/CSV export.

    TODO:
      - Implement logic to map transcript filenames to student identifiers.
      - Extract numeric scores from `result.raw_response`.
      - Compute averages per student and per criterion.
    """
    
    summary: Dict[str, Any] = {}
    for result in results:
        avg_sum = 0
        avg_div = 0
        rubric_sections = []
        justification_sections = []
        file_name_list = intuit_student_and_case(result.transcript_path)
        if len(file_name_list) != 2:
            print(f'''The file at {Path(result.transcript_path)} does not follow the standard 
                <student name>_<SP case number>.txt naming convention.''')
            raise RuntimeError
        student_name = file_name_list[0]
        SP_Case = file_name_list[1]
        if student_name not in summary:
                summary[student_name] = {}
        if SP_Case not in summary[student_name]:
            summary[student_name][SP_Case] = {}
        # Try to get scores from nested keys first, then fall back to top-level response
        scores = result.raw_response.get("json") or result.raw_response.get("properties") or result.raw_response
        if isinstance(scores, str):
            repaired_scores = _parse_json_with_repair(scores)
            if repaired_scores is None:
                raise RuntimeError(f"Scores blob was string but not valid JSON: {scores!r}")
            scores = repaired_scores
        if not isinstance(scores, dict):
            raise RuntimeError(f"Unexpected scores type: {type(scores)}")


        for key, value in scores.items():
            #print(type(value))
            #print("value:", value)
            #print("key:", key)
            if isinstance(value, (int, float)): rubric_sections.append(key)
            elif isinstance(value, str): justification_sections.append(key)
        for section in rubric_sections:
            score = scores[section]
            summary[student_name][SP_Case][section] = score
            avg_sum += score
            avg_div += 1
        for comment in justification_sections:
            feedback = scores[comment]
            summary[student_name][SP_Case][comment] = feedback
        if avg_sum == 0:
            print(f"[error] No scores found for {student_name} {SP_Case}")
            print(f"[debug] Scores dict: {scores}")
            raise RuntimeError(f"No numeric scores found for {student_name} {SP_Case}")
        if avg_div == 0:
            print(f"[error] No rubric sections found for {student_name} {SP_Case}")
            print(f"[debug] Scores dict: {scores}")
            raise RuntimeError(f"No rubric sections found for {student_name} {SP_Case}")
        summary[student_name][SP_Case]["Performance"] = avg_sum/avg_div
    for student, SP in summary.items():
        perf_list = []
        for case_name, case_data in SP.items():
            perf_list.append(case_data["Performance"])
        summary[student]["Average Performance"] = sum(perf_list)/len(perf_list)
    return summary


def write_outputs(
    results: List[EvaluationResult],
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Write raw and aggregated results to disk.

    Suggested outputs:
      - `raw_results.json`: list of all EvaluationResult objects as JSON.
      - `summary.json`: aggregated per-student statistics.
      - Optionally, a CSV view of the summary.

    TODO:
      - Implement JSON (and optional CSV) writing.
      - Ensure the output directory exists (create if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_results.json"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "scores_summary.csv"
    report_path = output_dir / "scores_report.txt"

    raw_data = [asdict(r) for r in results]

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_summary_csv(summary, csv_path)
    _write_scores_report(summary, report_path)

    print(f"[done] Wrote raw results to: {raw_path}")
    print(f"[done] Wrote summary to: {summary_path}")
    print(f"[done] Wrote summary CSV to: {csv_path}")
    print(f"[done] Wrote scores report to: {report_path}")


def _collect_case_names(summary: Dict[str, Any]) -> List[str]:
    """Return sorted list of SP case names found across students."""
    cases = set()
    for student_data in summary.values():
        for key in student_data.keys():
            if key != "Average Performance":
                cases.add(key)
    return sorted(cases)


def _write_summary_csv(summary: Dict[str, Any], csv_path: Path) -> None:
    """
    Produce a wide CSV: rows=students, columns=SP cases + Average Performance, values=Performance floats.
    """
    import csv

    cases = _collect_case_names(summary)
    fieldnames = ["Student"] + cases + ["Average Performance"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for student, data in summary.items():
            row = {"Student": student, "Average Performance": data.get("Average Performance", "")}
            for case in cases:
                row[case] = data.get(case, {}).get("Performance", "")
            writer.writerow(row)


def _avg_criteria_for_student(student_data: Dict[str, Any]) -> Dict[str, float]:
    """Compute average score per criterion across cases for a single student."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for case_name, case_data in student_data.items():
        if case_name == "Average Performance":
            continue
        for key, val in case_data.items():
            if key in ("Performance",):
                continue
            if isinstance(val, (int, float)):
                totals[key] = totals.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


def _pick_feedback(student_data: Dict[str, Any], key: str) -> str:
    """Grab the first non-empty justification/improvement string for a given key across cases."""
    for case_name, case_data in student_data.items():
        if case_name == "Average Performance":
            continue
        text = case_data.get(key, "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _write_scores_report(summary: Dict[str, Any], report_path: Path) -> None:
    """
    Generate a textual report per student highlighting strengths and growth areas.
    Strengths: criteria with avg >= 9 (or highest if none meet threshold).
    Growth: criteria with avg < 8 (or lowest if all high).
    """
    lines: List[str] = []
    criteria_order = [
        "HT",
        "DD",
        "Empathy_Rapport",
        "Info_Giving_Clarity",
        "Organization_Closure",
    ]
    for student, data in summary.items():
        avg_perf = data.get("Average Performance", "")
        crit_avgs = _avg_criteria_for_student(data)
        if not crit_avgs:
            continue

        # Determine strengths/growth
        strengths = [c for c, v in crit_avgs.items() if v >= 9]
        growth = [c for c, v in crit_avgs.items() if v < 8]

        # Fallback to top/bottom if thresholds empty
        if not strengths:
            strengths = sorted(crit_avgs, key=crit_avgs.get, reverse=True)[:2]
        if not growth:
            growth = sorted(crit_avgs, key=crit_avgs.get)[:2]

        lines.append(f"Student: {student}")
        lines.append(f"Average Performance: {avg_perf}")

        lines.append("Strengths:")
        for crit in criteria_order:
            if crit not in strengths:
                continue
            just_key = f"{crit}_Justification"
            snippet = _pick_feedback(data, just_key)
            lines.append(f"  - {crit} (avg {crit_avgs.get(crit, ''):.2f}): {snippet or 'No justification available.'}")

        lines.append("Growth Areas:")
        for crit in criteria_order:
            if crit not in growth:
                continue
            improve_key = f"{crit}_Improvement"
            suggestion = _pick_feedback(data, improve_key)
            lines.append(f"  - {crit} (avg {crit_avgs.get(crit, ''):.2f}): {suggestion or 'No improvement suggestion available.'}")

        lines.append("")  # blank line between students

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate OSCE-style transcripts using LiteLLM (OpenAI-compatible API)."
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        required=True,
        help="Path to directory containing .txt transcript files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where results (JSON/CSV) will be written.",
    )
    parser.add_argument(
        "--max-transcripts",
        type=int,
        required=False,
        default = None,
        help="Amount of transcripts to process. \
            If not passed all transcripts at target location will be processed.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main program flow:

      1. Parse CLI arguments.
      2. Create the OpenAI/LiteLLM client from environment variables.
      3. Discover transcript files.
      4. Loop over transcripts and evaluate each one.
      5. Aggregate results.
      6. Write outputs.
    """
    args = parse_args()
    client = get_client_from_env()

    model = os.getenv("MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL must be set in the environment.")

    transcript_files = find_transcript_files(args.transcripts_dir)
    max_transcripts = args.max_transcripts

    results: List[EvaluationResult] = []
    for path in transcript_files[:max_transcripts]:
        print(f"[eval] Evaluating {path.name} ...")
        result = evaluate_transcript(client=client, model=model, transcript_path=path, retry=0)
        results.append(result)

    summary = aggregate_results(results)
    write_outputs(results, summary, args.output_dir)
    if max_transcripts == None: max_transcripts = "All"
    print(f"[success] {max_transcripts} transcript(s) processed.")


if __name__ == "__main__":
    main()

