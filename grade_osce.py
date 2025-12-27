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
from grade_osce_helpers import build_prompt_from_transcript, _parse_json_with_repair, _normalize_response_payload, load_transcript_text




def evaluate_transcript(transcript_path: Path, retry, client, model, MAX_RETRIES=5,):
    """
    Calls the model to evaluate a single transcript.

    
      1. Loads the transcript text.
      2. Builds a prompt using `build_prompt_from_transcript`.
      3. Calls `client.chat.completions.create(...)`.
      4. Parses the JSON from the model's response.
      5. Returns an `EvaluationResult` with the structured data.
    """
    output_dict = {
                    'errors': [],
                   'warnings': []
                }
    try:
        text = load_transcript_text(transcript_path)
    except Exception as e:
        output_dict['errors'].append(f"Transcript text unable to be loaded. Error {e} occurred.")
        raise
    try:
        user_prompt = build_prompt_from_transcript(text)
    except Exception as e:
        output_dict['errors'].append(f"Prompt was unable to be built. Error {e} occurred.")
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
        output_dict['errors'].append(f"ERROR: error {e} occurred.")
        if retry <= MAX_RETRIES:
            output_dict['errors'].append("Retrying transcript evaluation")
            retry += 1
            return evaluate_transcript(transcript_path, retry, client, model, MAX_RETRIES=5,)
        else:
            output_dict['errors'].append(f"Failure not resolved after {retry} attempts. Exiting.")
            raise
            
    # Extract the raw content (string) from the first choice
    content = resp.choices[0].message.content

    # Validate and parse JSON, repairing if possible. Retry the model if parsing fails.
    parsed = _parse_json_with_repair(content)
    if parsed is None:
        output_dict['warnings'].append(f"[warn] Failed to parse JSON for {transcript_path.name} (attempt {retry + 1})")
        if retry <= MAX_RETRIES:
            output_dict['warnings'].append("Retrying transcript evaluation due to invalid JSON.")
            return evaluate_transcript(transcript_path, retry + 1, client, model, MAX_RETRIES=5,)
        parsed = {"error": "invalid_json", "raw_response": content}
    else:
        normalized = _normalize_response_payload(parsed)
        if normalized is None:
            output_dict['warnings'].append(f"[warn] Failed to parse nested JSON for {transcript_path.name} (attempt {retry + 1})")
            if retry <= MAX_RETRIES:
                output_dict['warnings'].append("Retrying transcript evaluation due to nested JSON parse failure.")
                return evaluate_transcript(transcript_path, retry + 1, client, model, MAX_RETRIES=5,)
            parsed = {"error": "invalid_json", "raw_response": content}
        else:
            parsed = normalized




    result_dict = {
            "transcript_path": str(transcript_path),
            "model": model,
            "raw_response": {
                **parsed,
                "_meta": {
                    "errors": output_dict['errors'],
                    "warnings": output_dict['warnings'],
                    "retry_count": retry
                }
            },
            "retry": retry
        }

    return result_dict


def intuit_student_and_case(path):
   '''
    Given a transcript file path, returns the student identifier and case identifier.
   '''

   return Path(path).stem.split("_")

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

def main():
    client = get_client_from_env()
    evaluate_transcript(
        transcript_path=Path("transcripts/sample_transcript.txt"),
        retry=0,        
        client=client,
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        MAX_RETRIES=5,
    )

if __name__ == "__main__":
    main()