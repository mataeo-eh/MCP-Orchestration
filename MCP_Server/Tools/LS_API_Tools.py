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

mcp = FastMCP("LearningSpace API (Demo)")

@mcp.tool()
def get_activities(input_dir: str = None):
    """
    Get a list of activities from the LearningSpace API.
    Returns:
        A list of activities.
        Response includes activity details like ID, title, start, and end dates.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    activities_path = Path(input_dir) / "Activities/"
    response = []

    for activity_file in activities_path.glob("*.json"):
        with open(activity_file, "r", encoding="utf-8") as f:
            activity_data = json.load(f)
            response.append(activity_data)

    print(response, "\n")
    return response

# Test the tool's underlying function
result = get_activities.fn()  # Some MCP libraries expose .fn or .func




if __name__ == "__main__":
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


    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "HI! How are you?"},
        ],
    )

    print(resp.choices[0].message.content)