from __future__ import annotations
from fastmcp import FastMCP
from datetime import datetime
import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
import requests
from System_Prompts import DIARIZATION_SYSTEM_PROMPT, SOAP_SYSTEM_PROMPT, FILENAME_SYSTEM_PROMPT, SOAP_JSON_SYSTEM_PROMPT, CLINICAL_RISK_SYSTEM_PROMPT, DASHBOARD_SUMMARY_SYSTEM_PROMPT, MOCK_RUBRIC_PROMPT, MOCK_RUBRIC_RAW




mcp = FastMCP("LearningSpace API (Demo)")








if __name__ == "__main__":
    mcp.run(transport="stdio")