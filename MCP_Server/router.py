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
from ..System_Prompts import MCP_ROUTER_PROMPT




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
        {"role": "system", "content": MCP_ROUTER_PROMPT},
        {"role": "user", "content": "HI! How are you?"},
    ],
)

print(resp.choices[0].message.content)



