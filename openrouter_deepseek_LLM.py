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

API_KEY = os.getenv("OpenRouter_deepseek_API_key")

#mcp = FastMCP("LearningSpace API (Demo)")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "THE_GRADE", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="tngtech/deepseek-r1t2-chimera:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)