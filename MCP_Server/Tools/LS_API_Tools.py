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
from openai import OpenAI, AsyncOpenAI
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
    activities = []

    for activity_file in activities_path.glob("*.json"):
        with open(activity_file, "r", encoding="utf-8") as f:
            activity_data = json.load(f)
            activities.append(activity_data)
    return activities

@mcp.tool()
def get_cases(input_dir: str = None):
    '''
    Get a list of activities from the Learningspace API
    Returns:
        Retrieve a list of cases.
        Response includes case details such as patient name, presenting complaint, case ID, and case number.
    '''
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    cases_path = Path(input_dir) / "Cases/"
    cases = []

    for case_file in cases_path.glob("*.json"):
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            cases.append(case_data)
    return cases

@mcp.tool()
def get_events_activity(activity_id: str, input_dir: str = None):
    """
    Get events related to a specific activity from the LearningSpace API.
    Args:
        activity_id: The ActivityID to retrieve events for.
    Returns:
        A list of events for the specified activity.
        Response includes event details like event ID, activity ID, event name, date, location, and status.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    events_path = Path(input_dir) / "Events_Activity/"
    events = []

    for event_file in events_path.glob("*.json"):
        with open(event_file, "r", encoding="utf-8") as f:
            event_data = json.load(f)
            events.append(event_data)
    return events

@mcp.tool()
def get_nbome_events(input_dir: str = None):
    """
    Get events marked as ready for NBOME integration from the LearningSpace API.
    Returns:
        A list of NBOME-ready events.
        Response includes event details like event ID, activity ID, event name, NBOME ready status, and submission status.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    events_path = Path(input_dir) / "Nbome_Events/"
    events = []

    for event_file in events_path.glob("*.json"):
        with open(event_file, "r", encoding="utf-8") as f:
            event_data = json.load(f)
            events.append(event_data)
    return events

@mcp.tool()
def get_student_assessments_nbome(event_id: str, input_dir: str = None):
    """
    Get student assessment results for a specific event from the LearningSpace API.
    Args:
        event_id: The EventID to retrieve student assessments for.
    Returns:
        Student assessment results for the specified event in NBOME format.
        Response includes assessment details like student ID, student name, NBOME ID, case details, scores, and pass status.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    assessments_path = Path(input_dir) / "Student_Assessments_Nbome/"
    assessments = []

    for assessment_file in assessments_path.glob("*.json"):
        with open(assessment_file, "r", encoding="utf-8") as f:
            assessment_data = json.load(f)
            assessments.append(assessment_data)
    return assessments

@mcp.tool()
def get_student_video_recordings_nbome(event_id: str, input_dir: str = None):
    """
    Get student video recordings for a specific NBOME event from the LearningSpace API.
    Args:
        event_id: The EventID to retrieve student video recordings for.
    Returns:
        Student video recordings for the specified NBOME event.
        Response includes video recording details like recording ID, student ID, case details, recording path, and duration.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    recordings_path = Path(input_dir) / "Student_Video_Recordings_Nbome/"
    recordings = []

    for recording_file in recordings_path.glob("*.json"):
        with open(recording_file, "r", encoding="utf-8") as f:
            recording_data = json.load(f)
            recordings.append(recording_data)
    return recordings

@mcp.tool()
def get_students(input_dir: str = None):
    """
    Get a list of students from the LearningSpace API.
    Returns:
        A list of students.
        Response includes student details like student ID, full name, NBOME ID, email, enrollment year, and status.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    students_path = Path(input_dir) / "Students/"
    students = []

    for student_file in students_path.glob("*.json"):
        with open(student_file, "r", encoding="utf-8") as f:
            student_data = json.load(f)
            students.append(student_data)
    return students

@mcp.tool()
def get_student_video_recordings_event_case(event_id: str, case_id: str, student_id: str, input_dir: str = None):
    """
    Get video recordings for a specific student, event, and case from the LearningSpace API.
    Args:
        event_id: The EventID to retrieve video recordings for.
        case_id: The CaseID to retrieve video recordings for.
        student_id: The StudentID to retrieve video recordings for.
    Returns:
        Video recordings for the specified student, event, and case.
        Response includes video recording details like recording ID, event ID, case ID, student details, transcript path, and recording duration.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"
    recordings_path = Path(input_dir) / "Student_Video_Recordings_Event_Case/"
    recordings = []

    for recording_file in recordings_path.glob("*.json"):
        with open(recording_file, "r", encoding="utf-8") as f:
            recording_data = json.load(f)
            recordings.append(recording_data)
    return recordings

@mcp.tool()
def get_video_files(video_recording_id: str, input_dir: str = None):
    """
    Get video files associated with a specific video recording from the LearningSpace API.
    Args:
        video_recording_id: The VideoRecordingID to retrieve video files for.
    Returns:
        Video files associated with the specified video recording.
        Response includes video file details like file ID, file name, file URL, file size, duration, format, and resolution.
        Note: This endpoint returns placeholder data as actual video files are not included in the mock implementation.
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / "MOCK_LS_API_ENDPOINTS"

    # Return placeholder response based on the README documentation
    return {
        "success": True,
        "total": 1,
        "offset": 0,
        "limit": 100,
        "data": [
            {
                "videoFileId": f"VF-{video_recording_id}",
                "videoRecordingId": video_recording_id,
                "fileName": f"recording_{video_recording_id}.mp4",
                "fileUrl": f"https://storage.example.com/videos/recording_{video_recording_id}.mp4",
                "fileSize": 125829120,
                "duration": 1245,
                "format": "mp4",
                "resolution": "1920x1080",
                "createdAt": "2024-09-15T08:30:00Z"
            }
        ]
    }






def debug():
    # Test the tool's underlying function
    result = get_activities.fn()  # Some MCP libraries expose .fn or .func
    print("============================")
    print("Debugging call for GET activities")
    print(result)
    print("============================")




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

    debug()

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