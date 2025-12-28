# Video Files Endpoint

This directory is a placeholder for the Video Files API endpoint.

## Endpoint

**GET** `/api/data/v1/video-files/video-recording/{VideoRecordingID}`

## Description

This endpoint retrieves video file information associated with a specific video recording. When requesting video file content, the API provides links to the video recording files in MP4 format.

## Note

Mock data files for this endpoint have not been implemented as they would require actual video file references. This endpoint would typically return URLs or file paths to MP4 video recordings.

## Example Response Structure

```json
{
  "success": true,
  "total": 1,
  "offset": 0,
  "limit": 100,
  "data": [
    {
      "videoFileId": "VF-001",
      "videoRecordingId": "VR-001",
      "fileName": "recording_001.mp4",
      "fileUrl": "https://storage.example.com/videos/recording_001.mp4",
      "fileSize": 125829120,
      "duration": 1245,
      "format": "mp4",
      "resolution": "1920x1080",
      "createdAt": "2024-09-15T08:30:00Z"
    }
  ]
}
```
