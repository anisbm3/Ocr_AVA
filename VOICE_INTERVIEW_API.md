# 🎙️ FastRTC Voice Interview API Documentation

This document describes the Gradio API endpoints for the FastRTC Voice Interview feature in the Python application.

## Overview

The Voice Interview system uses FastRTC for real-time audio processing, enabling voice-based AI interviews. The system conducts structured interviews through multiple sections:

1. **Pre-Introduction** - Welcome and name request
2. **Introduction** - Name verification
3. **HR Questions** - General HR-related questions
4. **Behavioral Questions** - Situational behavioral questions
5. **Technical Questions** - Role-specific technical questions
6. **Situational Questions** - Problem-solving scenarios

---

## API Endpoints

### Base URL
```
http://localhost:7861/gradio_api/call
```

### Authentication
No authentication required (local development only).

---

## 1. Setup Voice Interview

**Endpoint:** `/lambda`

Initializes the voice interview session with CV and job description data.

### Request

```bash
curl -X POST http://localhost:7861/gradio_api/call/lambda \
  -s -H "Content-Type: application/json" \
  -d '{
    "data": [
      "{\"personalInfo\":{\"name\":\"John Doe\",\"email\":\"john@example.com\"},\"experience\":[...]}",
      "{\"title\":\"Software Engineer\",\"requirements\":[\"Python\",\"FastAPI\"]}"
    ]
  }' \
  | awk -F'"' '{ print $4 }' \
  | read EVENT_ID; curl -N http://localhost:7861/gradio_api/call/lambda/$EVENT_ID
```

### Parameters

| Index | Type   | Required | Description                                      |
|-------|--------|----------|--------------------------------------------------|
| [0]   | string | No       | CV JSON - Parsed CV data from extraction tab     |
| [1]   | string | No       | Job Description JSON - Job requirements and info |

### Response

```json
{
  "data": ["✅ Voice interview ready! Click Start to begin."]
}
```

### Response Values

| Status | Message                                                    |
|--------|-----------------------------------------------------------|
| ✅     | Voice interview ready! Click Start to begin.              |
| ❌     | FastRTC is not available. Please install fastrtc package. |
| ❌     | Failed to initialize interview simulator                   |
| ❌     | Error creating voice stream: {error_details}              |

---

## 2. Get Voice Interview Report

**Endpoint:** `/get_voice_interview_report`

Retrieves the evaluation report after the interview is completed.

### Request

```bash
curl -X POST http://localhost:7861/gradio_api/call/get_voice_interview_report \
  -s -H "Content-Type: application/json" \
  -d '{"data": []}' \
  | awk -F'"' '{ print $4 }' \
  | read EVENT_ID; curl -N http://localhost:7861/gradio_api/call/get_voice_interview_report/$EVENT_ID
```

### Parameters

None required.

### Response (Markdown format)

```markdown
# 📋 Voice Interview Evaluation Report

## Pre-Introduction Section

**Score:** 85/100

**Strengths:** Candidate showed excellent communication skills...

**Weaknesses:** Could improve on initial greeting...

**Overview:** Good start to the interview...

---

## HR Section

**Score:** 78/100

...
```

### Response Structure (JSON internally)

The report is stored as JSON and formatted to Markdown:

```json
[
  {
    "section": "Pre-Introduction Section",
    "score": "85/100",
    "strength": "Candidate showed excellent communication skills...",
    "weaknesses": "Could improve on initial greeting...",
    "general overview": "Good start to the interview..."
  },
  {
    "section": "HR Section",
    "score": "78/100",
    "strength": "...",
    "weaknesses": "...",
    "general overview": "..."
  }
]
```

### Status Messages

| Status | Message                                                 |
|--------|--------------------------------------------------------|
| ⏳     | No evaluation available yet. Complete the interview first. |
| ❌     | Error getting evaluation: {error_details}               |

---

## 3. Get Voice Interview History

**Endpoint:** `/get_voice_interview_history`

Retrieves the full conversation history from the interview session.

### Request

```bash
curl -X POST http://localhost:7861/gradio_api/call/get_voice_interview_history \
  -s -H "Content-Type: application/json" \
  -d '{"data": []}' \
  | awk -F'"' '{ print $4 }' \
  | read EVENT_ID; curl -N http://localhost:7861/gradio_api/call/get_voice_interview_history/$EVENT_ID
```

### Parameters

None required.

### Response (JSON)

```json
[
  {
    "section": "Pre-Introduction Section",
    "role": "User said",
    "content": "Hello, I'm here for the interview"
  },
  {
    "section": "Pre-Introduction Section",
    "role": "You said",
    "content": "Welcome! I'm Alice, the HR manager. May I have your name please?"
  },
  {
    "section": "Introduction Section",
    "role": "User said",
    "content": "My name is John Doe"
  },
  {
    "section": "Introduction Section",
    "role": "You said",
    "content": "Nice to meet you, John! Let's begin with some HR questions..."
  }
]
```

### Status Messages

| Status | Message                                  |
|--------|------------------------------------------|
| -      | No conversation history yet.             |
| -      | No conversation history file found.      |
| ❌     | Error getting history: {error_details}   |

---

## JavaScript/TypeScript Integration Examples

### Using fetch API

```typescript
// Setup Voice Interview
async function setupVoiceInterview(cvJson: string, jobDescJson: string): Promise<string> {
  // Step 1: POST to initiate
  const response = await fetch('http://localhost:7861/gradio_api/call/lambda', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [cvJson, jobDescJson]
    })
  });
  
  const result = await response.json();
  const eventId = result.event_id;
  
  // Step 2: GET results using event_id
  const resultResponse = await fetch(
    `http://localhost:7861/gradio_api/call/lambda/${eventId}`
  );
  
  // Parse SSE response
  const text = await resultResponse.text();
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      const data = JSON.parse(line.substring(5));
      return data[0]; // Status message
    }
  }
  
  throw new Error('Failed to get response');
}

// Get Interview Report
async function getVoiceInterviewReport(): Promise<string> {
  const response = await fetch('http://localhost:7861/gradio_api/call/get_voice_interview_report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [] })
  });
  
  const result = await response.json();
  const eventId = result.event_id;
  
  const resultResponse = await fetch(
    `http://localhost:7861/gradio_api/call/get_voice_interview_report/${eventId}`
  );
  
  const text = await resultResponse.text();
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      const data = JSON.parse(line.substring(5));
      return data[0]; // Markdown report
    }
  }
  
  throw new Error('Failed to get report');
}

// Get Interview History
async function getVoiceInterviewHistory(): Promise<any[]> {
  const response = await fetch('http://localhost:7861/gradio_api/call/get_voice_interview_history', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [] })
  });
  
  const result = await response.json();
  const eventId = result.event_id;
  
  const resultResponse = await fetch(
    `http://localhost:7861/gradio_api/call/get_voice_interview_history/${eventId}`
  );
  
  const text = await resultResponse.text();
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      const data = JSON.parse(line.substring(5));
      return JSON.parse(data[0]); // Parse JSON history
    }
  }
  
  throw new Error('Failed to get history');
}
```

### Using axios (with helper)

```typescript
import axios from 'axios';

const GRADIO_BASE_URL = 'http://localhost:7861/gradio_api/call';

async function callGradioAPI(endpoint: string, data: any[] = []): Promise<any> {
  // Step 1: POST to initiate the call
  const postResponse = await axios.post(`${GRADIO_BASE_URL}/${endpoint}`, {
    data
  });
  
  const eventId = postResponse.data.event_id;
  
  // Step 2: GET results (SSE stream)
  const getResponse = await axios.get(`${GRADIO_BASE_URL}/${endpoint}/${eventId}`, {
    responseType: 'text'
  });
  
  // Parse SSE response
  const lines = getResponse.data.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      return JSON.parse(line.substring(5));
    }
  }
  
  throw new Error('No data received');
}

// Usage examples
const setupResult = await callGradioAPI('lambda', [cvJson, jobDescJson]);
const report = await callGradioAPI('get_voice_interview_report');
const history = await callGradioAPI('get_voice_interview_history');
```

---

## WebRTC Stream Access

The actual voice interview stream uses WebRTC for real-time audio. The stream URL format is:

```
http://localhost:7861/?__theme=light
```

### Embedding in iframe

```html
<iframe 
  src="http://localhost:7861/?__theme=light" 
  width="100%" 
  height="600px" 
  allow="microphone; camera"
  style="border: none;"
></iframe>
```

### Important Notes

1. **Microphone Permission**: The browser will request microphone access
2. **Same-Origin Policy**: May need CORS configuration for cross-origin access
3. **Stream URL**: The stream UI is embedded within the Gradio interface

---

## File Locations

The interview system uses these files in `python/interview_simulation/`:

| File | Description |
|------|-------------|
| `cv.json` | Candidate's CV data (written by setup) |
| `job_description.json` | Job requirements (written by setup) |
| `conversation_history.json` | Full interview conversation log |
| `report_interview.json` | Evaluation report after completion |

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| FastRTC not available | Package not installed | Run `pip install fastrtc` |
| GROQ_API_KEY missing | Environment variable not set | Add to `.env` file |
| Empty CV/Job data | Files not properly written | Check file permissions |
| WebRTC connection failed | Browser/network issue | Check microphone permissions |

### Environment Variables Required

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Interview Flow Diagram

```
┌─────────────────┐
│  Setup Interview │
│  POST /lambda    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pre-Introduction │
│ (Welcome, Name)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Introduction   │
│ (Name Verify)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HR Questions   │
│ (3 follow-ups)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Behavioral    │
│ (3 follow-ups)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Technical     │
│ (3 follow-ups)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Situational    │
│ (3 follow-ups)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation      │
│ Report Generated│
└─────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ GET /get_voice_interview_report │
│ GET /get_voice_interview_history│
└─────────────────────────────────┘
```

---

## Testing with cURL

### Quick Test Script (PowerShell)

```powershell
# Test Setup
$setupResponse = Invoke-RestMethod -Uri "http://localhost:7861/gradio_api/call/lambda" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"data": ["{}", "{}"]}'

$eventId = $setupResponse.event_id
Write-Host "Event ID: $eventId"

# Get Result
$result = Invoke-WebRequest -Uri "http://localhost:7861/gradio_api/call/lambda/$eventId"
Write-Host "Result: $($result.Content)"
```

### Quick Test Script (Bash)

```bash
#!/bin/bash

# Test Setup
EVENT_ID=$(curl -s -X POST http://localhost:7861/gradio_api/call/lambda \
  -H "Content-Type: application/json" \
  -d '{"data": ["{}", "{}"]}' | jq -r '.event_id')

echo "Event ID: $EVENT_ID"

# Get Result
curl -N http://localhost:7861/gradio_api/call/lambda/$EVENT_ID
```

---

## Version Information

- **FastRTC Version**: Required
- **Gradio Version**: 4.x+
- **Python Version**: 3.9+
- **Groq Model**: llama-3.3-70b-versatile
