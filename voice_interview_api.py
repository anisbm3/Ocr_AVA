"""
Native Voice Interview API

This module provides WebSocket-based real-time voice interview functionality
that can be used directly from a web frontend without relying on Gradio's Stream UI.

Endpoints:
- POST /api/voice-interview/setup - Initialize interview session
- WebSocket /api/voice-interview/stream - Real-time audio streaming
- GET /api/voice-interview/report - Get interview evaluation report
- GET /api/voice-interview/history - Get conversation history
- GET /api/voice-interview/status - Check interview status
"""

import os
import json
import asyncio
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Try to import audio processing libraries
try:
    from fastrtc import get_stt_model, get_tts_model
    AUDIO_MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ FastRTC audio models not available")
    AUDIO_MODELS_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️ Groq not available")
    GROQ_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class InterviewConfig:
    """Interview configuration constants"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 1.5  # seconds
    SILENCE_THRESHOLD = 0.5
    MIN_SPEECH_DURATION_MS = 250
    MIN_SILENCE_DURATION_MS = 2000


# ============================================================================
# Request/Response Models
# ============================================================================

class SetupRequest(BaseModel):
    cvData: Optional[Dict[str, Any]] = None
    jobDescription: Optional[Dict[str, Any]] = None


class SetupResponse(BaseModel):
    success: bool
    sessionId: str
    status: str
    message: str


class StatusResponse(BaseModel):
    active: bool
    currentSection: str
    totalSections: int
    questionsAsked: int


# ============================================================================
# Interview Session Manager
# ============================================================================

class VoiceInterviewSession:
    """
    Manages a voice interview session with state tracking and audio processing.
    """
    
    def __init__(self, session_id: str, cv_data: dict = None, job_desc: dict = None):
        self.session_id = session_id
        self.cv_data = cv_data or {}
        self.job_desc = job_desc or {}
        self.created_at = datetime.now()
        
        # Interview state
        self.current_section = "pre_intro"
        self.sections = [
            "pre_intro", "intro", "hr", "behavioral", "technical", "situational"
        ]
        self.section_states = {s: "pending" for s in self.sections}
        self.section_states["pre_intro"] = "active"
        
        self.exit_counter = 0
        self.is_complete = False
        self.history = []
        self.report = []
        
        # Audio processing
        if AUDIO_MODELS_AVAILABLE:
            self.stt_model = get_stt_model()
            self.tts_model = get_tts_model()
        else:
            self.stt_model = None
            self.tts_model = None
        
        # Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if GROQ_AVAILABLE else None
        
        # File paths
        self.base_dir = Path(__file__).parent / "interview_simulation"
        self.report_file = self.base_dir / "report_interview.json"
        self.history_file = self.base_dir / "conversation_history.json"
        
        # Save CV and job description
        self._save_interview_data()
        
        # Clear previous session data
        self._clear_session_files()
    
    def _save_interview_data(self):
        """Save CV and job description to files"""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            cv_path = self.base_dir / "cv.json"
            with open(cv_path, "w", encoding="utf-8") as f:
                json.dump(self.cv_data, f, ensure_ascii=False, indent=2)
            
            job_path = self.base_dir / "job_description.json"
            with open(job_path, "w", encoding="utf-8") as f:
                json.dump(self.job_desc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving interview data: {e}")
    
    def _clear_session_files(self):
        """Clear history and report files for new session"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error clearing session files: {e}")
    
    def _add_history(self, section: str, role: str, content: str):
        """Add entry to conversation history"""
        entry = {"section": section, "role": role, "content": content}
        self.history.append(entry)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def _save_report(self, report: list):
        """Save evaluation report"""
        self.report = report
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving report: {e}")
    
    def _get_section_prompt(self, section: str, user_input: str) -> str:
        """Get the LLM prompt for current section"""
        base_context = f"""Job Description:
{json.dumps(self.job_desc, ensure_ascii=False, indent=2)}

Candidate CV:
{json.dumps(self.cv_data, ensure_ascii=False, indent=2)}

Instruction:
Always make your questions meet the job description criteria and make harder questions on the skills mentioned in the CV.

History of the conversation:
{json.dumps(self.history, ensure_ascii=False, indent=2)}
"""
        
        section_prompts = {
            "pre_intro": """**ALWAYS ANSWER IN THE FORMAT 0,YOUR_ANSWER OR 1,YOUR_ANSWER, ALWAYS A BINARY DIGIT IN THE BEGINNING OF YOUR RESPONSE, make your responses short and do not ask the user for any code. You are not allowed to make placeholders for anything, if you are not provided with information, do not mention it. You also need to focus on the interview and not indulge the user in any other topic. You will be provided with a conversation history where you will be able to remember progress** The user is initiating a conversation. Respond by welcoming him to the interview in a professional manner, introduce yourself as Alice, the HR manager and begin by asking his name.""",
            
            "intro": f"""The user responded: {user_input}. Classify the user's latest message: if it contains a realistic human name or refuses to give a name, greet and introduce them to an hr question in the format 1,question; if it contains a ridiculous/unrealistic name or no valid name, give them a follow-up question in the format 0,question. Always output only 0,question or 1,question with no extra text, and treat insults or irrelevant replies as invalid names. If the user refuses to give a name, proceed to ask them the HR question directly.""",
            
            "hr": f"""The user responded: {user_input}. If that was a complete answer to the question related to the HR question, or if the user showed in any way that he has no answer, proceed to give him a behavioral question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.""",
            
            "behavioral": f"""The user responded: {user_input}. If that was a complete answer to the question related to the behavioral question, or if the user showed in any way that he has no answer, proceed to give him a technical question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.""",
            
            "technical": f"""The user responded: {user_input}. If that was a complete answer to the question related to the technical question, or if the user showed in any way that he has no answer, proceed to give him a problem-based / scenario-based question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.""",
            
            "situational": f"""The user responded: {user_input}. If that was a complete answer to the problem-based / scenario-based question, or if the user showed in any way that he has no answer, proceed to thank the user and end the interview. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text."""
        }
        
        return base_context + "\nNext task:\n" + section_prompts.get(section, "")
    
    def _call_llm(self, prompt: str) -> str:
        """Call the Groq LLM"""
        if not self.groq_client:
            return "1,I apologize, but the AI service is not available. Please try again later."
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "dismiss"
        except Exception as e:
            print(f"Groq API error: {e}")
            return "dismiss"
    
    def _evaluate_interview(self) -> list:
        """Generate evaluation report"""
        if not self.groq_client:
            return [{"error": "AI service not available"}]
        
        evaluation_prompt = f"""Based on the following conversation history, evaluate the candidate's performance in the interview. Provide feedback on strengths and areas for improvement for each section, as well as the score in valid JSON format only. The JSON should look like this example:

[{{"section": "section_name", "score": "x/100", "strength": "", "weaknesses": "", "general overview": ""}}]

Conversation History:
{json.dumps(self.history, ensure_ascii=False, indent=2)}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=500,
            )

            if response.choices and response.choices[0].message.content:
                evaluation = response.choices[0].message.content.strip()
                clean_eval = evaluation.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_eval)
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        return [{"error": "Evaluation failed"}]
    
    def _advance_section(self):
        """Move to the next interview section"""
        current_idx = self.sections.index(self.current_section)
        self.section_states[self.current_section] = "completed"
        self.exit_counter = 0
        
        if current_idx + 1 < len(self.sections):
            self.current_section = self.sections[current_idx + 1]
            self.section_states[self.current_section] = "active"
        else:
            self.is_complete = True
            # Generate evaluation
            evaluation = self._evaluate_interview()
            self._save_report(evaluation)
    
    async def process_text_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process text input and generate response.
        Used for both transcribed audio and direct text input.
        """
        if self.is_complete:
            return {
                "response": "The interview has been completed. Thank you for your time!",
                "section": "completed",
                "is_complete": True
            }
        
        section_name_map = {
            "pre_intro": "Pre-Introduction Section",
            "intro": "Introduction Section", 
            "hr": "HR Section",
            "behavioral": "Behavioural Section",
            "technical": "Technical Section",
            "situational": "Situational Section"
        }
        
        section_display = section_name_map.get(self.current_section, self.current_section)
        
        # Handle max retries for current section
        if self.exit_counter >= 3:
            self._advance_section()
            if self.is_complete:
                response_text = "Thank you for your time. The interview is now complete. We will get back to you soon. Goodbye!"
            else:
                # Get transition message
                prompt = self._get_section_prompt(self.current_section, user_input)
                response = self._call_llm(prompt)
                response_text = response[2:] if len(response) > 2 else response
            
            return {
                "response": response_text,
                "section": self.current_section,
                "is_complete": self.is_complete
            }
        
        # Get LLM response for current section
        prompt = self._get_section_prompt(self.current_section, user_input)
        response = self._call_llm(prompt)
        
        # Parse response (format: "0,message" or "1,message")
        if len(response) >= 2 and response[0] in ['0', '1']:
            flag = response[0]
            response_text = response[2:] if len(response) > 2 else ""
            
            # Record history
            self._add_history(section_display, "User said", user_input)
            self._add_history(section_display, "You said", response_text)
            
            if flag == '1':
                # Move to next section
                self._advance_section()
            else:
                # Increment retry counter
                self.exit_counter += 1
        else:
            response_text = response
            self._add_history(section_display, "User said", user_input)
            self._add_history(section_display, "You said", response_text)
        
        return {
            "response": response_text,
            "section": self.current_section,
            "is_complete": self.is_complete
        }
    
    def get_welcome_message(self) -> str:
        """Get the initial welcome message"""
        prompt = self._get_section_prompt("pre_intro", "Hello")
        response = self._call_llm(prompt)
        
        # Parse and store
        response_text = response[2:] if len(response) >= 2 and response[0] in ['0', '1'] else response
        self._add_history("Pre-Introduction Section", "You said", response_text)
        
        return response_text
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert audio bytes to text"""
        if not self.stt_model:
            return ""
        
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return self.stt_model.stt((InterviewConfig.SAMPLE_RATE, audio_array))
        except Exception as e:
            print(f"STT error: {e}")
            return ""
    
    async def text_to_speech(self, text: str):
        """Convert text to audio chunks (generator)"""
        if not self.tts_model:
            yield b""
            return
        
        try:
            for audio_chunk in self.tts_model.stream_tts_sync(text):
                yield audio_chunk
        except Exception as e:
            print(f"TTS error: {e}")
            yield b""
    
    def get_status(self) -> StatusResponse:
        """Get current interview status"""
        return StatusResponse(
            active=not self.is_complete,
            currentSection=self.current_section,
            totalSections=len(self.sections),
            questionsAsked=len([h for h in self.history if h["role"] == "You said"])
        )
    
    def get_report(self) -> dict:
        """Get interview report"""
        try:
            if self.report_file.exists():
                with open(self.report_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading report: {e}")
        return []
    
    def get_history(self) -> list:
        """Get conversation history"""
        return self.history


# ============================================================================
# Global Session Store
# ============================================================================

active_sessions: Dict[str, VoiceInterviewSession] = {}


def get_session(session_id: str) -> Optional[VoiceInterviewSession]:
    """Get session by ID"""
    return active_sessions.get(session_id)


def create_session(cv_data: dict = None, job_desc: dict = None) -> VoiceInterviewSession:
    """Create new interview session"""
    import uuid
    session_id = str(uuid.uuid4())
    session = VoiceInterviewSession(session_id, cv_data, job_desc)
    active_sessions[session_id] = session
    return session


# ============================================================================
# FastAPI Application
# ============================================================================

def create_voice_interview_api() -> FastAPI:
    """Create FastAPI app for voice interview"""
    
    api = FastAPI(
        title="Voice Interview API",
        description="WebSocket-based real-time voice interview API",
        version="1.0.0"
    )
    
    # CORS
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @api.post("/api/voice-interview/setup", response_model=SetupResponse)
    async def setup_interview(request: SetupRequest):
        """Initialize a new voice interview session"""
        try:
            session = create_session(
                cv_data=request.cvData,
                job_desc=request.jobDescription
            )
            
            # Get welcome message
            welcome = session.get_welcome_message()
            
            return SetupResponse(
                success=True,
                sessionId=session.session_id,
                status="ready",
                message=welcome
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.get("/api/voice-interview/status/{session_id}")
    async def get_status(session_id: str):
        """Get interview session status"""
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.get_status()
    
    @api.get("/api/voice-interview/report/{session_id}")
    async def get_report(session_id: str):
        """Get interview evaluation report"""
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        report = session.get_report()
        if not report:
            return {"status": "pending", "message": "Interview not yet complete"}
        return {"status": "complete", "report": report}
    
    @api.get("/api/voice-interview/history/{session_id}")
    async def get_history(session_id: str):
        """Get conversation history"""
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"history": session.get_history()}
    
    @api.post("/api/voice-interview/message/{session_id}")
    async def send_message(session_id: str, message: dict):
        """Send text message and get response (for text-based mode)"""
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user_input = message.get("text", "")
        if not user_input:
            raise HTTPException(status_code=400, detail="No message text provided")
        
        result = await session.process_text_input(user_input)
        return result
    
    @api.websocket("/api/voice-interview/stream/{session_id}")
    async def websocket_stream(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for real-time audio streaming.
        
        Protocol:
        - Client sends: {"type": "audio", "data": "<base64_audio>"}
        - Client sends: {"type": "text", "data": "<text_message>"}
        - Server sends: {"type": "response", "text": "...", "section": "...", "audio": "<base64>"}
        - Server sends: {"type": "status", "section": "...", "complete": false}
        """
        await websocket.accept()
        
        session = get_session(session_id)
        if not session:
            await websocket.send_json({"type": "error", "message": "Session not found"})
            await websocket.close()
            return
        
        try:
            # Send initial status
            await websocket.send_json({
                "type": "status",
                "section": session.current_section,
                "complete": session.is_complete
            })
            
            while True:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    # Process audio data
                    audio_base64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Speech to text
                    text = session.speech_to_text(audio_bytes)
                    
                    if text:
                        # Process text and get response
                        result = await session.process_text_input(text)
                        
                        # Generate audio response
                        audio_chunks = []
                        async for chunk in session.text_to_speech(result["response"]):
                            if chunk:
                                audio_chunks.append(chunk)
                        
                        audio_response = base64.b64encode(b''.join(audio_chunks)).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "response",
                            "text": result["response"],
                            "userText": text,
                            "section": result["section"],
                            "complete": result["is_complete"],
                            "audio": audio_response
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Could not understand audio"
                        })
                
                elif msg_type == "text":
                    # Process text directly
                    text = data.get("data", "")
                    if text:
                        result = await session.process_text_input(text)
                        
                        # Generate audio response if TTS available
                        audio_response = ""
                        if session.tts_model:
                            audio_chunks = []
                            async for chunk in session.text_to_speech(result["response"]):
                                if chunk:
                                    audio_chunks.append(chunk)
                            audio_response = base64.b64encode(b''.join(audio_chunks)).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "response",
                            "text": result["response"],
                            "section": result["section"],
                            "complete": result["is_complete"],
                            "audio": audio_response
                        })
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif msg_type == "end":
                    # End interview early
                    session.is_complete = True
                    evaluation = session._evaluate_interview()
                    session._save_report(evaluation)
                    await websocket.send_json({
                        "type": "complete",
                        "message": "Interview ended"
                    })
                    break
        
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            print(f"WebSocket error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass
    
    return api


# Standalone runner
if __name__ == "__main__":
    import uvicorn
    
    api = create_voice_interview_api()
    uvicorn.run(api, host="0.0.0.0", port=7862)
