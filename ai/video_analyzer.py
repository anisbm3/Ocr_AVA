"""
Enhanced Video CV Agent using Qwen2.5-VL for visual analysis
Combines Whisper for audio transcription with Qwen for frame analysis
"""
import os
import cv2
import json
import logging
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64

# Import Qwen Vision Agent
from ai.qwen_vision_agent import QwenVisionAgent

# Import base agent if available
try:
    from base_agent import BaseAgent, AgentResult
except ImportError:
    print("⚠ Base agent not available, using minimal implementation")
    from dataclasses import dataclass, field
    from abc import ABC, abstractmethod


    @dataclass
    class AgentResult:
        success: bool
        data: Dict[str, Any]
        confidence_score: float = 0.0
        error_message: str = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        def to_dict(self) -> Dict[str, Any]:
            """Return a serializable representation of the result."""
            return {
                "success": self.success,
                "confidence": self.confidence_score,
                "error_message": self.error_message,
                **(self.data or {}),
                "metadata": self.metadata,
            }

        def __getitem__(self, key: str) -> Any:
            """Allow bracket access like a dict to the inner data."""
            return (self.data or {}).get(key)

        def get(self, key: str, default: Any = None) -> Any:
            """Dict-like .get() which looks up keys in the inner data."""
            return (self.data or {}).get(key, default)


    class BaseAgent(ABC):
        def __init__(self, agent_name: str):
            self.agent_name = agent_name

        def create_success_result(self, data: Dict[str, Any], confidence_score: float = 1.0, metadata: Dict[str, Any] = None) -> AgentResult:
            return AgentResult(
                success=True,
                data=data,
                confidence_score=confidence_score,
                metadata=metadata or {}
            )

        def create_error_result(self, error_message: str, confidence_score: float = 0.0) -> AgentResult:
            return AgentResult(
                success=False,
                data={},
                confidence_score=confidence_score,
                error_message=error_message
            )

logger = logging.getLogger(__name__)

# Check for Whisper availability
WHISPER_AVAILABLE = False
try:
    import whisper

    WHISPER_AVAILABLE = True
    print("✓ Whisper available for audio transcription")
except ImportError:
    print("⚠ Install Whisper: pip install openai-whisper")

# Check for video processing
VIDEO_AVAILABLE = False
try:
    import cv2

    VIDEO_AVAILABLE = True
    print("✓ OpenCV available for video processing")
except ImportError:
    print("⚠ Install OpenCV: pip install opencv-python")


class QwenVideoCVAgent(BaseAgent):
    """
    Agent for analyzing video CVs using Qwen2.5-VL for frame analysis
    and Whisper for audio transcription
    """

    def __init__(self, lm_studio_host: str = "http://localhost:1234"):
        super().__init__("QwenVideoCVAgent")

        # Initialize Qwen Vision Agent for frame analysis
        self.qwen_vision = QwenVisionAgent(lm_studio_host)

        # Load Whisper model for audio transcription
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("✓ Whisper model loaded for audio transcription")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                self.whisper_model = None

    async def execute_task(self, task_data: Dict[str, Any]) -> AgentResult:
        """Execute video CV analysis task"""
        try:
            video_path = task_data.get("video_path")
            job_description = task_data.get("job_description", "")

            if not video_path or not os.path.exists(video_path):
                return self.create_error_result("Video file not found")

            # Analyze video CV
            analysis_result = await self.analyze_video_cv(video_path, job_description)

            return self.create_success_result(
                data=analysis_result,
                confidence_score=analysis_result.get("confidence", 0.7),
                metadata={"video_path": video_path}
            )

        except Exception as e:
            logger.error(f"Video CV analysis error: {e}")
            return self.create_error_result(f"Analysis failed: {str(e)}")

    async def analyze_video_cv(self, video_path: str, job_description: str = "") -> Dict[str, Any]:
        """
        Analyze video CV using Qwen2.5-VL for visual analysis and Whisper for audio

        Args:
            video_path: Path to video file
            job_description: Job description for context

        Returns:
            Complete video analysis results
        """
        result = {
            "video_path": video_path,
            "file_size": os.path.getsize(video_path),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.0
        }

        # Extract video metadata
        metadata = self.extract_video_metadata(video_path)
        result.update(metadata)

        # Extract and analyze key frames with Qwen2.5-VL
        frames_analysis = await self.analyze_key_frames_with_qwen(video_path)
        result["frames_analysis"] = frames_analysis

        # Extract audio transcript with Whisper
        transcript = await self.extract_audio_transcript(video_path)
        result["transcript"] = transcript

        # Combine visual and audio analysis
        if frames_analysis and transcript:
            combined_analysis = await self.combine_analyses(
                frames_analysis, transcript, job_description
            )
            result.update(combined_analysis)
            result["confidence"] = 0.85
        elif transcript:
            result["confidence"] = 0.65
        elif frames_analysis:
            result["confidence"] = 0.60
        else:
            result["confidence"] = 0.30

        return result

    def extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        metadata = {
            "duration_seconds": 0,
            "fps": 0,
            "resolution": "unknown",
            "format": Path(video_path).suffix
        }

        if not VIDEO_AVAILABLE:
            return metadata

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            metadata.update({
                "duration_seconds": round(duration, 2),
                "fps": round(fps, 2),
                "resolution": f"{width}x{height}",
                "frame_count": frame_count
            })

            cap.release()
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")

        return metadata

    async def analyze_key_frames_with_qwen(
            self,
            video_path: str,
            num_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Extract key frames and analyze them using Qwen2.5-VL

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract and analyze

        Returns:
            Frame analysis results
        """
        frames_analysis = {
            "frames": [],
            "summary": "",
            "insights": []
        }

        if not VIDEO_AVAILABLE or not self.qwen_vision.is_connected:
            logger.warning("Video processing or Qwen not available")
            return frames_analysis

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame intervals
            if total_frames > num_frames:
                interval = total_frames // num_frames
                frame_indices = [i * interval for i in range(num_frames)]
            else:
                frame_indices = range(total_frames)

            # Extract and analyze each frame
            for idx, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Save frame temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
                    frame_path = temp_frame.name
                    cv2.imwrite(frame_path, frame)

                # Analyze frame with Qwen2.5-VL
                timestamp = frame_idx / fps if fps > 0 else 0

                analysis_prompt = """Analyze this video frame from a job interview or CV video. Describe:
1. The person's appearance and professionalism
2. Their body language and demeanor
3. The setting/background
4. Any visible text or credentials
5. Overall impression for a job candidate"""

                frame_analysis = self.qwen_vision.analyze_video_frame(frame_path, analysis_prompt)

                frames_analysis["frames"].append({
                    "frame_index": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "analysis": frame_analysis.get("text", ""),
                    "confidence": frame_analysis.get("confidence", 0.0)
                })

                # Clean up temporary file
                try:
                    os.unlink(frame_path)
                except:
                    pass

            cap.release()

            # Generate overall summary from all frames
            if frames_analysis["frames"]:
                frames_analysis["summary"] = self._summarize_frame_analyses(
                    frames_analysis["frames"]
                )

        except Exception as e:
            logger.error(f"Error analyzing key frames: {e}")

        return frames_analysis

    def _summarize_frame_analyses(self, frames: List[Dict]) -> str:
        """Summarize multiple frame analyses"""
        if not frames:
            return "No frames analyzed"

        # Collect all analyses
        all_analyses = [f["analysis"] for f in frames if f.get("analysis")]

        if not all_analyses:
            return "No analysis data available"

        # Simple summary (you can enhance this with LLM)
        summary = f"Analyzed {len(frames)} frames from the video. "
        summary += "Overall impression based on visual analysis: "
        summary += all_analyses[len(all_analyses) // 2][:200] + "..."  # Middle frame as representative

        return summary

    async def extract_audio_transcript(self, video_path: str) -> str:
        """Extract and transcribe audio from video using Whisper"""
        if not self.whisper_model:
            return "Audio transcription not available. Whisper not loaded."

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Audio transcription not available. FFmpeg not installed. Install FFmpeg to enable audio analysis."

        try:
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            # Use ffmpeg to extract audio
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-ab', '160k', '-ac', '1', '-ar', '16000',
                '-vn', temp_audio_path, '-y'
            ], check=True, capture_output=True)

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_audio_path)
            transcript = result["text"]

            # Clean up
            os.unlink(temp_audio_path)

            return transcript

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            return "Audio extraction failed. FFmpeg error."
        except Exception as e:
            logger.error(f"Error extracting audio transcript: {e}")
            return f"Transcription error: {str(e)}"

    async def combine_analyses(
            self,
            frames_analysis: Dict,
            transcript: str,
            job_description: str
    ) -> Dict[str, Any]:
        """
        Combine visual and audio analyses for final assessment

        Args:
            frames_analysis: Visual analysis from Qwen
            transcript: Audio transcript from Whisper
            job_description: Job description for matching

        Returns:
            Combined analysis
        """
        combined = {
            "visual_summary": frames_analysis.get("summary", ""),
            "audio_transcript": transcript,
            "overall_assessment": ""
        }

        # Extract skills from transcript
        combined["skills"] = self.extract_skills_from_transcript(transcript)

        # Extract experience years (simple heuristic)
        combined["experience_years"] = self.extract_experience_years(transcript)

        # Generate overall assessment
        assessment = f"Candidate demonstrates "

        if len(combined["skills"]) > 5:
            assessment += "strong technical knowledge with multiple skills mentioned. "
        elif len(combined["skills"]) > 2:
            assessment += "relevant technical skills. "
        else:
            assessment += "some technical awareness. "

        if len(transcript.split()) > 200:
            assessment += "Good communication and detailed responses. "

        combined["overall_assessment"] = assessment

        return combined

    def extract_skills_from_transcript(self, transcript: str) -> List[str]:
        """Extract skills mentioned in transcript"""
        if not transcript:
            return []

        # Common technical skills to look for
        common_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'django', 'flask', 'fastapi', 'sql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd',
            'machine learning', 'data science', 'ai', 'deep learning',
            'tensorflow', 'pytorch', 'api', 'rest', 'graphql', 'microservices',
            'agile', 'scrum', 'devops', 'testing', 'debugging'
        ]

        transcript_lower = transcript.lower()
        found_skills = []

        for skill in common_skills:
            if skill in transcript_lower:
                found_skills.append(skill)

        return found_skills

    def extract_experience_years(self, transcript: str) -> float:
        """Extract years of experience from transcript (simple heuristic)"""
        import re

        # Look for patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\s+years?\s+(?:working|in|with)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, transcript.lower())
            if matches:
                return float(matches[0])

        return 0.0


def test_qwen_video_agent():
    """Test Qwen Video CV Agent"""
    import asyncio

    async def run_test():
        agent = QwenVideoCVAgent()

        # Test with sample video (provide actual path)
        test_video = "sample_cv_video.mp4"

        if os.path.exists(test_video):
            print(f"📹 Analyzing video: {test_video}")

            result = await agent.execute_task({
                "video_path": test_video,
                "job_description": "Looking for Python developer with 3+ years experience"
            })

            print("\n=== Video CV Analysis Results ===")
            if hasattr(result, 'to_dict'):
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(json.dumps(result, indent=2, default=str))
        else:
            print(f"⚠ Test video not found: {test_video}")
            print("Please provide a valid video path for testing")

    asyncio.run(run_test())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    test_qwen_video_agent()