"""
Interview Simulation Module

This module provides an AI-powered interview simulation using FastRTC and Groq.
It conducts structured interviews with multiple sections including:
- Pre-Introduction
- Introduction (name verification)
- HR Questions
- Behavioral Questions
- Technical Questions
- Situational Questions

The module generates evaluation reports for candidates.
"""

from .interview_system import InterviewSimulator, run_interview

__all__ = ["InterviewSimulator", "run_interview"]
