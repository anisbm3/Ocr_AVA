"""
AI Interviewer System
Voice-based interview with section progression and evaluation
"""

import os
import json
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime
from groq import Groq


class AIInterviewer:
    """AI Interviewer with multi-section voice interview capabilities"""
    
    # Interview sections
    SECTIONS = [
        "pre_intro",
        "intro",
        "hr",
        "behavioural",
        "technical",
        "situational"
    ]
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize AI Interviewer with Groq client"""
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        
        # Interview state
        self.current_section_index = 0
        self.exit_section_counter = 0
        self.history = []
        self.report = []
        
        # CV and Job Description
        self.cv_data = {}
        self.job_desc_data = {}
        
        # Section prompts
        self.section_prompts = {
            "pre_intro": """**ALWAYS ANSWER IN THE FORMAT 0,YOUR_ANSWER OR 1,YOUR_ANSWER, ALWAYS A BINARY DIGIT IN THE BEGINNING OF YOUR RESPONSE, make your responses short and do not ask the user for any code. You are not allowed to make placeholders for anything, if you are not provided with information, do not mention it. You also need to focus on the interview and not indulge the user in any other topic. You will be provided with a conversation history where you will be able to remember progress** The user is initiating a conversation. Respond by welcoming him to the interview in a professional manner, introduce yourself as Alice, the HR manager and begin by asking his name.""",
            
            "intro": {
                "main": "The user responded: {reply}. Compare this response with the candidate's name from the CV. If the name matches, proceed to give him a HR question in the format 1,question. If the name does not match or if the user refused to provide a name, give the user a follow-up question in the format 0,question. Only that format, no additional text.",
                "interrupted": "The user has refused to provide a name or has given an unrealistic name. Greet them then proceed to introduce to him a behavioral question."
            },
            
            "hr": {
                "main": "The user responded: {reply}. If that was a complete answer to the question related to the HR question, or if the user showed in any way that he has no answer, proceed to give him a technical question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.",
                "interrupted": "The user has been unable to provide a complete answer to the HR question. Proceed to introduce to him a behavioural question."
            },
            
            "behavioural": {
                "main": "The user responded: {reply}. If that was a complete answer to the question related to the behavioral question, or if the user showed in any way that he has no answer, proceed to give him a technical question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.",
                "interrupted": "The user has been unable to provide a complete answer to the behavioral question. Proceed to introduce to him a technical question."
            },
            
            "technical": {
                "main": "The user responded: {reply}. If that was a complete answer to the question related to the technical question, or if the user showed in any way that he has no answer, proceed to give him a problem-based / scenario-based question in the format 1,question. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.",
                "interrupted": "The user has been unable to provide a complete answer to the technical question. Proceed to introduce to him to a problem-based / scenario-based question."
            },
            
            "situational": {
                "main": "The user responded: {reply}. If that was a complete answer to the problem-based / scenario-based question, or if the user showed in any way that he has no answer, proceed to thank the user and end the interview. If you feel it was an incomplete answer, give the user a follow-up question in the format 0,question. Only that format, no additional text.",
                "interrupted": "Thank you for your time. We will get back to you soon. Goodbye!"
            }
        }
    
    def reset_interview(self):
        """Reset interview state for a new session"""
        self.current_section_index = 0
        self.exit_section_counter = 0
        self.history = []
        self.report = []
    
    def load_cv(self, cv_json: str):
        """Load CV data from JSON string"""
        try:
            self.cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
            return True
        except json.JSONDecodeError as e:
            print(f"Error loading CV: {e}")
            return False
    
    def load_job_description(self, job_desc_json: str):
        """Load job description from JSON string"""
        try:
            self.job_desc_data = json.loads(job_desc_json) if isinstance(job_desc_json, str) else job_desc_json
            return True
        except json.JSONDecodeError as e:
            print(f"Error loading job description: {e}")
            return False
    
    def add_to_history(self, section: str, role: str, content: str):
        """Add conversation entry to history"""
        entry = {
            "section": section,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
    
    def get_current_section(self) -> str:
        """Get the name of the current section"""
        if self.current_section_index < len(self.SECTIONS):
            return self.SECTIONS[self.current_section_index]
        return "completed"
    
    def advance_section(self):
        """Move to the next interview section"""
        self.current_section_index += 1
        self.exit_section_counter = 0
    
    def call_llm(self, section: str, prompt: str, user_reply: str) -> str:
        """
        Call Groq LLM with context
        Returns response from model
        """
        cv_text = json.dumps(self.cv_data, ensure_ascii=False, indent=2)
        job_desc_text = json.dumps(self.job_desc_data, ensure_ascii=False, indent=2)
        history_text = json.dumps(self.history, ensure_ascii=False, indent=2)
        
        complete_prompt = (
            f"Job Description:\n{job_desc_text}\n\n"
            f"Candidate CV:\n{cv_text}\n\n"
            f"Instruction:\n"
            f"Always make your questions meet the job description criteria and make harder questions on the skills mentioned in the CV.\n\n"
            f"History of the conversation:\n{history_text}\n\n"
            f"Next task:\n{prompt}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": complete_prompt}],
                max_tokens=250,
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                print(f"[{section}] User: {user_reply}")
                print(f"[{section}] AI: {answer}")
                
                # Add to history
                self.add_to_history(section, "user", user_reply)
                self.add_to_history(section, "assistant", answer)
                
                return answer
            
            return "dismiss"
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return "dismiss"
    
    def process_response(self, user_text: str) -> str:
        """
        Process user text input and generate AI response
        Returns the AI's response text
        """
        if not user_text or user_text.strip() == "":
            return "I didn't catch that. Could you please repeat?"
        
        current_section = self.get_current_section()
        
        # Check if interview is completed
        if current_section == "completed":
            return "Interview has been completed. Thank you!"
        
        # Pre-intro section (initial greeting)
        if current_section == "pre_intro":
            prompt = self.section_prompts["pre_intro"]
            response = self.call_llm(current_section, prompt, user_text)
            
            if response != "dismiss":
                self.advance_section()
                # Remove the leading "0," or "1," if present
                return response[2:] if len(response) > 2 and response[0].isdigit() else response
            return response
        
        # Handle other sections
        section_config = self.section_prompts.get(current_section)
        
        if not section_config:
            return "Section configuration error"
        
        # Check if we should interrupt (too many failed attempts)
        if self.exit_section_counter >= 3:
            interrupted_prompt = section_config.get("interrupted", "")
            response = self.call_llm(current_section, interrupted_prompt, user_text)
            self.advance_section()
            
            # Check if this was the last section
            if self.get_current_section() == "completed":
                self._generate_evaluation()
            
            return response
        
        # Normal section processing
        main_prompt = section_config.get("main", "")
        if "{reply}" in main_prompt:
            main_prompt = main_prompt.format(reply=user_text)
        
        response = self.call_llm(current_section, main_prompt, user_text)
        
        if response == "dismiss":
            return "I encountered an error. Please try again."
        
        # Parse response for flag (0 or 1)
        flag = response[0] if len(response) > 0 and response[0].isdigit() else "0"
        question = response[2:] if len(response) > 2 else response
        
        if flag == "1":
            # Move to next section
            self.exit_section_counter = 0
            self.advance_section()
            
            # Check if interview completed
            if self.get_current_section() == "completed":
                self._generate_evaluation()
        else:
            # Stay in same section, increment counter
            self.exit_section_counter += 1
        
        return question
    
    def _generate_evaluation(self):
        """Generate evaluation report based on interview history"""
        evaluation_prompt = (
            f"Based on the following conversation history, evaluate the candidate's performance in the interview. "
            f"Provide feedback on strengths and areas for improvement for each section, as well as the score in valid JSON format only. "
            f"The JSON should look like this example:\n\n"
            f'[{{"section": "section_name", "score": "x/100", "strength": "", "weaknesses": "", "general_overview": ""}}]\n\n'
            f"Conversation History:\n" + json.dumps(self.history, ensure_ascii=False, indent=2)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=500,
            )
            
            if response.choices and response.choices[0].message.content:
                evaluation = response.choices[0].message.content.strip()
                try:
                    # Clean and parse JSON
                    clean_eval = evaluation.replace('```json', '').replace('```', '').strip()
                    evaluation_json = json.loads(clean_eval)
                    self.report = evaluation_json
                    print("Evaluation report generated successfully")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse evaluation as JSON: {e}")
                    self.report = {"error": "Invalid evaluation format", "raw": evaluation}
            else:
                self.report = {"error": "No evaluation available"}
        
        except Exception as e:
            print(f"Groq API error during evaluation: {e}")
            self.report = {"error": f"Evaluation failed: {str(e)}"}
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Get the evaluation report"""
        return self.report
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the complete conversation history"""
        return self.history
    
    def export_session(self) -> Dict[str, Any]:
        """Export complete interview session data"""
        return {
            "cv_data": self.cv_data,
            "job_description": self.job_desc_data,
            "conversation_history": self.history,
            "evaluation_report": self.report,
            "completed": self.get_current_section() == "completed",
            "timestamp": datetime.now().isoformat()
        }
