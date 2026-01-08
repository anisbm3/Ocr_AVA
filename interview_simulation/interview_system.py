"""
Interview Simulation System using FastRTC and Groq

This module implements a voice-based AI interview simulator that conducts
structured interviews with candidates using speech-to-text and text-to-speech.
"""

import os
import json
from fastrtc import (
    ReplyOnPause, 
    Stream, 
    AlgoOptions, 
    SileroVadOptions, 
    get_stt_model, 
    get_tts_model
)
from groq import Groq
import dotenv
import gradio as gr

dotenv.load_dotenv()


class InterviewSimulator:
    """
    AI Interview Simulator that conducts structured voice interviews.
    
    The interview flows through multiple sections:
    1. Pre-Introduction - Welcome and name request
    2. Introduction - Name verification
    3. HR Questions
    4. Behavioral Questions  
    5. Technical Questions
    6. Situational Questions
    
    Each section allows up to 3 follow-up attempts before moving on.
    """
    
    def __init__(self, cv_path: str = None, job_desc_path: str = None):
        """
        Initialize the Interview Simulator.
        
        Args:
            cv_path: Path to the candidate's CV JSON file
            job_desc_path: Path to the job description JSON file
        """
        # Initialize STT and TTS models
        self.stt_model = get_stt_model()
        self.tts_model = get_tts_model()
        
        # File paths
        base_dir = os.path.dirname(__file__)
        self.report_file = os.path.join(base_dir, "report_interview.json")
        self.history_file = os.path.join(base_dir, "conversation_history.json")
        self.cv_file = cv_path or os.path.join(base_dir, "cv.json")
        self.job_desc_file = job_desc_path or os.path.join(base_dir, "job_description.json")
        
        # Section states: 1 = active, 0 = inactive, -1 = completed
        self.pre_intro_section = 1
        self.intro_section = 0
        self.hr_section = 0
        self.behavioural_section = 0
        self.technical_section = 0
        self.situational_section = 0
        
        # Exit counter for follow-up attempts
        self.exit_section = 0
        
        # Pre-intro message
        self.pre_intro_section_message = (
            "**ALWAYS ANSWER IN THE FORMAT 0,YOUR_ANSWER OR 1,YOUR_ANSWER, "
            "ALWAYS A BINARY DIGIT IN THE BEGINNING OF YOUR RESPONSE, make your "
            "responses short and do not ask the user for any code. You are not "
            "allowed to make placeholders for anything, if you are not provided "
            "with information, do not mention it. You also need to focus on the "
            "interview and not indulge the user in any other topic. You will be "
            "provided with a conversation history where you will be able to "
            "remember progress** The user is initiating a conversation. Respond "
            "by welcoming him to the interview in a professional manner, introduce "
            "yourself as Alice, the HR manager and begin by asking his name."
        )
        
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize files
        self._clear_json(self.history_file)
        self._clear_json(self.report_file)
        self.history = self._load_json(self.history_file)
        
        # Load CV and job description
        self.cv_text = json.dumps(
            self._load_json(self.cv_file), 
            ensure_ascii=False, 
            indent=2
        )
        self.job_desc_text = json.dumps(
            self._load_json(self.job_desc_file), 
            ensure_ascii=False, 
            indent=2
        )
    
    def _load_json(self, file_path: str) -> dict:
        """Load JSON from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed loading JSON from {file_path}: {e}")
            return {}
    
    def _clear_json(self, file_path: str) -> None:
        """Clear a JSON file by writing an empty list."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            print("Failed clearing history:", e)
    
    def _save_json(self, data: any, file_path: str) -> None:
        """Save data to a JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Failed saving history:", e)
    
    def _add_history(self, section: str, role: str, content: str) -> None:
        """Add an entry to the conversation history."""
        entry = {"section": section, "role": role, "content": content}
        self.history.append(entry)
        self._save_json(self.history, self.history_file)
    
    def _add_report(self, entry: any) -> None:
        """Add an entry to the evaluation report."""
        report = self._load_json(self.report_file)
        if isinstance(entry, list):
            report.extend(entry)
        else:
            report.append(entry)
        self._save_json(report, self.report_file)
    
    def _evaluate(self) -> dict:
        """Generate an evaluation report based on the conversation history."""
        print("=" * 50)
        print("STARTING EVALUATION REPORT GENERATION")
        print("=" * 50)
        
        evaluation_prompt = (
            f"Based on the following conversation history, evaluate the candidate's "
            f"performance in the interview. Provide feedback on strengths and areas "
            f"for improvement for each section, as well as the score in valid JSON "
            f"format only. The JSON should look like this example:\n\n"
            f"[{{\"section\": \"section_name\", \"score\": \"x/100\", \"strength\": \"\", "
            f"\"weaknesses\": \"\", \"general overview\": \"\"}}]\n\n"
            f"Conversation History:\n" + json.dumps(self.history, ensure_ascii=False, indent=2)
        )

        try:
            print(f"Sending evaluation request to Groq with {len(self.history)} conversation entries...")
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=2000,
            )

            if response.choices and response.choices[0].message.content:
                evaluation = response.choices[0].message.content.strip()
                print(f"Received evaluation response (length: {len(evaluation)})")
                try:
                    clean_eval = evaluation.replace('```json', '').replace('```', '').strip()
                    parsed_eval = json.loads(clean_eval)
                    print(f"Successfully parsed evaluation JSON with {len(parsed_eval)} sections")
                    return parsed_eval
                except json.JSONDecodeError as e:
                    print(f"Failed to parse evaluation as JSON: {e}")
                    print(f"Raw evaluation response: {evaluation[:500]}...")
                    # Fallback: save raw text
                    return [{"section": "Evaluation Error", "error": "JSON parsing failed", "raw_response": evaluation}]

            print("Groq returned empty evaluation response")
            return [{"section": "Evaluation Error", "error": "No evaluation available"}]

        except Exception as e:
            print(f"Groq API error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return [{"section": "Evaluation Error", "error": f"Evaluation failed: {str(e)}"}]
    
    def _groq_call(self, section: str, prompt: str, user_reply: str) -> str:
        """Make a call to the Groq LLM."""
        print(f"User said: {user_reply}")
        complete_prompt = (
            f"Job Description:\n{self.job_desc_text}\n"
            f"Candidate CV:\n{self.cv_text}\n"
            f"Instruction:\nAlways make your questions meet the job description "
            f"criteria and make harder questions on the skills mentioned in the CV.\n"
            f"History of the conversation:\n{json.dumps(self.history, ensure_ascii=False, indent=2)}\n"
            f"Next task:\n{prompt}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": complete_prompt}],
                max_tokens=250,
            )

            if response.choices and response.choices[0].message.content:
                q = response.choices[0].message.content.strip()
                print(f"Groq said: {q}")
                self._add_history(section, "User said", user_reply)
                self._add_history(section, "You said", q[2:])
                return q

            print("Groq returned empty response")
            return "dismiss"

        except Exception as e:
            print(f"Groq API error: {e}")
            return "dismiss"
    
    def _speak(self, text: str):
        """Convert text to speech and yield audio chunks."""
        for audio_chunk in self.tts_model.stream_tts_sync(text):
            yield audio_chunk
    
    def process_audio(self, audio):
        """
        Main audio processing function for the interview flow.
        
        This function handles speech-to-text, determines the current section,
        generates appropriate responses, and yields audio output.
        """
        reply = self.stt_model.stt(audio)
        if reply == "":
            yield from self._speak("I didn't catch that. Could you please repeat?")
            return
        
        # Section decider
        if self.pre_intro_section == -1 and self.intro_section == 0:
            self.intro_section = 1
        elif self.intro_section == -1 and self.hr_section == 0:
            self.hr_section = 1
        elif self.hr_section == -1 and self.behavioural_section == 0:
            self.behavioural_section = 1
        elif self.behavioural_section == -1 and self.technical_section == 0:
            self.technical_section = 1
        elif self.technical_section == -1 and self.situational_section == 0:
            self.situational_section = 1
        
        # Pre-Introduction Section
        if self.pre_intro_section == 1:
            self.pre_intro_section = -1
            question = self._groq_call("Pre-Introduction Section", self.pre_intro_section_message, reply)
            yield from self._speak(question[2:])
        
        # Introduction Section
        elif self.intro_section == 1:
            if self.exit_section + 1 == 4:
                self.exit_section = 0
                self.intro_section = -1
                interrupted_message = (
                    "The user has refused to provide a name or has given an unrealistic name. "
                    "Greet them then proceed to introduce to him a behavioral question."
                )
                response = self._groq_call("Introduction Section", interrupted_message, reply)
                yield from self._speak(response)
            else:
                intro_section_message = (
                    f"The user responded: {reply}. Classify the user's latest message: "
                    f"if it contains a realistic human name or refuses to give a name, "
                    f"greet and introduce them to an hr question in the format 1,question; "
                    f"if it contains a ridiculous/unrealistic name or no valid name, "
                    f"give them a follow-up question in the format 0,question. "
                    f"Always output only 0,question or 1,question with no extra text, "
                    f"and treat insults or irrelevant replies as invalid names. "
                    f"If the user refuses to give a name, proceed to ask them the HR question directly."
                )
                response = self._groq_call("Introduction Section", intro_section_message, reply)
                flag = response[0]
                question = response[2:]
                if flag == '1':
                    self.exit_section = 0
                    self.intro_section = -1
                yield from self._speak(question)
                if flag != '1':
                    self.exit_section += 1
        
        # HR Section
        elif self.hr_section == 1:
            if self.exit_section + 1 == 4:
                self.exit_section = 0
                self.hr_section = -1
                interrupted_message = (
                    "The user has been unable to provide a complete answer to the HR question. "
                    "Proceed to introduce to him a behavioural question."
                )
                response = self._groq_call("HR Section", interrupted_message, reply)
                yield from self._speak(response)
            else:
                hr_section_message = (
                    f"The user responded: {reply}. If that was a complete answer to the "
                    f"question related to the HR question, or if the user showed in any way "
                    f"that he has no answer, proceed to give him a technical question in the "
                    f"format 1,question. If you feel it was an incomplete answer, give the user "
                    f"a follow-up question in the format 0,question. Only that format, no additional text."
                )
                response = self._groq_call("HR Section", hr_section_message, reply)
                flag = response[0]
                question = response[2:]
                if flag == '1':
                    self.exit_section = 0
                    self.hr_section = -1
                yield from self._speak(question)
                if flag != '1':
                    self.exit_section += 1
        
        # Behavioural Section
        elif self.behavioural_section == 1:
            if self.exit_section + 1 == 4:
                self.exit_section = 0
                self.behavioural_section = -1
                interrupted_message = (
                    "The user has been unable to provide a complete answer to the behavioral question. "
                    "Proceed to introduce to him a technical question."
                )
                response = self._groq_call("Behavioural Section", interrupted_message, reply)
                yield from self._speak(response)
            else:
                behavioural_section_message = (
                    f"The user responded: {reply}. If that was a complete answer to the "
                    f"question related to the behavioral question, or if the user showed in any way "
                    f"that he has no answer, proceed to give him a technical question in the "
                    f"format 1,question. If you feel it was an incomplete answer, give the user "
                    f"a follow-up question in the format 0,question. Only that format, no additional text."
                )
                response = self._groq_call("Behavioural Section", behavioural_section_message, reply)
                flag = response[0]
                question = response[2:]
                if flag == '1':
                    self.exit_section = 0
                    self.behavioural_section = -1
                yield from self._speak(question)
                if flag != '1':
                    self.exit_section += 1
        
        # Technical Section
        elif self.technical_section == 1:
            if self.exit_section + 1 == 4:
                self.exit_section = 0
                self.technical_section = -1
                interrupted_message = (
                    "The user has been unable to provide a complete answer to the technical question. "
                    "Proceed to introduce to him to a problem-based / scenario-based question."
                )
                response = self._groq_call("Technical Section", interrupted_message, reply)
                yield from self._speak(response)
            else:
                technical_section_message = (
                    f"The user responded: {reply}. If that was a complete answer to the "
                    f"question related to the technical question, or if the user showed in any way "
                    f"that he has no answer, proceed to give him a problem-based / scenario-based "
                    f"question in the format 1,question. If you feel it was an incomplete answer, "
                    f"give the user a follow-up question in the format 0,question. Only that format, "
                    f"no additional text."
                )
                response = self._groq_call("Technical Section", technical_section_message, reply)
                flag = response[0]
                question = response[2:]
                if flag == '1':
                    self.exit_section = 0
                    self.technical_section = -1
                yield from self._speak(question)
                if flag != '1':
                    self.exit_section += 1
        
        # Situational Section
        elif self.situational_section == 1:
            if self.exit_section + 1 == 4:
                self.exit_section = 0
                self.situational_section = -1
                yield from self._speak("Thank you for your time. We will get back to you soon. Goodbye!")
                self._finalize_interview()
            else:
                situational_section_message = (
                    f"The user responded: {reply}. If that was a complete answer to the "
                    f"problem-based / scenario-based question, or if the user showed in any way "
                    f"that he has no answer, proceed to thank the user and end the interview. "
                    f"If you feel it was an incomplete answer, give the user a follow-up question "
                    f"in the format 0,question. Only that format, no additional text."
                )
                response = self._groq_call("Situational Section", situational_section_message, reply)
                flag = response[0]
                question = response[2:]
                if flag == '1':
                    self.exit_section = 0
                    self.situational_section = -1
                    yield from self._speak(question)
                    self._finalize_interview()
                else:
                    self.exit_section += 1
                    yield from self._speak(question)
    
    def _finalize_interview(self) -> None:
        """Generate and save the evaluation report."""
        print("\n" + "=" * 50)
        print("INTERVIEW COMPLETED - FINALIZING")
        print("=" * 50)
        print(f"Total conversation entries: {len(self.history)}")
        print(f"Report file location: {self.report_file}")
        
        try:
            evaluation_result = self._evaluate()
            
            # Always save the report, even if there were errors
            self._add_report(evaluation_result)
            
            # Check if evaluation was successful
            if isinstance(evaluation_result, list) and len(evaluation_result) > 0:
                has_error = any(item.get('error') for item in evaluation_result if isinstance(item, dict))
                if not has_error:
                    print("✓ Evaluation report generated and saved successfully!")
                    print(f"✓ Report saved to: {self.report_file}")
                    print(f"✓ {len(evaluation_result)} sections evaluated")
                else:
                    print("⚠ Evaluation completed with errors - check report file")
            else:
                print("⚠ Evaluation result format unexpected")
                
        except Exception as e:
            print(f"✗ CRITICAL ERROR during finalization: {e}")
            import traceback
            traceback.print_exc()
            # Save error report
            error_report = [{
                "section": "Critical Error",
                "error": str(e),
                "timestamp": str(json.dumps(self.history[-5:] if len(self.history) >= 5 else self.history))
            }]
            self._add_report(error_report)
        
        print("=" * 50)
        print("FINALIZATION COMPLETE")
        print("=" * 50 + "\n")


def run_interview():
    """
    Launch the interview simulation with a Gradio UI.
    
    This function creates an InterviewSimulator instance and launches
    a Gradio interface for voice-based interaction.
    """
    simulator = InterviewSimulator()
    
    stream = Stream(
        ReplyOnPause(
            simulator.process_audio,
            algo_options=AlgoOptions(
                audio_chunk_duration=1.5,
                started_talking_threshold=0.2,
                speech_threshold=0.1
            ),
            can_interrupt=False,
            model_options=SileroVadOptions(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=2000
            )
        ),
        modality="audio",
        mode="send-receive"
    )

    # Custom CSS to hide watermarks
    css = """
    header, footer, .gradio-container header, .gradio-container footer, .svelte-drumkq {
        display: none !important;
    }
    """

    # Create a clean wrapper UI
    with gr.Blocks(css=css, theme=None) as app:
        gr.Markdown("")  # optional placeholder
        stream.ui.render()  # embed the stream UI

    app.launch(show_api=False, share=False)


if __name__ == "__main__":
    run_interview()
