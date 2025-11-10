"""
Keyword-Triggered Conversation System
Uses Rev AI for STT, Gemini for LLM, and OpenAI TTS for speech output.
Keywords: "question" triggers response, "end" terminates conversation.
"""

import os
import sys
import time
import json
import queue
import threading
import pyaudio
import subprocess
from datetime import datetime
from pathlib import Path

from rev_ai.streamingclient import RevAiStreamingClient
from rev_ai.models import MediaConfig
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from mutagen.mp3 import MP3

# Norm violation detection
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load environment variables
load_dotenv()

# Global state
listening = True
conversation_active = True
question_mode = False  # Whether we're in active Q&A mode
full_transcript = []
current_utterance = []
silence_time = time.time()
transcribing = False

# Global norm violation detector (loaded once at startup)
norm_detector = None


class NormViolationDetector:
    """Fast local model for detecting potential norm violations."""

    def __init__(self, model_name="unitary/toxic-bert"):
        print(f"📦 Loading norm detection model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # inference mode

        # Define what we consider violations
        self.SOFT_LABELS = {"toxic", "insult"}  # rude, unkind
        self.HARD_LABELS = {"threat", "identity_hate", "obscene"}  # serious violations

        print(f"   ✓ Model loaded. Monitoring for: {self.SOFT_LABELS | self.HARD_LABELS}")

    def score_toxicity(self, text: str) -> dict:
        """Return probability scores for each toxicity label."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return {self.model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}

    def check_violation(self, text: str, soft_thresh=0.85, hard_thresh=0.35, min_words=5):
        """Check if text violates norms. Returns (is_violation, severity, scores)."""
        # Skip very short phrases
        if len(text.split()) < min_words:
            return False, None, {}

        scores = self.score_toxicity(text)

        # Check for hard violations first
        if any(scores.get(lbl, 0.0) >= hard_thresh for lbl in self.HARD_LABELS):
            return True, "hard", scores

        # Check for soft violations
        if any(scores.get(lbl, 0.0) >= soft_thresh for lbl in self.SOFT_LABELS):
            return True, "soft", scores

        return False, None, scores


class SimpleMicrophoneStream:
    """Simplified microphone stream for single user."""

    def __init__(self, rate, chunk, device_idx):
        self._rate = rate
        self._chunk = chunk
        self._device_idx = device_idx
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._device_idx,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Callback to fill buffer with incoming audio."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, log_file=None):
        """Yields audio data for Rev AI."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            data = [chunk]

            # Get all available chunks
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            raw_data = b"".join(data)

            # Log audio if requested
            if log_file and not log_file.closed:
                try:
                    log_file.write(raw_data)
                    log_file.flush()
                except (BrokenPipeError, OSError):
                    pass

            yield raw_data


def list_microphones():
    """List all available microphone devices."""
    p = pyaudio.PyAudio()
    print("\n=== Available Microphone Devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Index {i}: {info['name']}")
    p.terminate()
    print()


def monitor_silence(stream, ffmpeg_process, result, full_phrase, in_question_mode, silence_threshold=2.0):
    """Monitor for silence to end the current utterance (only in Q&A mode)."""
    global silence_time, transcribing, listening

    while not stream.closed:
        time.sleep(0.05)

        # End if not listening or conversation ended
        if not listening or not conversation_active:
            print("Stopping audio stream...")
            result.append(None)
            stream.closed = True
            time.sleep(0.5)
            if ffmpeg_process:
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()
            if full_phrase:
                result.append(" ".join(full_phrase))
            break

        # Only check for silence timeout in Q&A mode
        if in_question_mode:
            if (time.time() - silence_time > silence_threshold and
                not transcribing and
                len(full_phrase) > 0):
                print("🤫 [SILENCE DETECTED] Processing utterance...")
                result.append(None)
                stream.closed = True
                time.sleep(0.5)
                if ffmpeg_process:
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.wait()
                if full_phrase:
                    result.append(" ".join(full_phrase))
                break


def stream_audio_to_revai(device_idx, output_folder, revai_api_key, revai_config, in_question_mode=False, check_callback=None):
    """Stream audio from microphone to Rev AI for transcription.

    Args:
        check_callback: Optional function to call with each FINAL text in idle mode.
                       Should return True to stop listening and intervene.
    """
    global silence_time, transcribing, listening

    rate = 44100
    chunk = int(rate / 10)

    result = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_log_path = output_folder / f"audio_{timestamp}.mp3"

    # Create a NEW Rev AI client for each utterance (clients are single-use)
    print("🔌 [CONNECTING] Starting Rev AI connection...")
    streamclient = RevAiStreamingClient(revai_api_key, revai_config)
    print("   ✓ Rev AI client created")

    with SimpleMicrophoneStream(rate, chunk, device_idx) as stream:
        try:
            print("   ✓ Microphone stream opened")
            # Start FFmpeg to log audio
            ffmpeg_process = subprocess.Popen([
                "ffmpeg", "-f", "s16le", "-ar", str(rate), "-ac", "1",
                "-i", "pipe:0", "-y", "-codec:a", "libmp3lame",
                "-b:a", "192k", str(audio_log_path)
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   ✓ Audio logging started")

            # Start Rev AI streaming
            print("   ⏳ Connecting to Rev AI servers...")
            response_gen = streamclient.start(stream.generator(ffmpeg_process.stdin))
            print("🎤 [LISTENING] Microphone active, speak now...")
            silence_time = time.time()
            full_phrase = []

            # Start silence monitor (only active in Q&A mode)
            threading.Thread(
                target=monitor_silence,
                args=(stream, ffmpeg_process, result, full_phrase, in_question_mode),
                daemon=True
            ).start()

            # Process Rev AI responses
            first_partial = True
            for raw_response in response_gen:
                response = json.loads(raw_response)

                if response["type"] == "partial":
                    transcribing = True
                    silence_time = time.time()
                    # Update partial phrase
                    text = "".join(e["value"] for e in response["elements"])
                    if text.strip():
                        if first_partial:
                            print("🎙️  [SPEECH DETECTED] Processing audio...")
                            first_partial = False
                        print(f"📝 [TRANSCRIBING] {text}")

                elif response["type"] == "final":
                    silence_time = time.time()
                    text = "".join(e["value"] for e in response["elements"])
                    if text.strip():
                        print(f"✅ [FINAL] {text}")
                        full_phrase.append(text)
                        result.append(text)

                        # In idle mode, check each FINAL immediately
                        if not in_question_mode and check_callback:
                            should_intervene = check_callback(" ".join(full_phrase))
                            if should_intervene:
                                print("🚨 [INTERVENTION NEEDED] Stopping listening...")
                                listening = False
                                # Stop the stream
                                stream.closed = True
                                if ffmpeg_process:
                                    ffmpeg_process.stdin.close()
                                    ffmpeg_process.wait()
                                break

                    transcribing = False

        except Exception as e:
            print(f"Error in audio streaming: {e}")
            import traceback
            traceback.print_exc()

    return result


def check_for_keywords(text):
    """Check if text contains trigger keywords."""
    text_lower = text.lower()

    # Check for end conversation
    if "terminate" in text_lower:
        return "END"
    # Check for starting question mode
    elif "question" in text_lower:
        return "QUESTION"
    # Check for exiting question mode (but staying in conversation)
    elif "nevermind" in text_lower or "never mind" in text_lower or "that's all" in text_lower:
        return "EXIT_QUESTION"
    else:
        return None


def check_for_intervention(text):
    """Check if text requires immediate intervention (keywords or norm violations).

    Returns True if intervention is needed, False otherwise.
    """
    global norm_detector

    keyword = check_for_keywords(text)

    # Keywords that trigger intervention
    if keyword in ["END", "QUESTION"]:
        return True

    # Check for norm violations using local model
    if norm_detector:
        is_violation, severity, scores = norm_detector.check_violation(text)
        if is_violation:
            print(f"⚠️  [NORM VIOLATION DETECTED] Severity: {severity}")
            print(f"   Scores: {scores}")
            return True

    return False


def play_audio_response(openai_client, response_text, output_folder):
    """Convert text to speech and play it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = output_folder / f"response_{timestamp}.mp3"

    print(f"\n🤖 Assistant: {response_text}\n")

    try:
        # Generate speech
        print("🎵 [GENERATING AUDIO] Creating speech file...")
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=response_text,
            instructions="Speak with a calm and helpful tone."
        ) as response:
            response.stream_to_file(audio_file)

        print("🔊 [SPEAKING] Playing response...")
        # Play audio (using system command - cross-platform)
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", str(audio_file)], check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", str(audio_file)], shell=True, check=True)
        else:  # Linux
            subprocess.run(["mpg123", str(audio_file)], check=True)

        # Get audio duration for timing
        audio = MP3(audio_file)
        print("✅ [DONE SPEAKING] Audio playback complete\n")
        return audio.info.length

    except Exception as e:
        print(f"❌ [ERROR] Playing audio: {e}")
        return 0


def main():
    global conversation_active, listening, question_mode, full_transcript, norm_detector

    print("\n=== Keyword-Triggered Conversation System ===")
    print("Keywords:")
    print("  - 'question' → enters Q&A mode (AI responds to everything)")
    print("  - 'nevermind' / 'that's all' → exits Q&A mode")
    print("  - 'terminate' → ends conversation and shows transcript\n")

    # List available microphones
    list_microphones()
    device_idx = int(input("Select microphone index: "))

    # Setup output folder
    output_folder = Path("./conversation_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nLogs will be saved to: {output_folder}\n")

    # Initialize norm violation detector
    print("Initializing norm violation detector...")
    try:
        norm_detector = NormViolationDetector()
        print("✓ Norm detector initialized\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not load norm detector: {e}")
        print("   Continuing without norm violation detection...\n")
        norm_detector = None

    # Initialize APIs
    print("Initializing APIs...")

    # Rev AI
    revai_api_key = os.getenv("REVAI_API_KEY")
    if not revai_api_key:
        print("ERROR: REVAI_API_KEY not found in environment")
        return
    revai_config = MediaConfig('audio/x-raw', 'interleaved', 44100, 'S16LE', 1)

    # Gemini
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return
    genai.configure(api_key=google_api_key)

    system_instruction = """You are a helpful AI assistant in a voice conversation.
When the user says the word "question", they are entering Q&A mode with you.
Once in Q&A mode, respond naturally to everything they say until they say "nevermind" or "that's all".
Keep your responses concise and conversational, as they will be spoken aloud.
The conversation will end when the user says "terminate"."""

    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash-exp',
        system_instruction=system_instruction,
        generation_config={"temperature": 0.7}
    )
    chat = model.start_chat(history=[])

    # OpenAI TTS
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    openai_client = OpenAI(api_key=openai_api_key)

    print("✓ All APIs initialized\n")
    print("🎤 Listening... (say 'question' to enter Q&A mode, 'terminate' to finish)\n")

    # Main conversation loop
    try:
        while conversation_active:
            print("=" * 70)
            # Start listening for user input
            listening = True
            # In idle mode, check each FINAL for keywords/norms. In Q&A mode, wait for silence.
            check_func = check_for_intervention if not question_mode else None
            result = stream_audio_to_revai(device_idx, output_folder, revai_api_key, revai_config,
                                          in_question_mode=question_mode, check_callback=check_func)

            # Process the utterance
            user_text = " ".join([r for r in result if r is not None])

            if not user_text.strip():
                print("⚠️  [NO SPEECH DETECTED] No transcription received, listening again...\n")
                continue

            # Log to transcript
            timestamp = datetime.now().strftime("%H:%M:%S")
            full_transcript.append({
                "timestamp": timestamp,
                "role": "user",
                "text": user_text
            })

            print(f"\n👤 [USER {timestamp}]: {user_text}")

            # Check for keywords
            keyword = check_for_keywords(user_text)

            # Handle END keyword - exit conversation
            if keyword == "END":
                print("\n🛑 'Terminate' keyword detected. Ending conversation...\n")
                conversation_active = False
                break

            # Handle QUESTION keyword - enter Q&A mode
            elif keyword == "QUESTION":
                print("\n🔔 [QUESTION MODE] Activated! AI will respond to everything now.\n")
                question_mode = True

                try:
                    # Get response from Gemini
                    print("🧠 [GEMINI] Generating response...")
                    response = chat.send_message(user_text)
                    response_text = response.text

                    # Log to transcript
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    full_transcript.append({
                        "timestamp": timestamp,
                        "role": "assistant",
                        "text": response_text
                    })

                    # Play audio response
                    listening = False  # Don't record while speaking
                    audio_duration = play_audio_response(openai_client, response_text, output_folder)
                    time.sleep(audio_duration + 0.5)
                    print("⏭️  [READY] Preparing to listen again...\n")

                except Exception as e:
                    print(f"❌ [ERROR] Generating response: {e}")

            # Handle EXIT_QUESTION keyword - exit Q&A mode
            elif keyword == "EXIT_QUESTION":
                print("\n👋 [Q&A MODE] Exiting... Generating goodbye message.\n")

                try:
                    # Get goodbye response from Gemini
                    print("🧠 [GEMINI] Generating goodbye...")
                    response = chat.send_message("The user said they're done with questions. Say a brief, friendly goodbye.")
                    response_text = response.text

                    # Log to transcript
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    full_transcript.append({
                        "timestamp": timestamp,
                        "role": "assistant",
                        "text": response_text
                    })

                    # Play goodbye audio
                    listening = False
                    audio_duration = play_audio_response(openai_client, response_text, output_folder)
                    time.sleep(audio_duration + 0.5)
                    print("⏭️  [READY] Preparing to listen again...\n")

                except Exception as e:
                    print(f"❌ [ERROR] Generating goodbye: {e}")

                print("💤 [Q&A MODE] Deactivated. Returning to idle mode.\n")
                question_mode = False

            # Check for norm violations (only in idle mode)
            elif not question_mode and norm_detector:
                is_violation, severity, scores = norm_detector.check_violation(user_text)
                if is_violation:
                    print(f"\n🚨 [NORM VIOLATION] Severity: {severity} - Intervening...\n")

                    try:
                        # Get intervention response from Gemini with norm context
                        print("🧠 [GEMINI] Generating intervention...")
                        intervention_prompt = f"""The user said: "{user_text}"

This violates our social norms. Provide a brief, supportive intervention that:
1. Gently corrects the behavior
2. Reminds them of the norm
3. Encourages positive interaction

Keep it under 2 sentences and conversational since it will be spoken aloud."""

                        response = chat.send_message(intervention_prompt)
                        response_text = response.text

                        # Log to transcript
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        full_transcript.append({
                            "timestamp": timestamp,
                            "role": "intervention",
                            "text": response_text,
                            "violation_scores": scores
                        })

                        # Play intervention audio
                        listening = False
                        audio_duration = play_audio_response(openai_client, response_text, output_folder)
                        time.sleep(audio_duration + 0.5)
                        print("⏭️  [READY] Returning to idle listening...\n")

                    except Exception as e:
                        print(f"❌ [ERROR] Generating intervention: {e}")

            # If in Q&A mode, respond to everything
            elif question_mode:
                print("\n💬 [Q&A MODE ACTIVE] Generating response...\n")

                try:
                    # Get response from Gemini
                    print("🧠 [GEMINI] Generating response...")
                    response = chat.send_message(user_text)
                    response_text = response.text

                    # Log to transcript
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    full_transcript.append({
                        "timestamp": timestamp,
                        "role": "assistant",
                        "text": response_text
                    })

                    # Play audio response
                    listening = False  # Don't record while speaking
                    audio_duration = play_audio_response(openai_client, response_text, output_folder)
                    time.sleep(audio_duration + 0.5)
                    print("⏭️  [READY] Preparing to listen again...\n")

                except Exception as e:
                    print(f"❌ [ERROR] Generating response: {e}")

            # Not in Q&A mode and no keywords - just listen
            else:
                print("💤 [IDLE] Not in Q&A mode. Say 'question' to activate...\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        conversation_active = False

    # Save and display transcript
    print("\n" + "="*60)
    print("FULL CONVERSATION TRANSCRIPT")
    print("="*60 + "\n")

    for entry in full_transcript:
        role_icon = "👤" if entry["role"] == "user" else "🤖"
        print(f"{role_icon} [{entry['timestamp']}] {entry['role'].upper()}:")
        print(f"   {entry['text']}\n")

    # Save transcript to file
    transcript_file = output_folder / "transcript.json"
    with open(transcript_file, 'w') as f:
        json.dump(full_transcript, f, indent=2)

    transcript_txt = output_folder / "transcript.txt"
    with open(transcript_txt, 'w') as f:
        for entry in full_transcript:
            f.write(f"[{entry['timestamp']}] {entry['role'].upper()}: {entry['text']}\n\n")

    print(f"📁 Transcript saved to: {transcript_file}")
    print(f"📁 Transcript (text) saved to: {transcript_txt}")
    print("\n✓ Conversation ended.\n")


if __name__ == "__main__":
    main()
