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

# Norm violation detection with Ollama
import subprocess

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
    """Local Llama2 model for detecting norm violations using Ollama."""

    def __init__(self, model_name="llama3.2:3b", norms_file="norms.txt"):
        print(f"📦 Loading norm detection with Ollama ({model_name})...")
        self.model_name = model_name

        # Load norms from file
        try:
            with open(norms_file, 'r') as f:
                self.norms = f.read()
            print(f"   ✓ Loaded norms from {norms_file}")
        except FileNotFoundError:
            print(f"   ⚠️  Warning: {norms_file} not found, using default norms")
            self.norms = """
1. Respect yourself and others - No self-deprecating language or insults
2. Positive collaboration - Take turns, listen actively
3. Constructive communication - Give helpful feedback, disagree respectfully
4. Inclusive behavior - Welcome different perspectives
"""

        # Test that Ollama is available
        try:
            subprocess.run(["ollama", "list"], capture_output=True, check=True, timeout=5)
            print(f"   ✓ Ollama is ready with model: {model_name}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not verify Ollama: {e}")

    def check_violation(self, text: str, min_words=5):
        """Check if text violates norms using Llama2. Returns (is_violation, severity, explanation)."""
        # Skip very short phrases
        if len(text.split()) < min_words:
            return False, None, {}

        # Construct prompt for Llama2
        prompt = f"""Given these social norms:
{self.norms}

Does this statement CLEARLY and DIRECTLY violate any of the norms? Only say YES if it's an obvious violation.
Be lenient - casual conversation, minor disagreements, and normal expressions are acceptable.

Statement: "{text}"

Answer with ONLY "YES" or "NO" and nothing else."""

        try:
            # Call Ollama with timeout
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=5.0  # 5 second timeout for fast response
            )

            response = result.stdout.strip().lower()

            # Check if violation detected
            if "yes" in response:
                # For now, treat all violations as "soft" (can enhance later)
                return True, "soft", {"llama_response": response}
            else:
                return False, None, {"llama_response": response}

        except subprocess.TimeoutExpired:
            print("   ⚠️  Llama2 timeout - skipping violation check")
            return False, None, {"error": "timeout"}
        except Exception as e:
            print(f"   ⚠️  Llama2 error: {e}")
            return False, None, {"error": str(e)}


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


# =============================================================================
# NEW: Persistent Rev AI session (keeps Rev connection alive; no restart per turn)
# =============================================================================
class PersistentRevAiTranscriber:
    """
    Keeps ONE Rev AI streaming connection alive for the entire conversation.
    Per "turn", we:
      - clear buffers
      - set mode (idle vs Q&A)
      - wait until either:
          * intervention trigger happens (idle)
          * silence timeout happens (Q&A)
    We do NOT close Rev or recreate the client per turn.
    """
    def __init__(self, device_idx, output_folder, revai_api_key, revai_config, silence_threshold=2.0):
        self.device_idx = device_idx
        self.output_folder = output_folder
        self.revai_api_key = revai_api_key
        self.revai_config = revai_config
        self.silence_threshold = silence_threshold

        self.rate = 44100
        self.chunk = int(self.rate / 10)

        self._stream = None
        self._ffmpeg_process = None
        self._response_thread = None
        self._silence_thread = None

        self._utterance_queue = queue.Queue()

        self._lock = threading.Lock()
        self._collecting = False
        self._in_question_mode = False
        self._check_callback = None

        self._full_phrase = []
        self._first_partial = True

        self._stop_event = threading.Event()

        # One client for the whole session
        print("🔌 [CONNECTING] Creating persistent Rev AI client...")
        self._client = RevAiStreamingClient(self.revai_api_key, self.revai_config)
        print("   ✓ Rev AI client created (persistent)")

    def _silence_monitor_loop(self):
        global silence_time, transcribing, listening, conversation_active

        while not self._stop_event.is_set():
            time.sleep(0.05)

            if not conversation_active:
                break

            with self._lock:
                if not self._collecting:
                    continue
                in_question = self._in_question_mode
                has_text = len(self._full_phrase) > 0

            # Only do silence-end in Q&A mode
            if in_question and has_text:
                if (time.time() - silence_time > self.silence_threshold and not transcribing):
                    print("🤫 [SILENCE DETECTED] Processing utterance...")
                    self._emit_current_utterance()

        # On stop, best-effort flush
        with self._lock:
            if self._collecting and len(self._full_phrase) > 0:
                self._emit_current_utterance()

    def _emit_current_utterance(self):
        """
        Emit the currently collected utterance to the main loop,
        then stop collecting until the next 'start_turn()'.
        """
        with self._lock:
            if not self._collecting:
                return
            text = " ".join(self._full_phrase).strip()
            if not text:
                return

            # Stop collecting until next turn is started
            self._collecting = False
            self._in_question_mode = False
            self._check_callback = None

            # Reset phrase buffer for next turn
            self._full_phrase = []
            self._first_partial = True

        self._utterance_queue.put(text)

    def _audio_generator_wrapper(self, mic_gen):
        """
        Wraps the mic generator so we can keep the Rev stream alive without
        transcribing the assistant audio: when global 'listening' is False,
        we feed zeros (silence) of the same length instead of real mic audio.
        """
        global listening

        for raw in mic_gen:
            if self._stop_event.is_set():
                return
            if listening:
                yield raw
            else:
                # feed silence to keep connection alive
                yield b"\x00" * len(raw)

    def _response_loop(self, response_gen):
        global silence_time, transcribing, listening, conversation_active

        for raw_response in response_gen:
            if self._stop_event.is_set() or not conversation_active:
                break

            try:
                response = json.loads(raw_response)
            except Exception:
                continue

            with self._lock:
                collecting = self._collecting
                in_question = self._in_question_mode
                check_cb = self._check_callback

            # If we're not collecting for a turn, we still update timers on partials
            # but we ignore transcript accumulation (keeps connection warm).
            if response.get("type") == "partial":
                transcribing = True
                silence_time = time.time()

                if not collecting:
                    continue

                text = "".join(e.get("value", "") for e in response.get("elements", []))
                if text.strip():
                    if self._first_partial:
                        print("🎙️  [SPEECH DETECTED] Processing audio...")
                        self._first_partial = False
                    print(f"📝 [TRANSCRIBING] {text}")

            elif response.get("type") == "final":
                silence_time = time.time()

                text = "".join(e.get("value", "") for e in response.get("elements", []))
                if not text.strip():
                    transcribing = False
                    continue

                if collecting:
                    print(f"✅ [FINAL] {text}")

                    with self._lock:
                        self._full_phrase.append(text)

                    # Idle mode: check each FINAL immediately for intervention
                    if (not in_question) and check_cb:
                        should_intervene = check_cb(" ".join(self._full_phrase))
                        if should_intervene:
                            print("🚨 [INTERVENTION NEEDED] Stopping listening...")
                            listening = False  # keep same behavior: don't record while assistant speaks
                            self._emit_current_utterance()

                transcribing = False

        # if loop ends, mark stop
        self._stop_event.set()

    def start(self):
        """
        Start mic capture + ffmpeg logging + Rev streaming ONCE.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_log_path = self.output_folder / f"audio_session_{timestamp}.mp3"

        # Open mic stream once
        self._stream = SimpleMicrophoneStream(self.rate, self.chunk, self.device_idx).__enter__()
        print("   ✓ Microphone stream opened (persistent)")

        # Start FFmpeg once (continuous session log)
        self._ffmpeg_process = subprocess.Popen([
            "ffmpeg", "-f", "s16le", "-ar", str(self.rate), "-ac", "1",
            "-i", "pipe:0", "-y", "-codec:a", "libmp3lame",
            "-b:a", "192k", str(audio_log_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✓ Audio logging started (persistent session file)")

        # Start Rev streaming once
        print("   ⏳ Connecting to Rev AI servers (persistent stream)...")
        mic_gen = self._stream.generator(self._ffmpeg_process.stdin)
        wrapped_gen = self._audio_generator_wrapper(mic_gen)
        response_gen = self._client.start(wrapped_gen)
        print("🎤 [LISTENING] Microphone active, speak now...")

        # Start threads: response reader + silence monitor
        self._response_thread = threading.Thread(target=self._response_loop, args=(response_gen,), daemon=True)
        self._response_thread.start()

        self._silence_thread = threading.Thread(target=self._silence_monitor_loop, daemon=True)
        self._silence_thread.start()

    def stop(self):
        """
        Stop everything at end of conversation.
        """
        self._stop_event.set()

        # Best-effort shutdown of mic + ffmpeg
        try:
            if self._stream and not self._stream.closed:
                self._stream.closed = True
        except Exception:
            pass

        try:
            if self._ffmpeg_process:
                try:
                    self._ffmpeg_process.stdin.close()
                except Exception:
                    pass
                try:
                    self._ffmpeg_process.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self._stream:
                # call __exit__ explicitly since we manually __enter__'d
                self._stream.__exit__(None, None, None)
        except Exception:
            pass

    def get_next_utterance(self, in_question_mode=False, check_callback=None, block=True, timeout=None):
        """
        Start collecting for ONE utterance, then block until emitted.

        - Idle mode (in_question_mode=False): emits as soon as check_callback returns True
        - Q&A mode (in_question_mode=True): emits when silence threshold passes
        """
        global silence_time

        with self._lock:
            self._full_phrase = []
            self._first_partial = True
            self._collecting = True
            self._in_question_mode = in_question_mode
            self._check_callback = check_callback

        silence_time = time.time()

        if block:
            return self._utterance_queue.get(timeout=timeout)
        return None


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

    # NEW: start ONE persistent Rev AI session
    transcriber = PersistentRevAiTranscriber(
        device_idx=device_idx,
        output_folder=output_folder,
        revai_api_key=revai_api_key,
        revai_config=revai_config,
        silence_threshold=2.0
    )
    transcriber.start()

    print("🎤 Listening... (say 'question' to enter Q&A mode, 'terminate' to finish)\n")

    # Main conversation loop
    try:
        while conversation_active:
            print("=" * 70)

            # Start listening for user input (same logic, but no Rev restart)
            listening = True

            # In idle mode: emit when intervention triggers.
            # In Q&A mode: emit when silence threshold triggers.
            check_func = check_for_intervention if not question_mode else None

            user_text = transcriber.get_next_utterance(
                in_question_mode=question_mode,
                check_callback=check_func,
                block=True
            )

            if not user_text or not user_text.strip():
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
                    listening = False  # Don't record while speaking (now feeds silence to Rev)
                    audio_duration = play_audio_response(openai_client, response_text, output_folder)
                    time.sleep(0.5)
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
                    time.sleep(0.5)
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
                        time.sleep(0.5)
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
                    time.sleep(0.5)
                    print("⏭️  [READY] Preparing to listen again...\n")

                except Exception as e:
                    print(f"❌ [ERROR] Generating response: {e}")

            # Not in Q&A mode and no keywords - just listen
            else:
                print("💤 [IDLE] Not in Q&A mode. Say 'question' to activate...\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        conversation_active = False

    finally:
        # NEW: stop persistent transcriber cleanly
        try:
            transcriber.stop()
        except Exception:
            pass

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
