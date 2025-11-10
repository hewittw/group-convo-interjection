# Keyword-Triggered Conversation System

A proof-of-concept voice conversation system that uses keyword detection to trigger AI responses. Built with Rev AI (STT), Google Gemini (LLM), and OpenAI TTS.

## Features

- **Real-time speech-to-text** streaming using Rev AI
- **Keyword detection** to trigger AI responses
- **Natural voice responses** using OpenAI text-to-speech
- **Full conversation transcript** saved after each session
- **Audio logging** of the entire conversation

## Keywords

- **"question"** - Triggers an AI response from Gemini
- **"end"** - Ends the conversation and displays the full transcript

## Setup

### 1. Install System Dependencies

**FFmpeg** is required for audio processing:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project directory with your API keys:

```env
REVAI_API_KEY=your_rev_ai_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Getting API Keys:**

- **Rev AI**: Sign up at https://www.rev.ai/ and get your access token
- **Google Gemini**: Get an API key from https://ai.google.dev/
- **OpenAI**: Get an API key from https://platform.openai.com/api-keys

## Usage

### Running the Program

```bash
python keyword_conversation.py
```

### Workflow

1. **Select Microphone**: The program will list all available microphones. Enter the index number of your preferred microphone.

2. **Start Speaking**: The system will continuously listen and transcribe your speech.

3. **Trigger Response**: Say the word **"question"** anywhere in your speech to get an AI response.
   - Example: "I have a question about Python programming"
   - The AI will respond and speak its answer

4. **End Conversation**: Say the word **"end"** to terminate the session.
   - The full transcript will be displayed
   - All logs saved to `./conversation_logs/[timestamp]/`

### Example Conversation

```
👤 User: Hello, I have a question about machine learning
🔔 Question keyword detected!
🤖 Assistant: Machine learning is a subset of artificial intelligence...

👤 User: Thanks! I think we're done here. End.
🛑 End keyword detected. Terminating conversation...

=== FULL CONVERSATION TRANSCRIPT ===
```

## Output Files

After each conversation, the following files are saved:

```
conversation_logs/
  └── [timestamp]/
      ├── transcript.json      # Structured transcript with timestamps
      ├── transcript.txt       # Human-readable transcript
      ├── audio_*.mp3          # Recorded user audio segments
      └── response_*.mp3       # AI response audio files
```

## Architecture

```
User speaks
    ↓
Rev AI (Speech-to-Text)
    ↓
Keyword Detection ("question" or "end")
    ↓
[If "question"] → Google Gemini (LLM) → OpenAI TTS → Play Audio
[If "end"] → Save transcript and exit
```

## Customization

### Changing Keywords

Edit the `check_for_keywords()` function in [keyword_conversation.py](keyword_conversation.py:238):

```python
def check_for_keywords(text):
    text_lower = text.lower()

    if "your_end_keyword" in text_lower:
        return "END"
    elif "your_trigger_keyword" in text_lower:
        return "QUESTION"
```

### Changing AI Model or Voice

- **Gemini Model**: Line 272 - Change `model_name='gemini-2.0-flash-exp'`
- **OpenAI Voice**: Line 228 - Change `voice="alloy"` (options: alloy, echo, fable, onyx, nova, shimmer)

### Adjusting Silence Detection

Change the silence threshold in the `monitor_silence()` function (line 121):

```python
silence_threshold=2.0  # seconds of silence before ending utterance
```

## Troubleshooting

### Microphone Not Found
- Run `python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`
- Make sure your microphone is connected and recognized by your OS

### FFmpeg Not Found
- Ensure FFmpeg is installed and in your system PATH
- Test with: `ffmpeg -version`

### API Key Errors
- Verify all API keys are correctly set in `.env`
- Check that your API keys have not expired and have available credits

### Audio Playback Issues
- **macOS**: Should work with `afplay` (built-in)
- **Linux**: Install `mpg123` with `sudo apt-get install mpg123`
- **Windows**: Audio files will open with default player

## Based On

This is a simplified proof-of-concept based on the architecture used in the `misty-sel-multi` robot interaction system, streamlined for standalone keyword-triggered conversations without robot hardware.

## License

MIT
