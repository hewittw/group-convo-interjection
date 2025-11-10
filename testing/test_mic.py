#!/usr/bin/env python3
"""
Dual Microphone Speech-to-Text Transcriber
Captures audio from two Boya mini mics (stereo channels) and transcribes speech to terminal
Left channel = USER 1, Right channel = USER 2
"""

import speech_recognition as sr
import pyaudio
import sys
import threading
import queue
import time
from datetime import datetime
import numpy as np
import wave
import tempfile
import os


def list_audio_devices():
    """List all available audio input devices"""
    print("\n=== Available Audio Input Devices ===")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"[{i}] {info['name']} (inputs: {info['maxInputChannels']})")
    p.terminate()
    print()


def save_audio_channel_to_wav(audio_data, sample_rate, temp_file):
    """Save mono audio data to a temporary WAV file"""
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def transcribe_audio_chunk(audio_data, sample_rate, user_name, result_queue):
    """Transcribe a chunk of audio data"""
    recognizer = sr.Recognizer()

    # Create temporary WAV file
    temp_file = tempfile.mktemp(suffix=".wav")

    try:
        # Save audio to temporary file
        save_audio_channel_to_wav(audio_data, sample_rate, temp_file)

        # Load audio file for recognition
        with sr.AudioFile(temp_file) as source:
            audio = recognizer.record(source)

        try:
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio)
            timestamp = datetime.now().strftime("%H:%M:%S")
            result_queue.put((user_name, timestamp, text))
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            result_queue.put((user_name, datetime.now().strftime("%H:%M:%S"),
                            f"[ERROR: Could not request results; {e}]"))
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def process_channel(channel_data_queue, user_name, sample_rate, stop_event, result_queue, min_chunk_size=32000):
    """Process audio chunks for a specific channel/user"""
    buffer = np.array([], dtype=np.int16)
    silence_threshold = 500  # Adjust based on your environment
    min_speech_length = 8000  # Minimum samples for speech (about 0.5 seconds at 16kHz)

    while not stop_event.is_set():
        try:
            chunk = channel_data_queue.get(timeout=0.1)
            buffer = np.append(buffer, chunk)

            # Check if we have enough data and detect if there's speech
            if len(buffer) >= min_chunk_size:
                # Calculate energy/volume
                energy = np.abs(buffer).mean()

                if energy > silence_threshold and len(buffer) >= min_speech_length:
                    # We have speech, transcribe it
                    threading.Thread(
                        target=transcribe_audio_chunk,
                        args=(buffer.copy(), sample_rate, user_name, result_queue),
                        daemon=True
                    ).start()

                # Clear buffer after processing
                buffer = np.array([], dtype=np.int16)

        except queue.Empty:
            continue


def print_transcriptions(result_queue, stop_event):
    """Print transcription results from the queue"""
    while not stop_event.is_set() or not result_queue.empty():
        try:
            user_name, timestamp, text = result_queue.get(timeout=0.1)
            print(f"\n[{timestamp}] {user_name}: {text}")
            sys.stdout.flush()
        except queue.Empty:
            continue


def main():
    print("=" * 80)
    print("DUAL MICROPHONE SPEECH-TO-TEXT TRANSCRIBER")
    print("=" * 80)

    # List available devices
    list_audio_devices()

    # Get device index from user
    try:
        device_index = int(input("Enter device index for stereo microphone (2 channels): "))
    except ValueError:
        print("Error: Please enter a valid device number")
        sys.exit(1)

    # Verify device exists and has at least 2 input channels
    p = pyaudio.PyAudio()
    try:
        info = p.get_device_info_by_index(device_index)

        if info['maxInputChannels'] < 2:
            print(f"Error: Device {device_index} needs at least 2 input channels (has {info['maxInputChannels']})")
            print("Please select a stereo device or use separate mono devices.")
            sys.exit(1)

        print("\n" + "=" * 80)
        print(f"Device: {info['name']}")
        print(f"Left Channel (1) = USER 1")
        print(f"Right Channel (2) = USER 2")
        print("=" * 80)
        print("\nStarting transcription... (Press Ctrl+C to stop)")
        print("Speak into the microphones to see transcriptions below:")
        print("=" * 80)

    except (OSError, ValueError) as e:
        print(f"Error: Invalid device index - {e}")
        sys.exit(1)

    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2  # Stereo
    RATE = 16000  # 16kHz sample rate

    # Create queues and events
    user1_queue = queue.Queue()
    user2_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    # Create processing threads
    thread1 = threading.Thread(
        target=process_channel,
        args=(user1_queue, "USER 1", RATE, stop_event, result_queue),
        daemon=True
    )

    thread2 = threading.Thread(
        target=process_channel,
        args=(user2_queue, "USER 2", RATE, stop_event, result_queue),
        daemon=True
    )

    printer_thread = threading.Thread(
        target=print_transcriptions,
        args=(result_queue, stop_event),
        daemon=True
    )

    # Start processing threads
    thread1.start()
    thread2.start()
    printer_thread.start()

    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )

    try:
        stream.start_stream()
        print("\n[System] Listening to both channels...")

        while True:
            # Read stereo audio data
            data = stream.read(CHUNK, exception_on_overflow=False)

            # Convert to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Split stereo into two mono channels
            left_channel = audio_data[0::2]   # USER 1
            right_channel = audio_data[1::2]  # USER 2

            # Put channel data in respective queues
            user1_queue.put(left_channel)
            user2_queue.put(right_channel)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Stopping transcription...")
    finally:
        stop_event.set()
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Wait for threads to finish
        thread1.join(timeout=2)
        thread2.join(timeout=2)
        printer_thread.join(timeout=2)

        print("Transcription stopped. Goodbye!")
        print("=" * 80)


if __name__ == "__main__":
    main()
