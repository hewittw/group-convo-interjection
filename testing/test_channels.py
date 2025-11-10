#!/usr/bin/env python3
"""
Audio Channel Level Monitor
Visualize the audio levels from left and right channels to verify they're separate
"""

import pyaudio
import numpy as np
import sys
import time


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


def create_level_bar(level, max_width=40):
    """Create a visual bar representation of audio level"""
    normalized = min(level / 5000, 1.0)  # Adjust 5000 based on your mic sensitivity
    bar_length = int(normalized * max_width)
    bar = '█' * bar_length + '░' * (max_width - bar_length)
    return bar


def main():
    print("=" * 80)
    print("CHANNEL LEVEL MONITOR - Test if channels are separate")
    print("=" * 80)

    list_audio_devices()

    try:
        device_index = int(input("Enter device index: "))
    except ValueError:
        print("Error: Please enter a valid device number")
        sys.exit(1)

    p = pyaudio.PyAudio()

    try:
        info = p.get_device_info_by_index(device_index)
        print(f"\nDevice: {info['name']}")
        print(f"Channels: {info['maxInputChannels']}")
        print("\nTEST: Try speaking into ONE mic at a time and see if only one bar moves!")
        print("=" * 80 + "\n")
    except (OSError, ValueError) as e:
        print(f"Error: Invalid device index - {e}")
        sys.exit(1)

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000

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

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Split stereo
            left_channel = audio_data[0::2]
            right_channel = audio_data[1::2]

            # Calculate levels
            left_level = np.abs(left_channel).mean()
            right_level = np.abs(right_channel).mean()

            # Create bars
            left_bar = create_level_bar(left_level)
            right_bar = create_level_bar(right_level)

            # Print levels
            sys.stdout.write('\033[2K\033[1G')
            print(f"LEFT  (USER 1): {left_bar} {left_level:>6.0f}", end='\r')
            sys.stdout.write('\n')
            sys.stdout.write('\033[2K\033[1G')
            print(f"RIGHT (USER 2): {right_bar} {right_level:>6.0f}", end='\r')
            sys.stdout.write('\033[1A')
            sys.stdout.flush()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
