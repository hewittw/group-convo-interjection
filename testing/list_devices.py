#!/usr/bin/env python3
"""
List all audio devices in detail
"""

import pyaudio

p = pyaudio.PyAudio()

print("=" * 80)
print("ALL AUDIO DEVICES (INPUT AND OUTPUT)")
print("=" * 80)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"\n[{i}] {info['name']}")
    print(f"    Max Input Channels:  {info['maxInputChannels']}")
    print(f"    Max Output Channels: {info['maxOutputChannels']}")
    print(f"    Default Sample Rate: {int(info['defaultSampleRate'])} Hz")

p.terminate()

print("\n" + "=" * 80)
print("INPUT DEVICES ONLY:")
print("=" * 80)

p = pyaudio.PyAudio()
input_devices = []

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        input_devices.append((i, info))
        print(f"[{i}] {info['name']} ({info['maxInputChannels']} channels)")

p.terminate()

print("\n" + "=" * 80)
if len(input_devices) >= 2:
    print("You have multiple input devices available!")
    print("If your two Boya mics show up as SEPARATE devices, we can use them separately.")
else:
    print("Note: Your Boya mics may be combined into one device.")
print("=" * 80)
