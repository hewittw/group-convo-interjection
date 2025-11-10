"""
Speech-to-Text ROS2 Node: 

This node captures audio from a USB microphone, transcribes it using Google Cloud Speech-to-Text,
and publishes user speech events and interruption events to the ROS2 network.
"""
import zmq
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_srvs.srv import Trigger
import queue
import re
import time
import datetime
import json
import sounddevice as sd
import numpy as np
from google.cloud import speech
from typing import Optional, List, Dict, Any

from py_pubsub.conversation_logger import get_conversation_logger

class AudioConfig:
    """Audio configuration constants."""
    STREAMING_LIMIT = 240000  # 4 minutes in milliseconds
    SAMPLE_RATE = 48000
    CHUNK_SIZE = int(SAMPLE_RATE // 5)
    DEVICE_NAME = "TKGOU PnP USB Microphone" # Microphone device
    # DEVICE_NAME = 3
    CREDENTIALS_PATH = "/home/icl/2026-hri-conversation-design/google-credentials.json"

class ResumableMicrophoneStream:
    """
    Manages a recording stream using sounddevice.
    
    This class handles audio input from a microphone device, buffering audio data
    and providing it as a generator for streaming recognition.
    """
    
    def __init__(self, rate: int, chunk_size: int, device: Optional[int] = None):
        """
        Initialize the microphone stream.
        
        Args:
            rate: Sample rate in Hz
            chunk_size: Size of audio chunks
            device: Device index (None for default)
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self.device = device
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = self._get_current_time()

    def __enter__(self):
        """Start the audio stream."""
        self.closed = False
        try:
            self.stream = sd.InputStream(
                samplerate=self._rate,
                channels=1,
                dtype='int16',
                callback=self._fill_buffer,
                device=self.device,
                blocksize=self.chunk_size
            )
            self.stream.start()
            return self
        except Exception as e:
            self.closed = True
            raise RuntimeError(f"Failed to start audio stream: {e}")
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Stop and clean up the audio stream."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.closed = True
        self._buff.put(None)

    def _fill_buffer(self, in_data: np.ndarray, frames: int, time_info, status) -> None:
        """
        Callback function to fill the audio buffer.
        
        Args:
            in_data: Audio data from the microphone
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            print(f"Sounddevice error: {status}")
        self._buff.put(in_data.copy())

    def generator(self):
        """
        Generate audio chunks for streaming recognition.
        
        Yields:
            bytes: Audio data chunks
        """
        while not self.closed:
            try:
                data = []
                chunk = self._buff.get(timeout=1.0)
                if chunk is None:
                    return
                data.append(chunk)
                
                # Collect additional chunks if available
                while True:
                    try:
                        chunk = self._buff.get(block=False)
                        if chunk is None:
                            return
                        data.append(chunk)
                    except queue.Empty:
                        break
                
                combined_data = b"".join(data)
                yield combined_data
            except queue.Empty:
                continue

    @staticmethod
    def _get_current_time() -> int:
        """Get current time in milliseconds."""
        return int(round(time.time() * 1000))


class SpeechToTextNode(Node):
    """
    ROS2 Node for Speech-to-Text functionality.
    
    This node listens to audio input, transcribes speech using Google Cloud Speech-to-Text,
    detects wake words and interruptions, and publishes appropriate messages.
    
    Publishers:
        - user_interruption (String): User interruption events with metadata
            - JSON IDs: {is_wakeword, interruption_time, transcribed_audio}
            - Subscribed in: intent_classifier
        - user_speech (String): Final transcribed user speech
            - JSON IDs: {transcribed_audio}
            - Subscribed in: robot_behavior_generator

    Service Clients:
        - get_is_talking (Trigger): Check if robot is currently speaking
    
    Input Device:
        - USB Microphone: TKGOU PnP USB Microphone
    
    Output:
        - ROS2 topics for speech events and interruptions
    """
    
    def __init__(self):
        """Initialize the Speech-to-Text node."""
        super().__init__('speech_to_text')
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize service clients
        self._setup_service_clients()
        
        # Initialize speech recognition
        self._setup_speech_recognition()
        
        # Initialize audio device
        self.device_index = self._find_device_index(AudioConfig.DEVICE_NAME)
        if self.device_index is None:
            self.get_logger().error(f"Microphone '{AudioConfig.DEVICE_NAME}' not found.")
            raise RuntimeError("Required microphone not found")
        
        # Initialize state variables
        self._init_state_variables()
        
        self.logger = get_conversation_logger()

        self.get_logger().info(f"Speech-to-Text node initialized with microphone: {AudioConfig.DEVICE_NAME}")

        self.listening_enabled = False  #0710


        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:12345")  # Adjust port as needed
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self.timer = self.create_timer(0.1, self.receive_msg)  # Timer to check for messages


        # context_end = zmq.Context()
        # self.socket_end = context_end.socket(zmq.SUB)
        # self.socket_end.connect("tcp://localhost:123456")  # Adjust port as needed
        # self.socket_end.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        # self.timer_end = self.create_timer(0.1, self.receive_msg_end)  # Timer to check for messages

    # def receive_msg_end(self):
    #     """Receive messages from the ZeroMQ socket."""
    #     try:
    #         message = self.socket_end.recv_string(flags=zmq.NOBLOCK)
    #         self.get_logger().info("Received message from ZeroMQ")
    #         if message.startswith("conversation ended"):
    #             self.get_logger().info("Received end message from ZeroMQ")
    #             self.listening_end_callback(Empty())
    #     except zmq.Again:
    #         # No message received, continue
    #         pass
        
    def receive_msg(self):
        """Receive messages from the ZeroMQ socket."""
        try:
            message = self.socket.recv_string(flags=zmq.NOBLOCK)
            self.get_logger().info("[SpeechToText] Received message from ZeroMQ")
            if message.startswith("conversation started"):
                self.get_logger().info("Received start message from ZeroMQ")
                self.listening_start_callback(Empty())
            elif message.startswith("conversation ended"):
                self.get_logger().info("Received end message from ZeroMQ")
                self.listening_end_callback(Empty())
        except zmq.Again:
            # No message received, continue
            pass

        # self.start_listen_subscription = self.create_subscription(
        #     Empty,
        #     '/conversation_start',  # FROM: UI?
        #     self.listening_start_callback,
        #     10
        # )

        # self.end_listen_subscription = self.create_subscription(
        #     Empty,
        #     '/conversation_end',  # FROM: UI?
        #     self.listening_end_callback,
        #     10
        # )


    def _setup_publishers(self) -> None:
        """Set up ROS2 publishers."""
        self.user_interruption_publisher = self.create_publisher(
            String, 
            'user_interruption', 
            10
        )
        
        self.user_speech_publisher = self.create_publisher(
            String, 
            'user_speech', 
            10
        )

    def _setup_service_clients(self) -> None:
        """Set up ROS2 service clients."""
        self.get_is_talking_client = self.create_client(Trigger, 'get_is_talking')
        
        # Wait for service to be available
        timeout_sec = 10.0
        if not self.get_is_talking_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('get_is_talking service not available after waiting')
            raise RuntimeError("Required service not available")
        
        self.get_logger().info('get_is_talking service is available')

    def _setup_speech_recognition(self) -> None:
        """Set up Google Cloud Speech-to-Text client and configuration."""
        try:
            self.speech_client = speech.SpeechClient.from_service_account_file(
                AudioConfig.CREDENTIALS_PATH
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Google Speech client: {e}")
            raise
        
        self.speech_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=AudioConfig.SAMPLE_RATE,
            language_code="en-US",
            max_alternatives=1
        )
        
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.speech_config, 
            interim_results=True,
            single_utterance=False
        )

    def _init_state_variables(self) -> None:
        """Initialize node state variables."""
        self.wakewords = ["luna", "stop", "hey luna"]
        self.is_interrupted = False
        self.interruption_time = ""
        self.wakeword_detected = False
        self.wakeword_detected_sent = False
        self.interruption_sent = False

    def _find_device_index(self, device_name: str) -> Optional[int]:
        """
        Find the device index for the given device name.
        
        Args:
            device_name: Name of the audio device to find
            
        Returns:
            Device index if found, None otherwise
        """
        try:
            devices = sd.query_devices()
            for index, device in enumerate(devices):
                if device_name in device['name']:
                    self.get_logger().info(f"Found microphone '{device_name}' at index {index}")
                    return index
            return None
        except Exception as e:
            self.get_logger().error(f"Error querying audio devices: {e}")
            return None

    def _detect_wakeword(self, transcript: str) -> bool:
        """
        Check if any wakeword is present in the transcript.
        
        Args:
            transcript: The transcribed text to check
            
        Returns:
            True if a wakeword is detected, False otherwise
        """
        transcript_lower = transcript.lower().strip()
        
        for wakeword in self.wakewords:
            # Check for exact match or wakeword at boundaries
            if (wakeword == transcript_lower or 
                transcript_lower.startswith(wakeword + " ") or
                transcript_lower.endswith(" " + wakeword) or
                (" " + wakeword + " ") in transcript_lower):
                return True
        
        return False

    def _is_robot_talking(self) -> bool:
        """
        Check if the robot is currently talking.
        
        Returns:
            True if robot is talking, False otherwise
        """
        try:
            request = Trigger.Request()
            future = self.get_is_talking_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            response = future.result()
            if response and response.success:
                parsed_response = json.loads(response.message)
                is_talking_value = parsed_response.get("is_talking", "false")
                return is_talking_value.lower() == "true"
            else:
                self.get_logger().warning('get_is_talking service call failed')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return False

    def _publish_user_interruption(self, is_wakeword: bool = False, transcribed_audio: str = "") -> None:
        """
        Publish user interruption event.
        
        Args:
            is_wakeword: Whether the interruption was triggered by a wakeword
            transcribed_audio: The transcribed audio content
        """
        msg = String()
        response = {
            'is_wakeword': is_wakeword,
            'interruption_time': self.interruption_time,
            'transcribed_audio': transcribed_audio
        }
        msg.data = json.dumps(response)
        self.user_interruption_publisher.publish(msg)
        
        event_type = "wakeword" if is_wakeword else "regular"

        self.get_logger().info(f"Published {event_type} interruption: '{transcribed_audio}'")

    def _publish_user_speech(self, transcribed_audio: str) -> None:
        """
        Publish user speech event.
        
        Args:
            transcribed_audio: The final transcribed audio content
        """
        msg = String()
        response = {
            'transcribed_audio': transcribed_audio,
        }
        msg.data = json.dumps(response)
        self.user_speech_publisher.publish(msg)
        self.get_logger().info(f"Published user speech: '{transcribed_audio}'")


    def _process_transcript(self, transcript: str, is_final: bool) -> None:
        """
        Process a transcript and handle speech events.
        
        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcription result
        """
        word_count = len(re.findall(r'\w+', transcript))
        is_wakeword = self._detect_wakeword(transcript)
        
        if not self._is_robot_talking() and (self.is_interrupted or self.interruption_sent):
            self._reset_state()

        if self._is_robot_talking():
            self.is_interrupted = True

            # Handle wakeword detection - ONLY stop, don't generate behavior
            if is_wakeword and not self.wakeword_detected_sent:
                self.interruption_time = str(datetime.datetime.utcnow())
                self.wakeword_detected = True
                self.wakeword_detected_sent = True
                
                # Publish interruption to stop robot, but don't publish to behavior generator
                self._publish_user_interruption(is_wakeword=True, transcribed_audio=transcript)
                
                # CRITICAL: Return early to prevent any speech publication for wakewords
                if is_final:
                    self._reset_state()
                return
            
            # Handle regular interruption (robot talking + substantial speech + no wakeword)
            elif word_count >= 2 and not self.interruption_sent:
                self.interruption_time = str(datetime.datetime.utcnow())
                self.interruption_sent = True
                self._publish_user_interruption(is_wakeword=False, transcribed_audio=transcript)

                if is_final:
                    self._reset_state()
                return
        else: 
            # Handle final transcriptions - only for NON-wakeword speech
            if is_final:
                # Publish speech only if substantial content
                if word_count >= 2:
                    self._publish_user_speech(transcript)
                else:
                    self.logger.log_user_short_speech(transcript)
                
                # Reset state
                self._reset_state()

    def _reset_state(self) -> None:
        """Reset node state variables."""
        self.is_interrupted = False
        self.wakeword_detected = False
        self.wakeword_detected_sent = False
        self.interruption_sent = False
        self.interruption_time = ""

    # def _toggle_listening(self, enabled:Bool): #0710
    #     self.listening_enabled = enabled
    #     self.get_logger().info(f"Speech2Text of listening is {self.listening_enabled}")

    def listening_start_callback(self, msg):
        self.listening_enabled = True
        self.get_logger().info("Listening begin in speech2text")

    def listening_end_callback(self, msg):
        self.listening_enabled = False
        self.get_logger().info("Listening end in speech2text")

    def _listen_print_loop(self, responses, stream: ResumableMicrophoneStream) -> None:
        """
        Process responses from Google Speech-to-Text API.
        
        Args:
            responses: Iterator of speech recognition responses
            stream: The microphone stream object
        """
        for response in responses:
            # Check streaming limit
            if ResumableMicrophoneStream._get_current_time() - stream.start_time > AudioConfig.STREAMING_LIMIT:
                stream.start_time = ResumableMicrophoneStream._get_current_time()
                self.get_logger().info("Streaming limit reached. Resetting stream.")
                break
            
            # Skip empty results
            if not response.results:
                continue
            
            result = response.results[0]
            
            # Skip results without alternatives
            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            
            # Process transcript if it contains text
            if transcript:
                # only process if listening is enabled
                if self.listening_enabled:
                    self._process_transcript(transcript, result.is_final)

    def start_streaming(self) -> None:
        """Start the audio streaming and speech recognition loop."""
        self.get_logger().info("Starting speech recognition streaming...")
        
        while rclpy.ok():
            if not self.listening_enabled:
                time.sleep(0.5)
                continue
            try:
                with ResumableMicrophoneStream(
                    AudioConfig.SAMPLE_RATE, 
                    AudioConfig.CHUNK_SIZE, 
                    self.device_index
                ) as stream:
                    
                    audio_generator = stream.generator()
                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator
                    )
                    
                    responses = self.speech_client.streaming_recognize(
                        self.streaming_config, 
                        requests
                    )
                    
                    self._listen_print_loop(responses, stream)
                    
            except Exception as e:
                self.get_logger().error(f"Streaming error: {e}")
                time.sleep(1.0)  # Brief pause before retrying


def main(args=None):
    """Main function to run the Speech-to-Text node."""
    rclpy.init(args=args)
    
    try:
        node = SpeechToTextNode()
        
        # Start streaming in a separate thread-like manner
        import threading
        streaming_thread = threading.Thread(target=node.start_streaming, daemon=True)
        streaming_thread.start()
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("Shutting down Speech-to-Text node...")
    except Exception as e:
        print(f"Error running Speech-to-Text node: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()