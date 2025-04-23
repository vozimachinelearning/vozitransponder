import os
import threading
import time
import wave
import numpy as np
from src.audio import AudioProcessor

class AudioCaptureThread(threading.Thread):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.running = True
        self.audio_processor = AudioProcessor(device_index=0)  # Use default audio device (index 0)
        
    def handle_audio(self, audio_data):
        """Process audio data and send it to the callback"""
        if self.callback and audio_data:
            self.callback(audio_data)
    
    def run(self):
        """Main thread loop that continuously processes audio"""
        # Start the audio processor
        self.audio_processor.start()
        
        # Check if speaker.wav exists, if not it will be created by the audio processor
        speaker_file = os.path.join(os.path.dirname(__file__), 'speaker.wav')
        if not os.path.exists(speaker_file):
            print(f"No speaker sample found. Will create one at {speaker_file}")
        else:
            print(f"Found existing speaker sample at {speaker_file}")
        
        print("Audio capture thread started. Listening for speech...")
        
        while self.running:
            try:
                # Get the next audio segment from the processor
                audio_data = self.audio_processor.get_next_audio()
                
                if audio_data:
                    # Handle the audio data (send to callback)
                    self.handle_audio(audio_data)
                
                # Sleep a bit to prevent CPU overuse
                time.sleep(0.1)
                
            except Exception as e:
                print(f"AudioCaptureThread error: {e}")
    
    def stop(self):
        """Stop the audio capture thread"""
        self.running = False
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop()

# Example usage
def transcription_callback(audio_data):
    """Example callback function that would be passed to the transcriber"""
    print(f"Received audio data: {len(audio_data)} bytes")
    # In a real application, this would pass the audio to the transcriber
    # from src.transcriber import Transcriber
    # transcriber.transcribe(audio_data)

# This allows the script to be run directly for testing
if __name__ == "__main__":
    # Create and start the audio capture thread
    audio_thread = AudioCaptureThread(callback=transcription_callback)
    audio_thread.start()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop the thread when Ctrl+C is pressed
        print("Stopping audio capture...")
        audio_thread.stop()
        audio_thread.join()
        print("Audio capture stopped")