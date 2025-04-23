import pyaudio
import audioop
import threading
import numpy as np
import time
import os
import wave
import queue
from collections import deque

class AudioRingBuffer:
    """Simple ring buffer for audio data"""
    def __init__(self, max_size=10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, data):
        with self.lock:
            self.buffer.append(data)
    
    def get_all(self):
        with self.lock:
            return b''.join(list(self.buffer))
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
    
    def close(self):
        self.clear()

class AudioProcessor:
    def __init__(self, device_index=0, sample_rate=16000, chunk_size=4092):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = 1  # Mono
        
        # Speech detection parameters
        self.base_threshold = 1.3
        self.silence_duration = 1
        self.min_speech_frames = int(self.sample_rate / self.chunk_size * 0.1)
        self.energy_window = 3
        
        # Dynamic threshold adjustment parameters
        self.max_threshold_ratio = 1.8
        self.min_threshold_ratio = 1.1
        self.threshold_decay = 0.95
        self.noise_adapt_speed = 0.15
        
        # State variables
        self.is_recording = False
        self.frames = []
        self.silent_chunks = 0
        self.energy_history = deque(maxlen=self.energy_window)
        self.rms_history = deque(maxlen=self.energy_window)
        self.current_noise_floor = 0
        self.threshold_ratio = 1.3
        self.dynamic_threshold = 0
        self.energy_threshold = 0
        
        # Setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the audio processing thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the audio processing thread"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.audio:
            self.audio.terminate()
            
    def _calculate_energy(self, audio_chunk):
        """Calculate energy level of audio chunk"""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        
    def _calibrate(self):
        """Calibrate audio levels"""
        print("Calibrating audio levels...")
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        
        # Take a few samples to calibrate
        noise_samples = []
        for _ in range(5):  # 5 samples for calibration
            data = self.stream.read(self.chunk_size * 4, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            energy = self._calculate_energy(data)
            noise_samples.append((rms, energy))
            self.rms_history.append(rms)
            self.energy_history.append(energy)
            
        self.current_noise_floor = np.mean([x[0] for x in noise_samples])
        self.dynamic_threshold = self.current_noise_floor * self.threshold_ratio
        self.energy_threshold = np.mean([x[1] for x in noise_samples]) * self.threshold_ratio
        
        print(f"Initial noise floor: {self.current_noise_floor:.2f}")
        print(f"Initial thresholds: {self.dynamic_threshold:.2f} / {self.energy_threshold:.2f}")
        
    def _process_audio(self):
        """Main audio processing loop"""
        try:
            self._calibrate()
            
            while self.running:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    rms = audioop.rms(data, 2)
                    current_energy = self._calculate_energy(data)
                    
                    # Update histories
                    self.rms_history.append(rms)
                    self.energy_history.append(current_energy)
                    
                    # Calculate rolling averages
                    avg_rms = np.mean(self.rms_history)
                    avg_energy = np.mean(self.energy_history)
                    
                    # Dynamic threshold adjustment
                    if not self.is_recording:
                        # Adapt noise floor
                        self.current_noise_floor = (1 - self.noise_adapt_speed) * self.current_noise_floor + \
                                                  self.noise_adapt_speed * min(avg_rms, self.current_noise_floor * 1.5)
                        
                        # Adjust threshold ratio based on signal variance
                        rms_std = np.std(self.rms_history)
                        if rms_std > self.current_noise_floor * 0.9:  # High variance
                            self.threshold_ratio = max(self.threshold_ratio * self.threshold_decay, self.min_threshold_ratio)
                        else:  # Low variance
                            self.threshold_ratio = min(self.threshold_ratio * 1.05, self.max_threshold_ratio)
                        
                        # Update thresholds
                        self.dynamic_threshold = self.current_noise_floor * self.threshold_ratio
                        self.energy_threshold = avg_energy * self.threshold_ratio
                    
                    # Voice detection with adaptive criteria
                    strong_signal = rms > self.dynamic_threshold * 1.1
                    consistent_energy = current_energy > self.energy_threshold
                    voice_detected = strong_signal or (consistent_energy and rms > self.dynamic_threshold * 0.8)
                    
                    if voice_detected:
                        if not self.is_recording:
                            self.is_recording = True
                            self.frames = []
                        self.frames.append(data)
                        self.silent_chunks = 0
                    elif self.is_recording:
                        self.frames.append(data)
                        if current_energy < self.energy_threshold * 0.5:  # Lenient silence detection
                            self.silent_chunks += 1
                        else:
                            self.silent_chunks = max(0, self.silent_chunks - 1)
                        
                        # Check if we've had enough silence to end the recording
                        silence_frames = int(self.sample_rate / self.chunk_size * self.silence_duration)
                        if self.silent_chunks >= silence_frames:
                            if len(self.frames) >= self.min_speech_frames:
                                # We have a valid audio segment
                                audio_data = b''.join(self.frames)
                                # Save speaker sample BEFORE setting _last_audio_data to ensure
                                # the TTS system uses the same voice sample for this utterance
                                speaker_file = self._save_speaker_sample(audio_data)
                                # Only after saving the sample, make the audio available for transcription
                                self._last_audio_data = audio_data  # Save for get_next_audio
                            
                            # Reset for next recording
                            self.frames = []
                            self.silent_chunks = 0
                            self.is_recording = False
                            
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
                    
        except Exception as e:
            print(f"Failed to start audio processing: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                
    def _save_speaker_sample(self, audio_data):
        """Save a speaker sample of exactly 6 seconds if it meets quality criteria"""
        try:
            # Convert audio data to numpy array for analysis
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check audio quality
            duration = len(audio_array) / self.sample_rate
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            
            # Quality criteria
            target_duration = 6.0  # Exactly 6 seconds
            min_duration = 4.0     # Minimum 4 seconds (we'll pad if between 4-6 seconds)
            min_rms = 500         # Minimum volume threshold
            
            if duration >= min_duration and rms > min_rms:
                # Check if speaker.wav exists
                speaker_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'speaker.wav')
                
                # Calculate how many samples we need for exactly 6 seconds
                target_samples = int(target_duration * self.sample_rate)
                
                if duration >= target_duration:
                    # If we have more than 6 seconds, take the best 6 seconds from the middle
                    # This avoids start/end artifacts and gets the clearest speech
                    middle_point = len(audio_array) // 2
                    half_target = target_samples // 2
                    good_sample = audio_array[middle_point - half_target:middle_point + half_target]
                else:
                    # If we have between 4-6 seconds, pad with zeros to reach exactly 6 seconds
                    # First take what we have (centered on the middle for best quality)
                    middle_point = len(audio_array) // 2
                    half_available = len(audio_array) // 2
                    good_sample = audio_array[middle_point - half_available:middle_point + half_available]
                    
                    # Calculate padding needed
                    padding_needed = target_samples - len(good_sample)
                    # Pad evenly on both sides
                    pad_left = padding_needed // 2
                    pad_right = padding_needed - pad_left
                    good_sample = np.pad(good_sample, (pad_left, pad_right), 'constant')
                
                # Ensure we have exactly 6 seconds (target_samples samples)
                if len(good_sample) != target_samples:
                    good_sample = good_sample[:target_samples]
                
                with wave.open(speaker_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(good_sample.tobytes())
                
                print(f"New speaker sample saved: exactly {target_duration} seconds")
                return speaker_file
            else:
                if duration < min_duration:
                    print(f"Sample too short ({duration:.1f}s), need at least {min_duration}s")
                if rms <= min_rms:
                    print("Sample too quiet, waiting for clearer speech")
                return None
                
        except Exception as e:
            print(f"Error saving speaker sample: {e}")
            return None
            
    def get_next_audio(self):
        """Get the most recent audio data for transcription"""
        if hasattr(self, '_last_audio_data') and self._last_audio_data:
            audio_data = self._last_audio_data
            self._last_audio_data = None  # Clear after retrieving
            return audio_data
        return None