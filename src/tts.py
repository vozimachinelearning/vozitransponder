from TTS.api import TTS
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import os
import wave
import re
import collections
import concurrent.futures
from functools import lru_cache

class TTSProcessor:
    def __init__(self):
        self.running = True
        self.sample_rate = 24000
        self.is_speaking = False
        self.tts_queue = queue.PriorityQueue(maxsize=20)  # Use priority queue
        self.next_sequence = 0  # Add sequence counter
        
        # Initialize sequence tracking and locks
        self.sequence_buffer = {}
        self.current_sequence = 0
        self.playback_lock = threading.Lock()
        self.played_sequences = set()
        self.deduplication_lock = threading.Lock()
        
        # Retry mechanism configuration
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds
        
        # Initialize thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.setup_audio_stream()
        
        # Initialize TTS model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TTS model on {self.device}...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1.1").to(self.device)
        print("TTS model loaded successfully")
        
        # Apply model pruning/optimization if on CUDA
        if self.device == "cuda":
            self._prune_model()
        
        # Path to speaker sample
        self.speaker_sample_path = os.path.join(os.getcwd(), 'speaker.wav')
        self.fallback_sample = None  # Store a working sample as fallback
        self.setup_audio_stream()
        self.validate_speaker_sample()  # Add initial validation
        
        # Cache for speaker embeddings
        self.speaker_embedding_cache = {}
        
        # Start worker thread as non-daemon
        self.worker_thread = threading.Thread(target=self._process_queue)  # Remove daemon=True
        self.worker_thread.start()
        
        # Add queue monitoring to __init__
        self.queue_monitor = threading.Thread(target=self._monitor_queue)
        self.queue_monitor.start()

    def setup_audio_stream(self):
        """Setup audio stream with error handling"""
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()

            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            print("Audio output configured for TTS")
        except Exception as e:
            print(f"Audio setup error: {e}")
            self.stream = None

    def validate_speaker_sample(self):
        """Validate speaker sample file"""
        try:
            if os.path.exists(self.speaker_sample_path):
                with wave.open(self.speaker_sample_path, 'rb') as wf:
                    if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                        print("Speaker sample validated successfully")
                        return True
            print("Speaker sample not found or invalid")
            return False
        except Exception as e:
            print(f"Speaker sample validation error: {e}")
            return False

    def _process_queue(self):
        while self.running:
            try:
                seq_num, text, language = self.tts_queue.get(timeout=1.0)  # Get sequence number
                if not self.running:
                    break
                if text:
                    self._synthesize_and_play(seq_num, text, language)  # Pass sequence number
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS processing error: {e}")
                if not self.running:
                    break
                time.sleep(0.1)

    def _synthesize_and_play(self, seq_num, text, language):
        try:
            print(f"Synthesizing speech chunk [{seq_num}]: {text[:50]}...")
            self.is_speaking = True
            
            # We're now using the sequence tracking variables initialized in __init__
            # No need to reinitialize them here, which would reset the state
            
            # Validate speaker sample before synthesis
            if not self.validate_speaker_sample():
                print("No valid speaker sample found, synthesis may not work")
                self.is_speaking = False
                return
            
            # Generation thread - this is the only place we'll call the TTS engine
            def generate_audio():
                try:
                    # Check if we have a cached speaker embedding
                    speaker_embedding = None
                    if self.speaker_sample_path in self.speaker_embedding_cache:
                        speaker_embedding = self.speaker_embedding_cache[self.speaker_sample_path]
                    
                    # Generate speech with optimized settings
                    with torch.no_grad():
                        if speaker_embedding is not None:
                            # Use cached speaker embedding if available
                            wav = self.tts.tts(
                                text=text,
                                language=language,
                                speaker_embedding=speaker_embedding
                            )
                        else:
                            # Generate and cache speaker embedding
                            wav = self.tts.tts(
                                text=text,
                                language=language,
                                speaker_wav=self.speaker_sample_path
                            )
                            # Try to cache the speaker embedding for future use
                            try:
                                if (hasattr(self.tts, 'synthesizer') and 
                                    hasattr(self.tts.synthesizer, 'speaker_manager') and 
                                    self.tts.synthesizer.speaker_manager is not None and
                                    hasattr(self.tts.synthesizer.speaker_manager, 'compute_embedding')):
                                    embedding = self.tts.synthesizer.speaker_manager.compute_embedding(self.speaker_sample_path)
                                    self.speaker_embedding_cache[self.speaker_sample_path] = embedding
                            except Exception as e:
                                print(f"Speaker embedding caching error (non-critical): {e}")
                                # If we encounter an error, we'll just continue without caching
                    
                    if wav is not None and len(wav) > 0:
                        # Convert to float32 and normalize
                        wav = np.array(wav, dtype=np.float32)
                        if np.max(np.abs(wav)) > 0:
                            wav = wav / np.max(np.abs(wav))
                        
                        # Process audio in parallel using thread pool
                        def process_audio(audio_data):
                            # Apply audio enhancements using static methods
                            enhanced = TTSProcessor._apply_lowpass_filter(
                                tuple(audio_data.flatten()), cutoff=10000, sample_rate=self.sample_rate
                            )
                            enhanced = TTSProcessor._apply_noise_gate(enhanced, threshold=0.000001)
                            
                            # Reshape back to original shape if needed
                            if isinstance(enhanced, tuple):
                                enhanced = np.array(enhanced)
                            
                            return enhanced
                        
                        # Submit audio processing to thread pool
                        future = self.thread_pool.submit(process_audio, wav)
                        wav = future.result()
                        
                        # Validate audio RMS level
                        if TTSProcessor._calculate_rms(wav) < 0.01:
                            print("Warning: Audio signal too weak after processing")
                        
                        with self.playback_lock:
                            self.sequence_buffer[seq_num] = wav
                            
                            # Play in-order sequences
                            while self.current_sequence in self.sequence_buffer:
                                with self.deduplication_lock:
                                    if self.current_sequence in self.played_sequences:
                                        del self.sequence_buffer[self.current_sequence]
                                        continue
                                    audio_chunk = self.sequence_buffer.pop(self.current_sequence)
                                    self.played_sequences.add(self.current_sequence)
                                    self._play_audio_chunk(audio_chunk)
                                self.current_sequence += 1
                    else:
                        print("Synthesis produced no audio data")
                    
                    # Mark speech as complete
                    self.is_speaking = False
                    print("Speech synthesis completed")
                except Exception as e:
                    print(f"Audio generation error: {e}")
                    self.is_speaking = False
            
            # Start generation thread
            gen_thread = threading.Thread(target=generate_audio)
            gen_thread.start()
            
            # Wait for a short time to allow the thread to start processing
            time.sleep(0.1)
            
            # We don't set is_speaking to False here because the audio generation
            # and playback is happening asynchronously in the thread

        except Exception as e:
            print(f"Speech synthesis error: {e}")
            self.is_speaking = False

    @staticmethod
    def split_text_into_chunks(text, max_length=220):
        """Split text into chunks respecting sentence boundaries
        
        Static method for better performance and reusability
        """
        chunks = []
        current_chunk = ''
        
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += ' ' + sentence
            else:
                if current_chunk:
                    # Remove trailing period to prevent TTS issues
                    processed_chunk = current_chunk.strip()
                    if processed_chunk.endswith('.'):
                        processed_chunk = processed_chunk[:-1] + ' '
                    chunks.append(processed_chunk)
                current_chunk = sentence
                
                # Split very long sentences
                while len(current_chunk) > max_length:
                    split_point = max_length
                    while split_point > 0 and current_chunk[split_point] not in ' ,.;!?':
                        split_point -= 1
                    processed_chunk = current_chunk[:split_point].strip()
                    if processed_chunk.endswith('.'):
                        processed_chunk = processed_chunk[:-1] + ' '
                    chunks.append(processed_chunk)
                    current_chunk = current_chunk[split_point:]
        
        if current_chunk:
            # Remove trailing period from the last chunk as well
            processed_chunk = current_chunk.strip()
            if processed_chunk.endswith('.'):
                processed_chunk = processed_chunk[:-1] + ' '
            chunks.append(processed_chunk)
        return chunks

    def speak(self, text, language):
        """Add text chunks to TTS queue with parallel processing"""
        if not text or not language:
            return
            
        # Process text chunking in parallel
        def process_and_queue_chunks():
            try:
                # Use static method for text chunking
                chunks = TTSProcessor.split_text_into_chunks(text.strip())
                
                # Add chunks to queue with sequence numbers
                for chunk in chunks:
                    try:
                        with self.playback_lock:  # Ensure thread-safe sequence number assignment
                            seq_num = self.next_sequence
                            self.next_sequence += 1
                        
                        self.tts_queue.put_nowait((seq_num, chunk, language))
                    except queue.Full:
                        print("TTS queue full - skipping chunk to prevent overload")
                        break
            except Exception as e:
                print(f"Chunk processing error: {e}")
        
        # Submit chunking task to thread pool for parallel processing
        self.thread_pool.submit(process_and_queue_chunks)

    def _prune_model(self):
        """Apply pruning and optimization techniques to the TTS model"""
        try:
            print("Applying model pruning and optimization...")
            # Apply torch JIT compilation if available
            if hasattr(torch, 'jit') and hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                try:
                    # Try to trace and optimize the model
                    print("Applying JIT optimization...")
                    # Note: Full JIT compilation might not be compatible with all TTS models
                    # So we're just applying basic optimizations
                    torch._C._jit_set_profiling_executor(False)
                    torch._C._jit_set_profiling_mode(False)
                    print("Applied JIT optimization settings")
                except Exception as e:
                    print(f"JIT optimization error (non-critical): {e}")
            
            # Apply half-precision if available
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Convert to half precision (FP16) for faster inference
                    if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                        print("Converting model to half precision...")
                        self.tts.synthesizer.model = self.tts.synthesizer.model.half()
                        print("Model converted to half precision")
                except Exception as e:
                    print(f"Half precision conversion error (non-critical): {e}")
            
            print("Model optimization complete")
            return True
        except Exception as e:
            print(f"Model pruning error (non-critical): {e}")
            return False
    
    def stop(self):
        """Stop TTS processing"""
        self.running = False
        self.is_speaking = False
        # Clear the queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        # Wait for worker thread to finish
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=3.0)
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

    @staticmethod
    @lru_cache(maxsize=128)
    def _apply_lowpass_filter(data, cutoff=6000, sample_rate=24000):
        """Apply low-pass filter using FFT
        
        Static method with caching for better performance
        """
        # Check for empty data to prevent index errors
        if not data or len(data) == 0:
            return np.array([], dtype=np.float32)
            
        # Convert data to tuple for caching (numpy arrays aren't hashable)
        if isinstance(data, np.ndarray):
            data_tuple = tuple(data.flatten())
        else:
            data_tuple = tuple(data)
            
        # Process the data with error handling
        try:
            fft_data = np.fft.rfft(data)
            frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
            # Ensure frequencies array is not empty before indexing
            if len(frequencies) > 0:
                fft_data[frequencies > cutoff] = 0
            return np.fft.irfft(fft_data).astype(np.float32)
        except Exception as e:
            print(f"FFT processing error (non-critical): {e}")
            # Return original data if FFT processing fails
            return np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)

    @staticmethod
    def _apply_noise_gate(data, threshold=0.1):
        """Apply noise gate to silent portions
        
        Static method for better performance
        """
        smoothed = np.convolve(np.abs(data), np.ones(100)/100, mode='same')
        return data * (smoothed > threshold).astype(np.float32)

    @staticmethod
    def _calculate_rms(data):
        """Calculate RMS level of audio signal
        
        Static method for better performance
        """
        return np.sqrt(np.mean(data**2))

    def _monitor_queue(self):
        """Monitor queue status and ensure ordered processing"""
        while self.running:
            try:
                with self.playback_lock:
                    if hasattr(self, 'sequence_buffer') and hasattr(self, 'current_sequence'):
                        current_seq = self.current_sequence
                        pending = [k for k in self.sequence_buffer.keys() if k < current_seq]
                        
                        # Cleanup old sequence numbers
                        for seq in pending:
                            del self.sequence_buffer[seq]
                        
                        # Reduced logging frequency
                        if time.time() % 5 < 0.1:  # Log every 5 seconds
                            print(f"Current sequence: {current_seq}")
                            print(f"Pending buffer entries: {list(self.sequence_buffer.keys())[:5]}")
                
                time.sleep(1)
            except Exception as e:
                print(f"Queue monitor error: {e}")
                time.sleep(0.5)
    def get_queue_state(self):
        """Return current queue items in sequence order"""
        with self.playback_lock:
            items = list(self.tts_queue.queue)
            return sorted(items, key=lambda x: x[0])

    def _play_audio_chunk(self, audio_chunk):
        """Play audio chunk with error handling"""
        try:
            if not self.stream or not self.stream.is_active():
                self.setup_audio_stream()
            
            if self.stream:
                # Only start the stream if it's not already active
                if not self.stream.is_active():
                    self.stream.start_stream()
                self.stream.write(audio_chunk.tobytes())
                # Don't stop the stream between chunks to maintain continuous playback
                # self.stream.stop_stream()  # Commented out to keep stream active
        except Exception as e:
            print(f"Audio playback error: {e}")
            self.setup_audio_stream()

    @property
    def current_sequence_number(self):
        """Get current processing sequence number"""
        with self.playback_lock:
            return self.current_sequence
