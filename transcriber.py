import whisper
import threading
import queue
import numpy as np

class Transcriber:
    def __init__(self, model_size="base"):
        self.model = None
        self.model_size = model_size
        self.task_queue = queue.Queue()
        self.running = True
        self.last_detected = None
        self.detection_threshold = 0.4  # Lower threshold for better detection
        self.buffer_size = 3
        self.transcription_ready = None  # Callback function
        self.initialize_model()
        self.worker_thread = threading.Thread(target=self._process_queue)  # Remove daemon=True
        self.worker_thread.start()

    def initialize_model(self):
        try:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    def transcribe(self, audio_data):
        if self.running and audio_data:
            print("Received audio data for transcription")
            # Clean up queue if it's getting too full
            while self.task_queue.qsize() > 10:
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            self.task_queue.put(audio_data)

    def _process_queue(self):
        while self.running:
            try:
                audio_data = self.task_queue.get(timeout=0.001)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float32 = audio_array.astype(np.float32) / 32768.0

                # Improved transcription settings
                result = self.model.transcribe(
                    audio_float32,
                    fp16=False,
                    language=None,  # Set to None to let Whisper auto-detect language
                    task="transcribe",
                    condition_on_previous_text=False,
                    initial_prompt=None,  # Removed prompt that was causing issues
                    temperature=0.0,
                    best_of=None
                )
                
                current_lang = result.get('language')
                confidence = result.get('language_probability', 0)
                text = result.get('text', '').strip()
                
                if text:  # Only process if there's actual text
                    detected_language = current_lang
                    if confidence > self.detection_threshold:
                        self.last_detected = current_lang
                    elif self.last_detected:
                        detected_language = self.last_detected
                    
                    transcription_result = {
                        'text': text,
                        'language': detected_language or 'unknown',
                        'confidence': confidence
                    }
                    
                    print(f"Transcription result: {transcription_result}")
                    
                    # Call the callback with the transcription result
                    if self.transcription_ready:
                        self.transcription_ready(transcription_result)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")

    def reset_language_detection(self):
        """Reset language detection state"""
        self.last_detected = None

    def stop(self):
        """Stop transcription processing"""
        self.running = False
        # Clear the queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

