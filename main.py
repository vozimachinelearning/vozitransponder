# Remove PyQt5 imports
import os
import sys
import threading  # Add regular threading
from src.transcriber import Transcriber
from src.translator import MarianTranslator, translate_text
import queue
import time
from src.tts import TTSProcessor  # Add this import at the top
# Import the AudioCaptureThread from hereaudio.py
from hereaudio import AudioCaptureThread

# Set the PATH environment variable to include ffmpeg binary directory
os.environ["PATH"] += os.pathsep + r"src\ffmpeg\bin"

# AudioCaptureThread class is now imported from hereaudio.py

class TranscriptionApp:
    def __init__(self):
        self.translator = MarianTranslator()
        self.translation_running = True
        
        # Initialize transcriber with callback
        self.transcriber = Transcriber(model_size="base")
        self.transcriber.transcription_ready = self.handle_transcription
        
        # Initialize audio capture using the imported AudioCaptureThread
        self.audio_thread = AudioCaptureThread(callback=self.audio_callback)
        self.audio_thread.start()

        # Translation queue setup
        self.translation_queue = queue.Queue()
        
        # Initialize TTS processor before starting translation worker
        self.tts_processor = TTSProcessor()
        
        # Make translation worker non-daemon
        self.translation_worker = threading.Thread(target=self._process_translations)
        self.translation_worker.start()
        
        print("All systems initialized and running")

    def audio_callback(self, audio_data):
        if audio_data:
            # Pass audio data directly to transcriber for processing
            self.transcriber.transcribe(audio_data)
            print("Audio data sent for transcription")

    def _process_translations(self):
        print("Translation worker started")
        while self.translation_running:
            try:
                task = self.translation_queue.get(timeout=0.1)
                if task:
                    text, source_lang, target_lang = task
                    print(f"\nStarting translation task:")
                    print(f"Source language: {source_lang}")
                    print(f"Target language: {target_lang}")
                    print(f"Text to translate: {text}")
                    
                    translated = translate_text(text, target_lang, source_lang)
                    if translated:
                        self.handle_translation_result(text, translated, source_lang, target_lang)
                    else:
                        print(f"Translation failed: No result returned")
                self.translation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Translation error: {str(e)}")
                if task:
                    print(f"Failed text: {text}")

    def handle_translation_result(self, original_text, translated_text, source_lang, target_lang):
        print("\n=== Translation Result ===")
        print(f"[Original ({source_lang})]: {original_text}")
        print(f"[Translation ({target_lang})]: {translated_text}")
        print("=======================\n")
        
        # Send translated text to TTS
        self.tts_processor.speak(translated_text, target_lang)

    def handle_transcription(self, result):
        if not result or not result.get('text'):
            return
            
        detected_lang = result.get('language', '').lower()
        text = result.get('text', '').strip()
        
        if not text or not detected_lang:
            return
            
        # Only translate between English and Spanish
        if detected_lang not in ['en', 'es']:
            print(f"Unsupported language detected: {detected_lang}")
            return
            
        # Determine target language based on detected language
        target_lang = 'en' if detected_lang == 'es' else 'es'
        
        # Add translation task to queue
        print(f"\nTranscription detected ({detected_lang}): {text}")
        print(f"Queueing translation from {detected_lang} to {target_lang}...")
        self.translation_queue.put((text, detected_lang, target_lang))

def main():
    app = TranscriptionApp()
    print("Starting application... Press Ctrl+C to exit")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        app.audio_thread.stop()
        app.translation_running = False
        app.tts_processor.stop()
        # Wait for threads to finish
        app.translation_worker.join(timeout=2)
        app.audio_thread.join(timeout=2)
        print("Application shutdown complete")
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        # Keep running even after critical errors
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
