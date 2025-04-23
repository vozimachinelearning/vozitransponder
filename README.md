# VoziTransponder

A real-time voice-to-voice translation system with voice cloning capabilities. This application captures speech in one language, translates it, and outputs the translation using a cloned voice that maintains the characteristics of a reference speaker.

## Features

- Real-time voice capture and processing
- Bidirectional translation between English and Spanish
- Voice cloning technology for natural-sounding output
- Offline-capable speech recognition using Whisper
- Local text-to-speech synthesis using XTTS
- Efficient audio processing with noise reduction and filtering

### Audio Processing
- Real-time audio capture with PyAudio
- Audio stream management and noise filtering
- Voice activity detection to prevent silent processing

### Speech Recognition
- Uses OpenAI's Whisper model locally
- Supports English and Spanish language detection
- Real-time transcription processing

### Translation
- Bidirectional translation between English and Spanish
- Uses MarianMT for local translation processing
- Queue-based translation management for smooth operation

### Text-to-Speech with Voice Cloning
- XTTS (XTreme Text-to-Speech) for high-quality voice synthesis
- Voice cloning from a reference audio sample
- Advanced audio processing features:
  - Low-pass filtering
  - Noise gating
  - Audio normalization
  - Continuous speech output management

## Technical Implementation

The system uses a multi-threaded architecture to handle:
- Audio capture
- Speech recognition
- Translation processing
- Text-to-speech generation

All processing is done locally for privacy and reduced latency. The system uses a reference audio file ('speaker.wav') for voice cloning, ensuring consistent voice output across translations.

## Requirements

- Python 3.x
- PyAudio for audio capture
- TTS library for voice synthesis
- Torch for ML models
- FFmpeg for audio processing

## Usage

1. Ensure a reference audio file ('speaker.wav') is present in the root directory
2. Run the main application:
   ```
   python main.py
   ```
3. Start speaking in either English or Spanish
4. The system will automatically:
   - Detect the language
   - Transcribe the speech
   - Translate to the other language
   - Output the translation using the cloned voice

## Note

The system currently supports bidirectional translation between English and Spanish only. Voice cloning quality depends on the reference audio sample quality and length.