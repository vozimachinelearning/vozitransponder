# Real-Time AI Translation and Transcription System

A powerful real-time translation and transcription application leveraging local AI capabilities for speech recognition, language translation, and text-to-speech synthesis.

## Overview

This application provides seamless real-time translation between languages using a combination of local AI models and cloud services. It features a user-friendly interface with support for multiple languages, different translation engines, and various text-to-speech options.

## Key Features

### Speech Recognition
- Real-time audio capture and processing
- Local Whisper model for accurate speech recognition
- Automatic language detection
- Continuous speech monitoring

### Translation Capabilities
- Dual translation systems:
  - Local Helsinki-NLP based model for offline translation
  - Google Translate API for online translation
- Support for multiple language pairs
- Real-time translation processing
- Queue-based translation management

### Text-to-Speech (TTS)
- Multiple TTS options:
  - XTTS (Local AI-powered TTS)
  - GTTS (Google Text-to-Speech)
- Language-specific voice synthesis
- Configurable TTS settings
  - voice and tone cloning

### User Interface
- Clean and intuitive interface
- Dark/Light theme support
- Real-time transcription display
- Language pair selection
- Easy TTS mode switching

## Local AI Components
- OpenAI whisper models
- Helsinki-NLP based model for offline translation
- coqui-tts XTTS-V2 for multilingual offline speech synthesis

### Whisper Speech Recognition
- Uses OpenAI's Whisper model locally
- Supports multiple model sizes (base model by default)
- High-accuracy transcription
- Built-in language detection
- Optimized for real-time processing

### Helsinki-NLP Semantic Translation
- Local neural machine translation
- Offline translation capabilities
- Supports multiple language pairs
- Integrated with XTTS for complete offline operation

### XTTS (Local Text-to-Speech)
- AI-powered voice synthesis
- High-quality natural speech generation
- Multi-language support
- Configurable voice parameters
- Voice cloning support ( per message to keep each message's tone and meaning)
- Local processing for privacy

## System Architecture
INTERPRETER/
│
├── main.py                     # Application entry point
│
├── src/
│   ├── UI.py                   # User interface components
│   ├── audio.py                # Audio capture and management
│   ├── transcriber.py          # Whisper model integration
│   ├── translator.py           # Translation management
│   ├── newtranslator.py        # Enhanced translation capabilities
│   ├── tts.py                  # Text-to-speech processing
│   ├── gtts_processor.py       # Google TTS integration
│   ├── virtual_audio_router.py # Audio routing utilities
│   │
│   └── ffmpeg/                 # Audio processing libraries
│       ├── bin/                # Executable binaries
│       └── include/            # Header files
│
├── models/                     # AI model storage (not shown in snippets)
│   ├── whisper/                # Whisper speech recognition models
│   ├── translation/            # Helsinki-NLP translation models
│   └── tts/                    # XTTS voice models
│
└── README.md                   # Project documentation

### Core Components
1. **Audio Processing**
   - Real-time async threaded audio capture
   - Audio stream management
   - Device input/output handling

2. **Transcription Engine**
   - Whisper model integration
   - Language detection
   - Text processing

3. **Translation System**
   - Dual translation engines (Helsinki-NLP and Google)
   - Translation queue management
   - Language pair handling (BIDIRECCTIONAL TRANSLATION)

4. **TTS Processing**
   - Multiple TTS backends
   - Text queue management
   - Audio output handling (system doesn't hear itself when it speaks)

### Threading and Performance
- Multi-threaded architecture
- Separate threads for:
  - Audio capture
  - Transcription processing (threadpool)
  - Translation processing (threadpool)
  - TTS generation
- Queue-based task management
- Efficient resource utilization

## Usage

### Language Selection
1. Choose two languages from the dropdown menus 
2. System automatically detects spoken language
3. Translations are processed in real-time

### TTS Options
- **Off**: Text-only mode
- **XTTS**: Local AI-powered voice synthesis
- **GTTS**: Google's cloud-based TTS

### Interface Controls
- Toggle Theme: Switch between light and dark modes
- Clear: Reset all text displays
- Search: Find specific translations

## Technical Details

### Dependencies
- PyQt5 for UI
- Whisper for speech recognition
- Torch.MarianNMT for local translation
- XTTS/GTTS for speech synthesis
- Deep Translator for Google translation

### Performance Considerations
- Efficient memory management
- Optimized audio processing
- Queue-based task handling
- Thread-safe operations
- no gpu needed (with a dedicated gpu wait times are reduced, but the system does not need it by default)

## Future Enhancements
- Additional language model support
- clustering features for faster but local and private Inference
- Enhanced offline capabilities
- Improved voice customization and stability (the system creates corrupt voices if the message is too short)
- Extended API integration options (paid or free)

## Contributing
Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.