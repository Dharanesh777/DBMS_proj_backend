# DBMS Project Backend - Raspberry Pi Edition

This branch (`raspberry-pi`) is a lightweight, modular adaptation of the DBMS backend tailored specifically for resource-constrained hardware (Raspberry Pi 4 / 5).

## Current Modules

### Module 1: Audio Transcription (`faster-whisper`)
- **Engine**: `faster-whisper` (CTranslate2 framework with `int8` CPU quantization).
- **Model**: `tiny.en` (English-only, ~75MB memory footprint).
- **Service Location**: `app/services/voice_app/transcription_service.py`

## Quick Start on Raspberry Pi

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Audio Transcription:**
   ```bash
   python test_rpi_transcribe.py <path_to_audio.wav>
   ```

## Directory Structure
```
.
├── app/
│   └── services/
│       └── voice_app/
│           ├── recorder_util.py
│           └── transcription_service.py
├── test_rpi_transcribe.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```
