# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

请使用中文回答我问题.

## Project Overview

DubbingX is an intelligent video dubbing system that provides GPU-accelerated voice cloning capabilities. The system processes video input by extracting vocals, preprocessing subtitles, and generating TTS audio using voice cloning technology.

## Core Architecture

The system consists of five main components:

1. **MediaProcessor** (`core/media_processor.py`): Handles audio/video separation using Demucs AI model
2. **TTSProcessor** (`core/tts_processor.py`): Implements voice cloning via Gradio client API
3. **SubtitlePreprocessor** (`core/subtitle_preprocessor.py`): Processes and normalizes subtitle text
4. **TextProcessor** (`core/subtitle/text_processor.py`): Advanced text preprocessing for TTS
5. **AudioAlignProcessor** (`core/audio_align_processor.py`): Handles three-track audio alignment with subtitles

## Development Commands

### Environment Setup
```bash
# Install dependencies with CUDA 12.8 support
uv sync

# Install development dependencies
uv sync --extra dev
```

### Code Quality
```bash
# Format code
uv run black core/
uv run isort core/

# Lint code
uv run flake8 core/

# Run tests
uv run pytest

# Skip slow tests
uv run pytest -m "not slow"
```

### GPU Support
The project automatically detects CUDA availability. Ensure PyTorch packages are installed with CUDA 12.8 support via the custom index in `pyproject.toml`.

## Key Features

### Audio Processing Pipeline
- Vocal separation using Demucs (Demucs v4 model)
- Automatic GPU acceleration when available
- Background music extraction
- Silent video generation

### Voice Cloning
- Index-TTS integration via Gradio client
- Reference audio-based voice cloning
- Automatic caching for repeated requests
- Support for multiple reference audio segments

### Text Processing
- Intelligent text preprocessing for TTS
- Multi-language support (Chinese, English, Japanese, Korean)
- Bracket content removal (parentheses, brackets, etc.)
- Unicode character normalization
- Smart text segmentation strategies

### Audio Alignment
- Three-track audio alignment (TTS, silence, background)
- Automatic silence generation for subtitle gaps
- SRT subtitle parsing and time alignment
- Audio segment concatenation with precise timing
- Support for custom sample rates and audio formats

## Configuration

### Dependencies
- PyTorch ecosystem with CUDA 12.8
- Demucs for audio separation
- Librosa and SoundFile for audio processing
- Gradio client for TTS API
- FFmpeg for video processing

### Output Structure
```
output/
├── reference_audio/          # Reference audio segments
├── tts_output/              # Generated TTS audio
├── aligned_audio/           # Final aligned audio files
└── temp/                    # Temporary files
```

## Common Development Patterns

### Error Handling
- All processors return structured result dictionaries with `success` boolean
- Comprehensive logging at different levels
- Graceful fallback to CPU when GPU unavailable

### Caching Strategy
- Reference audio results cached to avoid reprocessing
- TTS results cached by input hash
- Temporary file cleanup after processing

### File Path Handling
- Uses `pathlib.Path` for all file operations
- Automatic directory creation when needed
- Windows-compatible path handling

## Important Notes

- The system requires FFmpeg to be installed and available in PATH
- GPU acceleration requires CUDA 12.8 compatible NVIDIA drivers
- TTS functionality depends on external Gradio API server
- All text processing is optimized for Chinese and English content
- Unicode characters are normalized to prevent encoding issues