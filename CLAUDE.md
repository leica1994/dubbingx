# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

请使用中文回答我的问题.

## Project Overview

DubbingX is an intelligent video dubbing system with GPU acceleration and voice cloning capabilities. The system processes video files to separate audio components, generates reference audio segments from subtitles, performs TTS (Text-to-Speech) generation using voice cloning, and handles comprehensive subtitle preprocessing.

## Development Environment Setup

### 依赖管理
项目使用 **uv** 作为Python包管理器，配置了PyTorch CUDA 12.8阿里云镜像源：

```bash
# 一键安装所有依赖（包括PyTorch CUDA版本）
uv sync

# 安装开发依赖
uv sync --extra dev

# 添加新依赖
uv add package_name
```

### PyTorch配置
项目已配置PyTorch CUDA 12.8版本，对应你之前使用的pip命令：
```bash
# 之前的pip命令
pip install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu128

# 现在只需要
uv sync  # 自动安装CUDA版本的PyTorch
```

### Development Tools
```bash
# Code formatting
uv run black core/
uv run isort core/

# Linting
uv run flake8 core/

# Type checking
uv run mypy core/

# Testing
uv run pytest
uv run pytest --cov=core  # with coverage
uv run pytest -m "not slow"  # skip slow tests
```

## Core Architecture

### Main Components

1. **MediaProcessor** (`core/media_processor.py`)
   - Primary entry point for media processing pipeline
   - Handles video/audio separation using Demucs AI model
   - Generates reference audio segments from subtitles
   - Supports GPU acceleration with automatic fallback to CPU
   - Methods: `separate_media()`, `generate_reference_audio()`

2. **TTSProcessor** (`core/tts_processor.py`)
   - Integrates with Index-TTS via Gradio client API
   - Handles voice cloning and TTS generation
   - Manages temporary file cleanup and caching
   - Methods: `generate_tts_from_reference()`, `_call_gradio_tts()`

3. **SubtitlePreprocessor** (`core/subtitle_preprocessor.py`)
   - Comprehensive subtitle format conversion and cleaning
   - Integrates with intelligent text processing for TTS optimization
   - Supports ASS, SRT, VTT and other subtitle formats
   - Methods: `preprocess_subtitle()`, `_clean_subtitle_entries_with_ai()`

### Subtitle Processing Components

4. **SubtitleEntry** (`core/subtitle/subtitle_entry.py`)
   - Data class for individual subtitle entries
   - Provides time manipulation utilities
   - Methods: `duration_seconds()`, `shift_time()`, `scale_time()`

5. **SubtitleProcessor** (`core/subtitle/subtitle_processor.py`)
   - Extensive subtitle format conversion system
   - ASS file formatting and style extraction
   - Timeline regeneration from audio segments
   - Methods: `convert_format()`, `extract_ass_style_to_srt()`, `regenerate_subtitles_from_audio()`

6. **IntelligentTextProcessor** (`core/subtitle/text_processor.py`)
   - Advanced text preprocessing for TTS optimization
   - Unicode character cleaning, bracket removal
   - Smart percentage/number handling, language detection
   - Caching system for performance optimization
   - Methods: `process()`, `quick_clean_text()`, `batch_process_texts()`

## Data Flow Architecture

```
Video Input → MediaProcessor → Audio Separation (Demucs)
     ↓
Subtitle Input → SubtitlePreprocessor → Text Cleaning → Reference Audio Generation
     ↓
Reference Audio + Processed Text → TTSProcessor → Voice Cloned Audio
     ↓
Final Assembly → Synchronized Subtitle + Audio Output
```

## Key Features

### Audio Processing
- **Demucs Integration**: AI-powered source separation (vocals, background music)
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback  
- **Reference Audio Generation**: Creates training samples from subtitle timing
- **Audio Quality Analysis**: Dynamic range, silence detection, volume normalization

### Text Processing Intelligence
- **Unicode Safety**: Handles problematic characters that could cause encoding issues
- **Content Filtering**: Removes brackets, annotations, and TTS-unsuitable content
- **Smart Number Conversion**: "20%" → "百分之二十", ranges like "1-3" → "一至三"
- **Language Detection**: Automatic Chinese/English/Mixed language identification

### TTS Integration
- **Voice Cloning**: Uses Index-TTS for reference-based voice synthesis
- **Gradio Client**: RESTful API integration with configurable parameters
- **Silence Generation**: Handles empty subtitles with appropriate duration matching
- **Batch Processing**: Efficient processing of multiple audio segments

### Subtitle Format Support
- **Multi-format**: ASS, SRT, VTT, LRC, SBV and more
- **ASS Processing**: Style extraction, timeline synchronization, formatting cleanup
- **Timeline Regeneration**: Rebuilds subtitles based on actual TTS audio durations
- **Caching System**: Intelligent result caching with validation

## Important Configuration

### Dependencies
- **PyTorch**: CUDA 12.8 support via Aliyun mirror for GPU acceleration
- **Demucs**: AI audio source separation
- **Gradio Client**: TTS API integration  
- **FFmpeg**: Audio/video processing backend
- **Librosa/SoundFile**: Audio analysis and I/O

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support recommended for Demucs
- **Memory**: Sufficient RAM for audio processing (4GB+ recommended)
- **Storage**: Temporary space for audio segments and caches

## Common Development Patterns

### Error Handling
All major components use comprehensive exception handling with detailed logging:
```python
try:
    result = processor.process_media(input_path)
    if not result['success']:
        logger.error(f"Processing failed: {result['error']}")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
```

### Resource Management
Components implement proper cleanup for GPU memory and temporary files:
```python  
def __del__(self):
    self.clear_cache()  # Clean up resources
```

### Progress Tracking
Long-running operations support progress callbacks:
```python
def process_with_progress(callback=None):
    for i, item in enumerate(items):
        # Process item
        if callback:
            callback(i + 1, total)
```

## Testing Strategy

- **Unit Tests**: Component-level testing with pytest
- **GPU Tests**: Marked with `@pytest.mark.gpu` for selective execution
- **Audio Tests**: Marked with `@pytest.mark.audio` requiring sample files
- **Slow Tests**: Marked with `@pytest.mark.slow` for CI optimization

## Performance Considerations

- **Caching**: Extensive use of result caching to avoid reprocessing
- **GPU Memory**: Automatic cleanup and memory management
- **Batch Processing**: Optimized for handling multiple files efficiently
- **Temporary Files**: Systematic cleanup of intermediate files