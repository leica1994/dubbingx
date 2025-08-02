# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

请使用中文回答我的问题.

## Project Overview

DubbingX is an intelligent video dubbing system with GPU acceleration that specializes in voice cloning and TTS (Text-to-Speech) generation. The system processes video files to separate media components (audio, video, background music), processes subtitles intelligently, and generates high-quality dubbed audio using Index-TTS.

## Project Structure

```
dubbingx/
├── core/                          # Core processing modules
│   ├── media_processor.py         # Media separation (Demucs, FFmpeg)
│   ├── tts_processor.py           # TTS generation via Index-TTS API
│   ├── subtitle_preprocessor.py   # Subtitle preprocessing & cleaning  
│   └── subtitle/                  # Subtitle processing subsystem
│       ├── subtitle_entry.py      # SubtitleEntry data class
│       ├── subtitle_processor.py  # Multi-format subtitle conversion
│       └── text_processor.py      # AI-powered text cleaning
├── pyproject.toml                 # uv project configuration
└── uv.lock                       # Dependency lock file
```

## Quick Start

### 一键安装与运行
```bash
# 1. 安装uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装所有依赖 (默认GPU版本)
uv sync

# 3. 验证安装
uv run python check_deps.py

# 4. 运行示例
uv run python demo.py
```

**默认配置说明:**
- ✅ **PyTorch GPU版本**: 默认从阿里云镜像安装CUDA 12.8版本
- ✅ **一键安装**: 无需额外脚本，直接 `uv sync` 完成所有依赖安装
- ✅ **自动配置**: uv会自动管理PyTorch源和版本
- ✅ **CPU备选**: 如无GPU可运行 `uv sync --extra cpu`

## Development Commands

### Package Management
- **Install all dependencies**: `uv sync` (默认安装GPU版本的PyTorch，来源：阿里云镜像 CUDA 12.8)
- **Add dependency**: `uv add <package>`
- **Remove dependency**: `uv remove <package>`
- **Update dependencies**: `uv sync --upgrade`
- **Install CPU version**: `uv sync --extra cpu` (如果无GPU环境)
- **Install dev dependencies**: `uv sync --extra dev`
- **Verify installation**: `uv run python check_deps.py`

### Running Code
- **Run Python module**: `uv run -m <module>`
- **Execute script**: `uv run python <script.py>`
- **Run with specific Python**: `uv run --python 3.11 <command>`

### Environment Management
- **Activate virtual environment**: uv automatically manages the venv
- **Install with extras**: `uv sync --extra <extra_name>`
- **Development mode**: `uv sync --dev`

## Core Architecture

### 1. Media Processing Pipeline (`media_processor.py`)

**MediaProcessor** is the entry point for video processing:
- **GPU-accelerated audio separation** using Demucs (htdemucs model)
- **Audio extraction** from video using FFmpeg
- **Silent video creation** by removing audio track
- **Reference audio generation** from subtitle timing
- **Caching system** for processed results

**Key Methods:**
- `separate_media()` - Main pipeline: video → silent video + vocal audio + background music
- `generate_reference_audio()` - Creates audio segments based on subtitle timings
- **Auto-caching** with JSON metadata for resume capability

### 2. TTS Processing (`tts_processor.py`)

**TTSProcessor** handles voice cloning via Index-TTS:
- **Gradio Client integration** for Index-TTS API communication
- **Reference-based voice cloning** using audio segments
- **Silence generation** for empty subtitle entries
- **Batch processing** with automatic error handling
- **Result caching** to avoid redundant API calls

**Key Features:**
- Connects to Index-TTS at `http://127.0.0.1:7860` by default
- Supports multiple TTS parameters (top_p, temperature, etc.)
- Handles both text and empty subtitle segments appropriately

### 3. Subtitle Processing System (`subtitle/`)

#### SubtitleEntry (`subtitle_entry.py`)
- **Dataclass** for subtitle timing and content
- Time manipulation methods (`shift_time`, `scale_time`)
- Duration calculations and validation
- Supports style/actor metadata

#### SubtitleProcessor (`subtitle_processor.py`)
- **Multi-format support**: SRT, ASS, VTT, LRC, SBV, SAMI, TTML
- **ASS Style extraction** for multi-language subtitles
- **Timeline regeneration** from TTS audio segments
- **4 regeneration strategies**:
  - `proportional` - Equal scaling
  - `cumulative` - Preserve gaps
  - `gap_preserving` - Maintain relative gaps  
  - `adaptive` - Smart strategy selection (recommended)

#### Text Processing (`text_processor.py`)
- **AI-powered text cleaning** with regex optimization
- **Bracket content removal** (TTS doesn't need annotations)
- **Unicode safety** ensuring GBK encoding compatibility
- **Smart percentage/dash handling** with context awareness
- **Language detection** (Chinese, English, Mixed, etc.)
- **Intelligent text splitting** with multiple strategies
- **Comprehensive caching system** (memory + disk)

### 4. Subtitle Preprocessing (`subtitle_preprocessor.py`)

**SubtitlePreprocessor** orchestrates the subtitle pipeline:
- **Format detection and conversion** to standardized SRT
- **ASS file formatting** with style handling
- **AI-powered text cleaning** integration
- **Validation and statistics** generation

## Key Dependencies

### Core Processing
- **torch/torchaudio** - GPU acceleration and audio processing
- **demucs** - State-of-the-art source separation
- **ffmpeg-python** - Video/audio manipulation
- **librosa/soundfile** - Audio analysis and I/O

### TTS Integration  
- **gradio-client** - Index-TTS API communication

### Audio Enhancement
- **noisereduce** - Audio noise reduction
- **audiostretchy** - Time-stretching capabilities

### Text Processing
- **scikit-learn** - ML-based text analysis
- **loguru** - Enhanced logging

## Important Patterns & Conventions

### 1. Error Handling & Logging
All processors use consistent error handling:
```python
try:
    # Processing logic
    return {'success': True, 'data': result}
except Exception as e:
    self.logger.error(f"Operation failed: {str(e)}")
    return {'success': False, 'error': str(e)}
```

### 2. Caching Strategy
- **JSON metadata files** for human-readable caching
- **File modification time validation** for cache invalidation
- **Hierarchical cache checking**: memory → disk → regenerate

### 3. GPU Utilization
```python
self.use_gpu = torch.cuda.is_available()
if self.use_gpu:
    model = model.cuda()
    # ... GPU processing
    torch.cuda.empty_cache()  # Always cleanup
```

### 4. Configuration Pattern
All processors accept optional output directories and provide detailed result dictionaries with:
- `success` boolean
- `error` message (if failed)
- Detailed metadata and statistics
- File paths and processing info

## Development Guidelines

### When Adding New Features
1. **Follow the processor pattern** - return detailed result dictionaries
2. **Implement proper caching** using JSON for metadata
3. **Add comprehensive logging** with self.logger
4. **Handle GPU memory** properly with cleanup
5. **Support batch processing** where applicable

### Text Processing Integration
Use the `IntelligentTextProcessor` for any text cleaning:
```python
from core.subtitle.text_processor import quick_clean_text, process_text

# Quick cleaning
cleaned = quick_clean_text(raw_text)

# Full processing with metadata
result = process_text(raw_text, split_strategy=SplitStrategy.ADAPTIVE)
```

### Subtitle Timeline Manipulation
For subtitle timing adjustments:
```python
from core.subtitle.subtitle_processor import regenerate_subtitles_from_audio

success = regenerate_subtitles_from_audio(
    original_srt_path="input.srt",
    audio_segments_dir="tts_output/",
    output_srt_path="new_timing.srt",
    strategy="adaptive"  # or "proportional", "cumulative", "gap_preserving"
)
```

## Performance Considerations

- **GPU Memory Management**: Always call `torch.cuda.empty_cache()` after GPU operations
- **Caching**: Leverage the built-in caching systems to avoid reprocessing
- **Batch Processing**: Use batch methods for multiple items when available
- **Text Processing**: The `IntelligentTextProcessor` has optimized regex compilation and caching

## Common Workflows

1. **Full Video Dubbing Pipeline**:
   - MediaProcessor.separate_media() → vocal audio + silent video
   - SubtitlePreprocessor.preprocess_subtitle() → clean SRT
   - MediaProcessor.generate_reference_audio() → audio segments
   - TTSProcessor.generate_tts_from_reference() → TTS audio
   - SubtitleProcessor.regenerate_subtitles_from_audio() → new timing

2. **Subtitle Processing Only**:
   - SubtitleProcessor format conversion (ASS→SRT, etc.)
   - Text cleaning with IntelligentTextProcessor
   - Timeline adjustment based on audio segments