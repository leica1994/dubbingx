# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

请使用中文回答我问题.

## Project Overview

DubbingX is an intelligent video dubbing system that provides GPU-accelerated voice cloning capabilities. The system processes video input by extracting vocals, preprocessing subtitles, and generating TTS audio using voice cloning technology.

## Core Architecture

The system uses a modern pipeline-based architecture with 8 sequential processing steps:

### Pipeline Architecture
The system is built around a **StreamlinePipeline** (`core/dubbing_pipeline.py`) that orchestrates an 8-step processing workflow:

1. **Step 0**: Subtitle Preprocessing (`PreprocessSubtitleProcessor`)
2. **Step 1**: Media Separation (`SeparateMediaProcessor`) 
3. **Step 2**: Reference Audio Generation (`GenerateReferenceAudioProcessor`)
4. **Step 3**: TTS Generation (`GenerateTTSProcessor`)
5. **Step 4**: Audio Alignment (`AlignAudioProcessor`)
6. **Step 5**: Aligned Subtitle Generation (`GenerateAlignedSrtProcessor`)
7. **Step 6**: Video Speed Processing (`ProcessVideoSpeedProcessor`)
8. **Step 7**: Audio-Video Merging (`MergeAudioVideoProcessor`)

### Task Management System
The system uses a sophisticated task scheduling architecture:

- **TaskScheduler** (`core/pipeline/task_scheduler.py`): Manages parallel execution with per-step thread pools
- **QueueManager** (`core/pipeline/task_queue.py`): Handles task queuing and distribution
- **StepProcessor** (`core/pipeline/step_processor.py`): Base class for all processing steps
- **Task** and **TaskListener** system for monitoring and status updates

### GUI Integration
The GUI (`main.py`) provides:
- **DubbingGUI**: Main application window with PySide6
- **GUIStreamlinePipeline**: GUI-integrated pipeline with real-time status signals
- **VideoSubtitleMatcher**: Automatic video-subtitle file matching
- Real-time status tracking with 8-step progress visualization

## Development Commands

### Environment Setup
```bash
# Install dependencies with CUDA 12.8 support  
uv sync

# Install development dependencies
uv sync --extra dev

# Run the GUI application
uv run python main.py
```

### Code Quality
```bash
# Format code
uv run black core/ main.py
uv run isort core/ main.py

# Lint code
uv run flake8 core/ main.py

# Type checking
uv run mypy core/

# Run tests
uv run pytest

# Skip slow tests
uv run pytest -m "not slow"

# Run with coverage
uv run pytest --cov=core
```

### Development Workflow

#### Adding New Processing Steps
1. Create processor class inheriting from `StepProcessor` in `core/pipeline/processors/`
2. Register in `TaskScheduler.STEP_DEFINITIONS` 
3. Update GUI status tracking in `main.py` (add step column)
4. Update pipeline cache handling

#### Modifying GUI Components
- Main window: `DubbingGUI` class in `main.py`
- Status tracking: `update_task_step_status()` methods
- File matching: `VideoSubtitleMatcher` class
- Threading: `StreamlineBatchDubbingWorkerThread` for background processing

### GPU Support
The project automatically detects CUDA availability. PyTorch packages are installed with CUDA 12.8 support via the custom index in `pyproject.toml`.

### Thread Pool Configuration
The TaskScheduler uses per-step thread pools with these defaults:
- Step 0 (Subtitle): 8 threads
- Steps 1,3 (GPU tasks): 2 threads  
- Steps 2,4,5,6,7 (I/O tasks): 4 threads

## Key Features

### Pipeline Processing
- 8-step sequential workflow with parallel task execution
- Per-step thread pools for optimal resource utilization  
- Comprehensive caching system with resume-from-cache capability
- Real-time status updates via Qt signals
- Thread-safe status notification management

### Error Handling & Recovery
- Structured result dictionaries with `success` boolean
- Graceful error handling with task retry capability
- Failed step detection with subsequent step reset
- Comprehensive logging at DEBUG/INFO/ERROR levels

## Configuration

### Dependencies
- PyTorch ecosystem with CUDA 12.8 (via custom index)
- Demucs for audio separation
- Librosa and SoundFile for audio processing
- Gradio client for TTS API integration
- FFmpeg for video processing  
- PySide6 for GUI framework
- pathlib for cross-platform file handling

### Project Structure
```
dubbingx/
├── core/                          # Core business logic
│   ├── pipeline/                  # Processing pipeline system
│   │   ├── processors/            # Step-specific processors (8 steps)
│   │   ├── task_scheduler.py      # Main task coordination
│   │   ├── task_queue.py          # Queue management
│   │   ├── task.py               # Task definitions
│   │   └── step_processor.py     # Base processor class
│   ├── subtitle/                 # Subtitle processing
│   ├── cache/                    # Caching system
│   ├── util/                     # Utility functions
│   ├── dubbing_pipeline.py       # Main pipeline orchestrator
│   └── tts_processor.py          # TTS integration
├── main.py                       # GUI application entry
├── pyproject.toml               # Project configuration
└── outputs/                     # Generated files (created at runtime)
    ├── [video_name]/            # Per-video output directory
    │   ├── reference_audio/     # Reference audio segments
    │   ├── tts_output/          # Generated TTS audio
    │   ├── aligned_audio/       # Final aligned audio
    │   └── temp/                # Temporary files
    └── cache/                   # Global cache files
```

## Common Development Patterns

### Pipeline Processing Pattern
All processors follow this pattern:
```python
class CustomProcessor(StepProcessor):
    def process_task(self, task: Task) -> ProcessResult:
        try:
            # 1. Update status to processing
            self.notify_status_change(task.id, self.step_id, StepStatus.PROCESSING)
            
            # 2. Perform actual processing
            result = self._do_processing(task)
            
            # 3. Update status based on result
            if result.success:
                self.notify_status_change(task.id, self.step_id, StepStatus.COMPLETED)
            else:
                self.notify_status_change(task.id, self.step_id, StepStatus.FAILED)
                
            return result
        except Exception as e:
            self.notify_status_change(task.id, self.step_id, StepStatus.FAILED, str(e))
            return ProcessResult(success=False, error=str(e))
```

### Status Management
- Use thread-safe `StatusNotificationManager` for GUI updates
- Status progression: PENDING → PROCESSING → COMPLETED/FAILED
- Failed steps automatically reset subsequent steps to PENDING

### Caching Strategy  
- Task-level caching with `TaskCacheManager`
- Resume from cache using `step_results` and `step_details`
- Cache invalidation on input file changes
- Per-video cache files: `{video_name}_pipeline_cache.json`

## Important Notes

### System Requirements
- FFmpeg must be installed and available in PATH
- CUDA 12.8+ compatible NVIDIA drivers for GPU acceleration
- Windows/Linux/macOS support (primary development on Windows)

### External Dependencies
- Index-TTS Gradio API server (default: http://127.0.0.1:7860)
- TTS functionality depends on external API availability
- Demucs model files downloaded automatically on first use

### Threading and Concurrency
- GUI runs on main thread with Qt event loop
- Background processing uses `StreamlineBatchDubbingWorkerThread`
- Pipeline uses per-step `ThreadPoolExecutor` instances
- Thread-safe status updates via `StatusNotificationManager`

### File Handling
- Uses `pathlib.Path` for cross-platform compatibility
- Automatic output directory creation
- Safe filename sanitization with `sanitize_filename()`
- All text processing optimized for Chinese and English content
- Unicode normalization to prevent encoding issues

## Debugging Tips

### Common Issues
1. **GUI not responding**: Check if background worker thread is properly connected
2. **Status not updating**: Verify `step_status_changed` signal connections
3. **Cache inconsistency**: Use cache repair functionality or delete cache files
4. **TTS failures**: Verify Index-TTS API server is running and accessible

### Debug Commands
```bash
# Enable debug logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG); from main import main; main()"

# Test individual processors
uv run python -c "from core.pipeline.processors import PreprocessSubtitleProcessor; processor = PreprocessSubtitleProcessor(); print('Processor loaded successfully')"
```