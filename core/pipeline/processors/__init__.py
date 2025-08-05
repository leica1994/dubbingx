"""
具体步骤处理器包

包含8个具体的处理步骤实现：
1. PreprocessSubtitleProcessor - 字幕预处理
2. SeparateMediaProcessor - 媒体分离  
3. GenerateReferenceAudioProcessor - 生成参考音频
4. GenerateTTSProcessor - TTS语音生成
5. AlignAudioProcessor - 音频对齐
6. GenerateAlignedSrtProcessor - 生成对齐字幕
7. ProcessVideoSpeedProcessor - 视频速度调整
8. MergeAudioVideoProcessor - 音视频合并
"""

from .preprocess_subtitle_processor import PreprocessSubtitleProcessor
from .separate_media_processor import SeparateMediaProcessor
from .generate_reference_audio_processor import GenerateReferenceAudioProcessor
from .generate_tts_processor import GenerateTTSProcessor
from .align_audio_processor import AlignAudioProcessor
from .generate_aligned_srt_processor import GenerateAlignedSrtProcessor
from .process_video_speed_processor import ProcessVideoSpeedProcessor
from .merge_audio_video_processor import MergeAudioVideoProcessor

__all__ = [
    'PreprocessSubtitleProcessor',
    'SeparateMediaProcessor', 
    'GenerateReferenceAudioProcessor',
    'GenerateTTSProcessor',
    'AlignAudioProcessor',
    'GenerateAlignedSrtProcessor',
    'ProcessVideoSpeedProcessor',
    'MergeAudioVideoProcessor',
]